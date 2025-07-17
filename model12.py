import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb,pdb
from dataloader import create_split_dataloaders  # Assuming this is your custom dataloader module
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# U-Net Implementation adapted for RGB content prediction
class conv2d_inplace_spatial(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, pooling_function, activation = nn.GELU()):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            activation,
            pooling_function,
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Upscale(nn.Module):
    def __init__(self, scale_factor=(2, 2), mode='bilinear', align_corners=False):
        super(Upscale, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class Unet_RGB_Content(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.mpool_2 = nn.MaxPool2d((2, 2))

        self.down1 = conv2d_inplace_spatial(in_channels, 32, self.mpool_2)
        self.down2 = conv2d_inplace_spatial(32, 64, self.mpool_2)
        self.down3 = conv2d_inplace_spatial(64, 128, self.mpool_2)
        self.down4 = conv2d_inplace_spatial(128, 256, self.mpool_2)

        self.upscale_2 = Upscale(scale_factor=(2, 2), mode='bilinear', align_corners=False)

        self.up1 = conv2d_inplace_spatial(256, 128, self.upscale_2)
        self.up2 = conv2d_inplace_spatial(256, 64, self.upscale_2)
        self.up3 = conv2d_inplace_spatial(128, 32, self.upscale_2)
        
        # Changed from 1 channel (binary mask) to 3 channels (RGB content)
        self.up4_amodal_content = conv2d_inplace_spatial(64, 3, self.upscale_2, activation=nn.Identity())
        
        # Loss functions for RGB regression
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def encode(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x1, x2, x3, x4
    
    def decode(self, h1, h2, h3, h4):
        h4 = self.up1(h4)  # Upsample and reduce channels
        h34 = torch.cat((h3, h4), dim=1)  # Skip connection

        h34 = self.up2(h34)
        h234 = torch.cat((h2, h34), dim=1)  # Skip connection

        h234 = self.up3(h234)
        h1234 = torch.cat((h1, h234), dim=1)  # Skip connection
        
        # Output RGB content (no activation - raw values)
        amodal_content = self.up4_amodal_content(h1234)
        return amodal_content
    
    def forward(self, rgb_image, modal_mask):
        """
        Forward pass for RGB amodal content prediction
        
        Args:
            rgb_image: RGB image tensor (B, 3, H, W)
            modal_mask: Modal mask tensor (B, H, W) or (B, 1, H, W)
        
        Returns:
            amodal_content: RGB amodal content (B, 3, H, W)
        """
        # Ensure modal mask has channel dimension
        if modal_mask.dim() == 3:  # (B, H, W)
            modal_mask = modal_mask.unsqueeze(1)  # (B, 1, H, W)
        
        # Concatenate RGB image and modal mask
        model_input = torch.cat([rgb_image, modal_mask], dim=1)  # (B, 4, H, W)
        
        # Encode-decode
        x1, x2, x3, x4 = self.encode(model_input)
        amodal_content = self.decode(x1, x2, x3, x4)

        return amodal_content

class RGBMetricsCalculator:
    @staticmethod
    def calculate_psnr(pred_rgb, true_rgb):
        """Calculate Peak Signal-to-Noise Ratio"""
        # Convert to numpy and move to CPU
        pred_np = pred_rgb.detach().cpu().numpy()
        true_np = true_rgb.detach().cpu().numpy()
        
        psnr_values = []
        for i in range(pred_np.shape[0]):  # Batch dimension
            # Transpose from (C, H, W) to (H, W, C) for skimage
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            true_img = np.transpose(true_np[i], (1, 2, 0))
            
            # Clip values to [0, 1] range
            pred_img = np.clip(pred_img, 0, 1)
            true_img = np.clip(true_img, 0, 1)
            
            psnr_val = psnr(true_img, pred_img, data_range=1.0)
            psnr_values.append(psnr_val)
        
        return np.mean(psnr_values)
    
    @staticmethod
    def calculate_ssim(pred_rgb, true_rgb):
        """Calculate Structural Similarity Index"""
        pred_np = pred_rgb.detach().cpu().numpy()
        true_np = true_rgb.detach().cpu().numpy()
        
        ssim_values = []
        for i in range(pred_np.shape[0]):
            # Transpose from (C, H, W) to (H, W, C)
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            true_img = np.transpose(true_np[i], (1, 2, 0))
            
            # Clip values to [0, 1] range
            pred_img = np.clip(pred_img, 0, 1)
            true_img = np.clip(true_img, 0, 1)
            
            ssim_val = ssim(true_img, pred_img, data_range=1.0, channel_axis=2)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    @staticmethod
    def calculate_mse(pred_rgb, true_rgb):
        """Calculate Mean Squared Error"""
        mse = F.mse_loss(pred_rgb, true_rgb)
        return mse.item()
    
    @staticmethod
    def calculate_mae(pred_rgb, true_rgb):
        """Calculate Mean Absolute Error"""
        mae = F.l1_loss(pred_rgb, true_rgb)
        return mae.item()

class AmodalContentTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', use_logger=False, 
                 project_name='amodal-rgb-content', run_name=None, learning_rate=1e-4):
        """
        Trainer for U-Net RGB amodal content prediction model
        
        Args:
            model: Unet_RGB_Content instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to run on ('cuda' or 'cpu')
            use_logger: Whether to use wandb logging
            project_name: WandB project name
            run_name: WandB run name
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_logger = use_logger
        
        # Loss function - using L1 loss for better RGB reconstruction
        self.criterion = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Metrics calculator
        self.metrics_calc = RGBMetricsCalculator()
        
        # Initialize wandb if needed
        if self.use_logger:
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    'model': 'Unet_RGB_Content',
                    'task': 'Task_1.2_RGB_Amodal_Content',
                    'optimizer': 'Adam',
                    'learning_rate': learning_rate,
                    'loss_function': 'L1Loss',
                    'device': device
                }
            )
            wandb.watch(self.model)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_idx, (images, modal_masks, amodal_content) in enumerate(self.train_loader):
            images = images.to(self.device)
            modal_masks = modal_masks.to(self.device)
            amodal_content = amodal_content.to(self.device)  # Now RGB content instead of binary mask
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_content = self.model(images, modal_masks)  # (B, 3, H, W)
            loss = self.criterion(pred_content, amodal_content)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                psnr_val = self.metrics_calc.calculate_psnr(pred_content, amodal_content)
                ssim_val = self.metrics_calc.calculate_ssim(pred_content, amodal_content)
                mse_val = self.metrics_calc.calculate_mse(pred_content, amodal_content)
                mae_val = self.metrics_calc.calculate_mae(pred_content, amodal_content)
                
                total_loss += loss.item()
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_mse += mse_val
                total_mae += mae_val
                num_batches += 1
            
            # Log batch metrics if using wandb
            if self.use_logger and batch_idx % 10 == 0:
                wandb.log({
                    'batch_train_loss': loss.item(),
                    'batch_train_psnr': psnr_val,
                    'batch_train_ssim': ssim_val,
                    'batch_train_mse': mse_val,
                    'batch_train_mae': mae_val,
                    'batch_idx': batch_idx
                })
        
        # Return average metrics
        return {
            'train_loss': total_loss / num_batches,
            'train_psnr': total_psnr / num_batches,
            'train_ssim': total_ssim / num_batches,
            'train_mse': total_mse / num_batches,
            'train_mae': total_mae / num_batches
        }
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, modal_masks, amodal_content in self.val_loader:
                images = images.to(self.device)
                modal_masks = modal_masks.to(self.device)
                amodal_content = amodal_content.to(self.device)
                
                # Forward pass
                pred_content = self.model(images, modal_masks)
                
                # Calculate loss
                loss = self.criterion(pred_content, amodal_content)
                
                # Calculate metrics
                psnr_val = self.metrics_calc.calculate_psnr(pred_content, amodal_content)
                ssim_val = self.metrics_calc.calculate_ssim(pred_content, amodal_content)
                mse_val = self.metrics_calc.calculate_mse(pred_content, amodal_content)
                mae_val = self.metrics_calc.calculate_mae(pred_content, amodal_content)
                
                total_loss += loss.item()
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_mse += mse_val
                total_mae += mae_val
                num_batches += 1
        
        # Return average metrics
        return {
            'val_loss': total_loss / num_batches,
            'val_psnr': total_psnr / num_batches,
            'val_ssim': total_ssim / num_batches,
            'val_mse': total_mse / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    def train(self, num_epochs, validate_every_n_epochs=1):
        """Train the model for specified number of epochs"""
        best_val_psnr = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Print training metrics
            print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                  f"PSNR: {train_metrics['train_psnr']:.2f}, "
                  f"SSIM: {train_metrics['train_ssim']:.4f}, "
                  f"MSE: {train_metrics['train_mse']:.4f}, "
                  f"MAE: {train_metrics['train_mae']:.4f}")
            
            # Validate only every N epochs
            if (epoch + 1) % validate_every_n_epochs == 0:
                val_metrics = self.validate_epoch()
                
                print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                      f"PSNR: {val_metrics['val_psnr']:.2f}, "
                      f"SSIM: {val_metrics['val_ssim']:.4f}, "
                      f"MSE: {val_metrics['val_mse']:.4f}, "
                      f"MAE: {val_metrics['val_mae']:.4f}")
                
                # Log to wandb
                if self.use_logger:
                    wandb.log({
                        'epoch': epoch + 1,
                        **train_metrics,
                        **val_metrics
                    })
                
                # Save best model based on PSNR
                if val_metrics['val_psnr'] > best_val_psnr:
                    best_val_psnr = val_metrics['val_psnr']
                    torch.save(self.model.state_dict(), 'best_unet_rgb_content_model.pth')
                    print(f"✓ New best model saved! PSNR: {best_val_psnr:.2f}")
                    
                    # Log best model to wandb
                    if self.use_logger:
                        wandb.log({'best_val_psnr': best_val_psnr})
            else:
                # Log only training metrics when not validating
                if self.use_logger:
                    wandb.log({
                        'epoch': epoch + 1,
                        **train_metrics
                    })
        
        print(f"\nTraining completed! Best validation PSNR: {best_val_psnr:.2f}")
    
    def finish_logging(self):
        """Finish wandb logging session"""
        if self.use_logger and wandb.run is not None:
            wandb.finish()
    
    def visualize_predictions(self, num_samples=4, save_path=None):
        """Visualize RGB content predictions"""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            images, modal_masks, amodal_content = next(iter(self.val_loader))
            images = images.to(self.device)
            modal_masks = modal_masks.to(self.device)
            amodal_content = amodal_content.to(self.device)
            
            # Get predictions
            pred_content = self.model(images, modal_masks)
            
            # Move to CPU for visualization
            images = images.cpu()
            modal_masks = modal_masks.cpu()
            amodal_content = amodal_content.cpu()
            pred_content = pred_content.cpu()
            
            # Plot samples
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
            
            for i in range(min(num_samples, images.shape[0])):
                # RGB Image (denormalize ImageNet if needed)
                img = images[i].permute(1, 2, 0).numpy()
                # Assuming images are already in [0,1] range or properly normalized
                img = np.clip(img, 0, 1)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('RGB Image')
                axes[i, 0].axis('off')
                
                # Modal Mask
                if modal_masks[i].dim() == 3:  # (1, H, W)
                    mask_vis = modal_masks[i][0].numpy()
                else:  # (H, W)
                    mask_vis = modal_masks[i].numpy()
                axes[i, 1].imshow(mask_vis, cmap='gray')
                axes[i, 1].set_title('Modal Mask (Input)')
                axes[i, 1].axis('off')
                
                # Ground Truth Amodal RGB Content
                gt_content = amodal_content[i].permute(1, 2, 0).numpy()
                gt_content = np.clip(gt_content, 0, 1)
                axes[i, 2].imshow(gt_content)
                axes[i, 2].set_title('GT Amodal Content')
                axes[i, 2].axis('off')
                
                # Predicted Amodal RGB Content
                pred_vis = pred_content[i].permute(1, 2, 0).numpy()
                pred_vis = np.clip(pred_vis, 0, 1)
                axes[i, 3].imshow(pred_vis)
                axes[i, 3].set_title('Pred Amodal Content')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            else:
                plt.show()
            
            # Log to wandb if enabled
            if self.use_logger and wandb.run is not None:
                wandb.log({"rgb_predictions": wandb.Image(fig)})
            
            plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Create dataloaders - NOW USING data_path12.txt for RGB content
    dataloaders = create_split_dataloaders(
        data_path_file='data_path12.txt',  # Changed to RGB content dataset
        batch_size=8,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        verbose=False,
        save_plots=False,
        random_seed=42
    )
    
    # Create U-Net model for RGB content prediction
    model = Unet_RGB_Content(in_channels=4)
    
    # Create trainer with wandb logging
    trainer = AmodalContentTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_logger=False,  # Set to True to enable wandb logging
        project_name='amodal-rgb-content-unet',
        run_name='unet-rgb-experiment-1',
        learning_rate=1e-4
    )
    
    # Train the model
    trainer.train(num_epochs=20, validate_every_n_epochs=2)
    
    # Visualize RGB content predictions
    trainer.visualize_predictions(num_samples=4, save_path='rgb_content_predictions.png')
    
    # Finish wandb logging
    trainer.finish_logging()
    
    # Test the trained model
    print("\n" + "="*50)
    print("Testing on test set...")
    
    test_loader = dataloaders['test']
    model.eval()
    
    # Test using validation function (reuse for test set)
    trainer.val_loader = test_loader  # Temporarily change to test loader
    test_metrics = trainer.validate_epoch()
    print(f"Test Results - PSNR: {test_metrics['val_psnr']:.2f}, "
          f"SSIM: {test_metrics['val_ssim']:.4f}, "
          f"MSE: {test_metrics['val_mse']:.4f}, "
          f"MAE: {test_metrics['val_mae']:.4f}")
