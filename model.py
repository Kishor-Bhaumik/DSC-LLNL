import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
from dataloader import create_split_dataloaders  # Assuming this is your custom dataloader module
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score

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

class Unet_Image(nn.Module):
    def __init__(self, in_channels=4, mask_content_preds=False):
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
        
        self.up4_amodal_mask = conv2d_inplace_spatial(64, 1, self.upscale_2, activation=nn.Identity())
        
        # Optional arguments
        self.mask_content_preds = mask_content_preds

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        
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
        
        logits_amodal_mask = self.up4_amodal_mask(h1234)
        return logits_amodal_mask
    
    def forward(self, rgb_image, modal_mask):
        """
        Forward pass adapted for our training pipeline
        
        Args:
            rgb_image: RGB image tensor (B, 3, H, W)
            modal_mask: Modal mask tensor (B, H, W)
        
        Returns:
            logits_amodal_mask: Amodal mask logits (B, 1, H, W)
        """
        # Add channel dimension to modal mask
        modal_mask = modal_mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Concatenate RGB image and modal mask
        model_input = torch.cat([rgb_image, modal_mask], dim=1)  # (B, 4, H, W)
        
        # Encode-decode
        x1, x2, x3, x4 = self.encode(model_input)
        logits_amodal_mask = self.decode(x1, x2, x3, x4)

        return logits_amodal_mask

class MetricsCalculator:
    @staticmethod
    def calculate_miou(pred_mask, true_mask, threshold=0.5):
        """Calculate mean Intersection over Union"""
        pred_binary = (pred_mask > threshold).float()
        true_binary = true_mask.float()
        
        intersection = (pred_binary * true_binary).sum(dim=[1, 2])
        union = pred_binary.sum(dim=[1, 2]) + true_binary.sum(dim=[1, 2]) - intersection
        
        # Avoid division by zero
        iou = intersection / (union + 1e-8)
        return iou.mean().item()
    
    @staticmethod
    def calculate_accuracy(pred_mask, true_mask, threshold=0.5):
        """Calculate pixel-wise accuracy"""
        pred_binary = (pred_mask > threshold).float()
        true_binary = true_mask.float()
        
        correct = (pred_binary == true_binary).float()
        accuracy = correct.mean().item()
        return accuracy
    
    @staticmethod
    def calculate_jaccard_f1(pred_mask, true_mask, threshold=0.5):
        """Calculate Jaccard and F1 scores using sklearn"""
        pred_binary = (pred_mask > threshold).cpu().numpy().flatten()
        true_binary = true_mask.cpu().numpy().flatten()
        
        jaccard = jaccard_score(true_binary, pred_binary, average='binary', zero_division=0)
        f1 = f1_score(true_binary, pred_binary, average='binary', zero_division=0)
        
        return jaccard, f1

class AmodalSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', use_logger=False, 
                 project_name='amodal-segmentation', run_name=None, learning_rate=1e-4):
        """
        Trainer for U-Net amodal segmentation model
        
        Args:
            model: Unet_Image instance
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
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Initialize wandb if needed
        if self.use_logger:
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    'model': 'Unet_Image',
                    'optimizer': 'Adam',
                    'learning_rate': learning_rate,
                    'loss_function': 'BCEWithLogitsLoss',
                    'device': device
                }
            )
            wandb.watch(self.model)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_miou = 0.0
        total_accuracy = 0.0
        total_jaccard = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        for batch_idx, (images, modal_masks, amodal_masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            modal_masks = modal_masks.to(self.device)
            amodal_masks = amodal_masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images, modal_masks)  # (B, 1, H, W)
            
            # Calculate loss - squeeze to match amodal_masks shape (B, H, W)
            loss = self.criterion(logits.squeeze(1), amodal_masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
                
                miou = self.metrics_calc.calculate_miou(pred_probs, amodal_masks)
                accuracy = self.metrics_calc.calculate_accuracy(pred_probs, amodal_masks)
                jaccard, f1 = self.metrics_calc.calculate_jaccard_f1(pred_probs, amodal_masks)
                
                total_loss += loss.item()
                total_miou += miou
                total_accuracy += accuracy
                total_jaccard += jaccard
                total_f1 += f1
                num_batches += 1
            
            # Log batch metrics if using wandb
            if self.use_logger and batch_idx % 10 == 0:
                wandb.log({
                    'batch_train_loss': loss.item(),
                    'batch_train_miou': miou,
                    'batch_train_accuracy': accuracy,
                    'batch_train_jaccard': jaccard,
                    'batch_train_f1': f1,
                    'batch_idx': batch_idx
                })
        
        # Return average metrics
        return {
            'train_loss': total_loss / num_batches,
            'train_miou': total_miou / num_batches,
            'train_accuracy': total_accuracy / num_batches,
            'train_jaccard': total_jaccard / num_batches,
            'train_f1': total_f1 / num_batches
        }
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        total_accuracy = 0.0
        total_jaccard = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, modal_masks, amodal_masks in self.val_loader:
                images = images.to(self.device)
                modal_masks = modal_masks.to(self.device)
                amodal_masks = amodal_masks.to(self.device)
                
                # Forward pass
                logits = self.model(images, modal_masks)
                
                # Calculate loss
                loss = self.criterion(logits.squeeze(1), amodal_masks)
                
                # Calculate metrics
                pred_probs = torch.sigmoid(logits.squeeze(1))
                
                miou = self.metrics_calc.calculate_miou(pred_probs, amodal_masks)
                accuracy = self.metrics_calc.calculate_accuracy(pred_probs, amodal_masks)
                jaccard, f1 = self.metrics_calc.calculate_jaccard_f1(pred_probs, amodal_masks)
                
                total_loss += loss.item()
                total_miou += miou
                total_accuracy += accuracy
                total_jaccard += jaccard
                total_f1 += f1
                num_batches += 1
        
        # Return average metrics
        return {
            'val_loss': total_loss / num_batches,
            'val_miou': total_miou / num_batches,
            'val_accuracy': total_accuracy / num_batches,
            'val_jaccard': total_jaccard / num_batches,
            'val_f1': total_f1 / num_batches
        }
    
    def train(self, num_epochs, validate_every_n_epochs=1):
        """Train the model for specified number of epochs"""
        best_val_miou = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Print training metrics
            print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                  f"mIoU: {train_metrics['train_miou']:.4f}, "
                  f"Acc: {train_metrics['train_accuracy']:.4f}, "
                  f"J&F: {train_metrics['train_jaccard']:.4f}/{train_metrics['train_f1']:.4f}")
            
            # Validate only every N epochs
            if (epoch + 1) % validate_every_n_epochs == 0:
                val_metrics = self.validate_epoch()
                
                print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                      f"mIoU: {val_metrics['val_miou']:.4f}, "
                      f"Acc: {val_metrics['val_accuracy']:.4f}, "
                      f"J&F: {val_metrics['val_jaccard']:.4f}/{val_metrics['val_f1']:.4f}")
                
                # Log to wandb
                if self.use_logger:
                    wandb.log({
                        'epoch': epoch + 1,
                        **train_metrics,
                        **val_metrics
                    })
                
                # Save best model
                if val_metrics['val_miou'] > best_val_miou:
                    best_val_miou = val_metrics['val_miou']
                    torch.save(self.model.state_dict(), 'best_unet_amodal_model.pth')
                    print(f"✓ New best model saved! mIoU: {best_val_miou:.4f}")
                    
                    # Log best model to wandb
                    if self.use_logger:
                        wandb.log({'best_val_miou': best_val_miou})
            else:
                # Log only training metrics when not validating
                if self.use_logger:
                    wandb.log({
                        'epoch': epoch + 1,
                        **train_metrics
                    })
        
        print(f"\nTraining completed! Best validation mIoU: {best_val_miou:.4f}")
        
        if self.use_logger:
            wandb.finish()
    
    def visualize_predictions(self, num_samples=4, save_path=None):
        """Visualize model predictions"""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            images, modal_masks, amodal_masks = next(iter(self.val_loader))
            images = images.to(self.device)
            modal_masks = modal_masks.to(self.device)
            amodal_masks = amodal_masks.to(self.device)
            
            # Get predictions
            logits = self.model(images, modal_masks)
            pred_probs = torch.sigmoid(logits.squeeze(1))
            
            # Move to CPU for visualization
            images = images.cpu()
            modal_masks = modal_masks.cpu()
            amodal_masks = amodal_masks.cpu()
            pred_probs = pred_probs.cpu()
            
            # Plot samples
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
            
            for i in range(min(num_samples, images.shape[0])):
                # RGB Image (denormalize ImageNet)
                img = images[i].permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('RGB Image')
                axes[i, 0].axis('off')
                
                # Modal Mask
                axes[i, 1].imshow(modal_masks[i].numpy(), cmap='gray')
                axes[i, 1].set_title('Modal Mask (Input)')
                axes[i, 1].axis('off')
                
                # Ground Truth Amodal Mask
                axes[i, 2].imshow(amodal_masks[i].numpy(), cmap='gray')
                axes[i, 2].set_title('GT Amodal Mask')
                axes[i, 2].axis('off')
                
                # Predicted Amodal Mask
                axes[i, 3].imshow(pred_probs[i].numpy(), cmap='gray')
                axes[i, 3].set_title('Pred Amodal Mask')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            else:
                plt.show()
            
            # Log to wandb if enabled
            if self.use_logger:
                wandb.log({"predictions": wandb.Image(fig)})
            
            plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Assuming the dataloader module is available
    # from your_dataloader_module import create_split_dataloaders
    
    # Create dataloaders
    dataloaders = create_split_dataloaders(
        data_path_file='data_path.txt',
        batch_size=8,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        verbose=False,  # Disable verbose for training
        save_plots=False,
        random_seed=42
    )
    
    # Create U-Net model
    model = Unet_Image(in_channels=4, mask_content_preds=False)
    
    # Create trainer with wandb logging
    trainer = AmodalSegmentationTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_logger=True,  # Enable wandb logging
        project_name='amodal-segmentation-unet',
        run_name='unet-experiment-2',
        learning_rate=1e-4
    )

    # Train the model with validation every 4 epochs
    trainer.train(num_epochs=10, validate_every_n_epochs=4)
    
    # Visualize predictions
    trainer.visualize_predictions(num_samples=4, save_path='predictions.png')
    
    # Test the trained model
    print("\n" + "="*50)
    print("Testing on test set...")
    
    test_loader = dataloaders['test']
    model.eval()
    test_metrics = trainer.validate_epoch()  # Can reuse validation function
    print(f"Test Results - mIoU: {test_metrics['val_miou']:.4f}, "
          f"Accuracy: {test_metrics['val_accuracy']:.4f}, "
          f"Jaccard: {test_metrics['val_jaccard']:.4f}, "
          f"F1: {test_metrics['val_f1']:.4f}")
