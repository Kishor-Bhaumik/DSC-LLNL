import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb,os
from vid_dataloader import create_object_dataloaders
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from utils import MetricsCalculator, build_camera_spatial_graph, visualize_batch,  randomly_keep_tensors_vectorized
from gcn import CameraST_GCN
import pdb
from tqdm import tqdm
import argparse


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

class EncoderModule(nn.Module):
    """Standalone encoder module that can be extracted from the full model"""
    def __init__(self, in_channels=4, input_size=256):
        super().__init__()
        self.input_size = input_size
        self.mpool_2 = nn.MaxPool2d((2, 2))
        
        # Encoder layers
        self.down1 = conv2d_inplace_spatial(in_channels, 32, self.mpool_2)
        self.down2 = conv2d_inplace_spatial(32, 64, self.mpool_2)
        self.down3 = conv2d_inplace_spatial(64, 128, self.mpool_2)
        self.down4 = conv2d_inplace_spatial(128, 256, self.mpool_2)
        
        if input_size == 512:
            self.down5 = conv2d_inplace_spatial(256, 512, self.mpool_2)
            self.use_down5 = True
            bottleneck_in_channels = 512
        else:
            self.use_down5 = False
            bottleneck_in_channels = 256
            
        # Bottleneck compression
        self.bottleneck_compress = nn.Sequential(
            nn.Conv2d(bottleneck_in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )
    
    def forward(self, x):
        """Forward pass returning only the latent representation"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        if hasattr(self, 'use_down5') and self.use_down5:
            x5 = self.down5(x4)
            latent = self.bottleneck_compress(x5)
        else:
            latent = self.bottleneck_compress(x4)
            
        return latent
    
    def forward_with_intermediates(self, x):
        """Forward pass returning all intermediate features"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        if hasattr(self, 'use_down5') and self.use_down5:
            x5 = self.down5(x4)
            latent = self.bottleneck_compress(x5)
            return [x1, x2, x3, x4, x5, latent]
        else:
            latent = self.bottleneck_compress(x4)
            return [x1, x2, x3, x4, latent]


class Unet_Image_SmallLatent(nn.Module):
    def __init__(self, in_channels=4, mask_content_preds=False, input_size=256, gcn_layers=3,gcn_in_channel=588,
                 temporal_kernel_size=3, num_cameras=6, base_channels=64, use_edge_importance=True,window=6,use_stgcn=True):
        """
        Modified U-Net with smaller latent representation and extractable encoder
        
        Args:
            in_channels: Number of input channels (default: 4 for RGB+mask)
            mask_content_preds: Optional masking flag
            input_size: Expected input image size (assumes square images)
        """
        super().__init__()
        
        self.input_size = input_size
        self.latent_size = 16
        if use_stgcn:
            self.stgcn = CameraST_GCN(in_channels=gcn_in_channel, num_cameras=num_cameras, num_layers=gcn_layers,
                                    base_channels=base_channels,
                                    use_edge_importance=use_edge_importance,
                                    dropout=0.5, temporal_kernel_size=temporal_kernel_size, window=window)
        self.use_stgcn = use_stgcn

        # Create encoder as a separate module
        self.encoder = EncoderModule(in_channels, input_size)
        
        # Determine bottleneck channels based on encoder
        if hasattr(self.encoder, 'use_down5') and self.encoder.use_down5:
            bottleneck_in_channels = 512
        else:
            bottleneck_in_channels = 256
            
        # Bottleneck expand
        self.bottleneck_expand = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, bottleneck_in_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_in_channels),
            nn.GELU()
        )

        # Decoder path
        self.upscale_2 = Upscale(scale_factor=(2, 2), mode='bilinear', align_corners=False)
        
        if hasattr(self.encoder, 'use_down5') and self.encoder.use_down5:
            self.up0 = conv2d_inplace_spatial(512, 256, self.upscale_2)
            decoder_start_channels = 512  # 256 from up0 + 256 from skip
        else:
            decoder_start_channels = 256
            
        self.up1 = conv2d_inplace_spatial(decoder_start_channels, 128, self.upscale_2)
        self.up2 = conv2d_inplace_spatial(256, 64, self.upscale_2)  # 128 + 128 from skip
        self.up3 = conv2d_inplace_spatial(128, 32, self.upscale_2)  # 64 + 64 from skip
        
        self.up4_amodal_mask = conv2d_inplace_spatial(64, 1, self.upscale_2, activation=nn.Identity())
        
        # Optional arguments
        self.mask_content_preds = mask_content_preds

        
    def encode(self, x):
        """Encode input to latent representation using the encoder module"""
        return self.encoder.forward_with_intermediates(x)
    
    def decode(self, x1, x2, x3, x4, x5_or_latent, latent=None):
        """Decode from latent representation"""
        if latent is None:
            # 5-argument case
            latent = x5_or_latent
            x5 = None
        else:
            # 6-argument case
            x5 = x5_or_latent
            
        # Expand from 3 channels back to original channel count
        h = self.bottleneck_expand(latent)
        
        if x5 is not None:
            # Extra upsampling layer
            h = self.up0(h)
            h = torch.cat((x4, h), dim=1)
            h = self.up1(h)
        else:
            h = self.up1(h)
            
        h34 = torch.cat((x3, h), dim=1)
        h34 = self.up2(h34)
        
        h234 = torch.cat((x2, h34), dim=1)
        h234 = self.up3(h234)
        
        h1234 = torch.cat((x1, h234), dim=1)
        logits_amodal_mask = self.up4_amodal_mask(h1234)
        
        return logits_amodal_mask
    
    def forward(self,model_input, batch):
        """
        Forward pass adapted for training pipeline
        
        Args:
            rgb_image: RGB image tensor (B, 3, H, W)
            modal_mask: Modal mask tensor (B, H, W)
        
        Returns:
            logits_amodal_mask: Amodal mask logits (B, 1, H, W)
        """
        
        # Encode-decode
        encode_outputs = self.encode(model_input)
        if self.use_stgcn:
            latent = encode_outputs[-1]  
            latent = self.stgcn(latent, batch)
            encode_outputs[-1] = latent  # Update latent with ST-GCN output

        #import pdb;pdb.set_trace()  # Debugging breakpoint
        logits_amodal_mask = self.decode(*encode_outputs)

        return logits_amodal_mask

def get_data(rgb_batch, modal_batch, amodal_batch, device='cuda'):
    rgb_batch = rgb_batch.to(device)
    modal_batch = modal_batch.to(device)
    amodal_batch = amodal_batch.view(-1, 224,224).to(device)
    rgb_img = rgb_batch.view(-1, 3, 224, 224)
    modal_img_mask = modal_batch.view(-1, 224, 224)
    return rgb_img, modal_img_mask, amodal_batch, rgb_batch.size(0)


class AmodalSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device='cuda', use_logger=False, 
                 project_name='amodal-segmentation', run_name=None, learning_rate=1e-4, val_camera=6):
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
        self.test_loader = test_loader
        self.device = device
        self.use_logger = use_logger
        self.val_camera = val_camera
        self.mse_loss = nn.MSELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
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
        
        for batch_idx, (rgb_batch, modal_batchs, amodal_batchs, metadata_batch) in tqdm(enumerate(self.train_loader), 
                                                                                   total=len(self.train_loader),
                                                                                   desc="Training"):
            rgb_img, modal_img_mask, amodal_batch, B = get_data(rgb_batch, modal_batchs, amodal_batchs, self.device)

            logits = self.model(torch.cat([rgb_img, modal_img_mask.unsqueeze(1)], dim=1), B)
            loss= self.criterion(logits.squeeze(1), amodal_batch)
            ''' 
            if self.val_camera != 6:
                rgb_batchs, modal_batchs, amodal_less= \
                    randomly_keep_tensors_vectorized(rgb_batchs, modal_batchs, amodal_batchs, self.val_camera)
                
            
            rgb_img, modal_img_mask, amodal_batchs, B = get_data(rgb_batchs, modal_batchs, amodal_batchs, self.device)
            

            logits = self.model(torch.cat([rgb_img, modal_img_mask.unsqueeze(1)], dim=1), B)
            if self.val_camera != 6:
                b,hw= logits.shape[0], logits.shape[-1]
                win = self.window
                logits= logits.view(b,-1,win,hw, hw)
            loss= self.criterion(logits.squeeze(1), amodal_batchs if self.val_camera == 6 else amodal_less)
            '''
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
                
                miou = self.metrics_calc.calculate_miou(pred_probs, amodal_batch)
                accuracy = self.metrics_calc.calculate_accuracy(pred_probs, amodal_batch)
                jaccard, f1 = self.metrics_calc.calculate_jaccard_f1(pred_probs, amodal_batch)

                total_loss += loss.item()
                total_miou += miou
                total_accuracy += accuracy
                total_jaccard += jaccard
                total_f1 += f1
                num_batches += 1
            
            

            
        
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
            for batch_idx, (rgb_batch, modal_batchs, amodal_batchs, metadata_batch) in tqdm(enumerate(self.val_loader),
                                                                                   total=len(self.val_loader),
                                                                                   desc="Validating"):
                rgb_img, modal_img_mask, amodal_batch, B = get_data(rgb_batch, modal_batchs, amodal_batchs, self.device)
                

                logits = self.model(torch.cat([rgb_img, modal_img_mask.unsqueeze(1)], dim=1), B)
                
                # Calculate loss
                loss = self.criterion(logits.squeeze(1), amodal_batch)

                # Calculate metrics
                pred_probs = torch.sigmoid(logits.squeeze(1))

                miou = self.metrics_calc.calculate_miou(pred_probs, amodal_batch)
                accuracy = self.metrics_calc.calculate_accuracy(pred_probs, amodal_batch)
                jaccard, f1 = self.metrics_calc.calculate_jaccard_f1(pred_probs, amodal_batch)

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
    
    def test_epoch(self):
        """Test for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        total_accuracy = 0.0
        total_jaccard = 0.0
        total_f1 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (rgb_batch, modal_batchs, amodal_batchs, metadata_batch) in tqdm(enumerate(self.test_loader),
                                                                                total=len(self.test_loader),
                                                                                desc="Testing"):
                rgb_img, modal_img_mask, amodal_batch, B = get_data(rgb_batch, modal_batchs, amodal_batchs, self.device)
                
                logits = self.model(torch.cat([rgb_img, modal_img_mask.unsqueeze(1)], dim=1), B)
                
                # Calculate loss
                loss = self.criterion(logits.squeeze(1), amodal_batch)

                # Calculate metrics
                pred_probs = torch.sigmoid(logits.squeeze(1))

                miou = self.metrics_calc.calculate_miou(pred_probs, amodal_batch)
                accuracy = self.metrics_calc.calculate_accuracy(pred_probs, amodal_batch)
                jaccard, f1 = self.metrics_calc.calculate_jaccard_f1(pred_probs, amodal_batch)

                total_loss += loss.item()
                total_miou += miou
                total_accuracy += accuracy
                total_jaccard += jaccard
                total_f1 += f1
                num_batches += 1
        
        # Return average metrics
        return {
            'test_loss': total_loss / num_batches,
            'test_miou': total_miou / num_batches,
            'test_accuracy': total_accuracy / num_batches,
            'test_jaccard': total_jaccard / num_batches,
            'test_f1': total_f1 / num_batches
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
                    # torch.save(self.model.get_encoder().state_dict(), 'unet_encoder.pth')
                    # print(f"✓ New best model saved! mIoU: {best_val_miou:.4f}")
                    
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
        
        # Don't finish wandb here - let user control when to finish
        # if self.use_logger:
        #     wandb.finish()
    
    def finish_logging(self):
        """Finish wandb logging session"""
        if self.use_logger and wandb.run is not None:
            wandb.finish()
    
    def visualize_predictions(self, num_samples=4, save_path=None):
        """Visualize model predictions"""
        self.model.eval()
        
        with torch.no_grad():
            # Create an iterator from the DataLoader
            val_loader_iter = iter(self.test_loader)
            rgb_batch, modal_batchs, amodal_batchs, metadata_batch = next(iter( val_loader_iter))
            rgb_img, modal_img_mask, amodal_batch, B = get_data(rgb_batch, modal_batchs, amodal_batchs, self.device)
            logits = self.model(torch.cat([rgb_img, modal_img_mask.unsqueeze(1)], dim=1), B)
            pred_probs = torch.sigmoid(logits.squeeze(1))
            # Move to CPU for visualization
            
            images = rgb_img.cpu()
            modal_masks = modal_img_mask.cpu()
            amodal_masks = amodal_batch.cpu()
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
            if self.use_logger and wandb.run is not None:
                wandb.log({"predictions": wandb.Image(fig)})
            
            plt.close(fig)

# Example usage
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Train ST-GCN for Amodal Segmentation")
    parser.add_argument('--use_logger', type=bool, default=False, help='Enable wandb logging')
    parser.add_argument('--use_stgcn', type=bool, default=False, help='Use ST-GCN for temporal processing')
    parser.add_argument('--use_edge_importance',type=bool, default=True, help='Use edge importance weighting in ST-GCN')

    parser.add_argument('--val_camera', type=int, default=6, help='Number of cameras in the dataset')
    parser.add_argument('--gcn_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--window', type=int, default=5, help='Temporal window size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--val_epochs', type=int, default=4, help='Validate every N epochs')
    parser.add_argument('--temporal_kernel_size', type=int, default=3, help='Temporal kernel size for ST-GCN')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--stride', type=int, default=2, help='Stride for temporal convolution')
    parser.add_argument('--run_name', type=str, default='stgcn_experiment', help='WandB run name')
    args = parser.parse_args()
    # use seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    base_path='/home/kbhau001/LLNL_DSC_2025/MOVi-MC-AC/'

    # List all scene directories
    all_scenes = os.listdir(base_path)
    all_scenes = [d for d in all_scenes if os.path.isdir(os.path.join(base_path, d)) and len(d) > 20]

    # Split scenes (example)
    train_scenes = all_scenes[:-2]  # All but last 2
    val_scenes = all_scenes[-2:-1]  # Second last scene
    test_scenes = all_scenes[-1:]    # Last scene

    # Create object-centric dataloaders
    dataloaders = create_object_dataloaders(
        base_path=base_path,
        train_scenes=train_scenes,
        val_scenes=val_scenes, 
        test_scenes=test_scenes,
        window_size=args.window,
        stride=args.stride,
        batch_size=args.batch_size,   # 2 object sequences per batch
        shuffle=True,
        verbose=False,
        save_plots=False,
        plot_save_dir='./object_plots'
    )

    ####
    # img, modal , amodal, _ = next(iter(dataloaders['train']))
    # print(img.shape, modal.shape, amodal.shape)
    # img = img.view(-1, 3, 224, 224)
    # modal = modal.view(-1, 224, 224)
    # amodal = amodal.view(-1, 224, 224)
    # # save this pictues in subplots , there shouldbe 
    # visualize_batch(img, modal, amodal, max_display=4, save_path='sample_batch.png')
    # exit()



    
    # Create U-Net model
    model = Unet_Image_SmallLatent(in_channels=4, mask_content_preds=False, input_size=256, gcn_layers=args.gcn_layers,
                 gcn_in_channel=588,temporal_kernel_size=args.temporal_kernel_size, num_cameras=6, base_channels=64, 
                 use_edge_importance=True,window=args.window, use_stgcn=args.use_stgcn)

    # Create trainer with wandb logging
    trainer = AmodalSegmentationTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        device= args.device,
        use_logger= args.use_logger,  # Enable wandb logging
        project_name='stgcn',
        run_name=args.run_name,
        learning_rate=args.lr,
        val_camera = args.val_camera

    )
    
    # Train the model with validation every 5 epochs
    trainer.train(num_epochs=args.num_epochs, validate_every_n_epochs=args.val_epochs)

    # Visualize predictions (wandb session still active)
    trainer.visualize_predictions(num_samples=4, save_path='predictions.png')
    

    
    # Test the trained model
    print("\n" + "="*50)
    print("Testing on test set...")

    test_metrics = trainer.test_epoch()
    print(f"Test Results - Loss: {test_metrics['test_loss']:.4f}, "
        f"mIoU: {test_metrics['test_miou']:.4f}, "
        f"Accuracy: {test_metrics['test_accuracy']:.4f}, "
        f"Jaccard: {test_metrics['test_jaccard']:.4f}, "
        f"F1: {test_metrics['test_f1']:.4f}")

    # Log test results to wandb if enabled
    if args.use_logger:
        wandb.log({
            'test_loss': test_metrics['test_loss'],
            'test_miou': test_metrics['test_miou'],
            'test_accuracy': test_metrics['test_accuracy'],
            'test_jaccard': test_metrics['test_jaccard'],
            'test_f1': test_metrics['test_f1']
        })
        # Finish wandb logging
    trainer.finish_logging()