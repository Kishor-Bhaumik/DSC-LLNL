

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

class ImageMaskDataset(Dataset):
    def __init__(self, data_path_file, transform=None, verbose=False, save_plots=False, plot_save_dir='./plots'):
        """
        Dataset for loading RGB images with modal and amodal masks.
        
        Args:
            data_path_file (str): Path to the text file containing data paths
            transform (callable, optional): Transform to apply to RGB images
            verbose (bool): If True, plot images during iteration
            save_plots (bool): If True, save plots instead of displaying them
            plot_save_dir (str): Directory to save plots when save_plots=True
        """
        self.data_paths = []
        self.transform = transform
        self.verbose = verbose
        self.save_plots = save_plots
        self.plot_save_dir = plot_save_dir
        
        # Create plot directory if saving plots
        if self.save_plots:
            import os
            os.makedirs(self.plot_save_dir, exist_ok=True)
        
        # Read data paths from text file
        with open(data_path_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split(',')
                    if len(parts) == 4:
                        modal_path, amodal_path, rgb_path, obj_id = parts
                        self.data_paths.append({
                            'modal_path': modal_path.strip(),
                            'amodal_path': amodal_path.strip(),
                            'rgb_path': rgb_path.strip(),
                            'obj_id': int(obj_id.strip())
                        })
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_info = self.data_paths[idx]
        
        # Load RGB image
        rgb_image = Image.open(data_info['rgb_path']).convert('RGB')
        
        # Get target size from transform or use original size
        if self.transform:
            # Extract target size from transforms
            target_size = None
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    target_size = t.size if isinstance(t.size, (list, tuple)) else (t.size, t.size)
                    break
            if target_size is None:
                target_size = rgb_image.size[::-1]  # (H, W)
        else:
            target_size = rgb_image.size[::-1]  # (H, W)
        
        # Load and process modal mask
        modal_mask_img = Image.open(data_info['modal_path'])
        modal_mask_np = np.array(modal_mask_img)
        modal_mask_binary = np.where(modal_mask_np == data_info['obj_id'], 1, 0)
        
        # Resize modal mask to match image size
        modal_mask_pil = Image.fromarray(modal_mask_binary.astype(np.uint8))
        modal_mask_resized = modal_mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
        modal_mask = torch.from_numpy(np.array(modal_mask_resized)).float()
        
        # Load and process amodal mask
        amodal_mask_img = Image.open(data_info['amodal_path'])
        amodal_mask_np = np.array(amodal_mask_img) / 255.0  # Convert 0,255 to 0,1

        if len(amodal_mask_np.shape) == 2:
            amodal_mask_pil = Image.fromarray((amodal_mask_np * 255).astype(np.uint8))
            amodal_mask_resized = amodal_mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
            amodal_mask = torch.from_numpy(np.array(amodal_mask_resized) / 255.0).float()
        
        else:
            amodal_mask_img = Image.open(data_info['amodal_path']).convert('RGB')
            if self.transform:
                amodal_mask = self.transform(amodal_mask_img)
            else:
                amodal_mask = transforms.ToTensor()(amodal_mask_img)

        # Apply transforms to RGB image
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            rgb_image = transforms.ToTensor()(rgb_image)
        
        # Verbose plotting
        if self.verbose:
            if self.save_plots:
                self.save_sample_plot(rgb_image, modal_mask, amodal_mask, idx)
            else:
                self.plot_sample(rgb_image, modal_mask, amodal_mask, idx)
        
        return rgb_image, modal_mask, amodal_mask
    
    def save_sample_plot(self, rgb_image, modal_mask, amodal_mask, idx):
        """Save RGB image, modal mask, and amodal mask in subplots to file"""
        import os
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot RGB image
        if rgb_image.shape[0] == 3:  # If tensor is CxHxW
            rgb_np = rgb_image.permute(1, 2, 0).numpy()
            # Denormalize if ImageNet transforms were applied
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        mean = np.array(t.mean)
                        std = np.array(t.std)
                        rgb_np = rgb_np * std + mean
                        rgb_np = np.clip(rgb_np, 0, 1)
                        break
        else:
            rgb_np = rgb_image.numpy()
        
        axes[0].imshow(rgb_np)
        axes[0].set_title(f'RGB Image (Sample {idx})')
        axes[0].axis('off')
        
        # Plot modal mask
        axes[1].imshow(modal_mask.numpy(), cmap='gray')
        axes[1].set_title(f'Modal Mask (Obj ID: {self.data_paths[idx]["obj_id"]})')
        axes[1].axis('off')
        
        # Plot amodal mask
        axes[2].imshow(amodal_mask.numpy(), cmap='gray')
        axes[2].set_title('Amodal Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.plot_save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to prevent memory leaks
        print(f"Plot saved: {save_path}")

    def plot_sample(self, rgb_image, modal_mask, amodal_mask, idx):
        """Plot RGB image, modal mask, and amodal mask in subplots"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend temporarily
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot RGB image
        if rgb_image.shape[0] == 3:  # If tensor is CxHxW
            rgb_np = rgb_image.permute(1, 2, 0).numpy()
            # Denormalize if ImageNet transforms were applied
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        mean = np.array(t.mean)
                        std = np.array(t.std)
                        rgb_np = rgb_np * std + mean
                        rgb_np = np.clip(rgb_np, 0, 1)
                        break
        else:
            rgb_np = rgb_image.numpy()
        
        axes[0].imshow(rgb_np)
        axes[0].set_title(f'RGB Image (Sample {idx})')
        axes[0].axis('off')
        
        # Plot modal mask
        axes[1].imshow(modal_mask.numpy(), cmap='gray')
        axes[1].set_title(f'Modal Mask (Obj ID: {self.data_paths[idx]["obj_id"]})')
        axes[1].axis('off')
        
        # Plot amodal mask
        axes[2].imshow(amodal_mask.numpy(), cmap='gray')
        axes[2].set_title('Amodal Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Force display in Jupyter
        from IPython.display import display
        display(fig)
        plt.close(fig)  # Close to prevent memory leaks

def get_imagenet_transforms(image_size=224):
    """
    Get standard ImageNet transformations
    
    Args:
        image_size (int): Target image size for resizing
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_split_dataloaders(data_path_file, batch_size=16, shuffle=True, num_workers=4, 
                           image_size=224, verbose=False, save_plots=False, plot_save_dir='./plots',
                           train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42):
    """
    Create train/val/test DataLoaders for the image-mask dataset
    
    Args:
        data_path_file (str): Path to the text file containing data paths
        batch_size (int): Batch size for the DataLoaders
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        image_size (int): Target image size for resizing
        verbose (bool): If True, plot images during iteration
        save_plots (bool): If True, save plots instead of displaying them
        plot_save_dir (str): Directory to save plots when save_plots=True
        train_ratio (float): Proportion of data for training (default: 0.7)
        val_ratio (float): Proportion of data for validation (default: 0.1)
        test_ratio (float): Proportion of data for testing (default: 0.2)
        random_seed (int): Random seed for reproducible splits
    
    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    
    # Get ImageNet transforms
    transform = get_imagenet_transforms(image_size)
    
    # Create full dataset
    full_dataset = ImageMaskDataset(
        data_path_file=data_path_file,
        transform=transform,
        verbose=verbose,
        save_plots=save_plots,
        plot_save_dir=plot_save_dir
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Remaining samples go to test
    
    print(f"Dataset split:")
    print(f"  Total samples: {total_size}")
    print(f"  Train: {train_size} ({train_size/total_size:.1%})")
    print(f"  Val: {val_size} ({val_size/total_size:.1%})")
    print(f"  Test: {test_size} ({test_size/total_size:.1%})")
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'val': DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }
    
    return dataloaders

# Example usage:
if __name__ == "__main__":
    # Create split dataloaders with plot saving enabled
    dataloaders = create_split_dataloaders(
        data_path_file='data_path12.txt',
        batch_size=4,
        shuffle=True,
        verbose=False,       # Enable visualization
        save_plots=False,    # Save plots instead of displaying
        plot_save_dir='./sample_plots',  # Directory to save plots
        train_ratio=0.7,    # 70% for training
        val_ratio=0.1,      # 10% for validation
        test_ratio=0.2,     # 20% for testing
        random_seed=42      # For reproducible splits
    )
    
    # Access individual dataloaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Example: Iterate through train dataloader
    print("\n=== Training Data ===")
    for batch_idx, (images, modal_masks, amodal_masks) in enumerate(train_loader):
        print(f"Train Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Modal masks shape: {modal_masks.shape}")
        print(f"  Amodal masks shape: {amodal_masks.shape}")
        
        # Break after a few batches for demonstration
        if batch_idx >= 1:
            break
    
    # Example: Iterate through validation dataloader
    print("\n=== Validation Data ===")
    for batch_idx, (images, modal_masks, amodal_masks) in enumerate(val_loader):
        print(f"Val Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Modal masks shape: {modal_masks.shape}")
        print(f"  Amodal masks shape: {amodal_masks.shape}")
        
        # Break after a few batches for demonstration
        if batch_idx >= 1:
            break
    
    # Example: Iterate through test dataloader
    print("\n=== Test Data ===")
    for batch_idx, (images, modal_masks, amodal_masks) in enumerate(test_loader):
        print(f"Test Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Modal masks shape: {modal_masks.shape}")
        print(f"  Amodal masks shape: {amodal_masks.shape}")
        
        # Break after a few batches for demonstration
        if batch_idx >= 1:
            break
    
