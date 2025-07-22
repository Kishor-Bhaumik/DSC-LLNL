

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import random


def select_kept_indices(tensor, kept_indices):
    """
    Select only the kept indices from dimension 1
    
    Args:
        tensor: Input tensor of shape [batch, 6, ...]
        kept_indices: List of lists with indices to keep for each batch
    
    Returns:
        selected_tensor: Tensor with only kept indices, shape [batch, len(kept_indices[0]), ...]
    """
    batch_size = tensor.shape[0]
    selected_tensors = []
    
    for batch_idx in range(batch_size):
        # Select the kept indices for this batch
        indices = kept_indices[batch_idx]
        selected = tensor[batch_idx, indices]  # Shape: [len(indices), ...]
        selected_tensors.append(selected)
    
    # Stack back into batch
    result = torch.stack(selected_tensors, dim=0)
    return result

def randomly_keep_tensors_vectorized(tensor,modal, amodal, keep):
    """
    Randomly keep 'keep' number of tensors - SAME indices for all batches
    
    Args:
        tensor: Input tensor of shape [batch, 6, ...]
        keep: Number of tensors to keep (1 to 5)
    
    Returns:
        modified_tensor: Tensor with (6-keep) tensors zeroed out
        kept_indices: List of same indices for all batches
    """
    batch_size = tensor.shape[0]
    total_tensors = tensor.shape[1]  # Should be 6
    
    # Create a copy to avoid modifying original tensor
    modified_tensor = tensor.clone()
    modified_modal = amodal.clone()
    
    # Generate random indices to keep ONCE for all batches
    all_indices = list(range(total_tensors))  # [0, 1, 2, 3, 4, 5]
    keep_idx = random.sample(all_indices, keep)  # Randomly select 'keep' indices
    keep_idx.sort()  # Sort for easier reading
    
    # Create mask for zeroing
    zero_indices = [idx for idx in all_indices if idx not in keep_idx]
    
    # Apply the same pattern to ALL batches
    for batch_idx in range(batch_size):
        # Zero out the tensors that are not kept
        for zero_idx in zero_indices:
            modified_tensor[batch_idx, zero_idx] = 0
            modified_modal[batch_idx, zero_idx] = 0  # Assuming amodal is also a tensor of shape [batch, 6, ...]

    # Return same indices for all batches
    kept_indices = [keep_idx for _ in range(batch_size)]

    #modal_less = select_kept_indices(modal, kept_indices)
    amodal_less = select_kept_indices(amodal, kept_indices)

    return modified_tensor, modified_modal, amodal_less



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

    
def build_camera_spatial_graph(num_cameras=6):
    """
    Build adjacency matrix for camera spatial connections only
    (Temporal connections handled by ST-GCN's temporal conv layers)
    
    Args:
        num_cameras: Number of cameras (spatial dimension)
    
    Returns:
        torch.Tensor: Normalized Adjacency matrix of shape (num_cameras, num_cameras)
    """
    # Initialize adjacency matrix
    adjacency = np.zeros((num_cameras, num_cameras))
    
    # 1. Self-connections
    for i in range(num_cameras):
        adjacency[i, i] = 1
    
    # 2. Spatial edges: Fully connect all cameras
    for cam1 in range(num_cameras):
        for cam2 in range(num_cameras):
            if cam1 != cam2:
                adjacency[cam1, cam2] = 1
    
    # Convert to torch tensor
    adj_tensor = torch.tensor(adjacency, dtype=torch.float32)

    # 3. Normalize the Adjacency Matrix
    # Calculate degree matrix D
    degree_matrix = torch.diag(torch.sum(adj_tensor, dim=1))
    # Compute D^(-1/2)
    degree_inv_sqrt = torch.pow(degree_matrix, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    
    # Calculate A_hat = D^(-1/2) * A * D^(-1/2)
    A_hat = torch.matmul(torch.matmul(degree_inv_sqrt, adj_tensor), degree_inv_sqrt)
    
    return A_hat


import matplotlib.pyplot as plt
import torch
import numpy as np
import math

def visualize_batch(img, modal, amodal, max_display=8, cols_per_type=1, save_path=None):
    """
    Visualize batch data with flexible display options
    
    Args:
        img, modal, amodal: tensors from your dataloader
        max_display: maximum number of samples to display
        cols_per_type: how many columns for each type (useful for large batches)
        save_path: path to save the plot (optional)
    """
    batch_size = img.shape[0]
    display_size = min(batch_size, max_display)
    
    # Calculate grid dimensions
    if cols_per_type > 1:
        # Grid layout: multiple samples per row
        rows = math.ceil(display_size / cols_per_type)
        cols = 3 * cols_per_type  # 3 types × cols_per_type
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    else:
        # Simple layout: one sample per row
        fig, axes = plt.subplots(display_size, 3, figsize=(12, 4 * display_size))
        if display_size == 1:
            axes = axes.reshape(1, -1)
    
    for i in range(display_size):
        if cols_per_type > 1:
            # Calculate position in grid
            row = i // cols_per_type
            col_offset = (i % cols_per_type) * 3
            img_col, modal_col, amodal_col = col_offset, col_offset + 1, col_offset + 2
            
            if rows == 1:
                img_ax, modal_ax, amodal_ax = axes[img_col], axes[modal_col], axes[amodal_col]
            else:
                img_ax, modal_ax, amodal_ax = axes[row, img_col], axes[row, modal_col], axes[row, amodal_col]
        else:
            img_ax, modal_ax, amodal_ax = axes[i, 0], axes[i, 1], axes[i, 2]
        
        # Process and display image
        img_np = img[i].permute(1, 2, 0).cpu().numpy()
        if img_np.min() < 0:
            img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        
        img_ax.imshow(img_np)
        img_ax.set_title(f'Img {i}')
        img_ax.axis('off')
        
        # Display modal
        modal_ax.imshow(modal[i].cpu().numpy(), cmap='gray')
        modal_ax.set_title(f'Modal {i}')
        modal_ax.axis('off')
        
        # Display amodal
        amodal_ax.imshow(amodal[i].cpu().numpy(), cmap='gray')
        amodal_ax.set_title(f'Amodal {i}')
        amodal_ax.axis('off')
    
    # Hide unused subplots
    if cols_per_type > 1 and display_size < max_display:
        total_plots = rows * cols
        used_plots = display_size * 3
        for idx in range(used_plots, total_plots):
            row_idx = idx // cols
            col_idx = idx % cols
            if rows == 1:
                axes[col_idx].axis('off')
            else:
                axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    if batch_size > max_display:
        print(f"Showing {display_size}/{batch_size} samples from the batch")

# Usage examples:
# Basic usage (limit to 8 samples)
# visualize_batch(img, modal, amodal, max_display=8)

# For larger displays (2 samples per row, showing 16 total)
# visualize_batch(img, modal, amodal, max_display=16, cols_per_type=2)

# Save to file
# visualize_batch(img, modal, amodal, max_display=6, save_path='batch_viz.png')