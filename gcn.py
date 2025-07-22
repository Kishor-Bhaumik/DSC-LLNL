
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_camera_spatial_graph

class ConvTemporalGraphical(nn.Module):
    """Basic module for applying a graph convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),  # No temporal convolution here, just spatial
            bias=True
        )

    def forward(self, x, A):
        # x shape: (N, C, T, V)
        # A shape: (V, V)
        
        x = self.conv(x)  # (N, out_channels*kernel_size, T, V)
        
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        
        # Apply graph convolution using adjacency matrix
        x = torch.einsum('nkctv,vw->nkctw', x, A)
        x = x.view(n, kc, t, v)
        
        return x.contiguous(), A


class ST_GCN_Block(nn.Module):
    """Spatial-temporal graph convolution block"""
    
    def __init__(self, in_channels, out_channels, temporal_kernel_size=9, 
                 stride=1, dropout=0, residual=True):
        super().__init__()
        
        padding = ((temporal_kernel_size - 1) // 2, 0)
        
        # Spatial graph convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels)
        
        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (temporal_kernel_size, 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A




class CameraST_GCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for Camera-based Video Understanding"""
    
    def __init__(self, in_channels=588, num_cameras=6, num_layers=3, 
                 base_channels=64, use_edge_importance=True, dropout=0.5, 
                 temporal_kernel_size=3, window=6):
        super().__init__()
        
        self.temporal_window = window

        self.num_cameras = num_cameras
        self.in_channels = in_channels 
        
        # Build camera adjacency matrix
        A = build_camera_spatial_graph(num_cameras)
        self.register_buffer('A', A)
        
        # Data normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_cameras)
        
        # Build ST-GCN layers
        self.st_gcn_networks = nn.ModuleList()
        
        # First layer (no residual)
        self.st_gcn_networks.append(
            ST_GCN_Block(in_channels, base_channels, temporal_kernel_size=temporal_kernel_size, 
                        residual=False, dropout=dropout)
        )
        
        # Subsequent layers
        current_channels = base_channels
        for i in range(1, num_layers):
            # Keep channels simple - can double every few layers if needed
            next_channels = base_channels
            self.st_gcn_networks.append(
                ST_GCN_Block(current_channels, next_channels, temporal_kernel_size=temporal_kernel_size, 
                            dropout=dropout)
            )
            current_channels = next_channels
        
        # Final layer to match input channels for output
        self.st_gcn_networks.append(
            ST_GCN_Block(current_channels, in_channels, temporal_kernel_size=temporal_kernel_size, 
                        dropout=dropout)
        )
        
        # Edge importance weighting
        if use_edge_importance:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x, batch_sz):
        self.B = batch_sz
        #pdb.set_trace()  # Debugging breakpoint
        x = x.view(self.B, self.num_cameras,self.temporal_window, -1)
        # Input: (B, 6, 5, 588) -> convert to ST-GCN format
        # Target: (B, 588, 5, 6, 1)
        
        B, cameras, frames, features = x.shape
        
        # Reshape to ST-GCN format: (B, C, T, V, M)
        x = x.permute(0, 3, 2, 1).unsqueeze(-1)  # (B, 588, 5, 6, 1)
        
        # Data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (B, M, V, C, T)
        x = x.view(N * M, V * C, T)  # (B*1, 6*588, 5)
          # Debugging breakpoint
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (B, M, C, T, V)
        x = x.view(N * M, C, T, V)  # (B, 588, 5, 6)
        
        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        
        # Reshape back to original format: (B, 6, 5, 588)
        x = x.view(N, M, C, T, V)  # (B, 1, 588, 5, 6)
        x = x.squeeze(1)  # (B, 588, 5, 6)
        x = x.permute(0, 3, 2, 1)  # (B, 6, 5, 588)

        return x.contiguous().view(-1, 3, 14, 14) # self.decoder(x).squeeze(1)  # (B, 1, H, W)
    
