#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class UVNeuralTexture(nn.Module):
    """
    UV-Based Neural Texture Field
    
    This module replaces per-point SH colors with a learned 2D texture map.
    Gaussian points sample colors from this texture based on their UV coordinates.
    
    Key features:
    - Fixed parameter count (independent of Gaussian point count)
    - Spatial coherence (adjacent points sample adjacent texels)
    - Editability (texture can be exported, modified, and reloaded)
    """
    
    def __init__(self, texture_size=1024, initial_color="average"):
        super().__init__()
        self.texture_size = texture_size
        
        # Create learnable texture map (RGB, 3 channels)
        # Texture coordinates: (H, W, 3) where last channel is RGB
        self.texture_map = nn.Parameter(
            torch.zeros(1, 3, texture_size, texture_size)
        )
        
        # Initialize texture
        self._initialize_texture(initial_color)
        
        # UV coordinate cache for efficiency
        self.uv_cache = None
        self.face_uv_cache = None
    
    def _initialize_texture(self, initial_color="average"):
        """Initialize texture with reasonable starting values"""
        if initial_color == "average":
            # Initialize with average skin tone
            avg_skin_color = torch.tensor([0.75, 0.65, 0.55], device='cuda')  # RGB
            with torch.no_grad():
                self.texture_map.data.fill_(0.5)  # Start with gray
                self.texture_map.data[:, 0, :, :] = avg_skin_color[0]
                self.texture_map.data[:, 1, :, :] = avg_skin_color[1]
                self.texture_map.data[:, 2, :, :] = avg_skin_color[2]
        elif initial_color == "random":
            with torch.no_grad():
                self.texture_map.data.uniform_(0.3, 0.8)  # Random but reasonable colors
    
    def set_uv_coordinates(self, face_uvcoords, textures_idx):
        """
        Set UV coordinates for the texture sampling
        
        Args:
            face_uvcoords: UV coordinates per face (N_faces, 3, 3)
            textures_idx: Texture indices (N_faces, 3)
        """
        self.face_uvcoords = face_uvcoords
        self.textures_idx = textures_idx
    
    def sample_texture(self, uv_coords):
        """
        Sample texture color at given UV coordinates using bilinear interpolation
        
        Args:
            uv_coords: UV coordinates (N_points, 2) in range [-1, 1]
            
        Returns:
            Sampled colors (N_points, 3) in range [0, 1]
        """
        # Convert from [-1, 1] range to [0, 1] range
        uv_normalized = (uv_coords + 1.0) / 2.0
        
        # Flip Y coordinate (texture convention)
        uv_normalized = torch.stack([
            uv_normalized[..., 0],
            1.0 - uv_normalized[..., 1]
        ], dim=-1)
        
        # Scale to texture size
        uv_scaled = uv_normalized * (self.texture_size - 1)
        
        # Sample using grid_sample (expects coordinates in [-1, 1] range)
        # Convert back to [-1, 1] range for grid_sample
        uv_grid_sample = uv_normalized * 2.0 - 1.0
        
        # Add batch and channel dimensions for grid_sample
        # grid_sample expects (N, C, H, W) input and (N, H, W, 2) coordinates
        uv_grid_sample = uv_grid_sample.unsqueeze(1)  # (N, 1, 2)
        
        # Sample texture
        sampled_colors = F.grid_sample(
            self.texture_map,
            uv_grid_sample.unsqueeze(0),  # Add batch dimension
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Remove extra dimensions and return
        return sampled_colors.squeeze(0).permute(1, 0)  # (N, 3)
    
    def get_texture_map(self):
        """Get the current texture map as a tensor"""
        return self.texture_map
    
    def set_texture_map(self, texture_data):
        """Set the texture map from external data (for editing)"""
        if texture_data.shape != self.texture_map.shape:
            raise ValueError(f"Texture data shape {texture_data.shape} doesn't match expected {self.texture_map.shape}")
        self.texture_map.data = texture_data.to(self.texture_map.device)
    
    def get_regularization_loss(self):
        """
        Regularization loss to encourage smooth texture
        """
        # TV loss (Total Variation) for spatial smoothness
        diff_x = torch.sum(torch.abs(self.texture_map[:, :, :, 1:] - self.texture_map[:, :, :, :-1]))
        diff_y = torch.sum(torch.abs(self.texture_map[:, :, 1:, :] - self.texture_map[:, :, :-1, :]))
        tv_loss = diff_x + diff_y
        
        return tv_loss