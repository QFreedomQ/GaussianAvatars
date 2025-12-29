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
    UV-Based Neural Texture Field with Multi-Scale and Seamless Improvements
    
    This module replaces per-point SH colors with a learned 2D texture map.
    Gaussian points sample colors from this texture based on their UV coordinates.
    
    Key features:
    - Multi-scale texture pyramid for adaptive resolution
    - Seamless texture synthesis to reduce UV artifacts
    - Fixed parameter count (independent of Gaussian point count)
    - Spatial coherence and editability
    """
    
    def __init__(self, texture_size=1024, initial_color="average", num_levels=3, enable_seamless=True):
        super().__init__()
        self.texture_size = texture_size
        self.num_levels = num_levels
        self.enable_seamless = enable_seamless
        
        # Create multi-scale texture pyramid
        self.texture_pyramid = nn.ModuleList()
        for i in range(num_levels):
            size = texture_size // (2 ** i)
            texture_map = nn.Parameter(torch.zeros(1, 3, size, size))
            self.texture_pyramid.append(texture_map)
            
            # Initialize each level
            self._initialize_texture_level(texture_map, initial_color, level=i)
        
        # Seamless convolution for artifact reduction
        if enable_seamless:
            self.seam_aware_conv = nn.Conv2d(3, 3, kernel_size=5, padding=2)
            # Initialize convolution weights
            nn.init.xavier_uniform_(self.seam_aware_conv.weight, gain=0.01)
            nn.init.constant_(self.seam_aware_conv.bias, 0)
        else:
            self.seam_aware_conv = None
        
        # UV coordinate cache for efficiency
        self.uv_cache = None
        self.face_uv_cache = None
    
    def _initialize_texture_level(self, texture_map, initial_color="average", level=0):
        """Initialize a single texture level with reasonable starting values"""
        if initial_color == "average":
            # Initialize with average skin tone
            avg_skin_color = torch.tensor([0.75, 0.65, 0.55], device='cuda')  # RGB
            with torch.no_grad():
                texture_map.data.fill_(0.5)  # Start with gray
                texture_map.data[:, 0, :, :] = avg_skin_color[0]
                texture_map.data[:, 1, :, :] = avg_skin_color[1]
                texture_map.data[:, 2, :, :] = avg_skin_color[2]
                
                # Higher levels (lower resolution) can have slightly different initialization
                if level > 0:  # Coarser levels
                    # Add subtle variation for multi-scale learning
                    noise = torch.randn_like(texture_map.data) * 0.05
                    texture_map.data = torch.clamp(texture_map.data + noise, 0, 1)
        elif initial_color == "random":
            with torch.no_grad():
                texture_map.data.uniform_(0.3, 0.8)  # Random but reasonable colors
    
    def set_uv_coordinates(self, face_uvcoords, textures_idx):
        """
        Set UV coordinates for the texture sampling
        
        Args:
            face_uvcoords: UV coordinates per face (N_faces, 3, 3)
            textures_idx: Texture indices (N_faces, 3)
        """
        self.face_uvcoords = face_uvcoords
        self.textures_idx = textures_idx
    
    def sample_texture(self, uv_coords, lod=0, distance=None):
        """
        Sample texture color at given UV coordinates using bilinear interpolation
        with multi-scale and seamless improvements
        
        Args:
            uv_coords: UV coordinates (N_points, 2) in range [-1, 1]
            lod: Level of detail (0 = highest resolution)
            distance: Optional distance for automatic LOD selection
            
        Returns:
            Sampled colors (N_points, 3) in range [0, 1]
        """
        # Automatic LOD selection based on distance
        if distance is not None:
            lod = self._calculate_lod(distance)
        
        # Get appropriate texture level
        texture_map = self.texture_pyramid[lod]
        
        # Convert from [-1, 1] range to [0, 1] range
        uv_normalized = (uv_coords + 1.0) / 2.0
        
        # Flip Y coordinate (texture convention)
        uv_normalized = torch.stack([
            uv_normalized[..., 0],
            1.0 - uv_normalized[..., 1]
        ], dim=-1)
        
        # Convert back to [-1, 1] range for grid_sample
        uv_grid_sample = uv_normalized * 2.0 - 1.0
        
        # Add batch and channel dimensions for grid_sample
        uv_grid_sample = uv_grid_sample.unsqueeze(1)  # (N, 1, 2)
        
        # Sample texture
        sampled_colors = F.grid_sample(
            texture_map,
            uv_grid_sample.unsqueeze(0),  # Add batch dimension
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Apply seamless convolution if enabled
        if self.enable_seamless and self.seam_aware_conv is not None:
            # Reshape for convolution
            sampled_colors = sampled_colors.squeeze(0).permute(1, 0).unsqueeze(0)  # (1, 3, N, 1)
            
            # Apply seamless convolution
            seamless_colors = self.seam_aware_conv(sampled_colors)
            
            # Reshape back
            seamless_colors = seamless_colors.squeeze(-1).permute(1, 0)  # (N, 3)
            return seamless_colors
        
        # Remove extra dimensions and return
        return sampled_colors.squeeze(0).permute(1, 0)  # (N, 3)
    
    def _calculate_lod(self, distance):
        """
        Calculate appropriate level of detail based on distance

        Args:
            distance: Distance from camera to surface

        Returns:
            LOD level (0 = highest resolution)
        """
        # Normalize distance and calculate LOD
        # Closer objects use higher resolution (LOD 0)
        # Farther objects use lower resolution (higher LOD)
        lod = min(int(distance / 2.0), self.num_levels - 1)
        return lod

    def get_texture_map(self, level=0):
        """Get the current texture map at specified level"""
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Texture level {level} out of range [0, {self.num_levels-1}]")
        return self.texture_pyramid[level]

    def set_texture_map(self, texture_data, level=0):
        """Set the texture map at specified level from external data (for editing)"""
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Texture level {level} out of range [0, {self.num_levels-1}]")

        expected_shape = self.texture_pyramid[level].shape
        if texture_data.shape != expected_shape:
            raise ValueError(f"Texture data shape {texture_data.shape} doesn't match expected {expected_shape}")
        self.texture_pyramid[level].data = texture_data.to(self.texture_pyramid[level].device)

        # If setting base level, propagate to coarser levels
        if level == 0:
            self._propagate_to_coarser_levels()
    
    def _propagate_to_coarser_levels(self):
        """Propagate base level texture to coarser levels"""
        for i in range(1, self.num_levels):
            # Downsample from level i-1 to level i
            coarse_texture = F.avg_pool2d(
                self.texture_pyramid[i-1], 
                kernel_size=2, 
                stride=2
            )
            self.texture_pyramid[i].data = coarse_texture.data
    
    def get_regularization_loss(self):
        """
        Regularization loss to encourage smooth texture across all levels
        """
        total_loss = 0
        
        # TV loss for each level
        for level, texture_map in enumerate(self.texture_pyramid):
            # Weight coarser levels less
            level_weight = 1.0 / (level + 1)
            
            diff_x = torch.sum(torch.abs(texture_map[:, :, :, 1:] - texture_map[:, :, :, :-1]))
            diff_y = torch.sum(torch.abs(texture_map[:, :, 1:, :] - texture_map[:, :, :-1, :]))
            tv_loss = (diff_x + diff_y) * level_weight
            total_loss += tv_loss
        
        # Consistency loss between levels
        for i in range(1, self.num_levels):
            # Encourage coarser levels to be similar to downsampled finer levels
            target_coarse = F.avg_pool2d(self.texture_pyramid[i-1], kernel_size=2, stride=2)
            consistency_loss = F.mse_loss(self.texture_pyramid[i], target_coarse)
            total_loss += consistency_loss * 0.1
        
        return total_loss