#
# Region-Adaptive Loss Weighting
# 
# This module implements region-aware loss weighting for facial avatars.
# Instead of using expensive VGG perceptual loss, it applies higher weights
# to important facial regions (eyes, mouth, nose) in the standard L1/SSIM loss.
#
# Advantages:
# - Near-zero computational overhead (only element-wise multiplication)
# - Focuses optimization on perceptually important regions
# - No additional GPU memory required
# - Easy to tune and visualize
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RegionAdaptiveLoss(nn.Module):
    """
    Region-adaptive loss weighting for face avatars.
    
    Applies higher loss weights to important facial regions (eyes, mouth, nose)
    to improve reconstruction quality where it matters most.
    
    Inspired by:
    - FaceScape: 3D Facial Dataset (CVPR 2020)
    - PIFu: Pixel-Aligned Implicit Function (ICCV 2019)
    
    Benefits:
    - Zero computational overhead (only multiplication)
    - Improved detail in critical regions
    - Better perceptual quality
    - No additional networks or parameters
    """
    
    def __init__(self, flame_model=None, weight_eyes=2.0, weight_mouth=2.0, 
                 weight_nose=1.5, weight_face=1.2):
        """
        Args:
            flame_model: FLAME model instance (optional)
            weight_eyes: Loss weight multiplier for eye regions
            weight_mouth: Loss weight multiplier for mouth region
            weight_nose: Loss weight multiplier for nose region
            weight_face: Loss weight multiplier for general face region
        """
        super().__init__()
        self.weight_eyes = weight_eyes
        self.weight_mouth = weight_mouth
        self.weight_nose = weight_nose
        self.weight_face = weight_face
        
        # FLAME vertex regions (approximate ranges based on FLAME topology)
        self.eye_left_verts = list(range(3997, 4067))
        self.eye_right_verts = list(range(3930, 3997))
        self.mouth_verts = list(range(2812, 3025))
        self.nose_verts = list(range(3325, 3450))
        
        print(f"[Region-Adaptive Loss] Initialized with weights: "
              f"eyes={weight_eyes}, mouth={weight_mouth}, nose={weight_nose}")
    
    def create_simple_weight_map(self, H, W, device):
        """
        Create a simple weight map based on facial region heuristics.
        Uses image center bias and vertical position bias.
        
        This is a fallback when FLAME binding is not available.
        
        Args:
            H: Image height
            W: Image width
            device: torch device
        
        Returns:
            weight_map: (1, H, W) weight map
        """
        weight_map = torch.ones((1, H, W), device=device)
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, H, device=device).view(-1, 1).expand(H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, -1).expand(H, W)
        
        # Center region (likely face)
        center_mask = (x_coords.abs() < 0.5) & (y_coords.abs() < 0.5)
        weight_map[:, center_mask] = self.weight_face
        
        # Upper-center (likely eyes)
        eye_mask = (x_coords.abs() < 0.4) & (y_coords > -0.3) & (y_coords < 0.1)
        weight_map[:, eye_mask] = self.weight_eyes
        
        # Lower-center (likely mouth)
        mouth_mask = (x_coords.abs() < 0.3) & (y_coords > 0.1) & (y_coords < 0.5)
        weight_map[:, mouth_mask] = self.weight_mouth
        
        return weight_map
    
    def create_weight_map_from_flame(self, rendered_image, camera, gaussians):
        """
        Create weight map by projecting FLAME semantic regions to image space.
        
        Args:
            rendered_image: Rendered image (3, H, W)
            camera: Camera object
            gaussians: Gaussian model with FLAME binding
        
        Returns:
            weight_map: (1, H, W) weight map
        """
        H, W = rendered_image.shape[1], rendered_image.shape[2]
        device = rendered_image.device
        
        # Start with uniform weights
        weight_map = torch.ones((1, H, W), device=device)
        
        # Check if FLAME binding is available
        if not hasattr(gaussians, 'verts') or gaussians.verts is None:
            # Fallback to simple heuristic-based weight map
            return self.create_simple_weight_map(H, W, device)
        
        try:
            # Get FLAME vertices in world space
            verts_3d = gaussians.verts  # (N_verts, 3)
            
            # Project to screen space (simplified projection)
            # In a full implementation, this would use proper camera matrices
            verts_2d_normalized = verts_3d[:, :2]  # Use XY coordinates as proxy
            
            # Convert to pixel coordinates
            verts_2d_px = torch.zeros((verts_3d.shape[0], 2), device=device)
            verts_2d_px[:, 0] = (verts_2d_normalized[:, 0] + 1.0) * W / 2.0
            verts_2d_px[:, 1] = (verts_2d_normalized[:, 1] + 1.0) * H / 2.0
            verts_2d_px = verts_2d_px.long()
            verts_2d_px[:, 0] = torch.clamp(verts_2d_px[:, 0], 0, W - 1)
            verts_2d_px[:, 1] = torch.clamp(verts_2d_px[:, 1], 0, H - 1)
            
            # Create masks for each region
            radius = max(H, W) // 30  # Adaptive radius based on image size
            
            # Eyes
            for vert_idx in self.eye_left_verts + self.eye_right_verts:
                if vert_idx < verts_2d_px.shape[0]:
                    x, y = verts_2d_px[vert_idx]
                    y_min, y_max = max(0, y - radius), min(H, y + radius)
                    x_min, x_max = max(0, x - radius), min(W, x + radius)
                    weight_map[0, y_min:y_max, x_min:x_max] = self.weight_eyes
            
            # Mouth
            for vert_idx in self.mouth_verts:
                if vert_idx < verts_2d_px.shape[0]:
                    x, y = verts_2d_px[vert_idx]
                    y_min, y_max = max(0, y - radius), min(H, y + radius)
                    x_min, x_max = max(0, x - radius), min(W, x + radius)
                    weight_map[0, y_min:y_max, x_min:x_max] = self.weight_mouth
            
            # Nose
            for vert_idx in self.nose_verts:
                if vert_idx < verts_2d_px.shape[0]:
                    x, y = verts_2d_px[vert_idx]
                    y_min, y_max = max(0, y - radius), min(H, y + radius)
                    x_min, x_max = max(0, x - radius), min(W, x + radius)
                    weight_map[0, y_min:y_max, x_min:x_max] = self.weight_nose
        
        except Exception as e:
            # If projection fails, fall back to simple weight map
            print(f"[Region-Adaptive Loss] Warning: FLAME projection failed ({e}), using heuristic weights")
            return self.create_simple_weight_map(H, W, device)
        
        return weight_map
    
    def forward(self, image, gt, weight_map=None):
        """
        Compute region-adaptive weighted L1 loss.
        
        Args:
            image: Rendered image (3, H, W)
            gt: Ground truth image (3, H, W)
            weight_map: Pre-computed weight map (1, H, W), optional
        
        Returns:
            Weighted L1 loss (scalar)
        """
        # If no weight map provided, use uniform weights
        if weight_map is None:
            return torch.abs(image - gt).mean()
        
        # Compute per-pixel error
        error = torch.abs(image - gt)
        
        # Apply regional weights
        weighted_error = error * weight_map
        
        # Normalize by total weight to keep loss magnitude similar to standard L1
        loss = weighted_error.sum() / (weight_map.sum() + 1e-8)
        
        return loss
    
    def visualize_weight_map(self, weight_map):
        """
        Create a visualization of the weight map for debugging.
        
        Args:
            weight_map: (1, H, W) weight map
        
        Returns:
            weight_map_rgb: (3, H, W) RGB visualization
        """
        # Normalize to [0, 1]
        weight_normalized = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-8)
        
        # Create RGB visualization (blue=low, red=high)
        weight_rgb = torch.zeros((3, weight_map.shape[1], weight_map.shape[2]), 
                                 device=weight_map.device)
        weight_rgb[0] = weight_normalized.squeeze(0)  # Red channel = high weights
        weight_rgb[2] = 1.0 - weight_normalized.squeeze(0)  # Blue channel = low weights
        
        return weight_rgb
