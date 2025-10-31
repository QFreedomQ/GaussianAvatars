#
# Lightweight Color Calibration Network
#
# This module implements a tiny MLP that calibrates the rendered image's colors
# to match the ground truth, correcting systematic color/exposure differences
# without incurring heavy computational costs.
#
# Inspired by NeRF in the Wild (CVPR 2021) and Mip-NeRF 360 (CVPR 2022).
#

import torch
import torch.nn as nn

class LightweightColorCalibration(nn.Module):
    """
    Lightweight color calibration network for Gaussian Avatars.
    
    Applies a small MLP per pixel to correct color/exposure discrepancies
    between rendered images and ground truth. Can be trained end-to-end with
    minimal overhead.
    """
    
    def __init__(self, hidden_dim=16, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        layers = []
        input_dim = 3
        output_dim = 3
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights to be close to identity mapping.
        """
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            
            # Encourage near-identity behavior initially
            last_linear = None
            for module in self.net:
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
    
    def forward(self, image):
        """
        Apply color calibration to the rendered image.
        
        Args:
            image: Rendered image tensor of shape (3, H, W) or (B, 3, H, W)
        
        Returns:
            Calibrated image tensor with the same shape as input
        """
        original_shape = image.shape
        
        if len(original_shape) == 3:
            # Convert to (1, 3, H, W)
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        assert C == 3, "Input image must have 3 channels (RGB)"
        
        # Flatten spatial dimensions: (B, 3, H, W) -> (B, H, W, 3)
        image_flat = image.permute(0, 2, 3, 1).contiguous()
        
        # Flatten to (B * H * W, 3)
        image_flat = image_flat.view(-1, 3)
        
        # Apply MLP
        calibrated_flat = self.net(image_flat)
        
        # Reshape back to (B, H, W, 3)
        calibrated = calibrated_flat.view(B, H, W, 3)
        
        # Permute back to (B, 3, H, W)
        calibrated = calibrated.permute(0, 3, 1, 2).contiguous()
        
        if len(original_shape) == 3:
            calibrated = calibrated.squeeze(0)
        
        return calibrated
    
    def regularization_loss(self):
        """
        Small L2 regularization on weights to avoid overfitting.
        
        Returns:
            Scalar regularization loss
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.net:
            if isinstance(module, nn.Linear):
                reg_loss = reg_loss + module.weight.pow(2).mean() * 1e-4
        return reg_loss
