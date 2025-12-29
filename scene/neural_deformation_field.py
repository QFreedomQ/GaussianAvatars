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

class NeuralDeformationField(nn.Module):
    """
    Generalized Neural Deformation Field
    
    This module replaces the lookup table approach for dynamic offsets with a learned
    MLP that maps from expression and pose parameters to vertex offsets.
    
    Input: Expression code (100D) + Neck pose (3D) + Jaw pose (3D) = 106D
    Output: Per-vertex offsets (N_verts x 3)
    """
    
    def __init__(self, n_expr=100, n_verts=5023, hidden_dim=256, num_layers=3):
        super().__init__()
        self.n_expr = n_expr
        self.n_verts = n_verts
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input dimension: expression (100) + neck pose (3) + jaw pose (3) = 106
        input_dim = n_expr + 3 + 3
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Final layer outputs per-vertex offsets (N_verts x 3)
        layers.append(nn.Linear(hidden_dim, n_verts * 3))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small values to prevent large deformations initially"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, expr_params, neck_pose, jaw_pose):
        """
        Forward pass of the neural deformation field
        
        Args:
            expr_params: Expression parameters (batch_size, 100)
            neck_pose: Neck pose parameters (batch_size, 3)
            jaw_pose: Jaw pose parameters (batch_size, 3)
            
        Returns:
            Vertex offsets (batch_size, N_verts, 3)
        """
        # Concatenate input parameters
        input_features = torch.cat([expr_params, neck_pose, jaw_pose], dim=-1)
        
        # Pass through MLP
        output = self.mlp(input_features)
        
        # Reshape to (batch_size, N_verts, 3)
        vertex_offsets = output.view(-1, self.n_verts, 3)
        
        return vertex_offsets
    
    def get_regularization_loss(self):
        """
        Regularization loss to encourage smooth deformations
        """
        # L2 regularization on MLP weights
        l2_loss = 0
        for param in self.mlp.parameters():
            l2_loss += torch.norm(param, p=2)
        
        return l2_loss