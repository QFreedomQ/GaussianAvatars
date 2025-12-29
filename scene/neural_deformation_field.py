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
    Generalized Neural Deformation Field with Conditional Architecture
    
    This module replaces the lookup table approach for dynamic offsets with a learned
    MLP that maps from expression and pose parameters to vertex offsets.
    
    Features:
    - Conditional architecture with identity embeddings
    - Multi-scale progressive training support
    - Input: Expression code (100D) + Neck pose (3D) + Jaw pose (3D) + Identity (64D) = 170D
    - Output: Per-vertex offsets (N_verts x 3)
    """
    
    def __init__(self, n_expr=100, n_verts=5023, hidden_dim=256, num_layers=3, 
                 identity_dim=64, use_identity_conditioning=True):
        super().__init__()
        self.n_expr = n_expr
        self.n_verts = n_verts
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.identity_dim = identity_dim
        self.use_identity_conditioning = use_identity_conditioning
        
        # Identity encoder for conditional architecture
        if use_identity_conditioning:
            self.identity_encoder = nn.Sequential(
                nn.Linear(300, 128),  # Shape parameters
                nn.ReLU(),
                nn.Linear(128, identity_dim)
            )
            # Initialize identity encoder
            self._initialize_identity_encoder()
            
            # Input dimension with identity conditioning
            input_dim = n_expr + 3 + 3 + identity_dim  # 106 + 64 = 170
        else:
            self.identity_encoder = None
            # Input dimension without identity conditioning
            input_dim = n_expr + 3 + 3  # 106
        
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
        
        # Progressive training parameters
        self.training_stage = 0
        self.expression_ranges = [
            (-0.5, 0.5),   # Stage 1: Small expressions
            (-1.0, 1.0),   # Stage 2: Medium expressions
            (-2.0, 2.0)    # Stage 3: Full range
        ]
    
    def _initialize_identity_encoder(self):
        """Initialize identity encoder weights"""
        for m in self.identity_encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _initialize_weights(self):
        """Initialize weights with small values to prevent large deformations initially"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, expr_params, neck_pose, jaw_pose, shape_params=None):
        """
        Forward pass of the neural deformation field with identity conditioning
        
        Args:
            expr_params: Expression parameters (batch_size, 100)
            neck_pose: Neck pose parameters (batch_size, 3)
            jaw_pose: Jaw pose parameters (batch_size, 3)
            shape_params: Optional shape parameters for identity conditioning (batch_size, 300)
            
        Returns:
            Vertex offsets (batch_size, N_verts, 3)
        """
        # Concatenate base input parameters
        input_features = [expr_params, neck_pose, jaw_pose]
        
        # Add identity conditioning if enabled and shape parameters provided
        if self.use_identity_conditioning and shape_params is not None:
            identity_embedding = self.identity_encoder(shape_params)
            input_features.append(identity_embedding)
        
        # Concatenate all features
        input_features = torch.cat(input_features, dim=-1)
        
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
        
        # Add identity encoder regularization if used
        if self.use_identity_conditioning and self.identity_encoder is not None:
            for param in self.identity_encoder.parameters():
                l2_loss += torch.norm(param, p=2) * 0.5  # Lower weight for identity encoder
        
        return l2_loss
    
    def set_training_stage(self, stage):
        """
        Set progressive training stage to control expression range
        
        Args:
            stage: Training stage (0, 1, or 2)
        """
        self.training_stage = min(stage, len(self.expression_ranges) - 1)
        print(f"[NeuralDeformationField] Set to training stage {self.training_stage}")
    
    def get_current_expression_range(self):
        """
        Get current expression range for progressive training
        
        Returns:
            tuple: (min_range, max_range)
        """
        return self.expression_ranges[self.training_stage]
    
    def advance_stage(self):
        """
        Advance to next training stage
        
        Returns:
            bool: True if stage was advanced, False if already at final stage
        """
        if self.training_stage < len(self.expression_ranges) - 1:
            self.training_stage += 1
            print(f"[NeuralDeformationField] Advanced to stage {self.training_stage}")
            return True
        return False