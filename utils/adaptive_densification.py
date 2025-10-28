#
# Adaptive Densification Strategy inspired by:
# 1. Deformable 3D Gaussians (arxiv 2023) - https://github.com/ingra14m/Deformable-3D-Gaussians
# 2. Dynamic 3D Gaussians (CVPR 2024) - https://github.com/JonathonLuiten/Dynamic3DGaussians
# 3. MonoGaussianAvatar (arxiv 2024) - Adaptive densification based on facial regions
#
# This module implements region-aware densification that focuses more on important
# facial regions (eyes, mouth, nose) while using coarser representation for less
# important areas (forehead, cheeks).
#

import torch
import numpy as np

class AdaptiveDensificationStrategy:
    """
    Region-aware adaptive densification for face avatars.
    
    Source: Based on Dynamic 3D Gaussians' adaptive densification strategy
    (https://github.com/JonathonLuiten/Dynamic3DGaussians/blob/main/scene/gaussian_model.py)
    and MonoGaussianAvatar's facial region importance weighting.
    
    Principle:
    1. Divide face mesh into semantic regions (eyes, mouth, nose, etc.)
    2. Assign different densification thresholds based on region importance
    3. More aggressive densification in high-detail regions
    4. Conservative densification in low-detail regions
    
    Benefits:
    - Better detail preservation in critical facial features
    - Reduced memory usage by avoiding over-densification in uniform regions
    - Improved rendering quality for expressions and eye movements
    - More efficient Gaussian distribution
    
    Impact on results:
    - Higher PSNR in facial feature regions (eyes, mouth)
    - Better temporal consistency in dynamic regions
    - Reduced total Gaussian count while maintaining quality
    """
    
    def __init__(self, num_faces, flame_model=None, importance_ratio=1.5):
        """
        Args:
            num_faces: Number of faces in the FLAME mesh
            flame_model: FLAME model instance for semantic region detection
            importance_ratio: Multiplier for important regions (default 1.5)
        """
        self.num_faces = num_faces
        self.flame_model = flame_model
        self.importance_ratio = importance_ratio
        
        # Initialize region importance weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(flame_model, torch.nn.Module) and hasattr(flame_model, 'faces'):
            faces_tensor = flame_model.faces
            if isinstance(faces_tensor, torch.Tensor):
                device = faces_tensor.device
        self.region_weights = torch.ones(num_faces, device=device)
        
        # If FLAME model is provided, compute semantic region weights
        if flame_model is not None:
            self._compute_semantic_weights()
    
    def _compute_semantic_weights(self):
        """
        Compute importance weights for each face based on FLAME semantic regions.
        
        FLAME face indices for key regions (approximate):
        - Eyes: face indices around eye vertices
        - Mouth: face indices around mouth vertices  
        - Nose: face indices around nose vertices
        - Face boundary: lower importance
        """
        try:
            # Get FLAME vertices
            faces = self.flame_model.faces
            if isinstance(faces, torch.Tensor):
                faces = faces.squeeze(0).cpu().numpy()
            else:
                faces = np.array(faces)
            
            # Define approximate vertex ranges for important regions (based on FLAME topology)
            # These ranges are approximate and based on FLAME vertex ordering
            eye_left_verts = list(range(3997, 4067))  # Left eye region
            eye_right_verts = list(range(3930, 3997))  # Right eye region
            mouth_verts = list(range(2812, 3025))  # Mouth region
            nose_verts = list(range(3325, 3450))  # Nose region
            
            important_verts = set(eye_left_verts + eye_right_verts + mouth_verts + nose_verts)
            
            # For each face, check if any vertex is in an important region
            face_weights = []
            for face_idx in range(len(faces)):
                face_verts = faces[face_idx]
                # Check overlap with important regions
                is_important = any(v in important_verts for v in face_verts)
                
                if is_important:
                    # High importance: lower threshold (more densification)
                    weight = self.importance_ratio
                else:
                    # Normal importance: standard threshold
                    weight = 1.0
                
                face_weights.append(weight)
            
            device = self.region_weights.device
            self.region_weights = torch.tensor(face_weights, device=device, dtype=torch.float32)
            
            print(f"[Adaptive Densification] Computed semantic weights for {len(faces)} faces")
            print(f"[Adaptive Densification] High-importance faces: {(self.region_weights > 1.0).sum().item()}")
        except Exception as e:
            print(f"[Adaptive Densification] Warning: Could not compute semantic weights: {e}")
            print(f"[Adaptive Densification] Using uniform weights")
    
    def get_adaptive_threshold(self, binding, base_threshold):
        """
        Get per-Gaussian adaptive densification threshold based on face binding.
        
        Args:
            binding: Tensor of face indices that each Gaussian is bound to (N,)
            base_threshold: Base densification threshold
        
        Returns:
            Per-Gaussian adaptive thresholds (N,)
        """
        # Get region weights for each Gaussian based on its binding
        if binding is None:
            return base_threshold
        if isinstance(binding, torch.Tensor) and binding.numel() > 0:
            max_index = self.region_weights.shape[0] - 1
            binding = binding.clamp(0, max_index)
            gaussian_weights = self.region_weights[binding]
        else:
            gaussian_weights = self.region_weights
        
        # Lower threshold = more aggressive densification
        # For important regions (weight=1.5), threshold becomes base_threshold/weight
        adaptive_thresholds = base_threshold / gaussian_weights
        
        return adaptive_thresholds
    
    def get_adaptive_prune_threshold(self, binding, base_opacity_threshold):
        """
        Get per-Gaussian adaptive pruning threshold based on face binding.
        
        Args:
            binding: Tensor of face indices that each Gaussian is bound to (N,)
            base_opacity_threshold: Base opacity threshold for pruning
        
        Returns:
            Per-Gaussian adaptive opacity thresholds (N,)
        """
        if binding is None:
            return base_opacity_threshold
        if isinstance(binding, torch.Tensor) and binding.numel() > 0:
            max_index = self.region_weights.shape[0] - 1
            binding = binding.clamp(0, max_index)
            gaussian_weights = self.region_weights[binding]
        else:
            gaussian_weights = self.region_weights
        
        # For important regions, use lower opacity threshold (less aggressive pruning)
        # For unimportant regions, use higher threshold (more aggressive pruning)
        adaptive_thresholds = torch.where(
            gaussian_weights > 1.0,
            base_opacity_threshold * 0.7,  # Keep more Gaussians in important regions
            base_opacity_threshold * 1.2   # Prune more aggressively in less important regions
        )
        
        return adaptive_thresholds


class SpatiallyAdaptiveDensification:
    """
    Spatially adaptive densification based on image-space gradients.
    
    Source: Inspired by Deformable 3D Gaussians
    (https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/scene/gaussian_model.py)
    
    Principle:
    - Track gradient statistics in image space
    - Identify regions with consistently high gradients
    - Apply adaptive thresholds based on spatial gradient distribution
    
    Benefits:
    - Focuses densification on areas with high-frequency details
    - Adapts to individual subject characteristics
    - Reduces over-densification in smooth regions
    """
    
    def __init__(self, num_gaussians):
        self.gradient_accum = torch.zeros((num_gaussians, 1), device='cuda')
        self.count = torch.zeros((num_gaussians, 1), device='cuda')
    
    def update_statistics(self, indices, gradients):
        """
        Update gradient statistics for specified Gaussians.
        
        Args:
            indices: Indices of visible Gaussians
            gradients: Gradient magnitudes for visible Gaussians
        """
        self.gradient_accum[indices] += gradients.unsqueeze(-1)
        self.count[indices] += 1
    
    def get_adaptive_threshold(self, base_threshold, percentile=70):
        """
        Compute adaptive threshold based on gradient distribution.
        
        Args:
            base_threshold: Base gradient threshold
            percentile: Percentile for adaptive thresholding
        
        Returns:
            Adaptive threshold value
        """
        valid_mask = self.count > 0
        avg_gradients = torch.zeros_like(self.gradient_accum)
        avg_gradients[valid_mask] = self.gradient_accum[valid_mask] / self.count[valid_mask]
        
        if avg_gradients[valid_mask].numel() > 0:
            threshold_percentile = torch.quantile(avg_gradients[valid_mask], percentile / 100.0)
            return max(base_threshold, threshold_percentile.item())
        else:
            return base_threshold
    
    def reset(self, num_gaussians):
        """Reset statistics for new Gaussian count."""
        self.gradient_accum = torch.zeros((num_gaussians, 1), device='cuda')
        self.count = torch.zeros((num_gaussians, 1), device='cuda')
