#
# Innovation 4: Adaptive Multi-Resolution Training with Optimized Sparse Evaluation
# Based on techniques from:
# - Instant-NGP (SIGGRAPH 2022): Progressive resolution training
# - 3D Gaussian Splatting: Efficient evaluation strategies
# - Common practice in neural rendering: Sparse evaluation with clustering
#

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import random


class ResolutionScheduler:
    """
    Manages progressive resolution training schedule.
    Starts at lower resolution and gradually increases to full resolution.
    """
    def __init__(
        self,
        start_resolution_ratio: float = 0.5,
        end_resolution_ratio: float = 1.0,
        start_iteration: int = 0,
        end_iteration: int = 15000,
        schedule_type: str = "linear"
    ):
        """
        Args:
            start_resolution_ratio: Initial resolution ratio (0.5 = 50% of original)
            end_resolution_ratio: Final resolution ratio (1.0 = full resolution)
            start_iteration: Iteration to start progressive training
            end_iteration: Iteration to reach full resolution
            schedule_type: "linear", "exponential", or "cosine"
        """
        self.start_ratio = start_resolution_ratio
        self.end_ratio = end_resolution_ratio
        self.start_iter = start_iteration
        self.end_iter = end_iteration
        self.schedule_type = schedule_type
        
    def get_resolution_ratio(self, iteration: int) -> float:
        """Get current resolution ratio based on iteration."""
        if iteration <= self.start_iter:
            return self.start_ratio
        if iteration >= self.end_iter:
            return self.end_ratio
            
        progress = (iteration - self.start_iter) / (self.end_iter - self.start_iter)
        
        if self.schedule_type == "linear":
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        elif self.schedule_type == "exponential":
            ratio = self.start_ratio * ((self.end_ratio / self.start_ratio) ** progress)
        elif self.schedule_type == "cosine":
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (1 - np.cos(progress * np.pi)) / 2
        else:
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
            
        return ratio
    
    def should_use_reduced_resolution(self, iteration: int) -> bool:
        """Check if current iteration should use reduced resolution."""
        return iteration < self.end_iter and self.start_ratio < self.end_ratio


class ViewClusterSampler:
    """
    Samples a diverse subset of views for efficient evaluation using stratified sampling.
    Ensures coverage across the dataset with lightweight logic suitable for large view counts.
    """
    def __init__(self, num_clusters: int = 10, random_seed: int = 42):
        """
        Args:
            num_clusters: Number of clusters for view sampling
            random_seed: Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        random.seed(random_seed)
        
    def fit_and_sample(self, cameras: List, sample_ratio: float = 0.3) -> Tuple[List[int], List]:
        """
        Sample cameras using stratified random sampling for speed.
        
        Args:
            cameras: List of camera objects
            sample_ratio: Ratio of cameras to sample
            
        Returns:
            indices: Indices of selected cameras
            sampled_cameras: Selected camera objects
        """
        total_cameras = len(cameras)
        num_samples = max(1, int(total_cameras * sample_ratio))
        
        if num_samples >= total_cameras:
            # Return all cameras if sampling ratio is high
            return list(range(total_cameras)), cameras
        
        # Stratified sampling: sample evenly across the camera list
        step = total_cameras / num_samples
        selected_indices = [int(i * step) for i in range(num_samples)]
        
        # Add some randomness to avoid always picking the same cameras
        random_offset = random.randint(0, max(1, int(step) - 1))
        selected_indices = [(idx + random_offset) % total_cameras for idx in selected_indices]
        selected_indices = sorted(list(set(selected_indices)))  # Remove duplicates
        
        sampled_cameras = [cameras[i] for i in selected_indices]
        
        return selected_indices, sampled_cameras


class SparseEvaluationScheduler:
    """
    Schedules sparse evaluation during training to save computation.
    Uses full evaluation only at important checkpoints.
    """
    def __init__(
        self,
        sparse_until_iter: int = 100000,
        sparse_view_ratio: float = 0.3,
        sparse_lpips_ratio: float = 0.5,
        full_eval_intervals: List[int] = None
    ):
        """
        Args:
            sparse_until_iter: Use sparse evaluation until this iteration
            sparse_view_ratio: Ratio of views to evaluate (0.3 = 30%)
            sparse_lpips_ratio: Ratio of views for LPIPS (expensive metric)
            full_eval_intervals: List of iterations to force full evaluation
        """
        self.sparse_until_iter = sparse_until_iter
        self.sparse_view_ratio = sparse_view_ratio
        self.sparse_lpips_ratio = sparse_lpips_ratio
        self.full_eval_intervals = full_eval_intervals or []
        
    def should_use_sparse_evaluation(self, iteration: int) -> bool:
        """Check if should use sparse evaluation."""
        if iteration in self.full_eval_intervals:
            return False
        return iteration < self.sparse_until_iter
    
    def should_compute_lpips(self, iteration: int, view_idx: int, total_views: int) -> bool:
        """Check if LPIPS should be computed for this view."""
        if iteration in self.full_eval_intervals or iteration >= self.sparse_until_iter:
            return True
        # Compute LPIPS only for a subset of views
        return view_idx < int(total_views * self.sparse_lpips_ratio)


def create_downsampled_camera(camera, resolution_ratio: float):
    """
    Create a downsampled version of the camera with reduced resolution.
    
    Args:
        camera: Original camera object
        resolution_ratio: Resolution scaling factor (0.5 = half resolution)
        
    Returns:
        Downsampled camera with adjusted image dimensions
    """
    from copy import copy
    
    # Create a shallow copy of the camera
    downsampled_cam = copy(camera)
    
    # Update resolution
    new_width = int(camera.image_width * resolution_ratio)
    new_height = int(camera.image_height * resolution_ratio)
    
    # Ensure dimensions are at least 1
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    downsampled_cam.image_width = new_width
    downsampled_cam.image_height = new_height
    
    # Downsample the ground truth image if available
    if hasattr(camera, 'original_image') and camera.original_image is not None:
        original_img = camera.original_image
        if isinstance(original_img, torch.Tensor):
            # Use bilinear interpolation for downsampling
            # original_img shape: (C, H, W)
            downsampled_img = torch.nn.functional.interpolate(
                original_img.unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            downsampled_cam.original_image = downsampled_img
    
    return downsampled_cam


def get_training_camera_with_adaptive_resolution(
    camera,
    iteration: int,
    resolution_scheduler: ResolutionScheduler
):
    """
    Get training camera with adaptive resolution based on current iteration.
    
    Args:
        camera: Original camera
        iteration: Current training iteration
        resolution_scheduler: Resolution scheduler instance
        
    Returns:
        Camera with appropriate resolution for current iteration
    """
    if resolution_scheduler is None or not resolution_scheduler.should_use_reduced_resolution(iteration):
        return camera
    
    ratio = resolution_scheduler.get_resolution_ratio(iteration)
    if ratio >= 1.0:
        return camera
        
    return create_downsampled_camera(camera, ratio)
