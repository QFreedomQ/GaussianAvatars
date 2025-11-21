"""
Adaptive Densification with View Coverage
动态密度过滤与视角覆盖自适应

参考: Gaussian Surfels (SIGGRAPH Asia 2023), Mip-Splatting (CVPR 2024)
原理: 根据视角覆盖度和梯度分布动态调整 densification 阈值，
      避免在已充分覆盖区域过度细分，提升效率和质量
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class ViewCoverageTracker:
    """
    追踪每个高斯点在不同视角下的可见性和覆盖度
    """
    def __init__(self, num_gaussians, device='cuda'):
        self.num_gaussians = num_gaussians
        self.device = device
        
        # 记录每个高斯点被看到的次数
        self.view_count = torch.zeros(num_gaussians, dtype=torch.int32, device=device)
        
        # 记录每个高斯点在屏幕上的平均投影大小（像素）
        self.avg_screen_size = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
        
        # 记录每个高斯点的梯度统计
        self.gradient_stats = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
    
    def update(self, visible_indices, screen_sizes, gradients=None):
        """
        更新覆盖度统计
        
        Args:
            visible_indices: (M,) 当前可见的高斯索引
            screen_sizes: (M,) 屏幕投影大小（radii2D）
            gradients: (M,) 可选的梯度大小
        """
        if len(visible_indices) == 0:
            return
        
        # 累加可见次数
        self.view_count.scatter_add_(
            0, visible_indices,
            torch.ones_like(visible_indices, dtype=torch.int32, device=self.device)
        )
        
        # 更新屏幕大小（指数移动平均）
        alpha = 0.9
        current_avg = torch.zeros_like(self.avg_screen_size)
        current_avg.scatter_add_(0, visible_indices, screen_sizes)
        
        # 仅更新可见点
        mask = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.device)
        mask[visible_indices] = True
        
        self.avg_screen_size[mask] = alpha * self.avg_screen_size[mask] + (1 - alpha) * current_avg[mask]
        
        # 更新梯度统计
        if gradients is not None:
            current_grad = torch.zeros_like(self.gradient_stats)
            current_grad.scatter_add_(0, visible_indices, gradients.abs())
            self.gradient_stats[mask] = alpha * self.gradient_stats[mask] + (1 - alpha) * current_grad[mask]
    
    def get_coverage_weights(self):
        """
        计算覆盖度权重（0-1），覆盖度高的点权重低（抑制 densification）
        """
        # 基于可见次数的权重（sigmoid 函数）
        view_weight = torch.sigmoid(-(self.view_count.float() - 10.0) / 5.0)  # 10次后开始抑制
        
        # 基于屏幕大小的权重（投影太大说明太少点覆盖，需要细分）
        size_weight = torch.sigmoid((self.avg_screen_size - 2.0) / 1.0)  # 2像素为阈值
        
        # 综合权重
        coverage_weight = (view_weight + size_weight) / 2.0
        
        return coverage_weight
    
    def reset(self):
        """重置统计（用于新一轮训练周期）"""
        self.view_count.zero_()
        self.avg_screen_size.zero_()
        self.gradient_stats.zero_()


class AdaptiveDensificationController:
    """
    自适应密度控制器
    动态调整 densification 和 pruning 的阈值
    """
    def __init__(
        self,
        base_grad_threshold=0.0002,
        min_grad_threshold=0.0001,
        max_grad_threshold=0.0005,
        coverage_factor=0.5,
        enable_adaptive=True,
    ):
        self.base_grad_threshold = base_grad_threshold
        self.min_grad_threshold = min_grad_threshold
        self.max_grad_threshold = max_grad_threshold
        self.coverage_factor = coverage_factor
        self.enable_adaptive = enable_adaptive
        
        self.coverage_tracker = None
        self.iteration_stats = defaultdict(list)
    
    def initialize_tracker(self, num_gaussians):
        """初始化覆盖度追踪器"""
        self.coverage_tracker = ViewCoverageTracker(num_gaussians)
    
    def update_coverage(self, visible_indices, screen_sizes, gradients=None):
        """更新覆盖度统计"""
        if self.coverage_tracker is not None:
            self.coverage_tracker.update(visible_indices, screen_sizes, gradients)
    
    def get_adaptive_threshold(self, gaussian_model):
        """
        获取自适应的梯度阈值
        
        Args:
            gaussian_model: GaussianModel 实例
        Returns:
            threshold: 标量或 (N,) 张量
        """
        if not self.enable_adaptive or self.coverage_tracker is None:
            return self.base_grad_threshold
        
        # 获取覆盖度权重
        coverage_weights = self.coverage_tracker.get_coverage_weights()
        
        # 基于覆盖度调整阈值（覆盖度高 → 阈值高 → 更难触发 densification）
        adaptive_threshold = self.base_grad_threshold + \
            self.coverage_factor * coverage_weights * (self.max_grad_threshold - self.base_grad_threshold)
        
        # 限制范围
        adaptive_threshold = torch.clamp(
            adaptive_threshold,
            min=self.min_grad_threshold,
            max=self.max_grad_threshold
        )
        
        return adaptive_threshold
    
    def should_densify_gaussian(self, gaussian_idx, gradient_magnitude):
        """
        判断单个高斯是否应该 densify
        
        Args:
            gaussian_idx: int
            gradient_magnitude: float
        Returns:
            bool
        """
        if not self.enable_adaptive:
            return gradient_magnitude > self.base_grad_threshold
        
        threshold = self.get_adaptive_threshold(None)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold[gaussian_idx].item()
        
        return gradient_magnitude > threshold
    
    def compute_densification_mask(self, gradients, gaussian_model):
        """
        计算需要 densify 的高斯掩码
        
        Args:
            gradients: (N,) 梯度大小
            gaussian_model: GaussianModel 实例
        Returns:
            mask: (N,) bool 张量
        """
        threshold = self.get_adaptive_threshold(gaussian_model)
        
        if isinstance(threshold, float):
            mask = gradients > threshold
        else:
            mask = gradients > threshold
        
        return mask
    
    def get_statistics(self):
        """获取统计信息（用于日志）"""
        if self.coverage_tracker is None:
            return {}
        
        return {
            'avg_view_count': self.coverage_tracker.view_count.float().mean().item(),
            'avg_screen_size': self.coverage_tracker.avg_screen_size.mean().item(),
            'coverage_weight_mean': self.coverage_tracker.get_coverage_weights().mean().item(),
        }
    
    def reset_tracker(self):
        """重置追踪器"""
        if self.coverage_tracker is not None:
            self.coverage_tracker.reset()


def compute_gradient_based_split_mask(
    gaussian_model,
    grad_threshold,
    scene_extent,
    size_threshold=None,
):
    """
    计算基于梯度的分裂掩码（改进版）
    
    Args:
        gaussian_model: GaussianModel 实例
        grad_threshold: 梯度阈值（标量或张量）
        scene_extent: 场景范围
        size_threshold: 可选的尺寸阈值
    Returns:
        mask: (N,) bool 张量
    """
    # 梯度条件
    grads = gaussian_model.xyz_gradient_accum / gaussian_model.denom
    grads[grads.isnan()] = 0.0
    
    if isinstance(grad_threshold, float):
        grad_mask = (grads >= grad_threshold).squeeze()
    else:
        grad_mask = (grads.squeeze() >= grad_threshold)
    
    # 尺寸条件
    if size_threshold is None:
        size_threshold = scene_extent * gaussian_model.percent_dense
    
    scales = gaussian_model.get_scaling
    size_mask = torch.max(scales, dim=1).values > size_threshold
    
    # 综合条件
    split_mask = torch.logical_and(grad_mask, size_mask)
    
    return split_mask


def compute_gradient_based_clone_mask(
    gaussian_model,
    grad_threshold,
    scene_extent,
    size_threshold=None,
):
    """
    计算基于梯度的克隆掩码（改进版）
    
    Args:
        gaussian_model: GaussianModel 实例
        grad_threshold: 梯度阈值
        scene_extent: 场景范围
        size_threshold: 可选的尺寸阈值
    Returns:
        mask: (N,) bool 张量
    """
    # 梯度条件
    grads = gaussian_model.xyz_gradient_accum / gaussian_model.denom
    grads[grads.isnan()] = 0.0
    
    if isinstance(grad_threshold, float):
        grad_mask = (grads >= grad_threshold).squeeze()
    else:
        grad_mask = (grads.squeeze() >= grad_threshold)
    
    # 尺寸条件（小于阈值则克隆）
    if size_threshold is None:
        size_threshold = scene_extent * gaussian_model.percent_dense
    
    scales = gaussian_model.get_scaling
    size_mask = torch.max(scales, dim=1).values <= size_threshold
    
    # 综合条件
    clone_mask = torch.logical_and(grad_mask, size_mask)
    
    return clone_mask
