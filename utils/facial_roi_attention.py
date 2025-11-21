"""
Facial ROI Attention for Cross-Identity Consistency
面部区域注意力机制（跨身份一致性增强）

参考: IDE-3D (ICCV 2023), FaceVerse (CVPR 2022)
原理: 对面部关键区域（眼睛、鼻子、嘴巴）施加额外注意力权重，
      在跨身份重演中保持这些区域的一致性和质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# FLAME 面部关键区域的顶点索引（近似）
# 基于 FLAME 拓扑结构的经验区域划分
FLAME_FACIAL_REGIONS = {
    'left_eye': list(range(1900, 2050)),
    'right_eye': list(range(2100, 2250)),
    'nose': list(range(500, 700)),
    'mouth': list(range(2800, 3100)),
    'left_eyebrow': list(range(1700, 1850)),
    'right_eyebrow': list(range(1850, 2000)),
    'chin': list(range(3200, 3400)),
    'cheek': list(range(1000, 1500)) + list(range(1500, 2000)),
}


def get_roi_mask_from_vertices(vertex_indices, total_vertices, device='cuda'):
    """
    从顶点索引生成 ROI 掩码
    
    Args:
        vertex_indices: List[int] 顶点索引列表
        total_vertices: int 总顶点数
        device: str
    Returns:
        mask: (total_vertices,) bool 张量
    """
    mask = torch.zeros(total_vertices, dtype=torch.bool, device=device)
    mask[vertex_indices] = True
    return mask


def get_roi_mask_from_faces(face_indices, binding, device='cuda'):
    """
    从面片索引和高斯绑定关系生成 ROI 掩码
    
    Args:
        face_indices: List[int] 面片索引列表
        binding: (N,) 高斯到面片的绑定索引
        device: str
    Returns:
        mask: (N,) bool 张量，标记哪些高斯在 ROI 内
    """
    face_mask = torch.zeros(binding.max() + 1, dtype=torch.bool, device=device)
    face_mask[face_indices] = True
    
    # 将面片掩码映射到高斯掩码
    gaussian_mask = face_mask[binding]
    return gaussian_mask


class FacialROIAttention:
    """
    面部区域注意力管理器
    为不同面部区域分配不同的损失权重
    """
    def __init__(
        self,
        region_weights=None,
        enable_roi_loss=True,
        lambda_roi=0.05,
    ):
        """
        Args:
            region_weights: Dict[str, float] 各区域权重
            enable_roi_loss: bool 是否启用 ROI 损失
            lambda_roi: float ROI 损失权重
        """
        if region_weights is None:
            # 默认权重：关键区域（眼睛、嘴巴）权重更高
            region_weights = {
                'left_eye': 2.0,
                'right_eye': 2.0,
                'nose': 1.5,
                'mouth': 2.5,  # 嘴巴最重要
                'left_eyebrow': 1.2,
                'right_eyebrow': 1.2,
                'chin': 0.8,
                'cheek': 1.0,
            }
        
        self.region_weights = region_weights
        self.enable_roi_loss = enable_roi_loss
        self.lambda_roi = lambda_roi
        
        # 缓存的区域掩码
        self.region_masks = {}
        self.gaussian_region_weights = None
    
    def initialize_from_flame_model(self, flame_model, binding):
        """
        从 FLAME 模型初始化区域掩码
        
        Args:
            flame_model: FlameHead 实例
            binding: (N_gaussians,) 高斯到面片的绑定
        """
        num_faces = len(flame_model.faces)
        num_gaussians = len(binding)
        
        # 初始化高斯权重为1.0
        self.gaussian_region_weights = torch.ones(num_gaussians, device=binding.device)
        
        # 为每个区域计算掩码并应用权重
        for region_name, vertex_indices in FLAME_FACIAL_REGIONS.items():
            if region_name not in self.region_weights:
                continue
            
            # 找到包含这些顶点的面片
            faces = flame_model.faces.cpu().numpy()
            region_face_indices = []
            
            for i, face in enumerate(faces):
                if any(v_idx in vertex_indices for v_idx in face):
                    region_face_indices.append(i)
            
            if len(region_face_indices) > 0:
                # 找到绑定到这些面片的高斯
                gaussian_mask = get_roi_mask_from_faces(
                    region_face_indices, binding, device=binding.device
                )
                
                # 应用区域权重
                self.gaussian_region_weights[gaussian_mask] = self.region_weights[region_name]
                self.region_masks[region_name] = gaussian_mask
        
        print(f"[FacialROIAttention] Initialized with {len(self.region_masks)} regions")
        for region_name, mask in self.region_masks.items():
            print(f"  {region_name}: {mask.sum().item()} gaussians (weight={self.region_weights[region_name]})")
    
    def get_pixel_weights(self, rendered_image, gt_image, camera_params=None):
        """
        生成像素级权重图（基于面部检测/分割）
        
        Args:
            rendered_image: (3, H, W)
            gt_image: (3, H, W)
            camera_params: 可选的相机参数
        Returns:
            weights: (H, W) 权重图
        """
        # 简化版本：假设中心区域为面部
        H, W = rendered_image.shape[1:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=rendered_image.device),
            torch.linspace(-1, 1, W, device=rendered_image.device),
            indexing='ij'
        )
        
        # 中心椭圆权重（面部通常在中心）
        distance = ((x / 0.6) ** 2 + (y / 0.8) ** 2)
        weights = torch.exp(-distance * 2.0)  # 高斯衰减
        
        # 归一化
        weights = weights / weights.max()
        weights = weights.clamp(min=0.5, max=2.0)  # 避免极端值
        
        return weights
    
    def compute_roi_weighted_loss(
        self,
        rendered_image,
        gt_image,
        loss_type='l1',
        pixel_weights=None,
    ):
        """
        计算 ROI 加权损失
        
        Args:
            rendered_image: (3, H, W) 或 (B, 3, H, W)
            gt_image: (3, H, W) 或 (B, 3, H, W)
            loss_type: str 'l1' 或 'l2'
            pixel_weights: (H, W) 可选的像素权重
        Returns:
            loss: 标量
        """
        if not self.enable_roi_loss:
            return torch.tensor(0.0, device=rendered_image.device)
        
        # 添加 batch 维度
        if rendered_image.dim() == 3:
            rendered_image = rendered_image.unsqueeze(0)
            gt_image = gt_image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 生成像素权重
        if pixel_weights is None:
            pixel_weights = self.get_pixel_weights(
                rendered_image[0], gt_image[0]
            )
        
        # 计算损失
        if loss_type == 'l1':
            error = torch.abs(rendered_image - gt_image)
        elif loss_type == 'l2':
            error = (rendered_image - gt_image) ** 2
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # 应用权重
        weighted_error = error * pixel_weights.unsqueeze(0).unsqueeze(0)
        
        loss = weighted_error.mean()
        
        return loss * self.lambda_roi
    
    def compute_cross_identity_consistency_loss(
        self,
        source_features,
        target_features,
        region_name='mouth',
    ):
        """
        计算跨身份特定区域的一致性损失
        用于跨身份重演时保持关键区域的外观
        
        Args:
            source_features: (N, D) 源主体特征
            target_features: (N, D) 目标主体特征
            region_name: str 区域名称
        Returns:
            loss: 标量
        """
        if region_name not in self.region_masks:
            return torch.tensor(0.0, device=source_features.device)
        
        mask = self.region_masks[region_name]
        
        # 仅在该区域内计算一致性
        source_roi = source_features[mask]
        target_roi = target_features[mask]
        
        # L2 一致性损失
        consistency_loss = F.mse_loss(source_roi, target_roi)
        
        return consistency_loss * self.lambda_roi
    
    def visualize_roi_weights(self, image_shape, output_path):
        """
        可视化 ROI 权重分布
        
        Args:
            image_shape: (H, W) 图像尺寸
            output_path: str 输出路径
        """
        import matplotlib.pyplot as plt
        
        H, W = image_shape
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dummy_image = torch.zeros(3, H, W, device=device)
        weights = self.get_pixel_weights(dummy_image, dummy_image)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(weights.cpu().numpy(), cmap='hot', interpolation='bilinear')
        plt.colorbar(label='ROI Weight')
        plt.title('Facial ROI Attention Weights')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ROI weights visualization saved to {output_path}")


def create_facial_mask_from_landmarks(landmarks, image_shape, dilation=10):
    """
    从面部关键点创建面部掩码（可用于更精确的 ROI）
    
    Args:
        landmarks: (68, 2) 面部关键点坐标 [x, y]
        image_shape: (H, W)
        dilation: int 膨胀像素数
    Returns:
        mask: (H, W) bool 张量
    """
    H, W = image_shape
    mask = torch.zeros(H, W, dtype=torch.bool)
    
    # 转换关键点到像素坐标
    landmarks_px = landmarks.clone()
    landmarks_px[:, 0] = (landmarks_px[:, 0] + 1) * W / 2
    landmarks_px[:, 1] = (landmarks_px[:, 1] + 1) * H / 2
    landmarks_px = landmarks_px.long()
    
    # 标记关键点位置
    for x, y in landmarks_px:
        if 0 <= x < W and 0 <= y < H:
            y_min = max(0, y - dilation)
            y_max = min(H, y + dilation + 1)
            x_min = max(0, x - dilation)
            x_max = min(W, x + dilation + 1)
            mask[y_min:y_max, x_min:x_max] = True
    
    return mask
