"""
Normal Constraint and Curvature Regularization
法线约束与曲率正则

参考: InstantAvatar (CVPR 2023), NeuralBody (ICCV 2021)
原理: 通过强制高斯点遵循底层 FLAME 网格的法线方向，以及对曲率施加平滑约束，
      提升几何一致性和表面质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_vertex_normals(vertices, faces):
    """
    计算三角网格的顶点法线
    
    Args:
        vertices: (N_verts, 3) 顶点坐标
        faces: (N_faces, 3) 面片索引
    Returns:
        normals: (N_verts, 3) 单位法向量
    """
    # 获取每个面的三个顶点
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # 计算面法线（叉积）
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = F.normalize(face_normals, dim=1, eps=1e-6)
    
    # 累加到顶点法线（面积加权）
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)
    
    # 归一化
    vertex_normals = F.normalize(vertex_normals, dim=1, eps=1e-6)
    
    return vertex_normals


def compute_face_normals(vertices, faces):
    """
    计算三角网格的面法线
    
    Args:
        vertices: (N_verts, 3)
        faces: (N_faces, 3)
    Returns:
        face_normals: (N_faces, 3)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = F.normalize(face_normals, dim=1, eps=1e-6)
    
    return face_normals


def compute_gaussian_orientation_alignment_loss(gaussian_rotations, target_normals, weights=None):
    """
    计算高斯旋转与目标法线的对齐损失
    
    Args:
        gaussian_rotations: (N, 4) 四元数 (wxyz 格式)
        target_normals: (N, 3) 目标法线方向
        weights: (N,) 可选权重
    Returns:
        loss: 标量
    """
    # 从四元数提取主轴方向（假设第一个轴为法线方向）
    # q = [w, x, y, z], 主轴 = q * [1, 0, 0] * q^-1
    w, x, y, z = gaussian_rotations[:, 0], gaussian_rotations[:, 1], \
                 gaussian_rotations[:, 2], gaussian_rotations[:, 3]
    
    # 旋转 [1, 0, 0] 向量
    nx = 1 - 2 * (y*y + z*z)
    ny = 2 * (x*y + w*z)
    nz = 2 * (x*z - w*y)
    
    gaussian_normals = torch.stack([nx, ny, nz], dim=1)
    gaussian_normals = F.normalize(gaussian_normals, dim=1, eps=1e-6)
    
    # 余弦相似度损失 (1 - cos(theta))
    cos_sim = (gaussian_normals * target_normals).sum(dim=1)
    loss = 1.0 - cos_sim.abs()  # abs 允许正负方向
    
    if weights is not None:
        loss = loss * weights
    
    return loss.mean()


def compute_laplacian_smoothness_loss(vertices, faces, vertex_offsets=None):
    """
    计算拉普拉斯平滑损失（曲率正则）
    
    Args:
        vertices: (N_verts, 3) 顶点坐标
        faces: (N_faces, 3) 面片索引
        vertex_offsets: (N_verts, 3) 可选的顶点偏移（用于动态偏移正则）
    Returns:
        loss: 标量
    """
    if vertex_offsets is not None:
        vertices = vertices + vertex_offsets
    
    # 构建邻接信息（稀疏版本）
    N = vertices.shape[0]
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)  # (3*N_faces, 2)
    
    # 双向边
    edges = torch.cat([edges, edges.flip(1)], dim=0)
    
    # 去重并排序
    edges = torch.unique(edges, dim=0)
    
    # 计算拉普拉斯坐标
    laplacian = torch.zeros_like(vertices)
    counts = torch.zeros(N, device=vertices.device)
    
    src_idx = edges[:, 0]
    dst_idx = edges[:, 1]
    
    # 累加邻居坐标
    laplacian.index_add_(0, src_idx, vertices[dst_idx])
    counts.index_add_(0, src_idx, torch.ones(edges.shape[0], device=vertices.device))
    
    # 平均
    laplacian = laplacian / (counts.unsqueeze(1) + 1e-8)
    
    # 拉普拉斯损失：V - mean(neighbors)
    laplacian_loss = (vertices - laplacian).pow(2).sum(dim=1).mean()
    
    return laplacian_loss


def compute_normal_consistency_loss(vertices_t0, vertices_t1, faces):
    """
    计算时序法线一致性损失（用于动态序列）
    鼓励相邻帧的法线变化平滑
    
    Args:
        vertices_t0: (N_verts, 3) 前一帧顶点
        vertices_t1: (N_verts, 3) 当前帧顶点
        faces: (N_faces, 3)
    Returns:
        loss: 标量
    """
    normals_t0 = compute_vertex_normals(vertices_t0, faces)
    normals_t1 = compute_vertex_normals(vertices_t1, faces)
    
    # 余弦距离
    cos_sim = (normals_t0 * normals_t1).sum(dim=1)
    loss = (1.0 - cos_sim).mean()
    
    return loss


class NormalRegularizer:
    """
    法线与曲率正则管理器
    集成到训练循环中使用
    """
    def __init__(
        self,
        lambda_normal_align=0.01,
        lambda_laplacian=0.001,
        lambda_normal_consistency=0.005,
    ):
        self.lambda_normal_align = lambda_normal_align
        self.lambda_laplacian = lambda_laplacian
        self.lambda_normal_consistency = lambda_normal_consistency
    
    def compute_loss(
        self,
        gaussian_model,
        mesh_vertices,
        mesh_faces,
        prev_mesh_vertices=None,
    ):
        """
        计算综合正则损失
        
        Args:
            gaussian_model: GaussianModel 实例
            mesh_vertices: (N_verts, 3) 当前网格顶点
            mesh_faces: (N_faces, 3) 面片索引
            prev_mesh_vertices: (N_verts, 3) 可选的前一帧顶点
        Returns:
            total_loss: 标量
            loss_dict: 各项损失字典（用于日志）
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. 法线对齐损失
        if self.lambda_normal_align > 0 and hasattr(gaussian_model, 'binding'):
            # 计算面法线
            face_normals = compute_face_normals(mesh_vertices, mesh_faces)
            
            # 获取高斯绑定的面ID
            binding = gaussian_model.binding
            target_normals = face_normals[binding]  # (N_gaussians, 3)
            
            # 获取高斯旋转（四元数）
            gaussian_rotations = gaussian_model.get_rotation  # (N_gaussians, 4)
            
            normal_align_loss = compute_gaussian_orientation_alignment_loss(
                gaussian_rotations, target_normals
            )
            
            loss_dict['normal_align'] = normal_align_loss.item()
            total_loss += self.lambda_normal_align * normal_align_loss
        
        # 2. 拉普拉斯平滑损失
        if self.lambda_laplacian > 0:
            # 如果有动态偏移，也对其施加平滑
            vertex_offsets = None
            if hasattr(gaussian_model, 'flame_param') and 'dynamic_offset' in gaussian_model.flame_param:
                # 获取当前帧的动态偏移
                timestep = getattr(gaussian_model, 'current_timestep', 0)
                vertex_offsets = gaussian_model.flame_param['dynamic_offset'][timestep]
            
            laplacian_loss = compute_laplacian_smoothness_loss(
                mesh_vertices, mesh_faces, vertex_offsets
            )
            
            loss_dict['laplacian'] = laplacian_loss.item()
            total_loss += self.lambda_laplacian * laplacian_loss
        
        # 3. 时序法线一致性损失
        if self.lambda_normal_consistency > 0 and prev_mesh_vertices is not None:
            normal_consistency_loss = compute_normal_consistency_loss(
                prev_mesh_vertices, mesh_vertices, mesh_faces
            )
            
            loss_dict['normal_consistency'] = normal_consistency_loss.item()
            total_loss += self.lambda_normal_consistency * normal_consistency_loss
        
        return total_loss, loss_dict
