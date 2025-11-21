"""
Expression-Adaptive Appearance Network
表达式自适应着色网络

参考: Neural Head Avatars from Monocular RGB Videos (CVPR 2023)
原理: 为不同 FLAME 表情参数学习条件外观偏移，使相同高斯点在不同表情下呈现不同颜色
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionAdaptiveColorMLP(nn.Module):
    """
    基于 FLAME 表情参数生成颜色偏移的多层感知器
    
    输入: expr_code (B, n_expr)  - FLAME 表情编码
    输出: color_offset (B, 3)   - RGB 颜色偏移 [-1, 1]
    """
    def __init__(self, n_expr=100, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.n_expr = n_expr
        layers = []
        
        # 输入层
        layers.append(nn.Linear(n_expr, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # 输出层 (3通道 RGB)
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Tanh())  # 限制输出范围 [-1, 1]
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化为零，避免初期扰动 baseline
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, expr_code):
        """
        Args:
            expr_code: (B, n_expr) 或 (n_expr,)
        Returns:
            color_offset: (B, 3) 或 (3,)
        """
        if expr_code.dim() == 1:
            expr_code = expr_code.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        color_offset = self.mlp(expr_code)
        
        if squeeze_output:
            color_offset = color_offset.squeeze(0)
        
        return color_offset


class ExpressionAdaptiveModule:
    """
    表达式自适应外观管理器
    用于在 FlameGaussianModel 中集成表达式条件颜色
    """
    def __init__(self, n_expr=100, hidden_dim=128, num_layers=3, lambda_expr_color=0.01):
        self.color_mlp = ExpressionAdaptiveColorMLP(
            n_expr=n_expr,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).cuda()
        
        self.lambda_expr_color = lambda_expr_color
        self.optimizer = None
    
    def setup_optimizer(self, lr=1e-4):
        """设置优化器"""
        self.optimizer = torch.optim.Adam(self.color_mlp.parameters(), lr=lr)
    
    def get_color_offset(self, expr_code, scale=1.0):
        """
        根据表情编码获取颜色偏移
        
        Args:
            expr_code: (n_expr,) FLAME 表情参数
            scale: 缩放因子，控制偏移强度
        Returns:
            color_offset: (3,) RGB 偏移
        """
        with torch.no_grad():
            offset = self.color_mlp(expr_code)
        return offset * scale * self.lambda_expr_color
    
    def compute_color_consistency_loss(self, rendered_colors, expr_codes, neutral_expr_code):
        """
        计算表情颜色一致性损失
        鼓励中性表情下的颜色与其它表情接近（避免过度偏移）
        
        Args:
            rendered_colors: (B, 3) 渲染颜色
            expr_codes: (B, n_expr) 当前帧表情编码
            neutral_expr_code: (n_expr,) 中性表情编码
        Returns:
            loss: 标量
        """
        neutral_offset = self.color_mlp(neutral_expr_code.unsqueeze(0))
        current_offset = self.color_mlp(expr_codes)
        
        # L2 正则化：鼓励偏移保持较小
        consistency_loss = F.mse_loss(current_offset, neutral_offset)
        
        return consistency_loss * self.lambda_expr_color
    
    def state_dict(self):
        return {
            'color_mlp': self.color_mlp.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'lambda_expr_color': self.lambda_expr_color,
        }
    
    def load_state_dict(self, state_dict):
        self.color_mlp.load_state_dict(state_dict['color_mlp'])
        if self.optimizer and state_dict['optimizer']:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lambda_expr_color = state_dict['lambda_expr_color']


def apply_expression_color_to_sh(sh_features, color_offset):
    """
    将表达式颜色偏移应用到球谐（SH）特征
    仅修改 DC 分量（前3个通道），保持高频细节
    
    Args:
        sh_features: (N, 3, (sh_degree+1)^2) 球谐系数
        color_offset: (3,) 或 (N, 3) RGB 偏移
    Returns:
        modified_sh: (N, 3, (sh_degree+1)^2)
    """
    modified_sh = sh_features.clone()
    
    # SH DC 分量在第0个位置
    if color_offset.dim() == 1:
        color_offset = color_offset.unsqueeze(0)  # (1, 3)
    
    # 只修改 DC 分量
    modified_sh[:, :, 0] += color_offset.squeeze(0)
    
    return modified_sh
