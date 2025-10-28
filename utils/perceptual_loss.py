#
# Perceptual Loss Module inspired by:
# 1. InstantAvatar (CVPR 2023) - https://github.com/tijiang13/InstantAvatar
# 2. NHA (CVPR 2023) - https://github.com/philgras/neural-head-avatars
# 3. LPIPS paper - "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
#
# This module implements a VGG-based perceptual loss for better preservation of
# high-frequency details and semantic features during avatar training.
#

import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss that compares features from multiple layers.
    
    Source: Based on InstantAvatar's perceptual loss implementation
    (https://github.com/tijiang13/InstantAvatar/blob/main/code/model/loss.py)
    
    Principle:
    - Uses pre-trained VGG19 network to extract multi-scale features
    - Computes L1 loss between features of rendered and ground truth images
    - Captures perceptual similarity better than pixel-wise losses
    
    Benefits:
    - Preserves high-frequency facial details (wrinkles, pores, etc.)
    - Improves semantic consistency across different expressions
    - Reduces artifacts in dynamic regions (mouth, eyes)
    """
    
    def __init__(self, layers=[1, 6, 11, 20, 29], normalize=True):
        """
        Args:
            layers: VGG19 layer indices to extract features from
                    [1: relu1_2, 6: relu2_2, 11: relu3_4, 20: relu4_4, 29: relu5_4]
            normalize: Whether to normalize input images to ImageNet stats
        """
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except:
            # Fallback for older torchvision versions
            vgg = models.vgg19(pretrained=True).features
        self.normalize = normalize
        
        # ImageNet normalization stats
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Split VGG into separate feature extractors for each layer
        self.slices = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev_layer, layer_idx + 1)]))
            prev_layer = layer_idx + 1
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # Layer weights from deep to shallow
        
    def forward(self, pred, target):
        """
        Args:
            pred: Rendered image (B, 3, H, W)
            target: Ground truth image (B, 3, H, W)
        
        Returns:
            Weighted sum of feature losses across all layers
        """
        if self.normalize:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, slice_net in enumerate(self.slices):
            x_pred = slice_net(x_pred)
            x_target = slice_net(x_target)
            loss += self.weights[i] * torch.nn.functional.l1_loss(x_pred, x_target)
        
        return loss


class LPIPSWrapper(nn.Module):
    """
    Wrapper for LPIPS loss to use during training in addition to evaluation.
    
    Source: Adapted from NHA (Neural Head Avatars)
    (https://github.com/philgras/neural-head-avatars)
    
    Principle:
    - Uses learned perceptual image patch similarity
    - Better aligned with human perception than VGG features alone
    
    Benefits:
    - State-of-art perceptual metric
    - Handles diverse image corruptions well
    """
    
    def __init__(self, lpips_fn):
        super(LPIPSWrapper, self).__init__()
        self.lpips_fn = lpips_fn
    
    def forward(self, pred, target):
        """
        Args:
            pred: Rendered image (3, H, W)
            target: Ground truth image (3, H, W)
        
        Returns:
            LPIPS distance
        """
        # LPIPS expects input in range [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        
        # Add batch dimension
        pred_scaled = pred_scaled.unsqueeze(0)
        target_scaled = target_scaled.unsqueeze(0)
        
        return self.lpips_fn(pred_scaled, target_scaled).mean()


class CombinedPerceptualLoss(nn.Module):
    """
    Combined perceptual loss using both VGG and LPIPS.
    
    Source: Strategy inspired by multiple recent avatar papers
    - PointAvatar (CVPR 2023)
    - FlashAvatar (ICCV 2023)
    
    Principle:
    - VGG loss: Fast, captures multi-scale features
    - LPIPS loss: Better perceptual alignment, but slower
    - Combine both for optimal quality
    """
    
    def __init__(self, lpips_fn=None, use_vgg=True, use_lpips=False, 
                 vgg_weight=1.0, lpips_weight=0.1):
        super(CombinedPerceptualLoss, self).__init__()
        
        self.use_vgg = use_vgg
        self.use_lpips = use_lpips
        self.vgg_weight = vgg_weight
        self.lpips_weight = lpips_weight
        
        if use_vgg:
            self.vgg_loss = VGGPerceptualLoss()
        
        if use_lpips and lpips_fn is not None:
            self.lpips_loss = LPIPSWrapper(lpips_fn)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Rendered image (3, H, W)
            target: Ground truth image (3, H, W)
        
        Returns:
            Combined perceptual loss
        """
        loss = 0.0
        
        if self.use_vgg:
            # VGG expects (B, C, H, W)
            pred_batch = pred.unsqueeze(0)
            target_batch = target.unsqueeze(0)
            loss += self.vgg_weight * self.vgg_loss(pred_batch, target_batch)
        
        if self.use_lpips:
            loss += self.lpips_weight * self.lpips_loss(pred, target)
        
        return loss
