# Modular Innovations for GaussianAvatars
# This package contains five lightweight innovations for efficient avatar training

from .region_adaptive_loss import RegionAdaptiveLoss
from .smart_densification import SmartDensificationMixin
from .progressive_training import ProgressiveResolutionScheduler
from .color_calibration import ColorCalibrationNetwork
from .contrastive_regularization import ContrastiveRegularization

__all__ = [
    'RegionAdaptiveLoss',
    'SmartDensificationMixin',
    'ProgressiveResolutionScheduler',
    'ColorCalibrationNetwork',
    'ContrastiveRegularization',
]
