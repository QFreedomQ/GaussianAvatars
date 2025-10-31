import torch
import torch.nn as nn

class ColorCalibrationNetwork(nn.Module):
    """Lightweight per-pixel color calibration network."""

    def __init__(self, hidden_dim=16, num_layers=3):
        super().__init__()
        layers = []
        in_dim = 3
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim if num_layers > 1 else in_dim, 3))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image):
        original_shape = image.shape
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape
        pixels = image.permute(0, 2, 3, 1).reshape(-1, 3)
        calibrated = self.net(pixels)
        calibrated = calibrated.view(B, H, W, 3).permute(0, 3, 1, 2)
        if len(original_shape) == 3:
            calibrated = calibrated.squeeze(0)
        return calibrated

    def regularizer(self, weight=1e-4):
        reg = 0.0
        for m in self.net:
            if isinstance(m, nn.Linear):
                reg = reg + weight * m.weight.pow(2).mean()
        return reg
