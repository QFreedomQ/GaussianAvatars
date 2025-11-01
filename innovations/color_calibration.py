import torch
import torch.nn as nn

class ColorCalibrationNetwork(nn.Module):
    """Lightweight per-pixel color calibration network using 1x1 convolutions."""

    def __init__(self, hidden_dim=16, num_layers=3):
        super().__init__()
        layers = []
        in_dim = 3
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(in_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_dim if num_layers > 1 else in_dim, 3, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image):
        original_shape = image.shape
        if image.dim() == 3:
            image = image.unsqueeze(0)
        calibrated = self.net(image)
        if len(original_shape) == 3:
            calibrated = calibrated.squeeze(0)
        return calibrated

    def regularizer(self, weight=1e-4):
        reg = 0.0
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                reg = reg + weight * m.weight.pow(2).mean()
        return reg
