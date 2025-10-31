import torch
import torch.nn as nn

class RegionAdaptiveLoss(nn.Module):
    """Region-aware weighting for reconstruction losses.

    This module generates a weight map that emphasizes semantically important
    facial regions. It is designed to be lightweight and usable even when the
    FLAME binding is not available; in that case it falls back to a heuristic
    mask centered around the face.
    """

    def __init__(self, flame_model=None, weight_eyes=2.0, weight_mouth=2.0,
                 weight_nose=1.5, weight_face=1.2, smoothing_kernel=11):
        super().__init__()
        self.flame_model = flame_model
        self.weight_eyes = weight_eyes
        self.weight_mouth = weight_mouth
        self.weight_nose = weight_nose
        self.weight_face = weight_face
        self.smoothing_kernel = smoothing_kernel

        # Pre-compute FLAME-based regions if possible
        self._region_indices = None
        if flame_model is not None and hasattr(flame_model, 'faces'):
            self._region_indices = self._build_flame_region_indices(flame_model)

    @staticmethod
    def _build_flame_region_indices(flame_model):
        """Return a dictionary mapping region names to vertex indices."""
        # Ranges are derived from the public FLAME topology
        return {
            'eyes_left': list(range(3997, 4067)),
            'eyes_right': list(range(3930, 3997)),
            'mouth': list(range(2812, 3025)),
            'nose': list(range(3325, 3450)),
        }

    def _project_vertices(self, verts, camera):
        """Project 3D vertices to normalized screen coordinates (-1, 1)."""
        # Camera utilities already process projection; reuse renderer transforms
        world_view = camera.world_view_transform.to(verts.device)
        projection = camera.full_proj_transform.to(verts.device)
        verts_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1)
        camera_space = (world_view @ verts_h.T).T
        clip_space = (projection @ camera_space.T).T
        ndc = clip_space[:, :3] / clip_space[:, 3:4]
        return ndc[:, :2]

    def _rasterize_region(self, coords, H, W, radius, device, weight):
        mask = torch.zeros((1, H, W), device=device)
        if coords.numel() == 0:
            return mask
        px = (coords[:, 0] * 0.5 + 0.5) * (W - 1)
        py = (-coords[:, 1] * 0.5 + 0.5) * (H - 1)
        px = px.long().clamp(0, W - 1)
        py = py.long().clamp(0, H - 1)
        for x, y in zip(px, py):
            y_min = max(0, y - radius)
            y_max = min(H, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(W, x + radius + 1)
            mask[:, y_min:y_max, x_min:x_max] = weight
        return mask

    def create_weight_map(self, image, camera, gaussians):
        """Generate a per-pixel weight map."""
        _, H, W = image.shape
        device = image.device
        weight_map = torch.ones((1, H, W), device=device)

        if self._region_indices is None or not hasattr(gaussians, 'verts'):
            return self._heuristic_map(H, W, device)

        verts = gaussians.verts.squeeze(0)
        coords = self._project_vertices(verts, camera)
        radius = max(H, W) // 60

        eyes = coords[self._region_indices['eyes_left'] + self._region_indices['eyes_right']]
        weight_map = torch.max(weight_map, self._rasterize_region(eyes, H, W, radius, device, self.weight_eyes))

        mouth = coords[self._region_indices['mouth']]
        weight_map = torch.max(weight_map, self._rasterize_region(mouth, H, W, radius, device, self.weight_mouth))

        nose = coords[self._region_indices['nose']]
        weight_map = torch.max(weight_map, self._rasterize_region(nose, H, W, radius, device, self.weight_nose))

        return weight_map

    def _heuristic_map(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device).view(-1, 1).expand(H, W)
        x = torch.linspace(-1, 1, W, device=device).view(1, -1).expand(H, W)

        face_mask = torch.exp(-((x * 1.2) ** 2 + y ** 2))
        weight_map = 1 + (self.weight_face - 1) * face_mask

        eye_mask = torch.exp(-(((x / 0.3) ** 2 + ((y + 0.2) / 0.15) ** 2)))
        mouth_mask = torch.exp(-(((x / 0.3) ** 2 + ((y - 0.4) / 0.2) ** 2)))
        nose_mask = torch.exp(-(((x / 0.2) ** 2 + (y / 0.3) ** 2)))

        weight_map = torch.maximum(weight_map, 1 + (self.weight_eyes - 1) * eye_mask)
        weight_map = torch.maximum(weight_map, 1 + (self.weight_mouth - 1) * mouth_mask)
        weight_map = torch.maximum(weight_map, 1 + (self.weight_nose - 1) * nose_mask)

        return weight_map.unsqueeze(0)

    def forward(self, pred, target, weight_map):
        error = torch.abs(pred - target)
        weighted = error * weight_map
        return weighted.sum() / (weight_map.sum() + 1e-8)
