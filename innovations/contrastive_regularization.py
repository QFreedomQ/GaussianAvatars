import torch
import torch.nn.functional as F

class ContrastiveRegularization:
    """Simple contrastive regularizer between nearby viewpoints.

    Maintains a small cache of downsampled rendered images and encourages
    consistency via cosine similarity.
    """

    def __init__(self, cache_size=2, downsample=8):
        self.cache_size = cache_size
        self.downsample = downsample
        self.cache = []

    def update_cache(self, image):
        with torch.no_grad():
            ds = F.adaptive_avg_pool2d(image.unsqueeze(0), self.downsample).squeeze(0)
            self.cache.append(ds.detach())
            if len(self.cache) > self.cache_size:
                self.cache.pop(0)

    def compute_loss(self, image):
        if len(self.cache) == 0:
            return torch.tensor(0.0, device=image.device)
        ds = F.adaptive_avg_pool2d(image.unsqueeze(0), self.downsample).squeeze(0)
        loss = 0.0
        for cached in self.cache:
            cos = F.cosine_similarity(ds.flatten(), cached.flatten(), dim=0)
            loss = loss + (1 - cos)
        return loss / len(self.cache)
