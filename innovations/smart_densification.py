import torch

class SmartDensificationMixin:
    """
    Mix-in class that provides percentile-based adaptive densification.
    
    Overrides densify_and_prune to select thresholds dynamically according to
    the gradient distribution, avoiding fixed thresholds that may not generalize
    across different stages of training.
    """

    def enable_smart_densification(self, percentile_clone=75.0, percentile_split=90.0):
        """Enable smart densification with specified percentiles."""
        self.use_smart_densification = True
        self.densify_clone_percentile = percentile_clone
        self.densify_split_percentile = percentile_split

    def densify_and_prune_smart(self, max_grad, min_opacity, extent, max_screen_size):
        """
        Densify and prune using percentile-based thresholds.
        
        Args:
            max_grad: Fallback gradient threshold
            min_opacity: Minimum opacity for pruning
            extent: Scene extent
            max_screen_size: Maximum screen-space size
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        valid_grads = grads_norm[grads_norm > 0]

        if valid_grads.numel() > 100:
            clone_threshold = torch.quantile(valid_grads, self.densify_clone_percentile / 100.0).item()
            split_threshold = torch.quantile(valid_grads, self.densify_split_percentile / 100.0).item()
            clone_threshold = max(clone_threshold, max_grad * 0.3)
            split_threshold = max(split_threshold, max_grad * 0.7)
        else:
            clone_threshold = max_grad * 0.5
            split_threshold = max_grad

        self.densify_and_clone(grads, clone_threshold, extent)
        self.densify_and_split(grads, split_threshold, extent)

        # Standard pruning
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_vs), big_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
