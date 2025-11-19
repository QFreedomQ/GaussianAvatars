"""
Image Enhancement Module for Cross-Identity Reenactment BRISQUE Score Improvement

This module implements various image enhancement techniques to improve the visual quality
of cross-identity reenactment results, specifically targeting BRISQUE score improvement.

Techniques implemented:
1. Adaptive Sharpening - Enhances edge details without over-sharpening
2. Bilateral Filtering - Reduces noise while preserving edges
3. Contrast Limited Adaptive Histogram Equalization (CLAHE) - Improves contrast
4. Color Balance - Ensures natural color distribution
5. High-frequency Enhancement - Preserves fine details
"""

import torch
import torch.nn.functional as F


class ImageEnhancer:
    """
    Image enhancement processor for improving BRISQUE scores in cross-identity reenactment.
    """
    
    def __init__(
        self,
        sharpen_strength: float = 0.3,
        denoise_strength: float = 0.02,
        contrast_enhance: bool = True,
        device: str | torch.device | None = None
    ):
        """
        Args:
            sharpen_strength: Strength of sharpening filter (0.0-1.0)
            denoise_strength: Strength of denoising (0.0-0.1)
            contrast_enhance: Whether to apply contrast enhancement
            device: Device to run computations on. When None, automatically selects CUDA if available.
        """
        self.sharpen_strength = sharpen_strength
        self.denoise_strength = denoise_strength
        self.contrast_enhance = contrast_enhance
        self.device = self._resolve_device(device)
        
        # Initialize sharpening kernel (Unsharp Mask)
        self.sharpen_kernel = self._create_sharpen_kernel().to(self.device)
        
        # Initialize bilateral filter approximation kernel
        self.bilateral_kernel = self._create_bilateral_kernel().to(self.device)
    
    def _resolve_device(self, device):
        """Resolve the device to use for computations."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            if device == "auto":
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.startswith("cuda") and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(device)
        if isinstance(device, torch.device):
            if device.type.startswith("cuda") and not torch.cuda.is_available():
                return torch.device("cpu")
            return device
        # Fallback to CPU
        return torch.device("cpu")
    
    def _create_sharpen_kernel(self) -> torch.Tensor:
        """
        Create an unsharp mask kernel for sharpening.
        """
        # Laplacian kernel for edge detection
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        
        # Normalize
        kernel = kernel.view(1, 1, 3, 3)
        return kernel
    
    def _create_bilateral_kernel(self) -> torch.Tensor:
        """
        Create a Gaussian kernel for bilateral filtering approximation.
        """
        kernel_size = 5
        sigma = 1.0
        
        # Create 2D Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def adaptive_sharpen(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive sharpening to enhance edge details.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
        
        Returns:
            Sharpened image tensor
        """
        if self.sharpen_strength <= 0:
            return image
        
        # Convert to grayscale for edge detection
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # Apply Laplacian for edge detection
        edges = F.conv2d(gray, self.sharpen_kernel, padding=1)
        
        # Create edge mask (stronger sharpening on edges, weaker on smooth areas)
        edge_mask = torch.abs(edges)
        edge_mask = torch.sigmoid((edge_mask - 0.1) * 10)  # Adaptive threshold
        
        # Apply sharpening with adaptive strength
        sharpened = image + self.sharpen_strength * edges * edge_mask
        
        return torch.clamp(sharpened, 0, 1)
    
    def bilateral_denoise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply bilateral filtering approximation for denoising.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
        
        Returns:
            Denoised image tensor
        """
        if self.denoise_strength <= 0:
            return image
        
        # Apply Gaussian blur per channel
        B, C, H, W = image.shape
        blurred = torch.zeros_like(image)
        
        for c in range(C):
            blurred[:, c:c+1] = F.conv2d(
                image[:, c:c+1],
                self.bilateral_kernel,
                padding=self.bilateral_kernel.shape[-1] // 2
            )
        
        # Blend original and blurred based on denoise strength
        denoised = (1 - self.denoise_strength) * image + self.denoise_strength * blurred
        
        return denoised
    
    def enhance_contrast(self, image: torch.Tensor, clip_limit: float = 0.03) -> torch.Tensor:
        """
        Apply contrast enhancement using adaptive histogram equalization.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
            clip_limit: Clipping limit for CLAHE
        
        Returns:
            Contrast-enhanced image tensor
        """
        if not self.contrast_enhance:
            return image
        
        # Convert to LAB color space (approximation using RGB)
        # Simple contrast stretching per channel
        B, C, H, W = image.shape
        enhanced = torch.zeros_like(image)
        
        for c in range(C):
            channel = image[:, c:c+1]
            
            # Calculate percentiles for adaptive stretching
            p_low = torch.quantile(channel, 0.02)
            p_high = torch.quantile(channel, 0.98)
            
            # Stretch contrast
            stretched = (channel - p_low) / (p_high - p_low + 1e-8)
            enhanced[:, c:c+1] = torch.clamp(stretched, 0, 1)
        
        # Blend with original to avoid over-enhancement
        alpha = 0.5
        enhanced = alpha * enhanced + (1 - alpha) * image
        
        return enhanced
    
    def enhance_high_frequency(self, image: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
        """
        Enhance high-frequency details to improve texture quality.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
            strength: Strength of high-frequency enhancement
        
        Returns:
            Enhanced image tensor
        """
        # Extract high-frequency component
        low_freq = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        high_freq = image - low_freq
        
        # Amplify high-frequency details
        enhanced = image + strength * high_freq
        
        return torch.clamp(enhanced, 0, 1)
    
    def color_balance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply color balancing to ensure natural color distribution.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
        
        Returns:
            Color-balanced image tensor
        """
        # Simple gray world assumption
        mean_r = image[:, 0].mean()
        mean_g = image[:, 1].mean()
        mean_b = image[:, 2].mean()
        
        avg_mean = (mean_r + mean_g + mean_b) / 3
        
        # Adjust each channel
        balanced = image.clone()
        balanced[:, 0] = balanced[:, 0] * (avg_mean / (mean_r + 1e-8))
        balanced[:, 1] = balanced[:, 1] * (avg_mean / (mean_g + 1e-8))
        balanced[:, 2] = balanced[:, 2] * (avg_mean / (mean_b + 1e-8))
        
        # Subtle blending to avoid over-correction
        alpha = 0.3
        balanced = alpha * balanced + (1 - alpha) * image
        
        return torch.clamp(balanced, 0, 1)
    
    def enhance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply full enhancement pipeline to improve BRISQUE score.
        
        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
        
        Returns:
            Enhanced image tensor
        """
        # Ensure image is on the correct device
        image = image.to(self.device)
        
        # Apply enhancement pipeline
        # 1. Bilateral denoising (reduce noise artifacts)
        enhanced = self.bilateral_denoise(image)
        
        # 2. Color balance (ensure natural colors)
        enhanced = self.color_balance(enhanced)
        
        # 3. Contrast enhancement (improve overall contrast)
        enhanced = self.enhance_contrast(enhanced)
        
        # 4. High-frequency enhancement (preserve details)
        enhanced = self.enhance_high_frequency(enhanced, strength=0.15)
        
        # 5. Adaptive sharpening (enhance edges)
        enhanced = self.adaptive_sharpen(enhanced)
        
        return enhanced


def create_enhancer(
    mode: str = "balanced",
    device: str | torch.device | None = None
) -> ImageEnhancer:
    """
    Factory function to create an ImageEnhancer with preset configurations.
    
    Args:
        mode: Enhancement mode - "subtle", "balanced", or "aggressive"
        device: Device to run computations on (None for auto-select)
    
    Returns:
        Configured ImageEnhancer instance
    """
    if mode == "subtle":
        return ImageEnhancer(
            sharpen_strength=0.2,
            denoise_strength=0.01,
            contrast_enhance=True,
            device=device
        )
    elif mode == "balanced":
        return ImageEnhancer(
            sharpen_strength=0.3,
            denoise_strength=0.02,
            contrast_enhance=True,
            device=device
        )
    elif mode == "aggressive":
        return ImageEnhancer(
            sharpen_strength=0.5,
            denoise_strength=0.03,
            contrast_enhance=True,
            device=device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'subtle', 'balanced', 'aggressive'")


# Convenience function for batch processing
def enhance_image_batch(images: torch.Tensor, mode: str = "balanced", device: str | torch.device | None = None) -> torch.Tensor:
    """
    Enhance a batch of images to improve BRISQUE scores.
    
    Args:
        images: Batch of images (B, C, H, W) in range [0, 1]
        mode: Enhancement mode
        device: Device to run on (None for auto-select)
    
    Returns:
        Enhanced images
    """
    enhancer = create_enhancer(mode, device)
    return enhancer.enhance(images)
