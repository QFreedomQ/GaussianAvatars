#!/usr/bin/env python3
"""
BRISQUE Boost Demo Script

Demonstrates the image enhancement capabilities for improving BRISQUE scores
in cross-identity reenactment results.

Usage:
    python demo_brisque_boost.py --input path/to/image.png --output path/to/output.png
    python demo_brisque_boost.py --input_dir path/to/images/ --output_dir path/to/enhanced/
"""

import os
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from argparse import ArgumentParser

from utils.image_enhancement import create_enhancer


def enhance_single_image(input_path, output_path, mode="balanced", show_comparison=False):
    """
    Enhance a single image and optionally compare before/after.
    
    Args:
        input_path: Path to input image
        output_path: Path to save enhanced image
        mode: Enhancement mode ("subtle", "balanced", "aggressive")
        show_comparison: Whether to display side-by-side comparison
    """
    # Load image
    img = Image.open(input_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    # Create enhancer and apply
    enhancer = create_enhancer(mode, device=device)
    enhanced_tensor = enhancer.enhance(img_tensor)
    
    # Save enhanced image
    enhanced_img = TF.to_pil_image(enhanced_tensor.squeeze(0).cpu())
    enhanced_img.save(output_path)
    
    print(f"Enhanced image saved to: {output_path}")
    
    # Optionally compute BRISQUE scores
    try:
        from piq import brisque
        
        with torch.no_grad():
            score_before = brisque(img_tensor, data_range=1.0).item()
            score_after = brisque(enhanced_tensor, data_range=1.0).item()
        
        improvement = score_before - score_after
        improvement_pct = (improvement / score_before) * 100
        
        print(f"\nBRISQUE Scores:")
        print(f"  Before: {score_before:.2f}")
        print(f"  After:  {score_after:.2f}")
        print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")
        
    except ImportError:
        print("\nNote: Install 'piq' to compute BRISQUE scores: pip install piq")
    
    if show_comparison:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img)
            axes[0].set_title(f'Original (BRISQUE: {score_before:.2f})')
            axes[0].axis('off')
            
            axes[1].imshow(enhanced_img)
            axes[1].set_title(f'Enhanced (BRISQUE: {score_after:.2f})')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Note: Install 'matplotlib' to display comparison: pip install matplotlib")


def enhance_directory(input_dir, output_dir, mode="balanced", file_pattern="*.png"):
    """
    Enhance all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        mode: Enhancement mode
        file_pattern: File pattern to match (default: "*.png")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching images
    images = sorted(input_dir.glob(file_pattern))
    
    if len(images) == 0:
        print(f"No images found matching pattern '{file_pattern}' in {input_dir}")
        return
    
    print(f"Found {len(images)} images to enhance")
    print(f"Mode: {mode}")
    
    # Create enhancer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enhancer = create_enhancer(mode, device=device)
    
    # Compute BRISQUE scores if available
    compute_brisque = False
    try:
        from piq import brisque
        compute_brisque = True
        scores_before = []
        scores_after = []
    except ImportError:
        print("Note: Install 'piq' to compute BRISQUE scores")
    
    # Process each image
    for img_path in tqdm(images, desc="Enhancing images"):
        # Load image
        img = Image.open(img_path)
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        
        # Enhance
        with torch.no_grad():
            enhanced_tensor = enhancer.enhance(img_tensor)
            
            if compute_brisque:
                score_before = brisque(img_tensor, data_range=1.0).item()
                score_after = brisque(enhanced_tensor, data_range=1.0).item()
                scores_before.append(score_before)
                scores_after.append(score_after)
        
        # Save
        output_path = output_dir / img_path.name
        enhanced_img = TF.to_pil_image(enhanced_tensor.squeeze(0).cpu())
        enhanced_img.save(output_path)
    
    print(f"\nEnhanced images saved to: {output_dir}")
    
    # Print statistics
    if compute_brisque and len(scores_before) > 0:
        import numpy as np
        
        mean_before = np.mean(scores_before)
        mean_after = np.mean(scores_after)
        improvement = mean_before - mean_after
        improvement_pct = (improvement / mean_before) * 100
        
        print(f"\nBRISQUE Score Statistics:")
        print(f"  Mean before: {mean_before:.2f} ± {np.std(scores_before):.2f}")
        print(f"  Mean after:  {mean_after:.2f} ± {np.std(scores_after):.2f}")
        print(f"  Average improvement: {improvement:.2f} ({improvement_pct:.1f}%)")


def compare_modes(input_path):
    """
    Compare all enhancement modes on a single image.
    
    Args:
        input_path: Path to input image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib required for comparison. Install with: pip install matplotlib")
        return
    
    # Load image
    img = Image.open(input_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    modes = ["off", "subtle", "balanced", "aggressive"]
    results = []
    
    # Compute BRISQUE if available
    compute_brisque = False
    try:
        from piq import brisque
        compute_brisque = True
    except ImportError:
        pass
    
    # Process each mode
    for mode in modes:
        if mode == "off":
            enhanced = img_tensor
        else:
            enhancer = create_enhancer(mode, device=device)
            with torch.no_grad():
                enhanced = enhancer.enhance(img_tensor)
        
        enhanced_img = TF.to_pil_image(enhanced.squeeze(0).cpu())
        
        score = None
        if compute_brisque:
            with torch.no_grad():
                score = brisque(enhanced, data_range=1.0).item()
        
        results.append((mode, enhanced_img, score))
    
    # Display comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (mode, enhanced_img, score) in enumerate(results):
        axes[idx].imshow(enhanced_img)
        title = f'{mode.capitalize()}'
        if score is not None:
            title += f'\nBRISQUE: {score:.2f}'
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('BRISQUE Boost Enhancement Modes Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="BRISQUE Boost Enhancement Demo")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output image path")
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--mode", type=str, default="balanced", 
                        choices=["subtle", "balanced", "aggressive"],
                        help="Enhancement mode")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Compare all modes on input image")
    parser.add_argument("--show_comparison", action="store_true",
                        help="Show before/after comparison")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare_modes:
        if not args.input:
            print("Error: --input required for --compare_modes")
            exit(1)
        compare_modes(args.input)
    
    elif args.input and args.output:
        enhance_single_image(args.input, args.output, args.mode, args.show_comparison)
    
    elif args.input_dir and args.output_dir:
        enhance_directory(args.input_dir, args.output_dir, args.mode)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Enhance single image")
        print("  python demo_brisque_boost.py --input input.png --output output.png --mode balanced")
        print("\n  # Enhance directory of images")
        print("  python demo_brisque_boost.py --input_dir renders/ --output_dir enhanced/ --mode balanced")
        print("\n  # Compare all modes")
        print("  python demo_brisque_boost.py --input input.png --compare_modes")
