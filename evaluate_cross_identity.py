#!/usr/bin/env python3
"""
Cross-Identity Reenactment Evaluation Script (BRISQUE Only)

This script evaluates cross-identity reenactment quality using the no-reference
BRISQUE metric, which captures perceptual realism without requiring ground truth.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from argparse import ArgumentParser


def compute_brisque_scores(render_dir: Path):
    """Compute BRISQUE scores for all rendered PNG images in a directory."""
    try:
        from piq import brisque
    except ImportError:
        print("Warning: 'piq' not installed. Install with: pip install piq")
        return None

    images = sorted(Path(render_dir).glob("*.png"))
    if len(images) == 0:
        print(f"Warning: No images found in {render_dir}")
        return None

    brisque_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for img_path in tqdm(images, desc="Computing BRISQUE"):
        img = TF.to_tensor(Image.open(img_path)).unsqueeze(0).to(device)
        score = brisque(img, data_range=1.0, reduction='none')
        brisque_scores.append(score.item())

    stats = {
        'BRISQUE_mean': float(np.mean(brisque_scores)),
        'BRISQUE_std': float(np.std(brisque_scores)),
        'BRISQUE_min': float(np.min(brisque_scores)),
        'BRISQUE_max': float(np.max(brisque_scores)),
        'BRISQUE_scores': brisque_scores,
    }
    return stats


def evaluate_cross_identity(model_path: str, target_name: str, iteration: int = -1):
    """Evaluate cross-identity reenactment results using BRISQUE only."""
    model_path = Path(model_path)

    if iteration == -1:
        target_dirs = list(model_path.glob(f"{target_name}/ours_*"))
        if len(target_dirs) == 0:
            target_dirs = list(model_path.glob("ours_*"))
        if len(target_dirs) == 0:
            raise ValueError(f"No rendering results found in {model_path}")
        reenact_dir = max(target_dirs, key=lambda p: int(p.name.split('_')[-1]))
    else:
        reenact_dir = model_path / target_name / f"ours_{iteration}"

    render_dir = reenact_dir / "renders"
    if not render_dir.exists():
        raise ValueError(f"Render directory not found: {render_dir}")

    print(f"\n{'='*70}")
    print("Evaluating Cross-Identity Reenactment (BRISQUE)")
    print(f"Model: {model_path}")
    print(f"Target: {target_name}")
    print(f"Render directory: {render_dir}")
    print(f"{'='*70}\n")

    results = {}

    print("Computing BRISQUE scores...")
    brisque_results = compute_brisque_scores(render_dir)
    if brisque_results is None:
        print("BRISQUE evaluation failed. No results will be saved.")
        return {}

    results.update(brisque_results)
    print(
        f"BRISQUE: {brisque_results['BRISQUE_mean']:.2f} ± {brisque_results['BRISQUE_std']:.2f} "
        f"(min={brisque_results['BRISQUE_min']:.2f}, max={brisque_results['BRISQUE_max']:.2f})"
    )

    # Save aggregated results (without per-frame list for compactness)
    results_file = reenact_dir / "cross_identity_metrics.json"
    with open(results_file, 'w') as f:
        results_to_save = {k: v for k, v in results.items() if k != 'BRISQUE_scores'}
        json.dump(results_to_save, f, indent=2)

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"BRISQUE Score: {results['BRISQUE_mean']:.2f}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate cross-identity reenactment (BRISQUE only)")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained model")
    parser.add_argument("-t", "--target_name", required=True, help="Target sequence name (e.g., '218_FREE')")
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to evaluate (-1 for latest)")

    args = parser.parse_args()
    evaluate_cross_identity(args.model_path, args.target_name, args.iteration)
