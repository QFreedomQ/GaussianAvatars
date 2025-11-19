#!/usr/bin/env python3
"""
Cross-Identity Reenactment Evaluation Script

Evaluates the quality of cross-identity reenactment results using no-reference metrics:
- BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator
- Temporal stability: Frame-to-frame consistency
- Optional: Identity preservation (requires insightface)
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


def compute_brisque_scores(render_dir):
    """
    Compute BRISQUE scores for all rendered images.
    
    Args:
        render_dir: Directory containing rendered PNG images
    
    Returns:
        Dictionary with BRISQUE statistics
    """
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
    
    return {
        'BRISQUE_mean': np.mean(brisque_scores),
        'BRISQUE_std': np.std(brisque_scores),
        'BRISQUE_min': np.min(brisque_scores),
        'BRISQUE_max': np.max(brisque_scores),
        'BRISQUE_scores': brisque_scores
    }


def compute_temporal_stability(render_dir):
    """
    Compute temporal stability metrics (frame-to-frame consistency).
    
    Args:
        render_dir: Directory containing rendered PNG images
    
    Returns:
        Dictionary with temporal stability statistics
    """
    from utils.image_utils import psnr as compute_psnr
    
    frames = sorted(Path(render_dir).glob("*.png"))
    if len(frames) < 2:
        print(f"Warning: Not enough frames for temporal analysis in {render_dir}")
        return None
    
    psnrs_inter = []
    l2_diffs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in tqdm(range(len(frames) - 1), desc="Computing temporal stability"):
        frame_t = TF.to_tensor(Image.open(frames[i])).unsqueeze(0).to(device)
        frame_t1 = TF.to_tensor(Image.open(frames[i + 1])).unsqueeze(0).to(device)
        
        # PSNR between consecutive frames
        psnr_val = compute_psnr(frame_t, frame_t1).item()
        psnrs_inter.append(psnr_val)
        
        # L2 difference
        l2_diff = torch.norm(frame_t - frame_t1).item()
        l2_diffs.append(l2_diff)
    
    return {
        'inter_frame_PSNR_mean': np.mean(psnrs_inter),
        'inter_frame_PSNR_std': np.std(psnrs_inter),
        'inter_frame_PSNR_variance': np.var(psnrs_inter),
        'inter_frame_L2_mean': np.mean(l2_diffs),
        'inter_frame_L2_std': np.std(l2_diffs),
    }


def compute_identity_consistency(source_ref_image, render_dir):
    """
    Compute identity preservation score using face recognition.
    
    Args:
        source_ref_image: Reference image of the source identity
        render_dir: Directory containing rendered frames
    
    Returns:
        Dictionary with identity consistency statistics
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("Warning: 'insightface' not installed. Skipping identity consistency.")
        print("Install with: pip install insightface")
        return None
    
    if not os.path.exists(source_ref_image):
        print(f"Warning: Source reference image not found: {source_ref_image}")
        return None
    
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Extract source identity embedding
    source_img = np.array(Image.open(source_ref_image))
    source_faces = app.get(source_img)
    if len(source_faces) == 0:
        print("Warning: No face detected in source image")
        return None
    source_embedding = source_faces[0].embedding
    
    # Extract embeddings from rendered frames
    frames = sorted(Path(render_dir).glob("*.png"))
    similarities = []
    
    for frame_path in tqdm(frames, desc="Computing identity consistency"):
        frame_img = np.array(Image.open(frame_path))
        faces = app.get(frame_img)
        if len(faces) > 0:
            frame_embedding = faces[0].embedding
            # Cosine similarity
            sim = np.dot(source_embedding, frame_embedding) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(frame_embedding)
            )
            similarities.append(sim)
    
    if len(similarities) == 0:
        print("Warning: No faces detected in rendered frames")
        return None
    
    return {
        'identity_score_mean': np.mean(similarities),
        'identity_score_std': np.std(similarities),
        'identity_score_min': np.min(similarities),
        'identity_score_max': np.max(similarities),
    }


def evaluate_cross_identity(
    model_path,
    target_name,
    source_ref_image=None,
    iteration=-1
):
    """
    Evaluate cross-identity reenactment results.
    
    Args:
        model_path: Path to the trained model
        target_name: Name of the target sequence (e.g., "218_FREE")
        source_ref_image: Optional reference image for identity consistency
        iteration: Iteration to evaluate (default: latest)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    # Find the appropriate directory
    model_path = Path(model_path)
    if iteration == -1:
        # Find latest iteration
        target_dirs = list(model_path.glob(f"{target_name}/ours_*"))
        if len(target_dirs) == 0:
            # Try without target name
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
    print(f"Evaluating Cross-Identity Reenactment")
    print(f"Model: {model_path}")
    print(f"Target: {target_name}")
    print(f"Render directory: {render_dir}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # 1. BRISQUE scores
    print("1. Computing BRISQUE scores...")
    brisque_results = compute_brisque_scores(render_dir)
    if brisque_results:
        results.update(brisque_results)
        print(f"   BRISQUE: {brisque_results['BRISQUE_mean']:.2f} ± {brisque_results['BRISQUE_std']:.2f}")
        print(f"   Range: [{brisque_results['BRISQUE_min']:.2f}, {brisque_results['BRISQUE_max']:.2f}]")
    
    # 2. Temporal stability
    print("\n2. Computing temporal stability...")
    temporal_results = compute_temporal_stability(render_dir)
    if temporal_results:
        results.update(temporal_results)
        print(f"   Inter-frame PSNR: {temporal_results['inter_frame_PSNR_mean']:.2f} ± {temporal_results['inter_frame_PSNR_std']:.2f} dB")
        print(f"   Inter-frame variance: {temporal_results['inter_frame_PSNR_variance']:.4f} (lower is better)")
    
    # 3. Identity consistency (optional)
    if source_ref_image:
        print("\n3. Computing identity consistency...")
        identity_results = compute_identity_consistency(source_ref_image, render_dir)
        if identity_results:
            results.update(identity_results)
            print(f"   Identity score: {identity_results['identity_score_mean']:.4f} ± {identity_results['identity_score_std']:.4f}")
    
    # Save results
    results_file = reenact_dir / "cross_identity_metrics.json"
    with open(results_file, 'w') as f:
        # Remove the detailed scores list for cleaner JSON
        results_clean = {k: v for k, v in results.items() if k != 'BRISQUE_scores'}
        json.dump(results_clean, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    if 'BRISQUE_mean' in results:
        print(f"BRISQUE Score:        {results['BRISQUE_mean']:.2f} (lower is better)")
    if 'inter_frame_PSNR_mean' in results:
        print(f"Temporal Stability:   {results['inter_frame_PSNR_mean']:.2f} dB")
    if 'identity_score_mean' in results:
        print(f"Identity Consistency: {results['identity_score_mean']:.4f} (higher is better)")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate cross-identity reenactment")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained model")
    parser.add_argument("-t", "--target_name", required=True, help="Target sequence name (e.g., '218_FREE')")
    parser.add_argument("--source_ref", default=None, help="Source reference image for identity consistency")
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to evaluate (-1 for latest)")
    
    args = parser.parse_args()
    
    evaluate_cross_identity(
        args.model_path,
        args.target_name,
        args.source_ref,
        args.iteration
    )
