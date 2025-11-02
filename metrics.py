#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import re
from PIL import Image
from plyfile import PlyData
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def get_gaussian_count(model_path, method):
    """
    Get the number of Gaussian points from the model checkpoint.
    
    Args:
        model_path: Path to the model directory
        method: Method name (e.g., "ours_30000")
    
    Returns:
        Number of Gaussian points, or None if not found
    """
    try:
        # Extract iteration number from the end of the method name (e.g., "ours_30000")
        match = re.search(r"(\d+)$", method)
        if not match:
            return None

        iteration = match.group(1)
        ply_path = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

        if not ply_path.exists():
            return None

        # Load the PLY file and count vertices (Gaussian points)
        plydata = PlyData.read(str(ply_path))
        num_points = len(plydata.elements[0].data)

        return num_points
    except Exception as e:
        print(f"Warning: Could not read Gaussian count: {e}")
        return None

def evaluate(model_paths, report_gaussians=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                mean_ssim = torch.tensor(ssims).mean()
                mean_psnr = torch.tensor(psnrs).mean()
                mean_lpips = torch.tensor(lpipss).mean()

                print("  SSIM : {:>12.7f}".format(mean_ssim.item()))
                print("  PSNR : {:>12.7f}".format(mean_psnr.item()))
                print("  LPIPS: {:>12.7f}".format(mean_lpips.item()))

                gaussian_count = None
                if report_gaussians:
                    gaussian_count = get_gaussian_count(scene_dir, method)
                    if gaussian_count is not None:
                        print("  GAUSSIANS: {:>12d}".format(gaussian_count))
                    else:
                        print("  GAUSSIANS: {:>12}".format("N/A"))

                print("")

                metrics_summary = {
                    "SSIM": mean_ssim.item(),
                    "PSNR": mean_psnr.item(),
                    "LPIPS": mean_lpips.item()
                }
                if report_gaussians and gaussian_count is not None:
                    metrics_summary["GAUSSIANS"] = gaussian_count

                full_dict[scene_dir][method].update(metrics_summary)
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--report_gaussians', action='store_true',
                        help='Include Gaussian point counts in the evaluation output.')
    args = parser.parse_args()
    evaluate(args.model_paths, args.report_gaussians)
