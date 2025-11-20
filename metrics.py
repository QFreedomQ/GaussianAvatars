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
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# 导入 LPIPS 类而不是函数，以便我们可以在 GPU 上预先加载模型
from lpipsPyTorch import LPIPS
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


# 已修改：此函数现在只返回文件路径，而不是将所有图像加载到 VRAM
def readImages(renders_dir, gt_dir):
    render_paths = []
    gt_paths = []
    image_names = []
    print(f"Reading from renders_dir: {renders_dir}")
    print(f"Reading from gt_dir: {gt_dir}")

    if not os.path.exists(renders_dir):
        print(f"Error: Renders directory not found: {renders_dir}")
        return [], [], []
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory not found: {gt_dir}")
        return [], [], []

    for fname in os.listdir(renders_dir):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname

        # 确保 render 和 gt 图像都存在
        if os.path.exists(render_path) and os.path.exists(gt_path):
            render_paths.append(render_path)
            gt_paths.append(gt_path)
            image_names.append(fname)
        else:
            if not os.path.exists(gt_path):
                print(f"Warning: Skipping {fname}, corresponding GT image not found at {gt_path}")

    print(f"Found {len(image_names)} matching image pairs.")
    return render_paths, gt_paths, image_names


# 已修改：此函数现在循环加载图像，以避免 OOM
def evaluate(model_paths):
    # (字典初始化已移入循环内部)
    print("")

    # 在循环外将 LPIPS 模型加载到 GPU 一次
    # 这可以防止 OOM 并加快处理速度
    try:
        lpips_model = LPIPS(net_type='vgg').cuda()
    except Exception as e:
        print(f"Error initializing LPIPS model: {e}")
        print(
            "Please ensure you have an internet connection to download LPIPS weights, or the weights are already cached.")
        return

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)

            # *** 新增：循环 "val" 和 "test" 两个目录 ***
            for eval_split_name in ["val", "test"]:
                print(f"\n--- Processing split: {eval_split_name} ---")

                # *** 修改：动态设置评估目录 ***
                eval_dir = Path(scene_dir) / eval_split_name

                if not os.path.exists(eval_dir):
                    print(f"Warning: Directory '{eval_split_name}' not found in {scene_dir}, skipping.")
                    continue  # 跳过这个 split，继续下一个 (例如 "test")

                # *** 修改：为每个 split 初始化独立的字典 ***
                split_full_dict = {}
                split_per_view_dict = {}
                split_full_dict_polytopeonly = {}
                split_per_view_dict_polytopeonly = {}

                # *** 修改：使用 eval_dir 而不是 test_dir ***
                for method in os.listdir(eval_dir):
                    print("Method:", method)

                    # *** 修改：使用 split 字典 ***
                    split_full_dict[method] = {}
                    split_per_view_dict[method] = {}
                    split_full_dict_polytopeonly[method] = {}
                    split_per_view_dict_polytopeonly[method] = {}

                    # *** 修改：使用 eval_dir ***
                    method_dir = eval_dir / method
                    gt_dir = method_dir / "gt"
                    renders_dir = method_dir / "renders"

                    # readImages 现在返回路径
                    render_paths, gt_paths, image_names = readImages(renders_dir, gt_dir)

                    if not image_names:
                        print(f"No images found for method {method} in {scene_dir}/{eval_split_name}. Skipping.")
                        continue

                    ssims = []
                    psnrs = []
                    lpipss = []

                    # 在循环中加载图像，一次一张
                    for idx in tqdm(range(len(render_paths)), desc="Metric evaluation progress"):
                        # 加载一对图像
                        render_img = Image.open(render_paths[idx])
                        gt_img = Image.open(gt_paths[idx])

                        # 将它们转换为张量并移至 GPU
                        render = tf.to_tensor(render_img).unsqueeze(0)[:, :3, :, :].cuda()
                        gt = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()

                        # 计算指标
                        ssims.append(ssim(render, gt))
                        psnrs.append(psnr(render, gt))
                        lpipss.append(lpips_model(render, gt))  # 使用预加载的模型

                        # [重要] 从 VRAM 中删除张量以释放内存
                        del render, gt, render_img, gt_img
                        torch.cuda.empty_cache()

                    # 检查是否实际计算了指标
                    if not ssims:
                        print(f"Warning: Metrics lists are empty for {method}. Check image pairs.")
                        continue

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")

                    # *** 修改：更新 split 字典 ***
                    split_full_dict[method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
                    split_per_view_dict[method].update(
                        {"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                         "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                         "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

                # *** 修改：根据 split 名称保存带前缀的 JSON 文件 ***
                # (这段代码现在在 method 循环之外，但在 split 循环之内)

                if not split_full_dict:
                    print(f"No methods processed for split '{eval_split_name}', not saving JSON files.")
                    continue

                output_results_file = str(Path(scene_dir) / f"{eval_split_name}_results.json")
                output_per_view_file = str(Path(scene_dir) / f"{eval_split_name}_per_view.json")

                print(f"Saving results to {output_results_file}")
                with open(output_results_file, 'w') as fp:
                    json.dump(split_full_dict, fp, indent=True)

                print(f"Saving per-view results to {output_per_view_file}")
                with open(output_per_view_file, 'w') as fp:
                    json.dump(split_per_view_dict, fp, indent=True)

        # 已更正：使用 "except Exception as e:" 来捕获并打印详细错误
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print(f"捕获到的详细错误信息: {e}")
            import traceback
            traceback.print_exc()  # 打印更完整的追溯信息


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
