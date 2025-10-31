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

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr, error_map
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips

from arguments import ModelParams, PipelineParams, OptimizationParams
from innovations import (
    RegionAdaptiveLoss,
    ProgressiveResolutionScheduler,
    ColorCalibrationNetwork,
    ContrastiveRegularization,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _downsample_if_needed(image, scale):
    if scale >= 0.999:
        return image
    new_h = max(1, int(image.shape[-2] * scale))
    new_w = max(1, int(image.shape[-1] * scale))
    return F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    use_amp = getattr(opt, "use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[AMP] Automatic Mixed Precision enabled")

    region_loss_fn = None
    if getattr(opt, "use_region_adaptive_loss", False):
        flame_model = gaussians.flame_model if isinstance(gaussians, FlameGaussianModel) else None
        region_loss_fn = RegionAdaptiveLoss(
            flame_model=flame_model,
            weight_eyes=opt.region_weight_eyes,
            weight_mouth=opt.region_weight_mouth,
            weight_nose=opt.region_weight_nose,
            weight_face=opt.region_weight_face,
        )
        print("[Innovation] Region-adaptive loss enabled")

    if getattr(opt, "use_smart_densification", False):
        gaussians.enable_smart_densification(opt.densify_percentile_clone, opt.densify_percentile_split)
        print(f"[Innovation] Smart densification enabled (clone={opt.densify_percentile_clone}%, split={opt.densify_percentile_split}%)")

    resolution_scheduler = None
    if getattr(opt, "use_progressive_resolution", False):
        resolution_scheduler = ProgressiveResolutionScheduler(opt.resolution_schedule, opt.resolution_milestones)
        print(f"[Innovation] Progressive resolution schedule {opt.resolution_schedule} with milestones {opt.resolution_milestones}")

    color_calibration = None
    color_optimizer = None
    if getattr(opt, "use_color_calibration", False):
        color_calibration = ColorCalibrationNetwork(hidden_dim=opt.color_net_hidden_dim, num_layers=opt.color_net_layers).to("cuda")
        color_optimizer = torch.optim.Adam(color_calibration.parameters(), lr=1e-4)
        print("[Innovation] Color calibration network enabled")

    contrastive_reg = None
    if getattr(opt, "use_contrastive_reg", False):
        contrastive_reg = ContrastiveRegularization(
            cache_size=opt.contrastive_cache_size,
            downsample=opt.contrastive_downsample,
        )
        print("[Innovation] Contrastive regularization enabled")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    loader_camera_train = DataLoader(
        scene.getTrainCameras(),
        batch_size=None,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    iter_camera_train = iter(loader_camera_train)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                custom_cam, msg = network_gui.receive()
                net_image = None
                if custom_cam is not None:
                    if gaussians.binding is not None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])

                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]

                    if gaussians.binding is not None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)
                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)
                        rgb_mesh = rgba_mesh[:3]
                        alpha_mesh = rgba_mesh[3:]
                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding is not None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        scale_factor = resolution_scheduler.get_scale(iteration) if resolution_scheduler else 1.0

        with torch.cuda.amp.autocast(enabled=use_amp):
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image_raw = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            image = color_calibration(image_raw) if color_calibration is not None else image_raw
            gt_image = viewpoint_cam.original_image.cuda()

            image_for_loss = _downsample_if_needed(image, scale_factor)
            gt_for_loss = _downsample_if_needed(gt_image, scale_factor)

            losses = {}
            if region_loss_fn is not None:
                weight_map = region_loss_fn.create_weight_map(image_for_loss, viewpoint_cam, gaussians)
                losses['l1'] = region_loss_fn(image_for_loss, gt_for_loss, weight_map) * (1.0 - opt.lambda_dssim)
            else:
                losses['l1'] = l1_loss(image_for_loss, gt_for_loss) * (1.0 - opt.lambda_dssim)

            losses['ssim'] = (1.0 - ssim(image_for_loss, gt_for_loss)) * opt.lambda_dssim

            if gaussians.binding is not None:
                if opt.metric_xyz:
                    losses['xyz'] = F.relu((gaussians._xyz * gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
                else:
                    losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

                if opt.lambda_scale != 0:
                    if opt.metric_scale:
                        losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                    else:
                        losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

                if opt.lambda_dynamic_offset != 0:
                    losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

                if opt.lambda_dynamic_offset_std != 0:
                    losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std

                if opt.lambda_laplacian != 0:
                    losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian

            if color_calibration is not None and opt.lambda_color_reg > 0:
                losses['color_reg'] = color_calibration.regularizer(opt.lambda_color_reg)

            if contrastive_reg is not None and opt.lambda_contrastive > 0:
                losses['contrastive'] = contrastive_reg.compute_loss(image) * opt.lambda_contrastive

            losses['total'] = sum(losses.values())

        if radii.dtype != torch.float32:
            radii = radii.float()

        scaler.scale(losses['total']).backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                for key in ['xyz', 'scale', 'dy_off', 'lap', 'dynamic_offset_std', 'color_reg', 'contrastive']:
                    if key in losses:
                        postfix[key] = f"{losses[key]:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:
                print(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if contrastive_reg is not None:
                contrastive_reg.update_cache(image.detach())

            if iteration < opt.iterations:
                scaler.step(gaussians.optimizer)
                if color_optimizer is not None:
                    scaler.step(color_optimizer)
                scaler.update()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if color_optimizer is not None:
                    color_optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        for key in ['xyz', 'scale', 'dy_off', 'lap', 'dynamic_offset_std', 'color_reg', 'contrastive']:
            if key in losses:
                tb_writer.add_scalar(f'train_loss_patches/{key}', losses[key].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        print(f"[ITER {iteration}] Evaluating")
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras': scene.getValCameras()},
            {'name': 'test', 'cameras': scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(
                    enumerate(
                        DataLoader(
                            config['cameras'],
                            shuffle=False,
                            batch_size=None,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                        )
                    ),
                    total=len(config['cameras']),
                ):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % max(1, len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                count = len(config['cameras'])
                psnr_test /= count
                l1_test /= count
                lpips_test /= count
                ssim_test /= count
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="Shared iteration interval for test and saving results.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations + 1, args.interval)))

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    safe_state(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
