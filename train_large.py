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
import time
import yaml
import os
import csv
import torch
import torchvision
from random import randint,gauss
from utils.loss_utils import l1_loss, ssim, src2ref, loss_reproj
from gaussian_renderer import render, render_large, network_gui
import sys
from lightning.pytorch.loggers import (
    TensorBoardLogger,
    WandbLogger,
)
from scene import LargeScene
from scene.datasets import GSDataset, CacheDataLoader, ChunkedCacheDataLoader, construct_cam_info
from utils.camera_utils import loadCam
from utils.general_utils import safe_state, parse_cfg
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.log_utils import tensorboard_log_image, wandb_log_image
from torch.utils.data import DataLoader, dataloader
from argparse import ArgumentParser, Namespace
from arguments import GroupParams
import json
import numpy as np
from pathlib import Path
from PIL import Image


def save_tensor_as_image(tensor, filename):
    #tensor: 3xHxW
    #1.转换成numpy
    np_image = tensor.detach().cpu().numpy()
    #2.调整维度，变成HxWx3
    np_image = np.transpose(np_image, (1, 2, 0))
    #3.映射到0-255
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
    #4.转成PIL Image
    img = Image.fromarray(np_image)
    #5.保存
    img.save(filename)

def get_expon_lr_func(
    lr_init, lr_final,lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init==0.0 and lr_final==0.0):
            #Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0.0, 1.0)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper

class DynamicResolutionScheduler:
    """Controls image resolution and densification windows during training."""

    def __init__(
        self,
        dataset,
        opt,
        no_dynamic_res=False,
        start_level=None,
        model_path=None,
        block_id=-1,
        coarse_train=False,
        lock_resolution_level=False,
    ) -> None:
        self.dataset = dataset
        self.model_path = Path(model_path) if model_path else None
        self.block_id = int(block_id)
        self.coarse_train = bool(coarse_train)
        self.lock_resolution_level = bool(lock_resolution_level)
        self.dynamic_resolution = bool(opt.dynamic_resolution) and not no_dynamic_res
        self.min_level = int(opt.resolution_start_level)
        self.max_level = int(opt.resolution_end_level)
        self.update_interval = max(1, int(opt.resolution_update_interval))
        self.metric_window = max(6, int(opt.resolution_metric_window))
        self.slope_ratio_threshold = float(opt.resolution_slope_ratio_threshold)
        self.curvature_ratio_threshold = float(opt.resolution_curvature_ratio_threshold)
        self.stable_windows_required = max(1, int(opt.resolution_stable_windows))
        self.densify_stage_start = int(opt.densify_stage_start)
        self.densify_stage_end = int(opt.densify_stage_end)
        self.extend_densify = bool(opt.extend_densify_on_resolution_change)

        if not self.dynamic_resolution:
            self.level = self.max_level
        elif start_level is not None:
            self.level = int(start_level)
        else:
            self.level = self.min_level
        self.level = max(self.min_level, min(self.level, self.max_level))

        self.iteration_at_level = 0
        self.extra_densify_iters = 0
        self.stable_windows = 0
        self.metric_history = []
        self.max_positive_slope = 0.0
        self.last_plateau_stats = None
        self.dataset.reset_down_ratio(self.level)
        self.resolution_events = []
        self.metric_records = []
        self._record_resolution_event(
            iteration=0,
            old_level=None,
            new_level=self.level,
            reason="initial",
        )

    @property
    def ratio(self):
        return self.level

    @property
    def records_densification_stats(self):
        return self.iteration_at_level <= self.densify_stage_end + self.extra_densify_iters

    @property
    def should_densify(self):
        upper = self.densify_stage_end + self.extra_densify_iters
        return self.densify_stage_start < self.iteration_at_level <= upper

    def step_iteration(self):
        self.iteration_at_level += 1

    def update_stability_metric(self, metric):
        self.metric_history.append(float(metric))
        is_stable = self._is_metric_plateau()
        if is_stable:
            self.stable_windows += 1
        else:
            self.stable_windows = 0

    def _is_metric_plateau(self):
        if len(self.metric_history) < self.metric_window:
            return False

        y = torch.tensor(self.metric_history[-self.metric_window:], dtype=torch.float32)
        x = torch.arange(self.metric_window, dtype=torch.float32)
        design = torch.stack((x * x, x, torch.ones_like(x)), dim=1)
        a, b, _ = torch.linalg.lstsq(design, y).solution.tolist()

        current_slope = 2.0 * a * float(x[-1]) + b
        current_curvature = 2.0 * a
        if current_slope > self.max_positive_slope:
            self.max_positive_slope = current_slope

        eps = 1e-12
        slope_ref = max(self.max_positive_slope, eps)
        normalized_slope = current_slope / slope_ref
        normalized_curvature = current_curvature / slope_ref
        self.last_plateau_stats = {
            "slope": float(current_slope),
            "curvature": float(current_curvature),
            "normalized_slope": float(normalized_slope),
            "normalized_curvature": float(normalized_curvature),
        }

        return (
            current_slope >= 0.0
            and normalized_slope < self.slope_ratio_threshold
            and normalized_curvature <= self.curvature_ratio_threshold
        )

    def compute_scale_frequency_metric(self, gaussians):
        scaling = gaussians.get_scaling
        opacity = gaussians.get_opacity
        value = 1.0 / torch.mean(scaling)
        weight = opacity ** 2
        weight = weight * scaling.prod(dim=1, keepdim=True)**2
        weight = weight / weight.sum()
        return (value * weight).sum().item()

    def update_resolution_if_needed(self, iteration, gaussians):
        if iteration % self.update_interval != 0:
            return False

        metric = self.compute_scale_frequency_metric(gaussians)
        self.update_stability_metric(metric)
        self._record_metric(iteration, metric)
        if self.last_plateau_stats is not None:
            stats = self.last_plateau_stats
            print(
                "resolution plateau stats: "
                f"level={self.ratio}/{self.max_level}, "
                f"metric={metric:.6f}, "
                f"slope_ratio={stats['normalized_slope']:.4f}, "
                f"curvature_ratio={stats['normalized_curvature']:.4f}, "
                f"stable_windows={self.stable_windows}/{self.stable_windows_required}"
            )
        return self.maybe_increase_resolution(iteration, metric)

    def maybe_increase_resolution(self, iteration, metric=None):
        if self.lock_resolution_level:
            return False
        if not self.dynamic_resolution or self.level >= self.max_level:
            return False
        if self.stable_windows < self.stable_windows_required:
            return False

        unfinished_densify_iters = max(0, self.densify_stage_end - self.iteration_at_level)
        old_level = self.level
        stats = self.last_plateau_stats.copy() if self.last_plateau_stats is not None else None
        self.level += 1
        self.dataset.reset_down_ratio(self.level)
        self._record_resolution_event(
            iteration=iteration,
            old_level=old_level,
            new_level=self.level,
            reason="plateau",
            metric=metric,
            stats=stats,
        )
        self.iteration_at_level = 0
        self.stable_windows = 0
        self.metric_history = []
        self.max_positive_slope = 0.0
        self.last_plateau_stats = None
        if self.extend_densify:
            self.extra_densify_iters += unfinished_densify_iters
        print(f"Increase training resolution level to {self.level}/{self.max_level}.")
        return True

    def _resolution_info(self):
        if len(self.dataset.cameras) == 0:
            orig_width = 0
            orig_height = 0
        else:
            camera = self.dataset.cameras[0]
            orig_width = int(getattr(camera, "width", 0))
            orig_height = int(getattr(camera, "height", 0))

        effective_scale = (
            float(self.dataset.image_ini_scale)
            * float(self.dataset.image_re_scale)
            / float(self.dataset.max_resolution_level)
        )
        return {
            "image_ini_scale": float(self.dataset.image_ini_scale),
            "image_re_scale": int(self.dataset.image_re_scale),
            "effective_image_scale": effective_scale,
            "image_width": int(orig_width * effective_scale),
            "image_height": int(orig_height * effective_scale),
        }

    def _record_resolution_event(
        self,
        iteration,
        old_level,
        new_level,
        reason,
        metric=None,
        stats=None,
    ):
        info = self._resolution_info()
        event = {
            "iteration": int(iteration),
            "block_id": self.block_id,
            "coarse_train": self.coarse_train,
            "reason": reason,
            "old_level": "" if old_level is None else int(old_level),
            "new_level": int(new_level),
            "max_level": self.max_level,
            "metric": "" if metric is None else float(metric),
            "stable_windows": self.stable_windows,
            "stable_windows_required": self.stable_windows_required,
            "iteration_at_level": self.iteration_at_level,
            **info,
        }
        if stats is not None:
            event.update(
                {
                    "slope": float(stats["slope"]),
                    "curvature": float(stats["curvature"]),
                    "normalized_slope": float(stats["normalized_slope"]),
                    "normalized_curvature": float(stats["normalized_curvature"]),
                }
            )
        else:
            event.update(
                {
                    "slope": "",
                    "curvature": "",
                    "normalized_slope": "",
                    "normalized_curvature": "",
                }
            )
        self.resolution_events.append(event)
        self._write_resolution_events()

    def _record_metric(self, iteration, metric):
        stats = self.last_plateau_stats or {}
        self.metric_records.append(
            {
                "iteration": int(iteration),
                "block_id": self.block_id,
                "coarse_train": self.coarse_train,
                "level": self.level,
                "max_level": self.max_level,
                "metric": float(metric),
                "slope": stats.get("slope", ""),
                "curvature": stats.get("curvature", ""),
                "normalized_slope": stats.get("normalized_slope", ""),
                "normalized_curvature": stats.get("normalized_curvature", ""),
                "stable_windows": self.stable_windows,
                "stable_windows_required": self.stable_windows_required,
                "iteration_at_level": self.iteration_at_level,
                **self._resolution_info(),
            }
        )
        self._write_metric_records()

    def _write_csv(self, path, rows):
        if path is None or len(rows) == 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_resolution_events(self):
        if self.model_path is None:
            return
        self._write_csv(self.model_path / "resolution_schedule.csv", self.resolution_events)
        with open(self.model_path / "resolution_schedule.json", "w") as f:
            json.dump(self.resolution_events, f, indent=2)

    def _write_metric_records(self):
        if self.model_path is None:
            return
        self._write_csv(self.model_path / "resolution_metrics.csv", self.metric_records)


def resolve_coarse_resolution_mode(opt):
    mode = getattr(opt, "coarse_resolution_mode", "dynamic")
    if isinstance(mode, str):
        mode = mode.lower()
        if mode == "dynamic":
            return opt.resolution_start_level, False
        if mode in ["min", "start", "low"]:
            return opt.resolution_start_level, True
        if mode.isdigit():
            return int(mode), True
        raise ValueError(
            "Unknown coarse_resolution_mode. Use 'dynamic', 'min', or an integer resolution level."
        )
    return int(mode), True


def resolve_initial_resolution_level(dataset, opt, coarse_train):
    if coarse_train:
        level, _ = resolve_coarse_resolution_mode(opt)
        return level

    coarse_level = opt.resolution_start_level
    scene_dir = Path(dataset.model_path).parent
    coarse_ratio_path = scene_dir / (scene_dir.name + "_coarse") / "coarse_ratio.json"
    if os.path.exists(coarse_ratio_path):
        with open(coarse_ratio_path, "r") as f:
            coarse_ratio_data = json.load(f)
        coarse_level = int(coarse_ratio_data.get("coarse_ratio", coarse_level))
        print(f"Loaded coarse resolution level {coarse_level} from {coarse_ratio_path}.")

    mode = opt.block_resolution_start
    if isinstance(mode, str):
        mode = mode.lower()
        if mode == "coarse":
            return coarse_level
        if mode in ["min", "start", "low"]:
            return opt.resolution_start_level
        if mode.isdigit():
            return int(mode)
        raise ValueError(
            "Unknown block_resolution_start. Use 'min', 'coarse', or an integer resolution level."
        )
    return int(mode)


def should_lock_resolution_level(opt, coarse_train):
    if not coarse_train:
        return False
    _, lock_level = resolve_coarse_resolution_mode(opt)
    return lock_level
                
def training(dataset, opt, pipe, no_dynamic_res, coarse_train, prune_outlier_iter, always_reproj, max_offset_k, testing_iterations, saving_iterations, refilter_iterations, checkpoint_iterations, checkpoint, max_cache_num, debug_from):
    first_iter = 0                      
    log_writer, image_logger = prepare_output_and_logger(dataset)

    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree,max_offset_k)
    scene = LargeScene(dataset, gaussians)
    chunk_cache_size = int(getattr(opt, "chunk_cache_size", 0))
    gs_dataset = GSDataset(
        scene.getTrainCameras(),
        scale=1 / dataset.resolution,
        no_dynamic_res=no_dynamic_res,
        preload=dataset.preload_images,
        cache_size=0 if chunk_cache_size > 0 else dataset.image_cache_size,
        max_resolution_level=opt.resolution_end_level,
        num_preload_workers=opt.data_loader_workers,
        znear=dataset.znear,
        zfar=dataset.zfar,
    )
    # if len(gs_dataset) > 0:
    #     print(f"Using maximum cache size of {max_cache_num} for {len(gs_dataset)} training images")
    #     data_loader = CacheDataLoader(
    #         gs_dataset,
    #         max_cache_num=max_cache_num,
    #         seed=42,
    #         batch_size=1,
    #         shuffle=True,
    #         num_workers=8,
    #         pin_memory=True
    #     )
    if chunk_cache_size > 0:
        print(
            "Using chunked image cache: "
            f"{chunk_cache_size} images for "
            f"{int(opt.chunk_cache_iterations)} iterations per chunk."
        )
        data_loader = ChunkedCacheDataLoader(
            gs_dataset,
            cache_size=chunk_cache_size,
            iterations_per_cache=opt.chunk_cache_iterations,
            seed=42,
            shuffle=True,
            num_workers=opt.data_loader_workers,
        )
    else:
        data_loader_kwargs = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": opt.data_loader_workers,
            "drop_last": False,
            "pin_memory": opt.pin_memory,
        }
        if opt.data_loader_workers > 0:
            data_loader_kwargs["persistent_workers"] = bool(opt.persistent_data_workers)
            data_loader_kwargs["prefetch_factor"] = int(opt.data_prefetch_factor)
        data_loader = torch.utils.data.DataLoader(gs_dataset, **data_loader_kwargs)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_time_render = 0.0
    ema_time_loss = 0.0
    ema_time_densify = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    
    initial_resolution_level = resolve_initial_resolution_level(dataset, opt, coarse_train)
    lock_resolution_level = should_lock_resolution_level(opt, coarse_train)
    
    Scheduler = DynamicResolutionScheduler(
        gs_dataset,
        opt,
        no_dynamic_res,
        initial_resolution_level,
        model_path=dataset.model_path,
        block_id=getattr(dataset, "block_id", -1),
        coarse_train=coarse_train,
        lock_resolution_level=lock_resolution_level,
    )
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=opt.iterations,
    )
    reproject_l1_weight = get_expon_lr_func(
        opt.reproject_l1_weight_init,
        opt.reproject_l1_weight_final,
        max_steps=opt.iterations,
    )
    
    while iteration <= opt.iterations:
        if len(gs_dataset) == 0:
            print("No training data found")
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, dataset)
            break    
        
        for dataset_index, (cam_info, gt_image, depth_inv) in enumerate(data_loader):    
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Render
            start = time.time()
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render_large(cam_info, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

            # Loss
            start = time.time()
            gt_image = gt_image.cuda(non_blocking=opt.pin_memory)
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            has_depth = torch.is_tensor(depth_inv)
            if has_depth:
                depth_rendered_inv = render_pkg["depth_inv"].squeeze(0)
                depth_gt_inv = depth_inv.cuda(non_blocking=opt.pin_memory)
                l1_loss_depth = torch.abs(depth_gt_inv - depth_rendered_inv).mean()
                loss += depth_l1_weight(iteration) * l1_loss_depth
            
            if (has_depth and Scheduler.records_densification_stats) or always_reproj:
                image_height, image_width = cam_info["image_height"], cam_info["image_width"]
                intrinsic, extrinsic = cam_info["intrinsic"].squeeze(0), cam_info["extrinsic"].squeeze(0)

                depth_rendered = (1.0 / (render_pkg["depth_inv"] + 1e-8)).squeeze(0)
                #resclale intrinsic
                intrinsic = intrinsic * Scheduler.ratio / Scheduler.max_level * 1 / dataset.resolution
                
                intrinsic = intrinsic.to(depth_rendered.device)
                image_height = image_height.item()
                image_width = image_width.item()    
            
                disturb = torch.tensor((0.3*image_width*torch.median(depth_rendered)/intrinsic[0,0],0.3*image_height*torch.median(depth_rendered)/intrinsic[0,0],0.0))
                dummy_camera,_,_ = construct_cam_info(image_height, image_width, extrinsic, intrinsic, disturb = disturb)
                temp_data_loader = DataLoader([dummy_camera],batch_size=1,shuffle=False, num_workers=0, pin_memory=True)
                for m in temp_data_loader:
                    dummy_render_pkg = render_large(m, gaussians, pipe, background)
                    dummy_rendered = torch.clamp(dummy_render_pkg["render"], 0.0, 1.0)
                    dummy_depth_rendered = (1.0 / (dummy_render_pkg["depth_inv"] + 1e-8)).squeeze(0)
                    dummy_radii = dummy_render_pkg["radii"]
                    dummy_visibility_filter = dummy_render_pkg["visibility_filter"]
                
                reprojected_depth, reprojected_image = src2ref(
                    intrinsic.to(depth_rendered.device).squeeze(0).float(),
                    extrinsic.to(depth_rendered.device).squeeze(0).float(),
                    depth_rendered,
                    dummy_camera["intrinsic"].to(depth_rendered.device).squeeze(0).float(),
                    dummy_camera["extrinsic"].to(depth_rendered.device).squeeze(0).float(),
                    dummy_depth_rendered,
                    dummy_rendered
                )

                loss_reproj_photo = loss_reproj(reprojected_depth, reprojected_image, gt_image)
                loss += loss_reproj_photo * reproject_l1_weight(iteration)

            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                grads = gaussians.xyz_gradient_accum / gaussians.denom
                grads[grads.isnan()] = 0.0
                ema_time = {
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "densify": ema_time_densify,
                    "num_points": radii.shape[0],
                    "mean_grad": grads.mean().item(),
                }

                lr = {}
                for param_group in gaussians.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                # Log and save
                # training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                #                 iter_start.elapsed_time(iter_end), testing_iterations, scene, render_large, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, dataset)


                # Densification
                if Scheduler.records_densification_stats:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if Scheduler.should_densify and iteration % opt.densification_interval == 0:
                        start = time.time()
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.densify_min_opacity,
                            scene.cameras_extent,
                            opt.densify_max_screen_size,
                        )
                        end = time.time()
                        ema_time_densify = 0.4 * (end - start) + 0.6 * ema_time_densify

                        # reset opacity
                        if (Scheduler.should_densify and iteration % opt.opacity_reset_interval == 0 and iteration < opt.densify_until_iter) or(dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                            print("reset opacity")
                
                #删除错误优化的高斯点，在任何阶段
                if iteration > opt.outlier_prune_start_iter and iteration % prune_outlier_iter == 0:
                    size_threshold = opt.outlier_prune_screen_size_scale * Scheduler.ratio
                    gaussians.prune_outlier(scene.cameras_extent, size_threshold)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # Log and save
                training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, render_large, (pipe, background))
                    
            iteration += 1
            
            if Scheduler.update_resolution_if_needed(iteration, gaussians):
                print("update_ratio_iteration:",iteration)
                break
            Scheduler.step_iteration()
            
            if iteration >= opt.iterations:
                break
        file_name = "coarse_ratio.json"
        path = os.path.join(dataset.model_path, file_name)
        os.makedirs(os.path.dirname(path),exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({"coarse_ratio":Scheduler.ratio}, f)

def prepare_output_and_logger(args):    
    if not args.model_path:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        args.model_path = os.path.join("./output/", config_name)
        if args.block_id >= 0:
            if args.block_id < args.block_dim[0] * args.block_dim[1] * args.block_dim[2]:
                args.model_path = f"{args.model_path}/cells/cell{args.block_id}"
                if args.logger_config is not None:
                    args.logger_config['name'] = f"{args.logger_config['name']}_cell{args.block_id}"
            else:
                raise ValueError("Invalid block_id: {}".format(args.block_id))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # build logger
    log_writer = None
    image_logger = None
    logger_args = {
        "save_dir": args.model_path
    }
    if args.logger_config is None or args.logger_config['logger'] == "tensorboard":
        try:
            log_writer = TensorBoardLogger(**logger_args)
            image_logger = tensorboard_log_image
        except ModuleNotFoundError as exc:
            print(
                "TensorBoard logging is disabled because neither tensorboard "
                "nor tensorboardX is installed."
            )
            log_writer = None
            image_logger = None
    elif args.logger_config['logger'] == "wandb":
        logger_args.update(name=args.logger_config['name'])
        logger_args.update(project=args.logger_config['project'])
        log_writer = WandbLogger(**logger_args)
        image_logger = wandb_log_image
    else:
        raise ValueError("Unknown logger: {}".format(args.logger_config['logger']))
    
    return log_writer, image_logger

def training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr, elapsed, testing_iterations, scene : LargeScene, gaussians,renderFunc, renderArgs):
    if log_writer:
        metrics_to_log = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_time/render": ema_time["render"],
            "train_time/loss": ema_time["loss"],
            "train_time/densify": ema_time["densify"],
            "train_time/num_points": ema_time["num_points"],
            "train_time/mean_grad": ema_time["mean_grad"],
            "iter_time": elapsed,
        }
        for key, value in lr.items():
            metrics_to_log["trainer/" + key] = value
        log_writer.log_metrics(metrics_to_log, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        gs_dataset = GSDataset(
            [scene.getTrainCameras()[idx%len(scene.getTrainCameras())] for idx in range(5,30,5)],
            scale=0.25,
            preload=False,
            znear=dataset.znear,
            zfar=dataset.zfar,
        )
        data_loader = DataLoader(gs_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        l1_test = 0.0
        psnr_test = 0.0
        
        for idx, (cam_info,gt_image,_) in enumerate(tqdm(data_loader,desc="Rendering progress")):
            image = torch.clamp(renderFunc(cam_info, gaussians, *renderArgs)["render"],0.0,1.0)
            gt_image = torch.clamp(gt_image.to("cuda"),0.0,1.0)
            l1_test += l1_loss(image,gt_image).mean().float()
            psnr_test += psnr(image,gt_image).mean().float()
        psnr_test /= len(gs_dataset)
        l1_test /= len(gs_dataset)
        print(psnr_test,l1_test)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 10_000, 16_000, 29_000,30_000,35_000,40_000,45_000,50_000,55_000,60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000,15_000,30_000,40_000,50_000,60_000])
    parser.add_argument("--refilter_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--max_cache_num", type=int, default=512)
    parser.add_argument("--max_offset_k", type=int, default=10000)
    parser.add_argument("--always_reproj", action="store_true", default=False)
    parser.add_argument("--prune_outlier_iter", type=int, default=800)
    parser.add_argument("--coarse_train", action="store_true", default=False)
    parser.add_argument("--no_dynamic_res", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)
        args.save_iterations.append(op.iterations)
    
    print("Optimizing " + lp.model_path)

    torch.inverse(torch.ones((1,1),device="cuda"))
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print("DEBUG: dataset.model_path =", lp.model_path)

    training(lp, op, pp, args.no_dynamic_res, args.coarse_train, args.prune_outlier_iter, args.always_reproj, args.max_offset_k, args.test_iterations, args.save_iterations, args.refilter_iterations, args.checkpoint_iterations, args.start_checkpoint, args.max_cache_num, args.debug_from)

    # All done
    print("\nTraining complete.")
