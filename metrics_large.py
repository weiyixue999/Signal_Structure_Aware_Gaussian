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
from lpipsPyTorch import lpips
import json
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr, color_correct
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, test_sets, correct_color=True):
    # hard code for better result
    correct_color = True

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    

    for scene_dir, test_set in zip(model_paths, test_sets):
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / test_set
        print(f"[DEBUG] test_dir: {test_dir}")
        if not test_dir.exists():
            print(f"[ERROR] test_dir does not exist: {test_dir}")
            continue
        for method in os.listdir(test_dir):
            try:
                print("Method:", method)
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                print(f"[DEBUG] renders_dir: {renders_dir}, gt_dir: {gt_dir}")
                if not renders_dir.exists() or not gt_dir.exists():
                    print(f"[ERROR] renders_dir or gt_dir does not exist for method {method}")
                    continue
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                print(f"[DEBUG] renders: {len(renders)}, gts: {len(gts)}, image_names: {len(image_names)}")

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    try:
                        if correct_color:
                            render_np = np.array(renders[idx], dtype=np.float32).transpose(0, 2, 3, 1)
                            gt_np = np.array(gts[idx], dtype=np.float32).transpose(0, 2, 3, 1)
                            render = torch.from_numpy(np.array(color_correct(render_np, gt_np), dtype=np.float32).transpose(0, 3, 1, 2)).contiguous()
                        else:
                            render = renders[idx]

                        render = render.cuda()
                        gt = gts[idx].cuda()
                        ssims.append(ssim(render, gt))
                        psnrs.append(psnr(render, gt))
                        lpipss.append(lpips(render, gt, net_type='vgg'))
                    except Exception as e:
                        print(f"[ERROR] Exception in metric loop idx={idx}: {e}")
                        import traceback; traceback.print_exc()
                        continue

                if len(ssims) > 0:
                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                else:
                    print("  SSIM : nan\n  PSNR : nan\n  LPIPS: nan")

                full_dict[scene_dir][method].update({"SSIM": float(torch.tensor(ssims).mean().item()) if len(ssims) > 0 else float('nan'),
                                                    "PSNR": float(torch.tensor(psnrs).mean().item()) if len(psnrs) > 0 else float('nan'),
                                                    "LPIPS": float(torch.tensor(lpipss).mean().item()) if len(lpipss) > 0 else float('nan')})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            except Exception as e:
                print(f"[ERROR] Unable to compute metrics for model {scene_dir}, method {method}: {e}")
                import traceback; traceback.print_exc()
                continue
        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--test_sets', '-t', required=False, nargs="+", type=str, default=["test"])
    parser.add_argument('--correct_color', '-c', action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args.model_paths, args.test_sets, args.correct_color)
