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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, pi

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def opacity_loss(gaussians):
    return gaussians.get_opacity.sum() * 1e-6

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def src2ref(ref_intrinsic, ref_extrinsic, ref_view_depth, dummy_intrinsic, dummy_extrinsic, dummy_view_depth, dummy_image):
    device = ref_extrinsic.device
    ref_height, ref_width = ref_view_depth.shape[0], ref_view_depth.shape[1]
    u, v = torch.meshgrid(torch.arange(ref_width, device=device), torch.arange(ref_height, device=device), indexing="xy") # u->right, v->down
    u, v = u.flatten().to(torch.float32) + 0.5, v.flatten().to(torch.float32) + 0.5

    z_ref = ref_view_depth.flatten()
    uv1 = torch.stack((u * z_ref, v * z_ref, z_ref), dim=0) # [3, H*W]
    xyz_ref = torch.matmul(torch.linalg.inv(ref_intrinsic), uv1)    # [3, H*W]
    xyz_ref_homo = torch.cat((xyz_ref, torch.ones((1, xyz_ref.shape[1]), device=device)), dim=0)    # [4, H*W]
    xyz_world = torch.matmul(torch.linalg.inv(ref_extrinsic), xyz_ref_homo) # [4, H*W]

    xyz_src = torch.matmul(dummy_extrinsic, xyz_world)[:3, :]  # [3, H*W]
    uv_src = torch.matmul(dummy_intrinsic, xyz_src)    # [3, H*W]
    u_src = (uv_src[0, :] / (uv_src[2, :]+1e-8)).view(ref_height, ref_width)
    v_src = (uv_src[1, :] / (uv_src[2, :]+1e-8)).view(ref_height, ref_width)
    
    u_src = 2.0 * (u_src / (ref_width - 1)) - 1.0
    v_src = 2.0 * (v_src / (ref_height - 1)) - 1.0
    grid = torch.stack((u_src, v_src), dim=-1).unsqueeze(0)  # [1, H, W, 2]
    reprojected_depth = torch.nn.functional.grid_sample(dummy_view_depth.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", align_corners=True)   # [1, 1, H, W]
    reprojected_depth = reprojected_depth.squeeze()

    reprojected_image = torch.nn.functional.grid_sample(dummy_image.unsqueeze(0), grid, mode="bilinear", align_corners=True)
    reprojected_image = reprojected_image.squeeze(0)

    return reprojected_depth, reprojected_image



# def src2ref(ref_intrinsic, ref_extrinsic, ref_view_depth, dummy_intrinsic, dummy_extrinsic, dummy_view_depth, dummy_image):
#     device = ref_extrinsic.device
#     # ensure float
#     ref_intrinsic = ref_intrinsic.to(device).float()
#     ref_extrinsic = ref_extrinsic.to(device).float()
#     dummy_intrinsic = dummy_intrinsic.to(device).float()
#     dummy_extrinsic = dummy_extrinsic.to(device).float()
#     ref_view_depth = ref_view_depth.to(device).float()
#     dummy_view_depth = dummy_view_depth.to(device).float()
#     dummy_image = dummy_image.to(device).float()

#     ref_h, ref_w = ref_view_depth.shape[-2], ref_view_depth.shape[-1]  # [H_ref, W_ref]

#     # Build pixel grid in ref image
#     u, v = torch.meshgrid(
#         torch.arange(ref_w, device=device, dtype=torch.float32),
#         torch.arange(ref_h, device=device, dtype=torch.float32),
#         indexing="xy"
#     )  # u shape [W_ref, H_ref] because of indexing="xy"
#     # transpose to shape [H_ref, W_ref]
#     u = u.t().flatten() + 0.5
#     v = v.t().flatten() + 0.5

#     z_ref = ref_view_depth.flatten()  # [H_ref * W_ref]

#     # backproject into ref camera coords then world
#     uv1 = torch.stack((u * z_ref, v * z_ref, z_ref), dim=0)  # [3, Npix]
#     inv_ref_intr = torch.linalg.inv(ref_intrinsic)
#     xyz_ref = inv_ref_intr.matmul(uv1)  # [3, Npix]
#     xyz_ref_h = torch.cat((xyz_ref, torch.ones((1, xyz_ref.shape[1]), device=device)), dim=0)  # [4, Npix]
#     inv_ref_ext = torch.linalg.inv(ref_extrinsic)
#     xyz_world = inv_ref_ext.matmul(xyz_ref_h)  # [4, Npix]

#     # world -> dummy (source) camera coordinates
#     xyz_src = dummy_extrinsic.matmul(xyz_world)[:3, :]  # [3, Npix]
#     uv_src = dummy_intrinsic.matmul(xyz_src)  # [3, Npix]

#     # pixel coords in source image (floating)
#     u_src_pixels = (uv_src[0, :] / (uv_src[2, :] + 1e-8)).view(ref_h, ref_w)
#     v_src_pixels = (uv_src[1, :] / (uv_src[2, :] + 1e-8)).view(ref_h, ref_w)

#     # determine source image size for proper normalization
#     # handle dummy_view_depth shapes
#     if dummy_view_depth.dim() == 2:
#         src_h, src_w = dummy_view_depth.shape
#     elif dummy_view_depth.dim() == 3:
#         # possibly [1,H,W] or [C,H,W] - prefer height/width from last two dims
#         src_h, src_w = dummy_view_depth.shape[-2], dummy_view_depth.shape[-1]
#     else:
#         raise ValueError("dummy_view_depth has unsupported dim: %d" % (dummy_view_depth.dim()))

#     # Normalize to [-1, 1] using source (dummy) image size (important!)
#     u_norm = 2.0 * (u_src_pixels / (src_w - 1)) - 1.0
#     v_norm = 2.0 * (v_src_pixels / (src_h - 1)) - 1.0

#     # grid for grid_sample must be [N, H_out, W_out, 2] with (x, y) coords
#     # grid_sample expects last dim ordering (x, y) <-> (u, v)
#     grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)  # [1, H_ref, W_ref, 2]

#     # prepare dummy_view_depth input for grid_sample: ensure shape [N, C, H_src, W_src]
#     if dummy_view_depth.dim() == 2:
#         depth_in = dummy_view_depth.unsqueeze(0).unsqueeze(0)  # [1,1,H_src,W_src]
#     elif dummy_view_depth.dim() == 3:
#         # if [1,H,W] treat as channel=1; if [C,H,W] treat as channels
#         if dummy_view_depth.shape[0] == 1:
#             depth_in = dummy_view_depth.unsqueeze(0)  # [1,1,H,W]
#         else:
#             depth_in = dummy_view_depth.unsqueeze(0)  # [1,C,H,W]
#     else:
#         raise ValueError("dummy_view_depth has unsupported dim for grid_sample")

#     # sample depth
#     reprojected_depth = torch.nn.functional.grid_sample(depth_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
#     # reprojected_depth -> [1, C_depth, H_ref, W_ref] ; we expect channel 1
#     reprojected_depth = reprojected_depth.squeeze()  # [H_ref, W_ref] if single channel

#     # prepare dummy_image input: ensure [1, C, H_src, W_src]
#     if dummy_image.dim() == 2:
#         # grayscale HxW -> treat as single channel
#         image_in = dummy_image.unsqueeze(0).unsqueeze(0)
#     elif dummy_image.dim() == 3:
#         # [C, H, W] -> [1, C, H, W]
#         image_in = dummy_image.unsqueeze(0)
#     elif dummy_image.dim() == 4:
#         # already [N, C, H, W] ; if N != 1, keep it but grid must match N
#         if dummy_image.shape[0] != 1:
#             # if multiple batch images are supplied, only support N=1 for now
#             raise ValueError("src2ref currently supports dummy_image with batch size 1")
#         image_in = dummy_image
#     else:
#         raise ValueError("dummy_image has unsupported dim: %d" % (dummy_image.dim()))

#     # sample image
#     reprojected_image = torch.nn.functional.grid_sample(image_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
#     # reprojected_image: [1, C, H_ref, W_ref]
#     reprojected_image = reprojected_image.squeeze(0)  # [C, H_ref, W_ref]

#     return reprojected_depth, reprojected_image



def loss_reproj(reprojected_depth, reprojected_image, image_gt):
    mask = (reprojected_depth == 0)
    mask = mask.expand_as(image_gt)
    loss = torch.abs(image_gt - reprojected_image)[mask].mean()
    return loss
