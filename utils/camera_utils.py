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
from scene.cameras import Camera, LightCam
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from transforms3d.euler import euler2mat, mat2euler
from PIL import Image
import cv2

WARNED = False

def loadCam(cam_info, image_scale, znear=0.01, zfar=100.0):
    image = Image.open(cam_info.image_path)
    new_width = int(image.width * image_scale)
    new_height = int(image.height * image_scale)
    resized_image_rgb = image.resize((new_width, new_height), resample=Image.BICUBIC, box=None, reducing_gap=None)
    resized_image_rgb = torch.from_numpy(np.array(resized_image_rgb)) / 255.0
    if len(resized_image_rgb.shape) == 3:
        resized_image_rgb = resized_image_rgb.permute(2, 0, 1)
    else:
        resized_image_rgb = resized_image_rgb.unsqueeze(dim=-1).permute(2, 0, 1)
    if cam_info.depth_filepath is not None:
        scale, offset = cam_info.depth_param["scale"], cam_info.depth_param["offset"]
        depth_mono_inv = cv2.imread(cam_info.depth_filepath, cv2.IMREAD_UNCHANGED)
        if depth_mono_inv.ndim != 2: depth_mono_inv = depth_mono_inv[..., 0]
        depth_mono_inv = cv2.resize(depth_mono_inv, (resized_image_rgb.shape[2], resized_image_rgb.shape[1]), interpolation=cv2.INTER_NEAREST) / 255.0
        depth_inv = depth_mono_inv * scale + offset
        depth_inv = torch.tensor(depth_inv)
    else:
        depth_inv = "none"
    
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3: 4, ...]
     
    image.close()
    
    return Camera(
         colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
         FoVx=cam_info.FovX, FoVy=cam_info.FovY,
         image=gt_image, gt_alpha_mask=loaded_mask,
         image_name=cam_info.image_name, depth_inv=depth_inv,
         znear=znear, zfar=zfar)

def loadCam_woImage(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return LightCam(R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, 
                     FoVy=cam_info.FovY, data_device=args.data_device,
                     width=resolution[0], height=resolution[1])

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
