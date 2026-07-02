import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_next_bytes,
)


DEPTH_ANYTHING_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def read_points3d_binary_with_ids(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_props = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point_id = int(point_props[0])
            xyz = np.asarray(point_props[1:4], dtype=np.float64)
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            _ = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            points[point_id] = xyz
    return points


def read_points3d_text_with_ids(path):
    points = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            points[point_id] = np.asarray(tuple(map(float, elems[1:4])), dtype=np.float64)
    return points


def load_colmap_sparse(sparse_dir):
    sparse_dir = Path(sparse_dir)
    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"
    if images_bin.exists() and points_bin.exists():
        return read_extrinsics_binary(str(images_bin)), read_points3d_binary_with_ids(points_bin)

    images_txt = sparse_dir / "images.txt"
    points_txt = sparse_dir / "points3D.txt"
    if images_txt.exists() and points_txt.exists():
        return read_extrinsics_text(str(images_txt)), read_points3d_text_with_ids(points_txt)

    raise FileNotFoundError(f"Could not find COLMAP images/points3D files in {sparse_dir}")


def import_depth_anything(depth_anything_root):
    if depth_anything_root:
        sys.path.insert(0, str(Path(depth_anything_root).resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    return DepthAnythingV2


def normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.float32)
    lo = np.percentile(depth[finite], 1.0)
    hi = np.percentile(depth[finite], 99.0)
    if hi <= lo:
        lo = float(depth[finite].min())
        hi = float(depth[finite].max())
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    depth = np.clip(depth, lo, hi)
    return (depth - lo) / (hi - lo)


def robust_affine_fit(x, y, min_points=30, trim_percentile=85.0, num_iters=4):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.shape[0] < min_points:
        return None

    keep = np.ones_like(x, dtype=bool)
    for _ in range(num_iters):
        if keep.sum() < min_points:
            break
        A = np.stack([x[keep], np.ones(keep.sum(), dtype=np.float64)], axis=1)
        scale, offset = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        residual = np.abs(scale * x + offset - y)
        threshold = np.percentile(residual[keep], trim_percentile)
        keep = residual <= max(threshold, 1e-12)

    if keep.sum() < min_points:
        return None
    A = np.stack([x[keep], np.ones(keep.sum(), dtype=np.float64)], axis=1)
    scale, offset = np.linalg.lstsq(A, y[keep], rcond=None)[0]
    residual = np.abs(scale * x[keep] + offset - y[keep])
    return {
        "scale": float(scale),
        "offset": float(offset),
        "num_sparse": int(keep.sum()),
        "mean_abs_error": float(residual.mean()),
    }


def sparse_inverse_depth_samples(image, points3d, depth_norm):
    point_ids = np.asarray(image.point3D_ids)
    valid = point_ids >= 0
    if not valid.any():
        return None, None

    xys = np.asarray(image.xys, dtype=np.float64)[valid]
    point_ids = point_ids[valid]
    xyz_world = []
    xys_kept = []
    for xy, point_id in zip(xys, point_ids):
        xyz = points3d.get(int(point_id))
        if xyz is None:
            continue
        xyz_world.append(xyz)
        xys_kept.append(xy)
    if not xyz_world:
        return None, None

    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    xys_kept = np.asarray(xys_kept, dtype=np.float64)
    rotation = qvec2rotmat(image.qvec)
    translation = np.asarray(image.tvec, dtype=np.float64)
    xyz_camera = (rotation @ xyz_world.T).T + translation
    z = xyz_camera[:, 2]

    h, w = depth_norm.shape
    px = np.rint(xys_kept[:, 0]).astype(np.int64)
    py = np.rint(xys_kept[:, 1]).astype(np.int64)
    valid = (z > 1e-6) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not valid.any():
        return None, None

    pred = depth_norm[py[valid], px[valid]]
    target_inv = 1.0 / z[valid]
    return pred, target_inv


def collect_image_paths(images_dir, extensions):
    paths = []
    for ext in extensions:
        paths.extend(Path(images_dir).glob(f"*{ext}"))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Generate DepthAnythingV2 inverse-depth supervision for this training code.")
    parser.add_argument("--scene-path", required=True, help="Scene split directory, e.g. data/rubble/train")
    parser.add_argument("--images", default="images", help="Image folder name inside scene path")
    parser.add_argument("--sparse", default="sparse/0", help="COLMAP sparse folder inside scene path")
    parser.add_argument("--output", default="depth_any", help="Output folder name inside scene path")
    parser.add_argument("--depth-anything-root", default="", help="Path to Depth-Anything-V2 repository")
    parser.add_argument("--checkpoint", required=True, help="DepthAnythingV2 checkpoint .pth")
    parser.add_argument("--encoder", choices=DEPTH_ANYTHING_CONFIGS.keys(), default="vits")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit. 0 means all images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-sparse-points", type=int, default=30)
    args = parser.parse_args()

    scene_path = Path(args.scene_path)
    images_dir = scene_path / args.images
    sparse_dir = scene_path / args.sparse
    output_dir = scene_path / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    images, points3d = load_colmap_sparse(sparse_dir)
    image_by_name = {Path(image.name).name: image for image in images.values()}
    image_paths = collect_image_paths(images_dir, [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"])
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    DepthAnythingV2 = import_depth_anything(args.depth_anything_root)
    model = DepthAnythingV2(**DEPTH_ANYTHING_CONFIGS[args.encoder])
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    depth_params = {}
    pending_fallback = []
    valid_scales = []
    valid_offsets = []

    for image_path in tqdm(image_paths, desc="DepthAnythingV2"):
        output_path = output_dir / f"{image_path.stem}.png"
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw is None:
            print(f"Warning: failed to read {image_path}")
            continue

        if output_path.exists() and not args.overwrite:
            depth_u8 = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
            depth_norm = depth_u8.astype(np.float32) / 255.0
        else:
            with torch.inference_mode():
                depth = model.infer_image(raw, args.input_size)
            depth_norm = normalize_depth(depth)
            depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_path), depth_u8)

        colmap_image = image_by_name.get(image_path.name)
        fit = None
        if colmap_image is not None:
            pred, target_inv = sparse_inverse_depth_samples(colmap_image, points3d, depth_norm)
            if pred is not None:
                fit = robust_affine_fit(pred, target_inv, min_points=args.min_sparse_points)

        key = image_path.stem
        if fit is None:
            pending_fallback.append(key)
            depth_params[key] = {
                "scale": 1.0,
                "offset": 0.0,
                "num_sparse": 0,
                "mean_abs_error": -1.0,
                "fallback": True,
            }
            continue

        depth_params[key] = fit
        depth_params[key]["fallback"] = False
        valid_scales.append(fit["scale"])
        valid_offsets.append(fit["offset"])

    if pending_fallback and valid_scales:
        fallback_scale = float(np.median(valid_scales))
        fallback_offset = float(np.median(valid_offsets))
        for key in pending_fallback:
            depth_params[key]["scale"] = fallback_scale
            depth_params[key]["offset"] = fallback_offset

    depth_param_path = sparse_dir / "depth_params.json"
    with open(depth_param_path, "w") as f:
        json.dump(depth_params, f, indent=2, sort_keys=True)

    print(f"Wrote {len(depth_params)} depth maps to {output_dir}")
    print(f"Wrote depth params to {depth_param_path}")
    print(f"Valid affine fits: {len(valid_scales)} / {len(depth_params)}")
    if valid_scales:
        print(
            "scale median/std:",
            float(np.median(valid_scales)),
            float(np.std(valid_scales)),
            "offset median/std:",
            float(np.median(valid_offsets)),
            float(np.std(valid_offsets)),
        )


if __name__ == "__main__":
    main()
