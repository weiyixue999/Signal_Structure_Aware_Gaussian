# Official implementation of “Signal Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction” (ICLR 2026).

This repository trains large-scene 3D Gaussian Splatting models with two main additions over vanilla 3DGS:

- Dynamic-resolution training: images start from a low resolution level and are promoted when the Gaussian scale metric becomes stable.
- Staged densification: densification statistics and pruning are tied to the current resolution stage, so each stage can grow/refine Gaussians before moving to a higher image resolution.

The typical pipeline is:

1. Train a coarse global Gaussian model.
2. Partition the scene into spatial blocks using the coarse model.
3. Train each block.
4. Merge block Gaussians.
5. Render and evaluate.

## Dataset Layout

The code expects COLMAP-style data:

```text
data/<scene>/
  train/
    images/
    sparse/0/
  val/
    images/
    sparse/0/
```

For the included rubble configuration:

```text
data/rubble/train
data/rubble/val
```

## Environment

Use a Python environment with PyTorch, CUDA, COLMAP/3DGS dependencies, and the renderer dependencies installed. The current development environment was run with:

```bash
conda activate SIG
```

Common Python dependencies include:

```bash
pip install lightning transforms3d lpips plyfile tqdm pyyaml pillow opencv-python
```

If pip needs your local proxy:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=http://127.0.0.1:7890
```

## Quick Start

The recommended entry point is the script under `scripts/`:

```bash
bash scripts/run_citygs.sh
```

The script contains the full pipeline:

- coarse global training
- scene partitioning
- per-block fine training
- block merging
- rendering
- metric evaluation

Check that the coarse-training line in `scripts/run_citygs.sh` is enabled if you do not already have a coarse checkpoint. If the coarse model was already trained, the script can start from partition/block training.

## Important Configuration

### Coarse Training

File:

```text
config/rubble_coarse.yaml
```

Useful fields:

```yaml
iterations: 15_000
resolution: 4
coarse_resolution_mode: "min"
chunk_cache_size: 500
chunk_cache_iterations: 2500
init_point_max_points: 0
init_point_extent_multiplier: 0.0
```

`coarse_resolution_mode` controls coarse image resolution:

```text
"dynamic"  start low and use the resolution scheduler
"min"      keep the coarse model at resolution_start_level
integer    lock coarse training to a specific level
```

`init_point_max_points: 0` keeps all original COLMAP points. Setting it to a positive value randomly downsamples the initial point cloud.

### Fine / Block Training

File:

```text
config/rubble_c9_r4.yaml
```

The coarse model path should usually point to the parent `point_cloud` directory:

```yaml
pretrain_path: "output/rubble_coarse/point_cloud"
```

The code automatically resolves the latest available iteration, e.g.:

```text
output/rubble_coarse/point_cloud/iteration_15000/point_cloud.ply
```

This avoids hard-coding `iteration_30000` when coarse training uses a different iteration count.

## Dynamic Resolution Scheduler

The scheduler is configured in the `optim_params` section of the fine config:

```yaml
dynamic_resolution: True
resolution_start_level: 1
resolution_end_level: 5
block_resolution_start: "min"
resolution_update_interval: 100
resolution_metric_window: 8
resolution_slope_ratio_threshold: 0.05
resolution_curvature_ratio_threshold: 0.0
resolution_stable_windows: 3
densify_stage_start: 500
densify_stage_end: 3000
extend_densify_on_resolution_change: True
```

Resolution levels are integer multipliers between `resolution_start_level` and `resolution_end_level`. With `resolution: 4` and `resolution_end_level: 5`, rubble uses:

```text
level 1: 0.25 * 1 / 5 = 0.05 -> 230x172
level 2: 0.25 * 2 / 5 = 0.10 -> 460x345
level 3: 0.25 * 3 / 5 = 0.15 -> 691x518
level 4: 0.25 * 4 / 5 = 0.20 -> 921x691
level 5: 0.25 * 5 / 5 = 0.25 -> 1152x864
```

The scheduler samples a scale-frequency metric every `resolution_update_interval` iterations. It fits a quadratic curve over the latest `resolution_metric_window` samples and checks:

```text
slope >= 0
normalized_slope < resolution_slope_ratio_threshold
normalized_curvature <= resolution_curvature_ratio_threshold
```

After `resolution_stable_windows` consecutive stable windows, the dataset switches to the next higher resolution level.

For block training, if resolution promotion is too slow, try:

```yaml
block_resolution_start: "coarse"
resolution_slope_ratio_threshold: 0.15
resolution_stable_windows: 1
```

## Image Loading and Chunk Cache

Raw images can be large, so reading and resizing every image on every iteration is expensive. This code supports chunked CPU caching:

```yaml
chunk_cache_size: 500
chunk_cache_iterations: 2500
```

Behavior:

1. Load and resize 500 images at the current resolution level.
2. Train for 2500 iterations by sampling from that cached chunk.
3. Release the chunk and cache the next 500 images.
4. If the resolution level changes, rebuild the chunk at the new resolution.

This is separate from the older LRU `image_cache_size`. For chunked caching, keep:

```yaml
image_cache_size: 0
```

## Rendering and Metrics

Rendering for evaluation uses the base training scale `1 / resolution` but disables the dynamic-resolution low level. For rubble:

```text
original val image: 4608x3456
resolution: 4
render/eval image: 1152x864
```

Outputs are written under:

```text
output/rubble_c9_r4/val/ours_<iteration>/renders
output/rubble_c9_r4/val/ours_<iteration>/gt
output/rubble_c9_r4/results.json
```

## Logs

Each training run writes scheduler logs:

```text
resolution_schedule.csv
resolution_schedule.json
resolution_metrics.csv
```

For coarse:

```text
output/rubble_coarse/resolution_schedule.csv
```

For blocks:

```text
output/rubble_c9_r4/cells/cell0/resolution_schedule.csv
output/rubble_c9_r4/cells/cell1/resolution_schedule.csv
...
```

`resolution_schedule.csv` records level changes. `resolution_metrics.csv` records every metric sample, including slope and curvature statistics.

## Common Issues

### Fine training cannot find `iteration_30000`

Do not hard-code the coarse iteration unless needed. Prefer:

```yaml
pretrain_path: "output/rubble_coarse/point_cloud"
```

The loader will automatically use the latest `iteration_*` directory.

### Coarse starts with fewer COLMAP points than expected

Use:

```yaml
init_point_max_points: 0
init_point_extent_multiplier: 0.0
```

This keeps the original COLMAP point count.

### Block training stays at low resolution

The stability threshold may be too conservative. Inspect:

```text
output/<scene>/cells/cell*/resolution_metrics.csv
```

Then relax:

```yaml
resolution_slope_ratio_threshold: 0.15
resolution_stable_windows: 1
```

or start blocks from the coarse level:

```yaml
block_resolution_start: "coarse"
```

### Training speed is near 20 it/s

The bottleneck is usually image decode/resize, not CPU-to-GPU copy. Enable chunk cache:

```yaml
chunk_cache_size: 500
chunk_cache_iterations: 2500
```

For even larger speedups, precompute an image pyramid offline and train directly from resized images.

## Notes

This repository is derived from 3D Gaussian Splatting style training code and is intended for large-scale scene experiments. Before release, check licenses of inherited files and third-party renderer components.
