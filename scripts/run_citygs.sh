set -euo pipefail

get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold && selected == "" { selected = $1 }
  END { if (selected != "") print selected }
  '
}

TEST_PATH="data/rubble/val"

COARSE_CONFIG="rubble_coarse"
CONFIG="rubble_c9_r4"

# Change this one variable to move all generated training outputs.
# Example: OUTPUT_ROOT=output_exp01 bash scripts/run_citygs.sh
OUTPUT_ROOT="${OUTPUT_ROOT:-output_new_2}"
COARSE_MODEL_PATH="$OUTPUT_ROOT/$COARSE_CONFIG"
MODEL_PATH="$OUTPUT_ROOT/$CONFIG"
COARSE_POINT_CLOUD_PATH="$COARSE_MODEL_PATH/point_cloud"

out_name="val"  # i.e. TEST_PATH.split('/')[-1]
max_block_id=3  # i.e. x_dim * y_dim * z_dim - 1
port=4041

# train coarse global gaussian model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python train_large.py --config config/$COARSE_CONFIG.yaml --model_path "$COARSE_MODEL_PATH" --max_offset_k 25 --prune_outlier_iter 800 --coarse_train 

# train CityGaussian
# obtain data partitioning
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python data_partition.py --config config/$CONFIG.yaml --model_path "$MODEL_PATH" --pretrain_path "$COARSE_POINT_CLOUD_PATH"

# optimize each block, please adjust block number according to config
pids=()
for num in $(seq 0 $max_block_id); do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting training block '$num'"
            CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --model_path "$MODEL_PATH" --pretrain_path "$COARSE_POINT_CLOUD_PATH" --max_offset_k 15 --prune_outlier_iter 800 --block_id $num --port $port &
            pids+=("$!")
            # Increment the port number for the next run
            ((port++))
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 20
            break 
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 20 
        fi
    done
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

# merge the blocks
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python merge.py --config config/$CONFIG.yaml --model_path "$MODEL_PATH" --iteration 30000

# rendering and evaluation
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python render_large.py --config config/$CONFIG.yaml --model_path "$MODEL_PATH" --custom_test "$TEST_PATH"

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python metrics_large.py -m "$MODEL_PATH" -t "$out_name"
