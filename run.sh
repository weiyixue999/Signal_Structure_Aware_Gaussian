set -euo pipefail

TEST_PATH="data/rubble/val"
COARSE_CONFIG="rubble_coarse"
CONFIG="rubble_c9_r4"

# Change this one variable to move all generated training outputs.
# Example: OUTPUT_ROOT=output_exp01 bash run.sh
OUTPUT_ROOT="${OUTPUT_ROOT:-test_rubble5}"
COARSE_MODEL_PATH="$OUTPUT_ROOT/$COARSE_CONFIG"
MODEL_PATH="$OUTPUT_ROOT/$CONFIG"
COARSE_POINT_CLOUD_PATH="$COARSE_MODEL_PATH/point_cloud"

OUT_NAME="val"
MAX_BLOCK_ID=3
PORT=4041

get_available_gpu() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -v threshold="$mem_threshold" -F', ' '
            $2 < threshold && selected == "" { selected = $1 }
            END { if (selected != "") print selected }
        '
}

wait_for_gpu() {
    local gpu_id
    while true; do
        gpu_id="$(get_available_gpu)"
        if [[ -n "$gpu_id" ]]; then
            echo "$gpu_id"
            return 0
        fi
        echo "No GPU available. Retrying in 20 seconds."
        sleep 20
    done
}

run_on_gpu() {
    local gpu_id="$1"
    shift
    echo "GPU $gpu_id is available."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@"
}

echo "Output root: $OUTPUT_ROOT"

# Train coarse global Gaussian model.
gpu_id="$(wait_for_gpu)"
run_on_gpu "$gpu_id" \
    python pipline/train_large.py \
    --config "config/$COARSE_CONFIG.yaml" \
    --model_path "$COARSE_MODEL_PATH" \
    --max_offset_k 25 \
    --prune_outlier_iter 800 \
    --coarse_train

# Partition the scene.
gpu_id="$(wait_for_gpu)"
run_on_gpu "$gpu_id" \
    python pipline/data_partition.py \
    --config "config/$CONFIG.yaml" \
    --model_path "$MODEL_PATH" \
    --pretrain_path "$COARSE_POINT_CLOUD_PATH"

# Optimize each block.
pids=()
for block_id in $(seq 0 "$MAX_BLOCK_ID"); do
    gpu_id="$(wait_for_gpu)"
    echo "Starting training block '$block_id' on GPU $gpu_id."
    CUDA_VISIBLE_DEVICES="$gpu_id" WANDB_MODE=offline \
        python pipline/train_large.py \
        --config "config/$CONFIG.yaml" \
        --model_path "$MODEL_PATH" \
        --pretrain_path "$COARSE_POINT_CLOUD_PATH" \
        --max_offset_k 15 \
        --prune_outlier_iter 800 \
        --block_id "$block_id" \
        --port "$PORT" &
    pids+=("$!")
    PORT=$((PORT + 1))
    sleep 20
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

# Merge the blocks.
gpu_id="$(wait_for_gpu)"
run_on_gpu "$gpu_id" \
    python pipline/merge.py \
    --config "config/$CONFIG.yaml" \
    --model_path "$MODEL_PATH" \
    --iteration 40000

# Render and evaluate.
gpu_id="$(wait_for_gpu)"
run_on_gpu "$gpu_id" \
    python pipline/render_large.py \
    --config "config/$CONFIG.yaml" \
    --model_path "$MODEL_PATH" \
    --custom_test "$TEST_PATH"

gpu_id="$(wait_for_gpu)"
run_on_gpu "$gpu_id" \
    python pipline/metrics_large.py \
    -m "$MODEL_PATH" \
    -t "$OUT_NAME"
