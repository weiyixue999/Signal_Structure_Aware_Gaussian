
export CUDA_LAUNCH_BLOCKING=1
# 定义获取可用GPU的函数
get_available_gpu() {
	local mem_threshold=500
	nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
	$2 < threshold { print $1; exit }
	'
}

# 变量定义（与run_citygs.sh保持一致）
TEST_PATH="data/rubble/val"
CONFIG="rubble_c9_r4"
out_name="val"

# merge the blocks
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python merge.py --config config/$CONFIG.yaml

# rendering
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH

# evaluation
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python metrics_large.py -m output/$CONFIG -t $out_name
