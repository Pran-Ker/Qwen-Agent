#!/bin/bash
# Startup script for VLLM server with custom Qwen3-VL model

MODEL_PATH="/Users/pran-ker/qwen3vl_exl"
MODEL_NAME="Qwen3-VL-Custom"
HOST="0.0.0.0"
PORT=8000

echo "=========================================="
echo "Starting VLLM Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Served as: $MODEL_NAME"
echo "Endpoint: http://$HOST:$PORT/v1"
echo "=========================================="

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len 8192 \
    --dtype auto \
    --limit-mm-per-prompt image=10 \
    --max-num-seqs 5 \
    --tensor-parallel-size 1

# Alternative configurations (uncomment to use):

# For multi-GPU (2 GPUs):
# python3 -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_PATH" \
#     --host "$HOST" \
#     --port "$PORT" \
#     --served-model-name "$MODEL_NAME" \
#     --trust-remote-code \
#     --max-model-len 8192 \
#     --dtype auto \
#     --tensor-parallel-size 2

# For lower memory usage:
# python3 -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_PATH" \
#     --host "$HOST" \
#     --port "$PORT" \
#     --served-model-name "$MODEL_NAME" \
#     --trust-remote-code \
#     --max-model-len 4096 \
#     --dtype float16 \
#     --gpu-memory-utilization 0.9
