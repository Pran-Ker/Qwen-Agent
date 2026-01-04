# Quick Start Guide - Custom Qwen3-VL Model

## Prerequisites Check

```bash
# Check if VLLM is installed
python3 -c "import vllm; print('VLLM installed')"

# Check GPU availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install Qwen-Agent if needed
pip3 install --user -e .
```

## 3-Step Setup

### Step 1: Start VLLM Server (Terminal 1)

```bash
cd /Users/pran-ker/Developer/Qwen-Agent
./start_vllm_server.sh
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### Step 2: Verify Server (Terminal 2)

```bash
curl http://localhost:8000/v1/models
```

Should return:
```json
{"object":"list","data":[{"id":"Qwen3-VL-Custom","object":"model","owned_by":"vllm"}]}
```

### Step 3: Run Agent

```bash
cd /Users/pran-ker/Developer/Qwen-Agent

# Option A: Web UI (recommended)
python3 examples/custom_qwen3vl_agent.py gui

# Option B: Command line
python3 examples/custom_qwen3vl_agent.py cli

# Option C: Single test
python3 examples/custom_qwen3vl_agent.py test
```

## Troubleshooting

### VLLM won't start
```bash
# Check GPU memory
nvidia-smi

# Try with lower memory
python3 -m vllm.entrypoints.openai.api_server \
    --model /Users/pran-ker/qwen3vl_exl \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen3-VL-Custom \
    --trust-remote-code \
    --max-model-len 4096
```

### Connection refused
```bash
# Check if port is in use
lsof -i :8000

# Check if server is running
ps aux | grep vllm
```

### Import errors
```bash
# Install missing dependencies
pip3 install --user vllm gradio uvicorn
```

## Configuration Files

- **VLLM Server**: `start_vllm_server.sh`
- **Agent Script**: `examples/custom_qwen3vl_agent.py`
- **Full Documentation**: `SETUP_CUSTOM_MODEL.md`

## Model Details

- **Path**: `/Users/pran-ker/qwen3vl_exl`
- **Size**: ~64GB (14 safetensors shards)
- **Type**: Qwen3-VL fine-tuned
- **Endpoint**: `http://localhost:8000/v1`
- **Model Name**: `Qwen3-VL-Custom`

## Quick Test

```python
import requests

response = requests.post(
    'http://localhost:8000/v1/chat/completions',
    json={
        'model': 'Qwen3-VL-Custom',
        'messages': [{'role': 'user', 'content': 'Hello!'}],
        'max_tokens': 100
    }
)
print(response.json())
```

## Next Steps

1. Review `SETUP_CUSTOM_MODEL.md` for detailed configuration
2. Customize `examples/custom_qwen3vl_agent.py` for your use case
3. Add custom tools if needed
4. Deploy to production environment

---

**GitHub Repo**: https://github.com/Pran-Ker/Qwen-Agent
