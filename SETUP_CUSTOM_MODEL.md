# Infrastructure Setup for Fine-Tuned Qwen3-VL Model

## Overview
This document outlines the configuration required to run your fine-tuned Qwen3-VL model (`/Users/pran-ker/qwen3vl_exl`) with the Qwen-Agent framework using VLLM for inference.

## Model Information
- **Location**: `/Users/pran-ker/qwen3vl_exl/`
- **Type**: Fine-tuned Qwen3-VL (14 safetensors shards, ~64GB total)
- **Format**: SafeTensors with config.json, tokenizer, and chat template

## Prerequisites

### 1. Install VLLM
```bash
pip3 install --user vllm
```

### 2. Install Qwen-Agent (Already in repo)
```bash
cd /Users/pran-ker/Developer/Qwen-Agent
pip3 install --user -e .
```

### 3. Verify GPU Availability
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 1: Start VLLM Server

### Basic VLLM Server Launch
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /Users/pran-ker/qwen3vl_exl \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen3-VL-Custom
```

### Recommended VLLM Server Launch (with optimizations)
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /Users/pran-ker/qwen3vl_exl \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen3-VL-Custom \
    --trust-remote-code \
    --max-model-len 8192 \
    --dtype auto \
    --tensor-parallel-size 1
```

### VLLM Server Parameters Explained
- `--model`: Path to your fine-tuned model
- `--host`: Server host (0.0.0.0 for external access, 127.0.0.1 for localhost only)
- `--port`: Server port (default 8000)
- `--served-model-name`: Name used in API calls
- `--trust-remote-code`: Required for custom models
- `--max-model-len`: Maximum sequence length (adjust based on GPU memory)
- `--dtype`: Data type (auto, float16, bfloat16)
- `--tensor-parallel-size`: Number of GPUs to use

### For Vision Models (Qwen3-VL)
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /Users/pran-ker/qwen3vl_exl \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen3-VL-Custom \
    --trust-remote-code \
    --max-model-len 8192 \
    --dtype auto \
    --limit-mm-per-prompt image=10 \
    --max-num-seqs 5
```

Additional vision parameters:
- `--limit-mm-per-prompt`: Limit multimodal inputs per prompt
- `--max-num-seqs`: Maximum number of sequences in a batch

## Step 2: Configure Qwen-Agent

### Option A: Create Custom Configuration File
Create `config_custom_model.py` in `/Users/pran-ker/Developer/Qwen-Agent/examples/`:

```python
# Configuration for custom fine-tuned Qwen3-VL model
LLM_CONFIG = {
    'model': 'Qwen3-VL-Custom',  # Must match --served-model-name
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'top_p': 0.8,
        'temperature': 0.7,
        'max_tokens': 2000,
        'max_input_tokens': 6500,
        'max_retries': 10,
    }
}
```

### Option B: Modify Existing Example
Update `/Users/pran-ker/Developer/Qwen-Agent/examples/assistant_qwen3.py` lines 46-63:

```python
llm_cfg = {
    'model': 'Qwen3-VL-Custom',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'max_tokens': 2000,
        'temperature': 0.7,
    }
}
```

## Step 3: Create Custom Agent Script

Create `/Users/pran-ker/Developer/Qwen-Agent/examples/custom_qwen3vl_agent.py`:

```python
import os
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

def init_custom_agent():
    llm_cfg = {
        'model': 'Qwen3-VL-Custom',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': 0.8,
            'temperature': 0.7,
            'max_tokens': 2000,
        }
    }

    tools = [
        'code_interpreter',  # Built-in code execution
        'image_gen',         # Built-in image generation (if needed)
    ]

    bot = Assistant(
        llm=llm_cfg,
        function_list=tools,
        name='Custom Qwen3-VL Agent',
        description='Fine-tuned Qwen3-VL model for specialized tasks'
    )

    return bot

def run_cli():
    bot = init_custom_agent()
    messages = []

    print("Custom Qwen3-VL Agent (type 'quit' to exit)")
    while True:
        query = input('\nUser: ')
        if query.lower() == 'quit':
            break

        messages.append({'role': 'user', 'content': query})
        response_text = ''

        for response in bot.run(messages=messages):
            if isinstance(response, list) and len(response) > 0:
                content = response[-1].get('content', '')
                print(f"\nAssistant: {content}")
                response_text = content

        messages.extend(response)

def run_gui():
    bot = init_custom_agent()
    chatbot_config = {
        'prompt.suggestions': [
            'Analyze this image for me',
            'Help me with code review',
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()

if __name__ == '__main__':
    # Choose interface
    # run_cli()  # Command line
    run_gui()    # Web UI
```

## Step 4: Test the Setup

### Test 1: Verify VLLM Server
```bash
curl http://localhost:8000/v1/models
```

Expected output:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-VL-Custom",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

### Test 2: Simple API Test
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-Custom",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Test 3: Run Custom Agent
```bash
cd /Users/pran-ker/Developer/Qwen-Agent
python3 examples/custom_qwen3vl_agent.py
```

## Configuration Checklist

- [ ] VLLM installed
- [ ] Qwen-Agent installed
- [ ] Model path verified: `/Users/pran-ker/qwen3vl_exl`
- [ ] VLLM server started on port 8000
- [ ] Server responding to `/v1/models` endpoint
- [ ] Custom agent script created
- [ ] Test queries working

## Important Notes

### Model Type Detection
Qwen-Agent auto-detects OpenAI-compatible endpoints when `model_server` starts with "http". No need to specify `model_type: 'oai'`.

### Tool Calling for Qwen3
**Do NOT use** these VLLM flags for Qwen3:
- `--enable-auto-tool-choice`
- `--tool-call-parser hermes`

Qwen-Agent handles tool parsing internally for better compatibility.

### Memory Requirements
- Your model is ~64GB across 14 shards
- Ensure sufficient GPU memory (at least 80GB VRAM recommended)
- For smaller GPUs, consider quantization or tensor parallelism

### Common Issues

**Issue**: VLLM server fails to start
- Check GPU memory: `nvidia-smi`
- Reduce `--max-model-len` parameter
- Use `--tensor-parallel-size 2` for multi-GPU

**Issue**: Connection refused
- Verify server is running: `curl http://localhost:8000/v1/models`
- Check firewall settings
- Ensure port 8000 is not in use: `lsof -i :8000`

**Issue**: Model generates poor responses
- Adjust `temperature` (lower = more deterministic)
- Adjust `top_p` (0.8-0.95 recommended)
- Verify correct chat template is loaded

## Next Steps

1. **Start VLLM server** with your model
2. **Run test queries** to verify inference works
3. **Create custom agent script** for your use case
4. **Add custom tools** if needed (see `/Users/pran-ker/Developer/Qwen-Agent/qwen_agent/tools/`)
5. **Deploy** via WebUI or integrate into your application

## Advanced Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Attention optimization
```

### Custom System Prompt
```python
bot = Assistant(
    llm=llm_cfg,
    function_list=tools,
    system_message='You are a specialized AI trained on [your domain]...'
)
```

### Multi-GPU Setup
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /Users/pran-ker/qwen3vl_exl \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## Resources

- Qwen-Agent Docs: https://github.com/QwenLM/Qwen-Agent
- VLLM Docs: https://docs.vllm.ai
- Your forked repo: https://github.com/Pran-Ker/Qwen-Agent
- Example configs: `/Users/pran-ker/Developer/Qwen-Agent/examples/`

## Support

For issues specific to:
- **VLLM inference**: Check VLLM GitHub issues
- **Qwen-Agent integration**: Check your forked repo or original Qwen-Agent repo
- **Model fine-tuning**: Refer to your training configuration

---

**Repository**: https://github.com/Pran-Ker/Qwen-Agent
**Model Path**: `/Users/pran-ker/qwen3vl_exl/`
**VLLM Endpoint**: `http://localhost:8000/v1`
