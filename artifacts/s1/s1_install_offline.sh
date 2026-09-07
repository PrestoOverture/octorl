#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /root
echo "=== python: $(which python3)  pip: $(which pip) ==="
echo "=== installing 170 packages offline ==="
pip install --no-index --find-links /root/wheels -r /root/wheels/to_download.txt 2>&1 | tail -12
rc=${PIPESTATUS[0]}
echo "=== pip exit: $rc ==="
echo "=== installing flash-attn ==="
pip install --no-deps /root/flash_attn-2.8.3.post1-cp312.whl 2>&1 | tail -4
echo "=== verify ==="
python3 - <<'PY'
import importlib
for m in ("torch","torchvision","vllm","verl","transformers","flash_attn","ray","xformers"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14} FAIL: {type(e).__name__}: {str(e)[:100]}")
import torch
print(f"  cuda: {torch.cuda.is_available()}  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
PY
