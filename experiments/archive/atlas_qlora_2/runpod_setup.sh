#!/usr/bin/env bash
# One-shot RunPod bootstrap.
# Pick a "PyTorch 2.4" or "PyTorch 2.5" template with a single GPU
# (RTX 4090 24GB, A6000 48GB, or A100 40/80GB are all fine for Llama-3.2-3B QLoRA).
#
# Usage on the pod (after cloning this folder):
#   bash runpod_setup.sh
#   export HF_TOKEN=hf_xxx              # for gated Llama models
#   export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
#   python train_qlora.py               # uses defaults; ~3733 train ex, 3 epochs
#
# Outputs:
#   checkpoints/atlas-llama32-3b-qlora/   (LoRA adapter + tokenizer)

set -euo pipefail

echo "[1/4] System packages…"
apt-get update -y
apt-get install -y --no-install-recommends git tmux htop curl ca-certificates
rm -rf /var/lib/apt/lists/*

echo "[2/4] Python deps…"
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/4] HF auth check…"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "  (warning) HF_TOKEN not set. Llama 3.2 is gated — set it before training."
  echo "  export HF_TOKEN=hf_xxx; export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN"
fi

echo "[4/4] GPU check…"
python - <<'PY'
import torch
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  Device count   : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i:>2}          : {p.name} | {p.total_memory/1e9:.1f} GB | "
              f"compute {p.major}.{p.minor}")
PY

echo "Setup OK. Run:  python train_qlora.py"
