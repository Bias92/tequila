# Merge LoRA → base model → GGUF (for harness deployment)

The harness in `tria-lab/atlas/src/harness.py` consumes a GGUF model
(`models/gguf/llama-3.2-3b-q4_k_m.gguf` per `config.py`). After QLoRA training
you have a LoRA adapter directory; convert it to GGUF in two steps.

## Step 1 — Merge LoRA into the base model (fp16)

Run on the same RunPod instance (or any CUDA box).

```bash
python - <<'PY'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER = "checkpoints/atlas-llama32-3b-qlora"
OUT = "checkpoints/atlas-llama32-3b-merged"

tok = AutoTokenizer.from_pretrained(ADAPTER)

# Load base in fp16 (NOT 4-bit) so we can dequantize-merge cleanly
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="cpu"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("Merged ->", OUT)
PY
```

## Step 2 — Convert to GGUF and quantize to Q4_K_M

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && make -j

# Convert HF -> GGUF (fp16)
python convert_hf_to_gguf.py \
  /workspace/checkpoints/atlas-llama32-3b-merged \
  --outfile /workspace/atlas-llama32-3b-f16.gguf \
  --outtype f16

# Quantize to Q4_K_M
./llama-quantize \
  /workspace/atlas-llama32-3b-f16.gguf \
  /workspace/atlas-llama32-3b-q4_k_m.gguf \
  Q4_K_M
```

## Step 3 — Drop into the harness

Copy `atlas-llama32-3b-q4_k_m.gguf` to `models/gguf/` in the atlas repo on
the Jetson. Update `src/config.py::LLMConfig.model_path` to point at the new
file (or just rename to match the existing default path).

## Sanity check

```bash
./llama-cli -m /workspace/atlas-llama32-3b-q4_k_m.gguf \
  -p "[Context]: Start of visit
[Utterance]: [Doctor] hi how are you?" -n 64 -t 0
```

Expected: a JSON object with `"relevant": false` (greetings are filtered).
