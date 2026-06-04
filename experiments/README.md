# ATLAS Experiments

This directory collects lightweight experiment code, reports, and plots copied
from local ATLAS experiment workspaces.

Large model artifacts are intentionally not committed:

- checkpoints
- LoRA adapter weights
- optimizer / scheduler states
- GGUF exports
- tarballs
- generated train/eval JSONL data

## Current Runs

### `qlora_3/`

Main QLoRA v3 training/evaluation bundle.

Included:

- `train_qlora.py`
- `eval_qlora.py`
- evaluation reports for valid/test1/test2/test3
- trainer state summary
- charts and confusion plots
- train/eval summary logs

### `solution2_probe/`

Sandbox probe for the C2 question answered/unanswered tracking idea.

Included:

- toy cases
- lexical C2 prototype
- hybrid raw+embedding C2 prototype
- real ACI silver-label probes
- broad ATLAS annotation stress test
- dual-threshold post-hoc simulation script
- feasibility report

## Archive

Failed QLoRA v1/v2 run folders were intentionally removed from this repo to
avoid confusing them with the current run of record. Keep only current `qlora_3`
artifacts here.

### `archive/atlas_src_updated/`

Snapshot of ATLAS `src` files used during previous QLoRA work.

### `archive/handoffs/` and `archive/atlas_handoff_12/`

Handoff notes from previous experiment/debugging sessions.
