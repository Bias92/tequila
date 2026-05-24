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
- feasibility report

## Archive

### `archive/atlas_qlora_1/`

Failed / obsolete QLoRA v1 code bundle, excluding local data and model
artifacts. Kept only for provenance. This run used an IID train/eval split, so
its evaluation is not a valid held-out result.

### `archive/atlas_qlora_2/`

Failed / obsolete QLoRA v2 code bundle, excluding local data and model
artifacts. Kept only for provenance. This run used a broken masking setup, so
it should not be treated as a current training recipe.

### `archive/atlas_qlora_1_results/`

Old v1 comparison reports. Kept for historical comparison only; not the current
run of record.

### `archive/atlas_src_updated/`

Snapshot of ATLAS `src` files used during previous QLoRA work.

### `archive/handoffs/` and `archive/atlas_handoff_12/`

Handoff notes from previous experiment/debugging sessions.

### `archive/train_qlora_v1_iid.py`

Archived failed IID training script.

### `archive/train_qlora_v2_mask_broken.py`

Archived failed v2 training script kept for divergence/debug context.
