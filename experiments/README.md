# ATLAS Experiments

This directory keeps only the current ATLAS experiment artifacts that are still
useful for discussion.

Large model artifacts are intentionally not committed:

- checkpoints
- LoRA adapter weights
- optimizer / scheduler states
- GGUF exports
- tarballs
- generated train/eval JSONL data

## Current Runs

### `qlora_3/`

C1 Agenda LLM QLoRA v3 training/evaluation bundle. This is the only C1 training
run kept here.

Included:

- `train_qlora.py`
- `eval_qlora.py`
- evaluation reports for valid/test1/test2/test3
- trainer state summary
- charts and confusion plots
- train/eval summary logs

### `solution2_probe/`

C2 question answered/unanswered tracking probe.

Included:

- toy cases
- lexical C2 prototype
- hybrid raw+embedding C2 prototype
- delayed matching C2 prototype
- real ACI silver-label probes
- broad ATLAS annotation stress test
- LLM judge pilot script
- feasibility report
