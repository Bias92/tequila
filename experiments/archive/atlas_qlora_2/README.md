# atlas_qlora — Llama-3.2-3B QLoRA on the agenda-setting JSONL

Self-contained folder for fine-tuning the agenda LLM. Drops onto a RunPod pod,
trains in a few hours on a single 4090/A6000/A100, produces a LoRA adapter you
can later merge into a GGUF for the deployment harness.

## What's in here

| File | Role |
|---|---|
| `train_qlora.py` | Main QLoRA training script (TRL `SFTTrainer`, completion-only loss) |
| `eval_qlora.py` | Inference-time eval: JSON parse rate, relevance acc, type macro-F1, optional BERTScore |
| `requirements.txt` | Pinned deps |
| `runpod_setup.sh` | One-shot pod bootstrap |
| `merge_and_export_gguf.md` | Adapter → merged HF → GGUF Q4_K_M (Jetson deploy path) |
| `data/train.jsonl` | 3733 examples, plan/assessment relabeled to follow_up/detail |
| `data/eval.jsonl` | 3733 examples, same cleaning |

## Training data shape (recap)

Each line:

```json
{"input":  "[Context]: <pipe-delimited prior summaries>\n[Utterance]: [Doctor|Patient] text",
 "output": "{\"relevant\": true, \"detail_list\": [{\"type\": ..., \"summary\": ...}]}"}
```

`output` is a JSON string. `relevant: false` examples have only the relevance bit.
The model is trained to emit `output` verbatim after the `[Utterance]:` block.

Allowed types (7): `agenda_item, detail, medication, social_history, follow_up,
question, question_unanswered`.

## Run on RunPod

1. Spin up a pod: PyTorch 2.4 / 2.5 template, single GPU (RTX 4090 24GB is fine).
2. SCP / `git clone` this folder into the pod.
3. `bash runpod_setup.sh`
4. `export HF_TOKEN=hf_xxx; export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN` (Llama 3.2 is gated)
5. `python train_qlora.py`

Defaults: rank 32, alpha 64, 3 epochs, lr 2e-4, batch 4 × grad-accum 4 (effective 16),
cosine schedule, paged AdamW 8-bit, gradient checkpointing.

Expected wall time on RTX 4090: ~1.5–2.5 h for 3 epochs over 3733 examples.

## Quick eval

```bash
python eval_qlora.py --limit 200            # smoke test on 200 eval rows
python eval_qlora.py                         # full eval (slower)
python eval_qlora.py --bertscore             # adds summary BERTScore (downloads deberta)
```

Reports `eval_report.json` with `summary{json_parse_rate, relevance_accuracy,
type_macro_f1, type_per_class}` and per-example logs.

## Deploy to harness

See `merge_and_export_gguf.md` for the LoRA → GGUF pipeline. The output drops
into `tria-lab/atlas/models/gguf/` and the existing harness picks it up via
`config.py::LLMConfig.model_path`.

## Known gotchas

- **`question_unanswered` has 0 training examples** — model won't learn this class.
  Either generate synthetic ones (annotation_gemini.py `--mode generate`) or
  expect this class to score F1=0 in the eval report.
- **train.jsonl ↔ eval.jsonl share all 172 source conversations.** Eval here is
  IID, not held-out. For a real held-out number, format `data/test1/`,
  `data/test2/`, `data/test3/` as JSONL too and evaluate against those.
- **The chat template is Llama 3.2's default.** If you switch to base
  Llama-3.2-3B (non-Instruct), edit `build_chat()` to plain concatenation.
- **`packing=False` is intentional.** Completion-only collator looks for the
  exact `<|start_header_id|>assistant<|end_header_id|>\n\n` marker; packing
  multiple examples into one sequence breaks the mask.

## Hyperparameter notes (vs spec)

The project spec v4 Phase 2 lists `lr=5e-5, ~5K examples`. Raised the default
here to `2e-4` because:

1. LoRA adapters tolerate higher LRs than full fine-tuning.
2. With only ~3733 examples and the JSON output being a tight format, lower LRs
   tend to under-fit in 3 epochs.

If divergence is observed (loss spikes, JSON parse rate drops mid-training),
re-run with `--lr 5e-5`.
