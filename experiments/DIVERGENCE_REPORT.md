# ATLAS — Spec v4 vs. Actual State: Divergence Report

Read-only audit. Compares `~/Desktop/atlas/README.md` (Project Spec v4, April 2026)
against actual code/data/training artifacts in `~/Desktop/atlas` and
`~/Desktop/tequila`. No files were modified. This is a divergence inventory only —
no fix recommendations, spec values are NOT treated as ground truth.

Citations: `README.md:L<n>` = atlas spec v4. Other paths are repo-relative to the
named repo (atlas = `~/Desktop/atlas`, tequila = `~/Desktop/tequila`).

The spec itself carries a banner (`README.md:L1-5`) admitting divergence and a
conference change (NeurIPS → AAAI). That banner is accurate; details below.

---

## 1. Conference / Deadline / Timeline

| | |
|---|---|
| **Spec v4** | `README.md:L10` — "**Target:** NeurIPS 2026 (~May 3 deadline)". `README.md:L450` — "abstract (~Apr 29), submit (~May 3)". 7-week plan `README.md:L397-451`. |
| **Actual** | Banner `README.md:L5` — "Conference target changed (NeurIPS → AAAI)." Today is 2026-05-19, past the spec's May 3 deadline. The Week-by-Week plan (`README.md:L397-451`, weeks 1–7, dataset freeze at 50–60 dialogues, cross-model 1B/7B, Pareto/latency figures) does not match what exists. No paper/ directory, no figures/tables dir, no submission artifact. |
| **Source** | `atlas/README.md` banner L1-5; absence of `paper/` in `atlas` tree; current date 2026-05-19. |

---

## 2. Models (Llama 3.2 3B/1B, Mistral 7B)

| | |
|---|---|
| **Spec v4** | `README.md:L121-127` model table: Llama 3.2 1B / 3B / Mistral 7B, all "GGUF Q4_K_M"; "Primary: Llama 3.2 3B. Others for generalization experiments." `README.md:L515` llama.cpp build. |
| **Actual** | Only **Llama-3.2-3B-Instruct** is used, via **HF Transformers + PEFT + bitsandbytes 4-bit NF4 (QLoRA)**, NOT GGUF / llama.cpp. No 1B run, no Mistral 7B run anywhere. `config.py:L40` still names `models/gguf/llama-3.2-3b-q4_k_m.gguf` and `n_gpu_layers=-1`, but no GGUF file exists in either repo and the harness never calls a real LLM (uses `MockLLM`, see §3/§9). Trained on the **Instruct** variant, not base. |
| **Source** | `tequila/experiments/qlora_3/train_qlora.py:L91` (`default="meta-llama/Llama-3.2-3B-Instruct"`), `:L125-137` (BitsAndBytesConfig NF4); `eval_qlora.py:L136`; `atlas/src/config.py:L38-45`; no `models/` dir in either repo. |

---

## 3. Component 1 — Agenda LLM (QLoRA setup, curriculum, results)

### 3a. Output schema: 3 fields → `detail_list`, 6 types → 7 types

| | |
|---|---|
| **Spec v4** | `README.md:L65-96` — "3 Fields Only": `{"relevant", "type", "summary"}`. `README.md:L93` six `type` values: `agenda_item, detail, medication, social_history, follow_up, question_unanswered`. |
| **Actual** | Output is `{"relevant": true, "detail_list": [{"type","summary"}, ...]}` — one utterance can emit multiple typed summaries. **7 types**: the six plus a new **`question`**. `prompts.py:L29-56` documents this as "[MODIFIED v3.1]: Types expanded from 6 to 7 (added 'question')" and "Output format changed from flat {type, summary} to detail_list". |
| **Source** | `tequila/.../train_qlora.py:L36-46` (SYSTEM_PROMPT lists 7 types incl. `question`, detail_list schema); `atlas/prompts.py:L44-56`, `:L29-37`; `atlas/src/harness.py:L33` & `LLMOutput.detail_list`. |

### 3b. QLoRA hyperparameters & curriculum

| | |
|---|---|
| **Spec v4** | `README.md:L129` — "QLoRA (rank 32–64). Three-phase curriculum". `README.md:L131-135`: Phase 1 ~50K synthetic+NoteChat @1e-4; Phase 2 ~5K MTS+ACI+PriMock+AMI @5e-5; Phase 3 ~500 Penn Medicine @1e-5. |
| **Actual** | **Single-phase** SFT. No curriculum, no NoteChat/MTS/PriMock/AMI/Penn data (see §6). Hyperparameters: `lora_r=32`, `lora_alpha=64`, `lora_dropout=0.05`, **lr=2e-4** cosine, warmup_ratio 0.03, **3 epochs**, batch 4 × grad_accum 4 (eff. 16), max_seq_len 2048, NF4 + double-quant + bf16 compute, LoRA on all 7 proj modules (`q,k,v,o,gate,up,down`), `paged_adamw_8bit`, completion-only loss (prompt masked to -100). Code explicitly overrides spec: `:L99-101` "QLoRA paper default. Spec Phase 2 says 5e-5; raise to 2e-4 for small data + LoRA adapter." Trained only on `data/challenge_data/train/train.jsonl` (3,733 lines), eval on `valid.jsonl` (1,012). |
| **Source** | `tequila/.../train_qlora.py:L96-114`, `:L142-211`, `:L49-75` (manual mask), `:L93-94` (data paths). |

### 3c. Training results (run of record = checkpoint-699)

| | |
|---|---|
| **Spec v4** | No concrete numbers (curriculum-based, examples-count targets only). |
| **Actual** | `trainer_state.json`: 699 steps = 2.998 epochs, `max_steps=699`, `train_batch_size=4`, `total_flos=6.49e16`. Train loss 2.21 (step 20) → ~0.16–0.19 (steps 600–680). Eval loss **rises**: 0.254 (step 200) → 0.264 (400) → 0.330 (600) — mild overfitting; `best_metric=null`, `best_model_checkpoint=null` (no best-model selection; `save_total_limit=2`). Held-out eval (`eval_qlora.py`) over 4 splits: JSON parse ≈0.999–1.000; relevance acc 0.868 / 0.867 / 0.873 / 0.878 (valid/test1/2/3); 8-class type macro-F1 0.633 / 0.642 / 0.654 / 0.637. Charts: `chart_4split_main.png`, `chart_per_class_f1.png` (7-class macro-F1 excl. `question_unanswered` = 0.733). |
| **Source** | `tequila/.../trainer_state.json` (full); `eval_report_{valid,test1,test2,test3}.json` `summary` blocks; `chart_*.png`. |

### 3d. Context format

| | |
|---|---|
| **Spec v4** | `README.md:L109-117` — context = pipe-delimited string of the model's own prior summaries. |
| **Actual** | Format matches in principle (`[Context]: ... \n [Utterance]: [Speaker] text`), but in `train.jsonl` early turns use the literal placeholder `"[Context]: Start of visit"` (also `harness.py:L163`). Training data is the precomputed JSONL, not generated by streaming the model's own summaries back (teacher-annotated, see §6). |
| **Source** | `atlas/data/challenge_data/train/train.jsonl` line 1; `atlas/src/harness.py:L163`. |

---

## 4. Component 2 — System Tracker (5 tasks)

| | |
|---|---|
| **Spec v4** | `README.md:L143-149` — 5 tasks: Linking (cos>0.8), First-mention (cos>0.85), Resolution (mentioned→discussed→resolved→unresolved), Visit phase (keyword), Nesting. `README.md:L156` agenda IDs like `"A001"`. |
| **Actual** | All 5 implemented in `atlas/src/state_tracker.py` (deterministic, CPU, no LLM). Divergences from spec wording: (a) agenda item `id` = the utterance_id (e.g. `line_0004`), NOT synthetic `A001` — code comment `:L271-285` explicitly: "Use utterance_id, NOT synthetic A001" (done to match gold `eval_only.linked_agenda_item_id`). (b) Resolution machine: `follow_up` type → `resolved`; `len(speakers)>=2` → `discussed`; `finalize()` adds a rule not in spec — items with `unanswered` that became `resolved` are downgraded to `discussed` (`:L119-131`). (c) Encoder is lazy/optional: if `sentence-transformers` is absent, linking is disabled and first-mention defaults to `True` (`:L188-231`, `:L320-329`). **The tracker is never exercised in tequila** — `eval_qlora.py:L10-11` states it does NOT compute system-level metrics; no harness-run artifact (tracker states / predictions output) exists in either repo. |
| **Source** | `atlas/src/state_tracker.py` (whole file); `tequila/.../eval_qlora.py:L10-11`; no harness output JSON anywhere. |

---

## 5. Component 3 — Context Manager / MLP (feature dim, training, baselines)

### 5a. Feature dimension: spec 395-d → actual 396-d, 6→7 type one-hot

| | |
|---|---|
| **Spec v4** | `README.md:L182-194` — "395-d": 384 emb + position + recency + speaker + token_count + medical entity count + **6-d type one-hot**; net `Linear(395,128)→ReLU→Linear(128,64)→ReLU→Linear(64,1)→Sigmoid`. |
| **Actual** | `config.py:L18` `feature_dim: int = 396  # 384 embedding + 12 scalar/onehot features (7 types)`. `policy.py:L74-79` builds `384 + [pos, recency, speaker, token_count, entity_count] + 7-d one-hot = 396`. The 7th one-hot slot is `question` (`TYPE_TO_INDEX` `:L50-58` has `"question": 6`). `policy.py` module docstring `:L1-14` still says "395-d" and "6 type one-hot" — stale vs. its own code. `medical entity count` is hardcoded `0.0` (`:L72`, scispaCy not invoked). MLP module outputs raw **logits** (no `Sigmoid` layer); sigmoid applied at inference (`:L38-47`, `:L101-104`) — differs from spec's "→ Sigmoid" in the net. |
| **Source** | `atlas/src/config.py:L16-27`; `atlas/src/policy.py:L1-14, L24-80`. |

### 5b. MLP training status

| | |
|---|---|
| **Spec v4** | `README.md:L208-235` — hindsight BERTScore supervision via `context_manager_labels.py`, Adam lr 1e-3 bs 256 50–100 epochs, ~30K examples / ~50 convs, 70/15/15 by conversation, ~5–10 min on Jetson. |
| **Actual** | `train_policy()` exists (`policy.py:L119-214`) but the **MLP is never trained**: no `eviction_policy.jsonl` / feature-label dataset, no policy checkpoint, no `models/policy/`, and `context_manager_labels.py` (the hindsight label generator, spec `README.md:L326`) does not exist in either repo. No training artifact for Component 3 anywhere. |
| **Source** | absence of `eviction_policy.jsonl`, `models/`, `src/context_manager_labels.py` in `atlas` tree; `atlas/src/policy.py:L119-214`. |

### 5c. 5 baseline strategies

| | |
|---|---|
| **Spec v4** | `README.md:L237-245` — 5 strategies: Growing/FIFO, Sliding, Random, Oldest-first, **Lowest-attention** (LLM attention weights). |
| **Actual** | `atlas/src/baselines.py`: `fifo`, `sliding` (count-based, K default 10), `random`, `oldest` implemented (4/5). `lowest_attention_strategy` is an explicit **PLACEHOLDER** that falls back to `oldest_first` — `:L152-167`: "PLACEHOLDER — NOT YET IMPLEMENTED ... MUST be replaced before this baseline can be included in final experiment results." So 4 real + 1 stub. |
| **Source** | `atlas/src/baselines.py:L45-200`. |

---

## 6. Component 4 — Annotation / Data

### 6a. Teacher model & pipeline

| | |
|---|---|
| **Spec v4** | `README.md:L304-327` — `annotation_pipeline.py` (batch/realtime/generate/noise/validate/format) + separate `context_manager_labels.py` = 7 steps. Teacher model unspecified. |
| **Actual** | File is `atlas/annotation_gemini.py` ("Annotation Pipeline v3 — Gemini Version"); teacher = **Gemini** `gemini-3.1-flash-lite-preview` (`:L58`, `:L577`). Modes: `batch, eval, realtime, generate, validate, noise, format` (`:L573`) — 7 modes including a dedicated **`eval`** mode (produces `eval_only` ground truth). Step 7 of spec (`context_manager_labels.py`, eviction-policy labels) is **absent**. Prompts in `atlas/prompts.py` are "v3.1" and self-document the divergences (`:L29-37`): detail_list output, `clinical_note` used as annotation ground truth, 6→7 types, medication tightened, agenda_item = first-mention-only. |
| **Source** | `atlas/annotation_gemini.py:L1-12, L58, L572-630`; `atlas/prompts.py:L1-56`. |

### 6b. Datasets actually present

| | |
|---|---|
| **Spec v4** | `README.md:L293-302` — Penn Medicine 16, ACI-BENCH 50/207, PriMock57 57, MTS-Dialog 1,201, AMI 100–150, Synthetic Streams A/B/C. |
| **Actual** | Only **ACI-BENCH / MEDIQA-Chat** material exists. `atlas/data/challenge_data/`: `train` 67, `valid` 20, `clinicalnlp_taskB_test1` 40, `clinicalnlp_taskC_test2` 40, `clef_taskC_test3` 40 conversations (D2N001–D2N207, the full 207 in official challenge splits) — each with `aci/` (raw) + `eval/` (annotated) + a `*.jsonl`. **No Penn Medicine, no PriMock57, no MTS-Dialog, no AMI, no synthetic Stream A/B/C.** Additionally `atlas/data/src_experiment_data/` holds an ASR-robustness track NOT in the spec dataset table: `{train,valid,test1,test2,test3}` × `{aci_asr, aci_asrcorr, virtscribe_asr, virtscribe_humantrans}` with `ACI*`/`VS*` IDs (광호's in-progress ASR variants; spec only had a generic "Stream C ASR-degraded synthetic"). |
| **Source** | `atlas` data tree (counts: train 67/67, valid 20/20, test1/2/3 40/40 each; src_experiment_data subdirs). |

### 6c. Three data products & `_eval_only`

| | |
|---|---|
| **Spec v4** | `README.md:L251-282` — Product A (LLM jsonl, strips eval), B (395-d features + label), C (3 fields + top-level `_eval_only`). |
| **Actual** | Product A exists as `train/valid/test*.jsonl` with flat `{"input","output"}` (output uses `detail_list`, not flat 3-field). Product **B is absent** (no feature/label jsonl — see §5b). Product C exists as `challenge_data/*/eval/*.json`: per-`detail_list`-entry key is **`eval_only`** (not top-level `_eval_only`), populated with `linked_agenda_item_id / first_mention / resolution_status / clinical_category`. The `aci/` files are raw (no annotations); the `eval/` files carry annotations + `eval_only`. A `train/validate/report.json` exists (88 samples, acceptance 0.886, avg_faithfulness 2.83, type_accuracy 0.954). |
| **Source** | `atlas/data/challenge_data/train/train.jsonl` L1; `atlas/data/challenge_data/valid/eval/D2N068.json` (eval_only nested per detail_list entry); `atlas/data/challenge_data/train/validate/report.json`. |

### 6d. File-structure divergence

`README.md:L456-504` describes `data/{raw,annotated/{human,teacher,converted},training,synthetic,noisy,validation,splits}`, `models/{gguf,policy}`, a 12-file `src/`, `experiments/{configs,logs,results}`, `paper/`. Actual `atlas` is largely flat: root has `README.md, annotation_gemini.py, prompts.py, check_types.py, eval_baselines.py, tsne.py`; `src/` has only 6 files (`config, harness, baselines, policy, state_tracker, metrics`). **Missing vs spec src/**: `compression.py, hardware.py, synthetic_gen.py, ami_conversion.py, context_manager_labels.py, annotation_pipeline.py, annotation_prompts.py`. No `models/`, `paper/`, `experiments/` (in atlas), `data/raw`, `data/annotated`, `data/synthetic`, `data/splits`.

---

## 7. Component 5 — Evaluation (per-metric status)

| Spec metric (`README.md:L351-376`) | Status |
|---|---|
| ROUGE-1/2/L | Implemented in `metrics.py:L10-22`; **not run** (the run-of-record evaluator `eval_qlora.py` does not compute ROUGE). |
| BERTScore (deberta-xlarge-mnli) | Implemented `metrics.py:L25-34` and `eval_qlora.py:L219-231` (opt-in `--bertscore`). Reports record only `n_summary_pairs_for_bertscore` (448/914/934/941); no BERTScore value stored in the 4 reports. |
| Detection P/R/F1 | Spec says "6-way type + irrelevant" (`README.md:L356`). Actual is **8-class**: 7 types + `irrelevant`, and `metrics.py:L41-44` lists BOTH `question` and `question_unanswered`. `eval_qlora.py` same 8 classes. |
| Agenda completeness | Implemented `metrics.py:L70-105`; not in `eval_qlora.py`, not run. |
| Temporal F1 (30/60/120s) | **Not implemented anywhere.** |
| Linking / First-mention / Resolution accuracy | Implemented `metrics.py:L108-123` + `generate_report`; **not run** (requires harness+tracker; `eval_qlora.py:L10-11` explicitly excludes). |
| Hardware (tegrastats / tok-s / KV cache) | `parse_tegrastats_log` is a **stub returning zeros** (`metrics.py:L126-131`); `compute_kv_cache_memory` is a bare formula. No hardware data collected. |

What was actually run (`tequila/.../eval_qlora.py`): JSON parse rate, relevance accuracy, 8-class type macro-F1 + per-class, optional BERTScore. Notable result: **`question_unanswered` F1 = 0.0, support = 0** in all 4 splits — the class never appears in gold or predictions; this drags the 8-class macro-F1 (0.63–0.65) vs 7-class (0.733, per `chart_per_class_f1.png`).

**Source:** `atlas/src/metrics.py` (whole); `tequila/.../eval_qlora.py:L1-12, L115-236`; `eval_report_*.json`.

---

## 8. Component 6 — Ablation (6 conditions × 5 budgets)

| | |
|---|---|
| **Spec v4** | `README.md:L382-393` — 6 conditions (Full / Policy only / No policy / Compression only / Hierarchical only / Policy+compression) × budget sweep B ∈ {256,512,1024,2048,4096}. |
| **Actual** | **Not run.** No learned MLP (§5b), no compression module (`src/compression.py` absent — spec `README.md:L206`, `L432`), so no ablation condition involving "Learned MLP" or "Compression" is realizable. `atlas/eval_baselines.py` does a partial **baseline-only** sweep: `oracle, fifo-{256,512,1024,2048}, sliding-{5,10}, random-1024, oldest-1024` — note **4096 is not in the experiment list** (though `config.py:L36` `budget_sweep` includes 4096), and it only aggregates buffer size / tokens / retention / latency (`aggregate()` `:L93-116`), NOT ROUGE/BERTScore/agenda-completeness/temporal. No saved output file from `eval_baselines.py` exists in either repo. |
| **Source** | `atlas/eval_baselines.py:L38-48, L93-116`; `atlas/src/config.py:L36`; absence of `src/compression.py`, policy checkpoint, ablation results. |

---

## 9. Hardware / Jetson

| | |
|---|---|
| **Spec v4** | `README.md:L11` Jetson AGX Orin (JetPack 6.x, CUDA 12.x); `README.md:L515` llama.cpp `-DGGML_CUDA=ON`; `README.md:L369-376` tegrastats / tok-s / KV-cache profiling. |
| **Actual** | Training ran on **RunPod** (`train_qlora.py:L11` docstring "Run on RunPod: see README.md") on a CUDA GPU via HF Transformers + bitsandbytes 4-bit — **not Jetson, not llama.cpp/GGUF**. The harness never runs a real LLM: `eval_baselines.py` and `harness.main()` use `MockLLM` fed precomputed annotations. Token counting is a char/4 placeholder explicitly flagged unsuitable for final experiments (`harness.py:L82-95`). `tegrastats` parsing is a zero-returning stub (`metrics.py:L126-131`). `config.py` GGUF/`n_gpu_layers`/`n_ctx` fields are unused. No tegrastats logs, no latency/throughput/memory profiling artifacts anywhere. |
| **Source** | `tequila/.../train_qlora.py:L11, L125-137`; `atlas/src/harness.py:L82-95, L98-131`; `atlas/eval_baselines.py:L74-79`; `atlas/src/metrics.py:L126-131`; `atlas/src/config.py:L38-45`. |

---

## 10. Additional facts (in code/data, not in spec v4)

1. **Two-repo split.** `atlas` (tria-lab/atlas, private) holds code + data; `tequila` (Bias92/tequila, public) holds the QLoRA training/eval artifacts (`experiments/qlora_3` + `experiments/archive`). Not mentioned in spec.
2. **`experiments/archive/` = prior failed runs kept as evidence.** `train_qlora_v1_iid.py`: used `DataCollatorForCompletionOnlyLM`, trained/eval on `data/train.jsonl`/`data/eval.jsonl` (IID split → leakage). `train_qlora_v2_mask_broken.py`: used plain `DataCollatorForLanguageModeling` with **no completion mask** (loss on prompt tokens too — "mask broken"). `qlora_3/train_qlora.py` is the corrected version: manual `-100` masking in `build_chat` + custom `completion_collator` + conversation-level holdout (`challenge_data/train` vs `valid`). (Source: `tequila/experiments/archive/*` headers + diff vs `qlora_3`.)
3. **`question` type is pervasive**, not just in prompts: `policy.TYPE_TO_INDEX` (7 entries), `harness.LLMOutput.type` docstring, `metrics.ALL_CLASSES` (8-class), `eval_qlora.VALID_TYPES`, `check_types.ALL_TYPES`. Spec's 6-type schema is superseded everywhere in code.
4. **`check_types.py`** (atlas root, Korean comments) — annotation type-frequency + ambiguity/edge-case auditor over `detail_list`. A data-QA tool with no spec counterpart.
5. **`tsne.py` / `tsne.png`** — summary-embedding visualization. Despite the name it uses **UMAP** (not t-SNE), all-MiniLM-L6-v2, colored by the 7 types. Gitignored (`atlas/.gitignore`: `.DS_Store, tsne.py, tsne.png`).
6. **ASR-robustness data track** (`data/src_experiment_data/`, §6b): VirtScribe (`VS*`) + ACI (`ACI*`) with `asr` / `asrcorr` / `humantrans` variants across train/valid/test1-3. This is a concrete robustness experiment scaffold; spec only had a vague synthetic "Stream C".
7. **Eval is 8-class with a dead class.** `question_unanswered` has support 0 in all four held-out splits — the trained model and the gold annotations never use it in eval, so its F1 is 0 and the headline 8-class macro-F1 (0.63–0.65) understates per-class quality vs. the 7-class figure (0.733).
8. **Spec divergence is self-documented** inside `atlas/prompts.py:L29-37` ("[MODIFIED v3.1]" list) and `train_qlora.py:L99-101` (lr override vs "Spec Phase 2"). The codebase acknowledges it has moved off spec v4 in-line.
9. **`config.py` deprecation note**: `BufferConfig.max_context_items` is marked DEPRECATED/unused by the harness (kept only for `annotation_pipeline --mode format` compat) — but `annotation_pipeline.py` does not exist; only `annotation_gemini.py` does.
10. **atlas git history** shows the divergence direction: recent commits `Update question_unanswered policy`, `Merge detail_list if divided`, `Fix docs into merged detail_list`, `Update eval_baselines.py data paths to challenge_data layout` — i.e. active movement toward detail_list + challenge_data layout, away from spec v4. (Source: `git -C ~/Desktop/atlas log --oneline`.) `tequila` has no commit history (no commits on `main`).

---

*End of report. Read-only; no repo files modified. Awaiting review.*
