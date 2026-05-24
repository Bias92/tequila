# Solution 2 Probe Report

Date: 2026-05-24

Original sandbox: `/Users/markov/Desktop/atlas_solution2_probe`

Tequila copy: `experiments/solution2_probe`

Main ATLAS repo: `/Users/markov/Desktop/atlas`

This sandbox was created to test the feasibility of "Solution 2" without
modifying the main ATLAS repository.

## One-Line Takeaway

Solution 2 is currently the most logically defensible direction for the
streaming architecture, and the prototype shows feasibility. However, the
prototype result is not an official accuracy claim because it uses silver labels,
not human-reviewed gold labels.

Recommended wording:

> Under the current streaming design, Solution 2 is the most logically defensible
> approach. A sandbox prototype also showed feasibility, but official evaluation
> requires human-reviewed question answered/unanswered labels in eval data.

## Solution 2 Definition

Solution 2 means:

```text
C1 Agenda LLM:
  Emits type="question" for question utterances.
  Does not emit type="question_unanswered".

C2 System Tracker:
  Keeps each question open as dialogue state.
  Marks it answered if a later utterance appears to answer it.
  Marks it unanswered at finalize time if no answer is found.
```

This is different from making C1 directly predict `question_unanswered`.
`question_unanswered` is a hindsight state: whether a question was never
answered can only be known after later dialogue has occurred.

## Important Repo-Grounded Context

Confirmed from the data audit:

- `question_unanswered` appears 0 times in actual `data/challenge_data` data.
- Existing C1 JSONL data already uses `question`, not `question_unanswered`.
- C2 currently has no real open-question-to-answered state machine.
- `eval/*.json` currently has `eval_only` metadata for linking, first mention,
  and resolution status, but it does not contain a per-question answered /
  unanswered gold label.
- Therefore, Solution 2 needs both code work and eval annotation work.

## What Was Built In The Sandbox

Files:

- `src/state_tracker_solution2.py`
  - Lexical-only C2 question tracker prototype.
- `src/state_tracker_solution2_hybrid.py`
  - Hybrid prototype using:
    - raw short-answer handling,
    - lexical overlap,
    - sentence-transformer embedding similarity.
- `run_probe.py`
  - Small hand-built / derived toy cases.
- `run_real_silver_probe.py`
  - Silver benchmark using patient questions from real ACI files.
- `run_real_hybrid_probe.py`
  - Compares lexical baseline vs hybrid prototype on real ACI-derived silver
    cases.

The main ATLAS repo was not modified.

## Test Setup

Data source used during the local probe:

```text
/Users/markov/Desktop/atlas/data/challenge_data/*/aci/*.json
```

The committed scripts default to a sibling `../atlas/data/challenge_data`
layout and also accept `ATLAS_DATA=/path/to/atlas/data/challenge_data`.

Splits scanned:

- `train`
- `valid`
- `clinicalnlp_taskB_test1`
- `clinicalnlp_taskC_test2`
- `clef_taskC_test3`

The real-data probe focused on patient-spoken `type="question"` entries.

Silver answer label construction:

- For each patient question, look ahead up to 4 utterances.
- If a later other-speaker utterance has an ACI annotation with answer-like type
  (`detail`, `follow_up`, `medication`, `social_history`), treat it as a silver
  answer candidate.
- If no annotated answer exists but the other speaker gives a short raw answer
  such as "yes", "no", "yeah", or "correct", treat that raw utterance as a
  silver answer candidate.

This is not gold labeling. It is a feasibility benchmark.

## Results

Toy cases:

```text
4/4 passed
```

Dataset scan:

```text
patient questions: 177
silver answerable within 4 utterances: 165
  annotated summary answers: 150
  raw short answers: 15
no silver answer candidate: 12
```

Lexical baseline:

```text
answered exact: 105/165 = 63.6%
answered status-only: 111/165 = 67.3%
counterfactual unanswered: 153/165 = 92.7%
no-silver unanswered: 12/12 = 100.0%
```

Hybrid raw + embedding prototype, threshold 0.40:

```text
answered exact: 155/165 = 93.9%
answered status-only: 160/165 = 97.0%
counterfactual unanswered: 105/165 = 63.6%
no-silver unanswered: 11/12 = 91.7%
```

Threshold sweep:

| Embedding Threshold | Exact Answered | Status Answered | Counterfactual Unanswered | No-Silver Unanswered |
|---:|---:|---:|---:|---:|
| 0.35 | 163/165 (98.8%) | 164/165 (99.4%) | 87/165 (52.7%) | 11/12 (91.7%) |
| 0.40 | 155/165 (93.9%) | 160/165 (97.0%) | 105/165 (63.6%) | 11/12 (91.7%) |
| 0.45 | 149/165 (90.3%) | 155/165 (93.9%) | 116/165 (70.3%) | 11/12 (91.7%) |
| 0.50 | 139/165 (84.2%) | 148/165 (89.7%) | 123/165 (74.5%) | 11/12 (91.7%) |
| 0.55 | 138/165 (83.6%) | 146/165 (88.5%) | 127/165 (77.0%) | 11/12 (91.7%) |
| 0.60 | 131/165 (79.4%) | 140/165 (84.8%) | 133/165 (80.6%) | 11/12 (91.7%) |

## Interpretation

What improved:

- The current lexical-style C2 is too weak for question answered detection.
- Adding raw short-answer handling and semantic similarity made answered
  detection much stronger on the silver benchmark.
- This supports the claim that Solution 2 is implementable, not merely a
  theoretical idea.

What did not become solved:

- Robust unanswered detection is still threshold-sensitive.
- Lower thresholds catch more real answers but also risk closing open questions
  too aggressively.
- The silver benchmark itself is noisy. Some "failures" are likely caused by the
  silver heuristic choosing the wrong answer utterance.

Therefore:

```text
Correctness signal: promising.
Robustness signal: not solved yet.
Official metric: not available until human-reviewed eval labels exist.
```

## Why This Still Supports Solution 2

Solution 2 is not being recommended because the prototype is perfect. It is
recommended because it matches the streaming architecture better:

- C1 classifies the current utterance.
- Whether a question remains unanswered is not fully observable at the moment
  the question is asked.
- C2 already owns dialogue state, linking, first mention, and resolution-like
  tracking.
- Therefore question answered/unanswered status naturally belongs to C2 state,
  not to C1 utterance type prediction.

The prototype result matters because it shows that a C2-style answer tracker is
plausible enough to pilot.

## What Is Needed For An Official Claim

Data:

- Add human-reviewed question status metadata only to `eval/*.json`.
- Do not put this field into C1 training JSONL.
- Suggested eval-only fields:

```json
{
  "question_status": "answered",
  "answered_by_utterance_id": "line_0014"
}
```

or:

```json
{
  "question_status": "unanswered",
  "answered_by_utterance_id": null
}
```

Code:

- Add per-question state to C2.
- Add answer matching logic.
- Add finalize logic that converts still-open questions to unanswered.
- Add metrics for question answered/unanswered tracking.

Evaluation:

- Report C1 question detection separately from C2 question resolution tracking.
- Do not claim the sandbox silver benchmark as final accuracy.

## Suggested Professor-Facing Summary

> We found that `question_unanswered` is not present in the actual data, while
> question utterances are already labeled as `question`. In a streaming setting,
> whether a question is unanswered is a dialogue-level outcome rather than an
> instantaneous utterance label. I therefore tested a sandbox version of Solution
> 2: C1 detects questions, and C2 tracks whether each question is later answered.
> A prototype using raw short-answer handling plus semantic matching improved
> silver answered detection from 63.6% to 93.9%. However, this is only a
> feasibility result because the benchmark is silver-labeled. To make an official
> claim, we should add human-reviewed `question_status` metadata to `eval/*.json`
> and evaluate C2 question tracking separately from C1 question detection.
