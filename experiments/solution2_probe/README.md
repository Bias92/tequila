# ATLAS Solution 2 Probe

This directory is a sandbox for testing the **Solution 2** idea for ATLAS
question handling.

It is intentionally separated from the main `tria-lab/atlas` repo. The code here
is a feasibility probe, not production tracker code and not an official accuracy
evaluation.

## What Was Tested

Current ATLAS architecture:

```text
C1 Agenda LLM:
  emits relevant/type/summary for the current utterance

C2 System Tracker:
  links summaries to agenda items
  tracks first mention and resolution status

C3 Context Manager:
  manages which summaries remain in the context buffer
```

The issue tested here:

```text
question_unanswered is a hindsight state.
At the moment a question is asked, C1 cannot know whether it will remain
unanswered later in the conversation.
```

Solution 2 reframes the task as:

```text
C1:
  emit type="question"

C2:
  keep that question open as dialogue state
  mark it answered if a later utterance appears to answer it
  mark it unanswered at finalize time if no answer is found
```

## Status Of The Evidence

Use the results in this folder as:

```text
feasibility probe / silver-label stress test
```

Do **not** use them as:

```text
official gold-label accuracy
```

Known limitations:

- The labels are silver labels built from existing annotations plus heuristics.
- No human-reviewed `question_status` gold labels exist in the ATLAS eval data
  yet.
- The probes use annotation summaries as C2 input, not C1 model predictions.
- The broad stress test includes duplicate or related annotation views
  (`aci/eval`, ASR/human-transcript variants), so it is not an independent test
  set.
- `dual-threshold` numbers below are a post-hoc simulation using low/high
  tracker runs. A production dual-threshold C2 state machine has not been
  implemented yet.

## Files

- `src/state_tracker_solution2.py`
  - lexical-only C2 question tracker prototype
- `src/state_tracker_solution2_hybrid.py`
  - hybrid tracker using lexical overlap, raw short-answer handling, and
    sentence-transformer embedding similarity
- `src/state_tracker_solution2_matching.py`
  - delayed matching tracker that scores local question-answer edges and chooses
    the best assignment before finalize
- `run_probe.py`
  - toy / counterfactual cases copied or derived from ATLAS examples
- `run_real_silver_probe.py`
  - early lexical-only patient-question probe over `challenge_data/*/aci`
- `run_real_hybrid_probe.py`
  - lexical vs hybrid probe over `challenge_data/*/aci`
- `run_broad_dual_probe.py`
  - broad stress test over `challenge_data/{aci,eval}` plus
    `src_experiment_data/*/aci`, including a dual-threshold post-hoc simulation
- `run_matching_probe.py`
  - broad stress test for delayed matching C2 variants
- `run_llm_judge_baseline.py`
  - Gemini judge pilot baseline for "just ask an LLM" comparisons
- `REPORT_SOLUTION2_PROBE.md`
  - original report from the first sandbox run
- `REPORT_MATCHING_PROBE.md`
  - current report for the graph/matching-style C2 probe

## Data Paths

The committed scripts default to an ATLAS repo beside this repo:

```text
/Users/markov/Desktop/atlas
```

Equivalent default from this directory:

```text
../../../atlas
```

Override paths if needed:

```bash
ATLAS_DATA=/path/to/atlas/data/challenge_data python3 run_real_hybrid_probe.py
ATLAS_ROOT=/path/to/atlas python3 run_broad_dual_probe.py
```

## Reproduction Commands

From this directory:

```bash
python3 run_probe.py
python3 run_real_silver_probe.py
python3 run_real_hybrid_probe.py
python3 run_broad_dual_probe.py
python3 run_matching_probe.py
```

`run_real_hybrid_probe.py` and `run_broad_dual_probe.py` load
`sentence-transformers/all-MiniLM-L6-v2` if `sentence_transformers` is available.
If the encoder cannot be loaded, embedding scores fall back to `0.0`, so hybrid
numbers will not match the verified run.

## Verified Results

Verified locally from `/Users/markov/Desktop/tequila/experiments/solution2_probe`
against `/Users/markov/Desktop/atlas`.

### 1. Toy Cases

Command:

```bash
python3 run_probe.py
```

Result:

```text
SUMMARY: 4/4 cases passed
```

### 2. Early Lexical-Only Probe

Command:

```bash
python3 run_real_silver_probe.py
```

This older script only treats annotated answer-like utterances as silver answers.
It does **not** include raw short answers such as "yes" or "no" in the silver
answerable count.

Result:

```text
patient questions: 177
silver answered candidates within 4 utterances: 154
no near answer candidate: 23

answered exact: 108/154 (70.1%)
answered status-only: 108/154 (70.1%)
counterfactual unanswered: 154/154 (100.0%)
```

### 3. Challenge ACI Hybrid Probe

Command:

```bash
python3 run_real_hybrid_probe.py
```

This script includes both annotated summary answers and raw short answers in the
silver answerable count.

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
answered exact: 105/165 (63.6%)
answered status-only: 111/165 (67.3%)
counterfactual unanswered: 153/165 (92.7%)
```

Hybrid raw + embedding, default threshold `0.40`:

```text
answered exact: 155/165 (93.9%)
answered status-only: 160/165 (97.0%)
counterfactual unanswered: 105/165 (63.6%)
```

Interpretation:

```text
The hybrid tracker catches many more silver answers than the lexical baseline,
but it also closes questions more aggressively. That is why counterfactual
unanswered drops from 92.7% to 63.6%.
```

### 4. Broad Stress Test

Command:

```bash
python3 run_broad_dual_probe.py
```

Inventory:

```text
challenge_aci        files=207  questions=1850  doctor=1673  patient=177  answerable=1772  raw_short=183
challenge_eval       files=207  questions=1880  doctor=1687  patient=193  answerable=1813  raw_short=171
src_experiment_aci   files=257  questions=2239  doctor=1959  patient=280  answerable=2124  raw_short=172
all_json_annotations files=671  questions=5969  doctor=5319  patient=650  answerable=5709  raw_short=526
```

The most useful summary is patient questions over `all_json_annotations`:

```text
patient questions: 650
silver answerable: 599
no silver answer: 51
```

Broad patient-question results:

| Tracker | Threshold | Answered Exact | Answered Status | Counterfactual Unanswered | No-Silver Unanswered |
|---|---:|---:|---:|---:|---:|
| lexical | - | 394/599 (65.8%) | 417/599 (69.6%) | 544/599 (90.8%) | 51/51 (100.0%) |
| hybrid raw+embedding | 0.45 | 544/599 (90.8%) | 562/599 (93.8%) | 424/599 (70.8%) | 48/51 (94.1%) |
| hybrid raw+embedding | 0.60 | 485/599 (81.0%) | 515/599 (86.0%) | 473/599 (79.0%) | 48/51 (94.1%) |
| hybrid no-raw | 0.45 | 489/599 (81.6%) | 518/599 (86.5%) | 473/599 (79.0%) | 51/51 (100.0%) |
| hybrid no-raw | 0.60 | 426/599 (71.1%) | 446/599 (74.5%) | 530/599 (88.5%) | 51/51 (100.0%) |

Interpretation:

```text
Lower thresholds improve answered recall but close questions more aggressively.
Higher thresholds and no-raw mode preserve unanswered questions better, but miss
more real answers.
```

### 5. Dual-Threshold Post-Hoc Simulation

The dual-threshold idea:

```text
score >= high:
  confirmed answered

low <= score < high:
  uncertain answer candidate
  keep the question open

score < low:
  still open
```

Important: this is **not** a committed production implementation. It is a
post-hoc simulation using a low-threshold run (`0.45`) and a high-threshold run
(`0.60`).

Patient questions over `all_json_annotations`, hybrid raw+embedding:

```text
answerable_total=599
no_silver_total=51

actual answerable buckets:
  confirmed_answered=515
  uncertain_candidate=47
  still_open=37

confirmed_answered:
  515/599 (86.0%)

candidate_or_confirmed_answered:
  562/599 (93.8%)

counterfactual_confirmed_unanswered:
  424/599 (70.8%)

counterfactual_not_confirmed_answered:
  473/599 (79.0%)

no-silver buckets:
  confirmed_unanswered=48
  false_confirmed_answered=3
```

Patient questions over `all_json_annotations`, hybrid no-raw:

```text
answerable_total=599
no_silver_total=51

actual answerable buckets:
  confirmed_answered=446
  uncertain_candidate=72
  still_open=81

confirmed_answered:
  446/599 (74.5%)

candidate_or_confirmed_answered:
  518/599 (86.5%)

counterfactual_confirmed_unanswered:
  473/599 (79.0%)

counterfactual_not_confirmed_answered:
  530/599 (88.5%)

no-silver buckets:
  confirmed_unanswered=51
```

Interpretation:

```text
Dual-threshold does not remove the answered/unanswered trade-off.
It makes the trade-off explicit by adding an uncertain candidate bucket.
```

## What "Counterfactual Unanswered" Means

For a question with a silver answer, the probe reruns the local dialogue window
after removing that answer utterance.

Example:

```text
Original:
  Patient: Can I take all my medications at the same time?
  Doctor: Yes, you can take them together.

Counterfactual:
  Patient: Can I take all my medications at the same time?
  [doctor answer removed]
```

A cautious tracker should leave the question unanswered in the counterfactual
run. This stress test checks whether the tracker closes questions too easily
based on unrelated later utterances.

## Bottom Line

Confirmed by the repo audit and these probes:

- Current ATLAS C2 does not have a real question answered/unanswered state
  machine.
- A lexical-only C2 is conservative but misses many answer cases.
- Hybrid semantic matching substantially improves silver answered detection.
- More aggressive matching increases false-close risk.
- Dual-threshold interpretation is feasible as a design direction, but it has
  not yet been implemented as production C2 code.

Safe professor-facing wording:

```text
This is not an official accuracy result. It is a silver-label feasibility probe.
Still, broader ATLAS annotation stress tests suggest that C2 question tracking
is plausible: lexical patient-question answered detection was 69.6%, while a
hybrid semantic tracker reached 86.0% answered status at a more balanced
threshold, with 79.0% counterfactual unanswered and 94.1% no-silver unanswered.
```

### 6. Delayed Matching Probe

Command:

```bash
python3 run_matching_probe.py
```

This probe replaces immediate threshold-based closure with local
question-answer matching. C2 first collects candidate edges between open
questions and later utterances, then chooses the best assignment before
finalizing unanswered questions.

Key patient-question results over `all_json_annotations`:

| Tracker | Setting | Answered Status | Counterfactual Unanswered | No-Silver Unanswered |
|---|---:|---:|---:|---:|
| hybrid raw+embedding | 0.60 | 515/599 (86.0%) | 473/599 (79.0%) | 48/51 (94.1%) |
| matching raw | min=0.45 emb=0.45 | 555/599 (92.7%) | 492/599 (82.1%) | 48/51 (94.1%) |
| matching no-raw | min=0.40 pen=0.05 | 528/599 (88.1%) | 525/599 (87.6%) | 51/51 (100.0%) |

Current interpretation:

```text
The matching tracker is the strongest C2 direction so far. It improves the
answered/unanswered balance because C2 no longer closes a question immediately
at the first threshold crossing.
```
