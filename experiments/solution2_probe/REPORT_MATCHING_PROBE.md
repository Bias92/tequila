# C2 Matching Probe

This is the current C2 direction after the early greedy-threshold probes.

## Problem

The first Solution 2 trackers closed a question as soon as a later utterance
crossed a similarity threshold.

That was simple, but brittle:

- a question could be closed too early;
- a single threshold created a strong answered/unanswered trade-off;
- the tracker did not choose the best question-answer pairing across the local
  dialogue window.

## New Formulation

The matching tracker treats C2 question tracking as a constrained matching
problem.

```text
open questions Q
later answer candidates U

score(q, u)
  = lexical overlap
  + embedding similarity
  + same-agenda bonus
  + answer-type bonus
  - temporal-distance penalty
```

Then C2 chooses the highest-scoring question-answer assignment before
finalizing unanswered questions.

```text
matched question   -> answered
unmatched question -> unanswered
```

This is still a probe, not production C2. The goal is to test whether a more
global C2 decision rule improves over greedy threshold matching.

## Files

- `src/state_tracker_solution2_matching.py`
  - delayed matching C2 tracker
- `run_matching_probe.py`
  - broad silver-label evaluation for matching variants

## Reproduction

Run from this directory:

```bash
python3 run_matching_probe.py
```

The script evaluates patient questions over:

```text
challenge_data/{train,valid,test1,test2,test3}/{aci,eval}
src_experiment_data/*/aci
```

The verified corpus count is:

```text
files: 671
patient questions: 650
silver answerable: 599
no silver answer: 51
```

## Result

Patient questions over `all_json_annotations`.

| Tracker | Setting | Answered exact | Answered status | Counterfactual unanswered | No-answer unanswered |
|---|---:|---:|---:|---:|---:|
| lexical-only | - | 394/599 (65.8%) | 417/599 (69.6%) | 544/599 (90.8%) | 51/51 (100.0%) |
| hybrid raw+embedding | 0.45 | 544/599 (90.8%) | 562/599 (93.8%) | 424/599 (70.8%) | 48/51 (94.1%) |
| hybrid raw+embedding | 0.60 | 485/599 (81.0%) | 515/599 (86.0%) | 473/599 (79.0%) | 48/51 (94.1%) |
| matching raw | min=0.45 emb=0.45 | 520/599 (86.8%) | 555/599 (92.7%) | 492/599 (82.1%) | 48/51 (94.1%) |
| matching no-raw | min=0.40 pen=0.05 | 505/599 (84.3%) | 528/599 (88.1%) | 525/599 (87.6%) | 51/51 (100.0%) |

## Takeaway

The matching version is the strongest C2 direction so far.

Compared with the previous balanced hybrid setting:

```text
hybrid raw+embedding @0.60
  answered status:          86.0%
  counterfactual unanswered:79.0%
  no-answer unanswered:     94.1%

matching no-raw, min=0.40, temporal penalty=0.05
  answered status:          88.1%
  counterfactual unanswered:87.6%
  no-answer unanswered:     100.0%
```

This suggests the weakness was not C2 itself. The weaker part was the greedy
threshold matcher.

## Safe Wording

Use this as a probe result:

```text
We reformulated C2 question tracking as local question-answer matching rather
than immediate threshold-based closure. On the same silver-label stress test,
matching improved the balance between answered detection and unanswered
preservation.
```

Do not claim this as final gold-label accuracy.
