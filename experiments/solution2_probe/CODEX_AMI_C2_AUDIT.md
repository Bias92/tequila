# AMI C2 Audit

Date: 2026-06-18

This note records a direct Codex audit of the AMI C2 experiment handoff.

## Bottom line

The AMI pilot does **not** prove that the C2 question-state idea is invalid.

It does show that the current matcher, when naively ported to AMI, does not beat the majority baseline in a meaningful way. That is a useful warning, not a kill shot.

For the current lab presentation, do **not** replace the ACI probe slides with the AMI numbers. Treat AMI as an external stress test still under validation.

## Why the AMI result is not a clean disproof

The AMI setup labels an elicit act as answered if it appears as an adjacency-pair source. If an elicit is not a source, it is treated as unanswered.

That mapping is noisy. Several sampled "not-source" elicits are followed by utterances that look like answers:

- `What did you get ?`
  - followed by: `Um, I just got the project announcement... Designing a remote control.`
- `Did you get the same thing ?`
  - followed by: `Yeah. ... we're gonna have individual work and then a meeting about it.`
- `Like on the shelf.`
  - followed by: `Um I dunno. ... That's a good question. ... it probably is our sale actually.`

So `not source` is not always the same thing as a clean unanswered question.

## What was run

Script:

```bash
cd /Users/markov/Desktop/tequila/experiments/solution2_probe
python3 run_ami_c2_eval.py --meetings 5 --window 20 --audit 5
```

Input summaries:

```text
/tmp/ami_summaries.json
2324 / 2324 summaries present
```

AMI pilot size:

```text
meetings = 5
acts = 4010
elicit questions = 340
source_elicit = 206
not_source = 134
target_only_not_source = 24
window = 20
```

## Results

Under the simplest `source means answered` label:

```text
all-answered baseline = 60.6%

best observed accuracy:
candidate=meaningful, threshold=0.30
accuracy = 61.2%
recall = 98.5%
specificity = 3.7%
```

Under stricter filtering that skips source/target-weird cases:

```text
all-answered baseline = 64.6%

best observed accuracy:
candidate=meaningful or task, threshold=0.30
accuracy = 65.6%
recall = 99.0%
specificity = 4.5%
```

Raising the threshold improves unanswered preservation but destroys answered recall:

```text
gold=strict_negative, candidate=meaningful

threshold 0.30: recall 99.0%, specificity 4.5%,  accuracy 65.6%
threshold 0.40: recall 93.0%, specificity 10.9%, accuracy 64.0%
threshold 0.50: recall 62.7%, specificity 44.5%, accuracy 56.3%
threshold 0.60: recall 27.4%, specificity 81.8%, accuracy 46.6%
```

## Interpretation

The current AMI pilot says:

1. The current C2 matcher over-closes on AMI.
2. Candidate filtering alone does not fix it.
3. The AMI labels need a small manual audit before they can be used as a decisive benchmark.
4. This should not be presented as a positive result.
5. This also should not be used as proof that C2 is worthless.

## ATLAS eval warning

Do not use ATLAS `eval_only.resolution_status` as a direct per-question answered/unanswered label.

In the ATLAS repo, `resolution_status` is defined as agenda-item status:

```text
mentioned | discussed | resolved | unresolved
```

The existing C2 code updates it at agenda-item level:

```text
follow_up -> resolved
two speakers -> discussed
otherwise mentioned/unresolved
```

That is not the same as `question_status`.

## Recommended framing

For the lab presentation:

- Keep the C2 QuestionState mechanism.
- Keep the ACI probe as a feasibility/stress-test.
- Say AMI is being explored as a stricter external benchmark.
- Do not claim AMI success yet.
- Do not claim AMI invalidates C2 yet.

