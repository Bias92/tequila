# ATLAS Solution 2 Probe

Small sandbox for testing the "Solution 2" idea without touching the main
`tria-lab/atlas` repo.

Question being tested:

```text
C1 emits type="question".
C2 keeps that question open.
If a later utterance looks like an answer, C2 marks it answered.
If the conversation ends while it is still open, C2 marks it unanswered.
```

This is not production code. It is a sanity probe for whether the idea is even
plausible on small examples copied/derived from the current ATLAS data.

Run the toy cases:

```bash
python run_probe.py
```

Run the real-data probes from this directory:

```bash
python run_real_silver_probe.py
python run_real_hybrid_probe.py
```

By default the real-data probes expect the ATLAS repo to live next to the
`tequila` repo:

```text
../atlas/data/challenge_data
```

Override that with:

```bash
ATLAS_DATA=/path/to/atlas/data/challenge_data python run_real_hybrid_probe.py
```

Current rule:

- A `question` opens a `QuestionState`.
- Later `detail`, `follow_up`, `medication`, or `social_history` from the other
  speaker can answer it.
- The answer candidate must be linked to the same agenda item by token overlap.
- At the end, any still-open question becomes `unanswered`.

Known limitation:

This only works for clear cases where the C1 summaries share topic words. It
will miss indirect answers, vague acknowledgements, and multi-turn repairs.
