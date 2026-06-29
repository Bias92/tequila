"""
ATLAS C1 -> C2 -> C3 integration demo.

Runs ONE annotated dialogue through the full streaming pipeline and prints a
per-utterance trace:
  C1 = trained QLoRA agenda LLM (real)  [or --c1 mock = gold-fed, instant]
  C2 = SystemTracker (deterministic resolution tracking)
  C3 = context-buffer eviction  (--c3 fifo default; learned policy if --c3 learned --c3_pt X)

Imports atlas/src modules (harness, state_tracker, policy, baselines) off --atlas_src.

Run on RunPod (model+adapter live there):
  python demo_c1c2c3.py \
    --atlas_src /workspace/atlas/src \
    --conv /workspace/atlas/data/challenge_data/valid/aci/D2N080.json \
    --c1 real --adapter /workspace/checkpoints/atlas-llama32-3b-qlora \
    --c3 fifo --budget 256
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas_src", required=True, help="path to atlas/src")
    ap.add_argument("--conv", required=True, help="annotated conversation json (utterances+annotations)")
    ap.add_argument("--c1", choices=["real", "mock"], default="real")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="/workspace/checkpoints/atlas-llama32-3b-qlora")
    ap.add_argument("--c3", choices=["fifo", "sliding", "oldest", "random", "learned"], default="fifo")
    ap.add_argument("--c3_pt", default=None, help="learned policy .pt (only for --c3 learned)")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--max_utts", type=int, default=None, help="cap utterances for a quick demo")
    args = ap.parse_args()

    sys.path.insert(0, args.atlas_src)
    from config import DEFAULT_CONFIG
    from harness import StreamingHarness, MockLLM, AgendaLLM, LLMOutput
    from state_tracker import SystemTracker
    from baselines import get_strategy

    DEFAULT_CONFIG.buffer.budget = args.budget
    conv = json.load(open(args.conv))
    if args.max_utts:
        conv = dict(conv)
        conv["utterances"] = conv["utterances"][: args.max_utts]

    # ---------- C1 ----------
    if args.c1 == "real":
        from eval_qlora import load_model, generate, parse_output
        model, tok = load_model(args.base_model, args.adapter)

        class RealC1(AgendaLLM):
            def predict(self, context, speaker, text):
                user = (f"[Context]: {context or 'Start of visit'}\n"
                        f"[Utterance]: [{(speaker or '').capitalize()}] {text}")
                obj, ok = parse_output(generate(model, tok, user))
                if not ok or not obj or not obj.get("relevant"):
                    return LLMOutput(relevant=False)
                dl = obj.get("detail_list") or (
                    [{"type": obj.get("type"), "summary": obj.get("summary")}] if obj.get("summary") else [])
                dl = [{"type": d["type"], "summary": d["summary"]}
                      for d in dl if d.get("type") and d.get("summary")]
                return LLMOutput(relevant=bool(dl), detail_list=dl) if dl else LLMOutput(relevant=False)
        llm = RealC1(DEFAULT_CONFIG)
        c1_label = f"real QLoRA ({args.adapter})"
    else:
        anns = {a["utterance_id"]: a for a in conv.get("annotations", [])}
        llm = MockLLM(DEFAULT_CONFIG, anns)
        c1_label = "mock (gold-fed)"

    # ---------- C2 ----------
    tracker = SystemTracker(DEFAULT_CONFIG.tracker)

    # ---------- C3 ----------
    if args.c3 == "learned":
        from policy import LearnedEvictionStrategy
        strategy = LearnedEvictionStrategy(model_path=args.c3_pt, config=DEFAULT_CONFIG.policy)
        c3_label = f"learned MLP ({args.c3_pt})"
    else:
        strategy = get_strategy(args.c3)
        c3_label = f"{args.c3} baseline"

    print(f"\n{'='*70}\nATLAS C1->C2->C3 demo  |  conv={conv.get('id')}\n"
          f"  C1 = {c1_label}\n  C2 = SystemTracker\n  C3 = {c3_label} (budget={args.budget})\n{'='*70}")

    harness = StreamingHarness(DEFAULT_CONFIG)
    harness.load_conversation_dict(conv)
    result = harness.run(llm, tracker=tracker, eviction_strategy=strategy)

    # ---------- per-utterance trace ----------
    utts = conv["utterances"]
    for i, utt in enumerate(utts):
        out = result.llm_outputs[i]
        try:
            buf_n = len(result.buffer_snapshots[i])
        except Exception:
            buf_n = "?"
        sp = utt.get("speaker", "?")
        txt = (utt.get("text", "") or "").strip().replace("\n", " ")
        line = f"[{i:02d}] ({sp[:7]:>7}) {txt[:64]}"
        if out.get("relevant"):
            types = [d.get("type") for d in (out.get("detail_list") or [])] or [out.get("type")]
            line += f"\n      C1-> RELEVANT {types}   C3 buffer={buf_n}"
        else:
            line += f"\n      C1-> (irrelevant)        C3 buffer={buf_n}"
        print(line)

    # ---- AGENDA SUMMARY (untruncated) — for over-lump / fragmentation judgment ----
    items = result.tracker_states[-1].get("agenda_items", [])
    print(f"\n{'='*70}\nAGENDA SUMMARY: {len(items)} agenda item(s)")
    for it in items:
        print(f"  - [{it['id']}] {it['topic'][:70]}  "
              f"(status={it['status']}; details={len(it['details'])} "
              f"q={len(it['questions'])} meds={len(it['medications'])} "
              f"fu={len(it['follow_ups'])} unans={len(it['unanswered'])})")

    print(f"\n{'-'*70}\nC2 final tracker state (truncated json):")
    print(json.dumps(result.tracker_states[-1], indent=2, ensure_ascii=False, default=str)[:1500])
    print(f"\nmetadata: {json.dumps(result.metadata, default=str)}")


if __name__ == "__main__":
    main()
