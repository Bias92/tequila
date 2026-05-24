# ATLAS QLoRA Results

이 폴더는 ATLAS 프로젝트 C1 (Agenda LLM) 의 QLoRA 학습 + 평가 결과 백업입니다.

작성일: 2026-05-05

---

## 폴더 구조

tequila/experiments/
- qlora_3/                    [현재 학습 결과, paper 1차 hold-out 숫자]
- archive/                    [핸드오프 + src snapshot만 보관]
- solution2_probe/            [C2 question tracking feasibility probe]
- README.md                   [experiments index]
- QLORA_RESULTS_README.md     [이 파일]

---

## atlas_qlora_3/ — 2026-05-05 학습 결과 (paper 1차)

신버전 데이터 (5/4 광호 머지) + completion-only mask 적용 + hold-out 평가.

### 결과 요약

| Split | n | Conv | Parse Rate | Rel Acc | Type Macro F1 |
|-------|------|------|------------|---------|---------------|
| Valid (D2N068~087) | 1012 | 20 | 1.0000 | 86.76% | 0.633 |
| Test1 (D2N088~127) | 2032 | 40 | 0.9985 | 86.71% | 0.642 |
| Test2 (D2N128~167) | 2225 | 40 | 0.9996 | 87.33% | 0.654 |
| Test3 (D2N168~207) | 2298 | 40 | 0.9996 | 87.82% | 0.637 |
| Mean | 7567 | 140 | 0.9994 | 87.16% | 0.642 |

7-class macro F1 (question_unanswered 제외, support 0): 0.733

### 내부 파일

- eval_report_valid.json   (557K)         hold-out valid (D2N068~087, 1012 utts)
- eval_report_test1.json   (1.1M)         test1 (D2N088~127, 2032 utts)
- eval_report_test2.json   (1.2M)         test2 (D2N128~167, 2225 utts)
- eval_report_test3.json   (1.2M)         test3 (D2N168~207, 2298 utts)
- train_qlora.py           (9K, 234줄)    [학습 코드, completion-only mask 패치 적용]
- eval_qlora.py                            [평가 코드]
- trainer_state.json                       [학습 상태 요약]
- train_log.txt / eval_summary.txt / results_summary.txt
- chart_*.png / dist_*.png                 [결과 시각화]

대용량 체크포인트, LoRA weights, optimizer state, GGUF, tarball, train/eval JSONL은
repo에 올리지 않음.

각 eval_report_*.json 안에는 summary 와 per_example 둘 다 있어서 utterance 단위 error analysis 가능.

### 학습 셋업

- Base model: meta-llama/Llama-3.2-3B-Instruct
- QLoRA: r=16, target=q/k/v/o/gate/up/down_proj
- 4-bit NF4 quantization, bf16 compute
- 3 epochs, batch=4, lr=2e-4 cosine
- Completion-only loss masking (build_chat 에서 user portion = -100)
- Custom collator (input_ids/attention_mask/labels 일관 padding)
- Train: D2N001~067 batch mode jsonl (1773 relevant utts)
- Validation during training: D2N068~087 (hold-out)
- Hardware: A100 SXM 80GB (RunPod), 22 분

---

## archive/ — 보조 자료

실패한 1차/2차 QLoRA 폴더와 결과는 repo에서 제거함. 현재 run of record는
`qlora_3/` 하나만 남김.

- atlas_handoff_12/           4/9 핸드오프 폴더 (handoff_01~12.md)
- atlas_src_updated/          4/9 src 업데이트
- handoffs/                   이전 핸드오프 .md
  - HANDOFF_TO_CLAUDE.md       (5/2 핸드오프)
  - QLORA_RETRAIN_HANDOFF.md   (5/4 핸드오프, 이번 세션 시작 전)

---

## paper limitation / caveats

- Train 셋 67 conv 작음 (광호님이 src_experiment_data batch annotation 추가 진행 중)
- Ground truth = Gemini 3.1 Flash Lite Preview annotation (validate-mode QA 92% acceptance)
- question_unanswered 라벨이 4 split 전부 support 0 (라벨 제외 또는 합성 데이터 필요)
- agenda_item / social_history 의 recall 낮음 (모델이 보수적, precision 은 높음)

---

## 다음 단계

1. 1B 모델 cross-model ablation
2. C3 (context manager) MLP 학습 데이터 생성 — paper 메인 contribution
3. C2/C3 baseline 비교 (기배님이 eval_baselines.py 작성, confirm 브랜치)
4. Cross-domain (AMI dataset)
5. Jetson 도착 시 GGUF Q4_K_M 양자화 + tegrastats hardware metrics
