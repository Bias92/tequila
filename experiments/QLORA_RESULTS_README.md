# ATLAS QLoRA Results

이 폴더는 ATLAS 프로젝트 C1 (Agenda LLM) 의 QLoRA 학습 + 평가 결과 백업입니다.

작성일: 2026-05-05

---

## 폴더 구조

atlas_qlora_results/
- atlas_qlora_3/              [오늘 학습 결과, paper 1차 hold-out 숫자]
- archive/                    [옛 자료, 1차 학습 + 핸드오프 등]
- inspect/                    [5/3 검증 시 만든 폴더]
- README.txt                  [옛 메모]
- README.md                   [이 파일]

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

- checkpoints/atlas-llama32-3b-qlora/    QLoRA 학습 산출물 (795M)
  - adapter_config.json                   LoRA r=16, target=q/k/v/o/gate/up/down_proj
  - adapter_model.safetensors             [학습된 LoRA weights, 194MB]
  - checkpoint-600/                       중간 저장 (epoch 1.72, eval_loss best 시점)
  - checkpoint-699/                       최종 저장 (epoch 3.0, eval_loss 0.326)
  - final_eval.json                       eval_loss, runtime 등
  - tokenizer.json, tokenizer_config.json Llama-3.2-3B tokenizer
  - special_tokens_map.json
  - training_args.bin                     SFTConfig 직렬화 (재현 가능)
  - README.md                             HF auto-generated
- eval_report_valid.json   (557K)         hold-out valid (D2N068~087, 1012 utts)
- eval_report_test1.json   (1.1M)         test1 (D2N088~127, 2032 utts)
- eval_report_test2.json   (1.2M)         test2 (D2N128~167, 2225 utts)
- eval_report_test3.json   (1.2M)         test3 (D2N168~207, 2298 utts)
- train_qlora.py           (9K, 234줄)    [학습 코드, completion-only mask 패치 적용]

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

## archive/ — 옛 자료 보관

- atlas_qlora_1/              5/2 1차 학습 폴더 (구버전 프롬프트, IID 누수)
  - train_qlora.py (207줄)     1차 원본 (mask 패치 전)
  - data/train.jsonl, eval.jsonl  1차 IID 데이터 (D2N001~067 양쪽 다)
- atlas_qlora_2/              5/2 1차 패치 폴더
  - train_qlora.py (208줄)     1차 패치 (mask 변경, completion-only 깨짐)
  - train_qlora.py.bak         패치 전 백업
  - checkpoints/                1차 LoRA adapter
- atlas_qlora_1_results/      5/3 1차 학습 산출물 (paper 비교용)
  - atlas-llama32-3b-q4_k_m.gguf  (1.9G)  1차 GGUF 양자화 (구버전 모델)
  - atlas_extra_models.tar.gz     (1.0G)  1차 추가 모델 (1B 등)
  - atlas_qlora_results_full.tar.gz (689M)
  - eval_report_finetuned.json    (120K)  1차 3B FT (구버전 IID, type F1 0.62)
  - eval_report_2ep.json          (120K)  1차 2 epoch ablation
  - eval_report_1b_finetuned.json (123K)  1차 1B FT (cross-model)
  - eval_report_fewshot.json      (132K)  1차 few-shot baseline
  - eval_report_vanilla.json      (125K)  1차 vanilla zero-shot
- atlas_handoff_12/           4/9 핸드오프 폴더 (handoff_01~12.md)
- atlas_src_updated/          4/9 src 업데이트
- handoffs/                   이전 핸드오프 .md
  - HANDOFF_TO_CLAUDE.md       (5/2 핸드오프)
  - QLORA_RETRAIN_HANDOFF.md   (5/4 핸드오프, 이번 세션 시작 전)

### 1차 학습 (5/3) vs 3차 학습 (5/5) 비교

1차 학습은 train/eval 모두 D2N001~067 사용해서 누수 있었음. 결과 부풀려짐.

| 항목 | 1차 (구버전 IID, 부풀려짐) | 3차 (신버전 hold-out, 정직) |
|------|---------------------------|----------------------------|
| Type macro F1 | 0.62 | 0.642 |
| Rel acc | 94% | 87.16% |
| JSON parse | 96% | 99.94% |

Rel acc 떨어진 게 누수 빠진 정직성 신호. F1 거의 동일 → 학습 자체는 잘 됐음.

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
