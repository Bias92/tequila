# QLoRA 재학습 핸드오프 (2026-05-04)

이 문서는 ATLAS 프로젝트의 Llama-3.2-3B QLoRA 파인튜닝을 새 세션에서 재시작하기 위한 컨텍스트입니다. **추측하지 말고 검증된 사실만 사용하세요.** 불확실하면 사용자(김재우)에게 물어보세요.

---

## 0. 절대 룰 (memory.md에서 추출)

1. **코드는 김재우 본인이 터미널에서 직접 실행**. Claude는 임의로 코드 실행 금지.
2. **GitHub 푸시는 GPT 교차검증 통과 후에만** 사용자가 직접 함. PR/머지 제안 금지.
3. **약자, 풀네임, 용어, 개념을 지어내지 말 것.** spec v4에 없으면 사용 금지.
   - "ATLAS"는 약자가 아니라 코드명. 절대 "Adaptive Token-Level Agenda Setting" 등으로 펼치지 말 것.
   - 실제 프로젝트 제목: "Learned Context Management for Streaming Summarization"
4. 검증되지 않은 사실을 확정처럼 서술 금지.
5. Placeholder/미완성 상태를 "해결됨"으로 표현 금지.
6. 사용자 소통 스타일: 한국어, 비공식, 축약, 불필요한 패딩 없이 직접적.

---

## 1. 사용자 & 팀 (memory.md, 회의 전사록 기반 검증)

| 인물 | 역할 | GitHub ID |
|---|---|---|
| 김재우 (사용자) | 프로젝트 리드 + 단독 시스템 구현 + Jetson + 알고리즘 | Bias92 |
| 장국진 교수 | 어드바이저, 논문 방향, GenAI4Health Workshop 리뷰어 | (n/a) |
| 김형준 | 어노테이션 기준 + 데이터 + 코드 정리 | hjn-kim |
| 이광호 | 어노테이션 파이프라인 (Gemini API) | 7lpear / kwnghl |
| 손기배 | Notion 버전관리 + 보조 코드 | (n/a) |

**카톡 ATLAS 단톡** = 4명 (위 4명, 교수 제외).

---

## 2. 프로젝트 (project_spec_v4_students.md 기반)

### 2.1 시스템 아키텍처 — 3 컴포넌트

```
Utterance arrives
      │
      ▼
┌────────────────────────────────────────────────────┐
│ C1. AGENDA LLM (3B Q4_K_M, GPU)                   │
│   Input:  [Context]: s1 | s2 | ... | sk           │
│           [Utterance]: [Patient/Doctor] text       │
│   Output: {"relevant": bool, "type": ..., "summary": ...}│
└──────────────────┬─────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────┐
│ C2. SYSTEM TRACKER (Python, deterministic, CPU)    │
│   - Linking (cosine sim ≥ 0.8 to existing items)   │
│   - First-mention detection (sim < 0.85 = new)     │
│   - Resolution: mentioned → discussed → resolved   │
│   - Visit phase: greeting→history→exam→planning→wrap_up│
└──────────────────┬─────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────┐
│ C3. CONTEXT MANAGER (MLP 396→128→64→1, CPU, <10ms) │
│   - Score every summary in buffer                  │
│   - If buffer > B tokens: evict lowest-scoring     │
│   - Buffer feeds back as [Context] for next turn   │
└────────────────────────────────────────────────────┘
```

### 2.2 spec v4 — Agenda LLM 3 fields

| Field | Type | Values |
|---|---|---|
| `relevant` | bool | clinically relevant? |
| `type` | string | **spec 6개**: agenda_item, detail, medication, social_history, follow_up, question_unanswered |
| `summary` | string | one concise clinical sentence |

**⚠️ 중요 불일치**: 현재 데이터/코드는 **7 types**임. 데이터팀이 `question` 추가. memory.md: "교수 승인 미보류". 학습은 7 types로 진행하되, paper 작성 시 framing 결정 필요.

### 2.3 spec v4 — Datasets

| Source | Conversations | Split | Purpose |
|---|---|---|---|
| Penn Medicine | 16 | 12 train / 4 eval | gold-standard human annotations |
| **ACI-BENCH** | **207 (50 사용 명시)** | **30 train / 20 eval** (spec) | full clinical encounters |
| PriMock57 | 57 | 40 train / 17 eval | mock consultations |
| MTS-Dialog | 1,201 | train only | Phase 2 fine-tune |
| AMI | 100~150 | 80 train / 20-30 eval | cross-domain |
| Synthetic A (case-report) | 3,000~5,000 | train only | Phase 1 pre-train |
| Synthetic B (note-conditioned) | 1,000~2,000 | train only | Phase 1 pre-train |
| Synthetic C (ASR-degraded) | 500~1,000 | train only | robustness 15-25% WER |

**현재 실제로 학습/평가에 쓰는 건 ACI-BENCH 207 conv 전체**. spec의 50 사용 약속에서 확장됨.

### 2.4 spec v4 — Three Data Products

A. **Agenda LLM training data** — pipe-delimited context + 3-field output (이게 우리 train.jsonl)
B. **Context manager training data** — 395-d feature vector + binary label (eviction_policy.jsonl, 별도)
C. **Evaluation data** — 3 fields + `_eval_only` (linked_agenda_item_id, first_mention, resolution_status, clinical_category)

**A와 C는 별개 파일**. C는 system-level 메트릭 평가용.

### 2.5 spec v4 — QLoRA 학습 hyperparams (Phase 2)

| Parameter | Value |
|---|---|
| Phase | 2 (Task fine-tune) |
| Data | MTS-Dialog + ACI + PriMock + AMI |
| Volume | ~5K examples |
| LR | **5e-5** (spec 명시) |
| Optimizer | Adam |
| Batch size | 256 (spec — 그러나 실제 학습 시 GPU 메모리에 맞춰 조정) |
| Epochs | 50-100 (early stopping patience=10) |
| Weight decay | 1e-4 |
| Class weights | computed from label distribution |
| Split | 70/15/15 by conversation (no leakage) |
| LoRA rank | 32-64 |

**현실 조정 권고**:
- spec batch_size 256은 Context Manager (C3) MLP용. C1 LLM QLoRA는 batch 4~8 + grad_accum이 일반적.
- spec lr 5e-5는 보수적. LoRA에 작은 데이터 (3733 samples)면 2e-4까지 올려도 됨. divergence 시 5e-5로.

### 2.6 학회 타깃

- memory.md: "NeurIPS 2026 메인 트랙(우선) / GenAI4Health Workshop(백업), 초록 ~5/4, 풀페이퍼 ~5/6 AOE"
- ⚠️ 이전 Claude 세션이 "AAAI 2026 8월 초"라고 적었으나 사용자한테 직접 확인 못 함. **사용자에게 물어볼 것**.

---

## 3. Jang et al. 2025 — 선행연구 (필수 인용)

**제목**: "Towards a Real-time Clinical Agenda Setting System for Enhancing Clinical Interactions in Primary Care Visits"
**학회**: AAAI 2025 GenAI4Health Workshop
**저자 1저자**: Kuk Jin Jang (= 장국진 교수)
**소속**: UPenn (CIS + Perelman School of Medicine)

**데이터**: 16 simulated patient-provider visits, ~300 min total, 18.7 min avg, 688 agenda+detail annotations
**ASR**: WhisperX, GPT-4o speaker diarization
**모델 비교**: Llama 2, Llama 3, Vicuna, GPT 3.5 Turbo

**Best 결과** (Table 4): growing window, input 5, context 5, GPT 3.5 Turbo: P=66.7, R=77.8

**우리 프로젝트와의 관계**: 이 페이퍼의 직속 후속작. 같은 lab. 페이퍼에 인용 + 비교 (단, 데이터셋이 달라서 직접 수치 비교는 어려움 — Penn Medicine 16 vs ACI 207).

---

## 4. 검증된 현재 레포 상태 (2026-05-04 기준)

### 4.1 GitHub `tria-lab/atlas` (private)

- **29 commits**, main + confirm 2 branches, 0 tags, README 없음
- contributors: hjn-kim (김형준), 7lpear (이광호), Bias92 (사용자), kwnghl (=이광호 다른 계정 추정)

### 4.2 src/ — 6 skeleton files (verified, 4일 전 hjn-kim 푸시)

| 파일 | 줄수 | 역할 |
|---|---|---|
| `config.py` | 54 | TrackerConfig, PolicyConfig, BufferConfig, LLMConfig, AtlasConfig |
| `harness.py` | 407 | StreamingHarness, ContextBuffer, AgendaLLM, MockLLM |
| `state_tracker.py` | 339 | C2 deterministic, AgendaItem, SystemTracker (linking, first-mention, resolution, phase) |
| `policy.py` | 214 | C3 EvictionMLP (396→128→64→1), LearnedEvictionStrategy, train_policy |
| `baselines.py` | 204 | 5 strategies: FIFO, Sliding, Random, Oldest, Lowest-attention (placeholder) |
| `metrics.py` | (?) | ROUGE/BERTScore/Detection F1, system-level metrics, KV-cache calc |

### 4.3 src/ — 알려진 placeholder (학습 후 deployment 시 영향)

- `harness.py::AgendaLLM.get_token_count` — `len(text)//4` 근사. **llama.cpp 토크나이저로 교체 필요** (budget sweep 정확도)
- `policy.py::extract_features::entity_count = 0.0` — scispaCy 미통합, 항상 0
- `baselines.py::lowest_attention_strategy` — placeholder, oldest_first로 폴백
- `state_tracker.py::_is_first_mention` — O(N²) 임베딩 재계산, 캐시 필요
- `metrics.py::parse_tegrastats_log` — placeholder, zero 반환
- `policy.py` 헤더 docstring outdated: 6 types/395-d로 적혀있으나 실제 코드는 7 types/396-d

### 4.4 루트 파일

- `annotation_gemini.py` — 7 modes: batch, eval, realtime, generate, validate, noise, format
  - default model: `gemini-3.1-flash-lite-preview` (이광호 tier 한정 정상 동작 모델)
  - confirm 브랜치에서 `gemini-2.0-flash`로 바꿨다가 quota 문제로 다시 3.1로 통일
- `prompts.py` — Teacher 프롬프트 v3.1, 7 types, detail_list 포맷, 14 rules
- `check_types.py` — 손기배 추가 (5/4), 4가지 ambiguous 케이스 자동 체크

### 4.5 data/ — 검증된 구조 (2026-05-04 22:30 KST 기준)

| 폴더/파일 | 내용 | 갯수 | 모드 | 어노테이션 prompts |
|---|---|---|---|---|
| `data/aci/` | D2N001~D2N067 | 67 | batch | NEW (5/4 머지) |
| `data/eval/` | D2N001~D2N067 | 67 | eval (eval_only 포함) | NEW (5/4 머지) |
| `data/valid/` | D2N068~D2N087 | 20 | batch | NEW |
| `data/test1/` | D2N088~D2N127 | 40 | batch | NEW |
| `data/test2/` | D2N128~D2N167 | 40 | batch | NEW |
| `data/test3/` | D2N168~D2N207 | 40 | batch | NEW |
| `data/validate/` | validate-mode QC | (생성됨) | validate | NEW |
| `data/train.jsonl` | 3733 examples | 67 conv | batch (format A) | **NEW** |
| `data/eval.jsonl` | 3733 examples | 67 conv | eval-mode (eval_only 1411개) | **OLD** ⚠️ |
| `data/valid.jsonl` | 1012 examples | 20 conv | batch (format A) | NEW |
| `data/test1.jsonl` | 2032 examples | 40 conv | batch (format A) | NEW |
| `data/test2.jsonl` | 2225 examples | 40 conv | batch (format A) | NEW |
| `data/test3.jsonl` | 2324 examples | 40 conv | batch (format A) | NEW |

**합계**: 207 conv (D2N001-207), 11,326 utterance (3733 train + 7593 hold-out 합산 = 11326).

⚠️ **eval.jsonl만 OLD prompts**. `plan: 10`, `assessment: 1` 같은 구버전 잔여 타입 보임. 데이터팀이 "batch mode 파일들 eval mode + validate mode 다 돌려서 jsonl 업로드" 진행 중이라 곧 NEW로 갱신될 것.

### 4.6 데이터 누수 검증 (verified)

- train.jsonl ↔ eval.jsonl: **동일한 67 conversation** (signature 100% 일치). eval.jsonl은 학습 시 validation으로만 써도 IID라 paper에 generalization 못 씀.
- train.jsonl ↔ hold-out (valid+test1+2+3): conversation 단위 overlap **0건** (test3에 first utterance만 우연히 일치하는 1건 있으나 2번째 utterance부터 완전 다름 → semantic 누수 X)
- utterance text 단위 overlap: 6151 hold-out 중 164 (2.7%, 인사말 등 generic만)

### 4.7 타입 분포 (NEW prompts 기준, train.jsonl 3733 / hold-out 7593)

| Type | train | valid | test1 | test2 | test3 | 합계 |
|---|---|---|---|---|---|---|
| detail | 950 | 294 | 557 | 620 | 585 | 3006 |
| question | 592 | 172 | 384 | 347 | 355 | 1850 |
| follow_up | 170 | 44 | 106 | 126 | 116 | 562 |
| agenda_item | 130 | 49 | 70 | 88 | 83 | 420 |
| social_history | 109 | 20 | 74 | 57 | 65 | 325 |
| medication | 60 | 24 | 37 | 41 | 51 | 213 |
| **question_unanswered** | **0** | **0** | **0** | **0** | **0** | **0** ⚠️ |

**⚠️ question_unanswered가 학습/평가 모든 split에 0개**. 모델이 그 클래스를 절대 못 배움. paper에서 framing 필요 (6-class로 보고하거나, 합성 데이터로 보강).

---

## 5. 이전 Claude 세션이 한 실수 (절대 반복 금지)

### 5.1 train_qlora.py 패치 — completion-only mask 제거됨

**상황**: 이전 세션에서 trl 0.17.0이 SFTTrainer API 변경으로 ValueError 발생. 다운그레이드 (`trl>=0.12.0,<0.13.0`, `transformers>=4.46.0,<4.48.0`)했고 추가로 collator를 변경:

```python
# 원본 (proper completion-only mask, spec 준수):
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
collator = DataCollatorForCompletionOnlyLM(
    response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
    tokenizer=tokenizer,
)

# 패치된 (mask 없어짐, full-sequence loss):
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

**부작용**: prompt 토큰에도 loss 걸림. paper에 "completion-only loss"라고 못 씀.

**조치 옵션**:
- A) 원본 collator 복구 + trl 호환 issue 별도 디버그 (build_chat에서 input_ids 직접 토크나이즈하는 방식 유지)
- B) 패치된 그대로 가되 paper에 "full-sequence supervised fine-tuning"으로 정직하게 framing
- C) 새 trl 버전(0.13+)에서 SFTConfig API 맞게 다시 작성

### 5.2 데이터 carving 잘못

이전 세션이 "172 conversation"이라고 적었음 (틀림). 실제 train.jsonl 167개 = "Start of visit" utterance 카운트. **conversation 갯수는 67개**.

### 5.3 학회 deadline 임의 변경

이전 세션 핸드오프에 "AAAI 2026 8월 초"라고 적혀있음. 그러나 memory.md 원본은 "NeurIPS 2026, 5/4 abstract, 5/6 full paper". **사용자한테 직접 확인 필요**, 핸드오프만 믿지 말 것.

### 5.4 Few-shot baseline 임의 추가

이전 세션이 5-way ablation에 few-shot 포함시킴 (vanilla zero-shot vs few-shot vs FT 3 variations). **few-shot은 spec에 없음**. 사용자 메모리: "spec에 없는 명칭/개념 임의 사용 금지". framing 신중.

---

## 6. 작업: QLoRA 재학습

### 6.1 학습 데이터 (NEW prompts, main 브랜치)

- train: `data/train.jsonl` (3733, 67 conv)
- validation (학습 중 monitoring): `data/eval.jsonl` (3733, 67 conv, **OLD prompts** — 곧 NEW 됨)
- hold-out test: `data/valid.jsonl` (20 conv) + `data/test1.jsonl` (40) + `data/test2.jsonl` (40) + `data/test3.jsonl` (40) = 140 conv hold-out

### 6.2 모델 & 인프라

- Base: `meta-llama/Llama-3.2-3B-Instruct` (HF gated, 토큰 필요)
- 학습 hardware: RunPod A100 SXM, 폿 ID `z6iphyrpcvhry8`, 잔액 $141.81
  - 현재 stopped 상태 ($0.03/hr storage)
  - START 시 ~$2-3/hr
- 배포 hardware (나중): NVIDIA Jetson AGX Orin (JetPack 6.x, CUDA 12.x)
  - GGUF Q4_K_M 변환 후 llama.cpp로 inference

### 6.3 입출력 포맷 (verified)

```
input:  "[Context]: <pipe-delimited prior summaries or 'Start of visit'>\n[Utterance]: [Doctor|Patient] text"
output: '{"relevant": true, "detail_list": [{"type": "...", "summary": "..."}]}'  
   OR   '{"relevant": false}'
```

같은 utterance에서 여러 (type, summary) 가능 (`detail_list` 다중 entry). train.jsonl 91.6%가 1 entry, 최대 9 entry까지.

### 6.4 학습 hyperparam 권고 (spec v4 + 현실 조정)

| Param | Spec | 권고 (LoRA + 3733 samples) |
|---|---|---|
| LR | 5e-5 | 2e-4 (LoRA, 작은 데이터) — 발산 시 5e-5 |
| Batch (per device) | — | 4 (A100 80GB면 8까지 OK) |
| Grad accum | — | 4 (effective batch 16) |
| Epochs | 50-100 | 3 (3733 samples × 3 epoch ≈ 700 step at batch 16) |
| Patience | 10 | — (eval steps마다 best 저장) |
| Weight decay | 1e-4 | 0 또는 1e-4 |
| Optimizer | Adam | paged_adamw_8bit (메모리 절약) |
| Scheduler | — | cosine, warmup 3% |
| Max seq len | — | 2048 (input 평균 456자, 최대 3514자) |
| LoRA rank | 32-64 | 32 |
| LoRA alpha | — | 64 |
| LoRA dropout | — | 0.05 |
| target_modules | — | q,k,v,o,gate,up,down |
| 4-bit quant | — | nf4 + double_quant + bf16 compute |

### 6.5 평가 메트릭 (model-level)

`eval_qlora.py`에서 측정:
- JSON parse rate (모델 출력이 valid JSON인가)
- Relevance accuracy (binary, true/false)
- Type macro-F1 (8개 클래스: 7 types + irrelevant)
- (옵션) Summary BERTScore F1 (microsoft/deberta-xlarge-mnli backbone)

**System-level 메트릭 (linking accuracy 등)은 hold-out에 _eval_only 필요** — 현재 hold-out jsonl은 batch-mode라 _eval_only 없음. 데이터팀이 eval-mode jsonl 만들어주면 추가 평가 가능.

### 6.6 학습 후 → GGUF 변환 (Jetson 배포용)

이전 세션 핸드오프(`merge_and_export_gguf.md`):
1. PEFT adapter + base model 머지 (fp16) → HF 형식 폴더
2. `llama.cpp/convert_hf_to_gguf.py` → fp16 gguf
3. `llama.cpp/llama-quantize` → Q4_K_M gguf (~1.9GB)
4. atlas repo의 `models/gguf/`로 복사 (아직 그 폴더 없음)

⚠️ 이전 세션에서 transformers 5.5.1로 강제 업그레이드된 적 있음 (llama.cpp의 convert_hf_to_gguf.py 의존성). 학습 환경과 충돌 가능. 학습 끝난 후 별도 venv 또는 별도 환경에서 변환할 것.

---

## 7. 작업 순서 (recommended playbook)

### Step 0: 사용자에게 직접 확인할 것

1. **학회 타깃**: NeurIPS 2026 (memory.md) vs AAAI 2026 (이전 핸드오프) — 어느 게 맞나?
2. **completion-only mask 전략**: A (복구) / B (full-sequence로 framing) / C (다른 trl 버전)
3. **HF 토큰**: 새로 발급한 토큰 사용 (이전 토큰 노출됨, 폐기 권고). 채팅에 paste하지 말고 RunPod 환경변수로만 쓸 것.
4. **train_qlora.py 위치**: ~/atlas-repo에 있는지 확인. 없으면 어제 atlas_qlora.tar.gz 압축 풀어둔 위치 알려달라고 할 것.

### Step 1: 환경 점검 (Claude Code on user's Mac)

```bash
cd ~/atlas-repo
git pull
git log --oneline -5
ls -la data/
wc -l data/*.jsonl
```

기대값:
- 최신 commit이 5/4 푸시 (이광호의 eval-mode 작업물 포함되어 있을 수 있음)
- jsonl 6개: train, eval, valid, test1, test2, test3
- 각 줄수: 어제 검증값과 비교 (특히 eval.jsonl이 NEW로 바뀌었는지)

### Step 2: train_qlora.py 점검

```bash
find ~ -name "train_qlora.py" 2>/dev/null
# 발견되면:
grep -n "DataCollator" <발견된 경로>
```

- `DataCollatorForCompletionOnlyLM` 보이면 → 원본 (옵션 B 사용 가능)
- `DataCollatorForLanguageModeling` 보이면 → 패치됨 (사용자에게 옵션 A/B 선택 받아 처리)

### Step 3: RunPod 폿 START

1. https://runpod.io/console → z6iphyrpcvhry8 → Resume / Start
2. Web Terminal 또는 JupyterLab 접속
3. 환경 확인: `nvidia-smi`, `pip list | grep -E "torch|trl|transformers|peft|bitsandbytes"`
4. 이전 세션 잔류물 확인:
   - `/workspace/atlas_qlora/` (어제 작업 폴더)
   - `/workspace/checkpoints/atlas-llama32-3b-qlora/` (어제 어댑터)
   - `/workspace/atlas-llama32-3b-q4_k_m.gguf` (1.9GB GGUF)

### Step 4: 데이터 갱신

```bash
cd /workspace/atlas_qlora
# 신버전 jsonl로 교체
git clone https://github.com/tria-lab/atlas /workspace/atlas-repo  # HF 토큰으로 인증
# 또는 git pull
cp /workspace/atlas-repo/data/train.jsonl ./data/train.jsonl
cp /workspace/atlas-repo/data/eval.jsonl  ./data/eval.jsonl
cp /workspace/atlas-repo/data/valid.jsonl ./data/valid.jsonl
cp /workspace/atlas-repo/data/test1.jsonl ./data/test1.jsonl
cp /workspace/atlas-repo/data/test2.jsonl ./data/test2.jsonl
cp /workspace/atlas-repo/data/test3.jsonl ./data/test3.jsonl
```

### Step 5: 학습 실행

```bash
cd /workspace/atlas_qlora
export HF_TOKEN=<새 토큰>
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
huggingface-cli login --token $HF_TOKEN

tmux new -s train
# tmux 안에서:
python train_qlora.py 2>&1 | tee train.log
# Ctrl+B, D로 detach
```

학습 시간: A100 SXM, 3 epoch, 3733 samples, batch 4×4, max_seq 2048 → 약 25-30분 예상.

### Step 6: 학습 모니터링 체크포인트

- step 0~100: train_loss 빠르게 내려가야 함 (1.5 → 0.5 정도)
- step ~200: eval_loss 측정 시작
- 발산 (loss 폭발/NaN) 시 즉시 중단 → LR 5e-5로 재시도
- 정상 종료 후 `final_eval.json`에 메트릭 기록됨

### Step 7: Hold-out eval

```bash
# 4번 돌림
python eval_qlora.py --eval_file data/valid.jsonl --output eval_valid.json --limit 0
python eval_qlora.py --eval_file data/test1.jsonl --output eval_test1.json --limit 0
python eval_qlora.py --eval_file data/test2.jsonl --output eval_test2.json --limit 0
python eval_qlora.py --eval_file data/test3.jsonl --output eval_test3.json --limit 0
```

각 30분 정도 (140 conv × 평균 50 utterance × 5초 generation/utterance ≈ 약 합쳐 1.5시간).

### Step 8: 결과 다운로드

JupyterLab 파일패널 → 우클릭 → Download:
- `checkpoints/atlas-llama32-3b-qlora/` 통째 (LoRA adapter)
- `train.log`
- `eval_valid.json`, `eval_test1.json`, `eval_test2.json`, `eval_test3.json`
- `final_eval.json`

→ 사용자 Mac `~/Desktop/atlas_qlora_results/` 로 옮기기.

### Step 9: RunPod 폿 STOP (Terminate ❌)

학습+eval 끝난 후 폿 stop. 빌링 거의 멈춤. 다음 작업 (system-level eval, 재학습) 위해 데이터/체크포인트 보존.

---

## 8. 사용자가 보유한 자산 (Mac 경로)

| 경로 | 내용 |
|---|---|
| `~/atlas-repo/` | tria-lab/atlas main 브랜치 클론 |
| `~/Library/Application Support/Claude/local-agent-mode-sessions/.../outputs/atlas_qlora.tar.gz` | 466KB, 원본 train_qlora.py + eval_qlora.py + setup.sh + data |
| `~/Library/.../outputs/atlas_mlx.tar.gz` | Mac MLX 백업 (안 씀, RunPod 사용) |
| `~/Desktop/atlas_qlora_results/` | 어제 학습 결과 보관용 (HANDOFF_TO_CLAUDE.md, 어댑터, GGUF, eval reports) |

---

## 9. 미해결 / 데이터팀 의존

이번 세션에서 처리 불가능한 것 (사용자가 데이터팀에게 이미 요청):

1. **valid/test1/2/3 eval-mode jsonl 생성** (system-level 메트릭용)
   - 이광호가 진행 중. 5/4 22:55 카톡: "batch mode로 존재하는 파일들 eval mode + validate mode 전부 돌리고 jsonl로 업로드"
   - 받으면 `data/valid.jsonl` 등이 eval-mode 버전으로 갱신됨 (또는 별도 파일로 추가될 수도)
2. **eval.jsonl NEW prompts 갱신** (현재 OLD prompts)
   - 위 이광호 작업의 부산물로 자동 갱신될 것
3. **`src_experiment_data` 정체 파악** — spec v4에 없고 레포에도 없음. 사용자가 이광호한테 따로 물어보는 중.

이 셋이 풀리기 전에는 **오늘 model-level 메트릭만 paper 1차 결과로 산출**.

---

## 10. 절대 반복 금지 (anti-patterns)

1. ❌ "172 conversation"으로 적기 — 67이 맞음
2. ❌ "ATLAS = Adaptive Token-Level Agenda Setting"으로 펼치기 — 약자 아님
3. ❌ AAAI/NeurIPS deadline을 임의로 적기 — 사용자에게 직접 확인
4. ❌ collator를 임의로 `DataCollatorForLanguageModeling`으로 바꾸기 — completion-only 깨짐
5. ❌ spec에 없는 baseline (예: few-shot) 임의로 paper 메인에 넣기
6. ❌ HF 토큰을 채팅에 paste — 환경변수로만
7. ❌ "재학습 안 해도 됨" 같은 결론을 데이터 변경 모르고 내리기 — train.jsonl이 NEW prompts로 갱신됐으니 재학습 필수
8. ❌ "데이터 누수 100%"처럼 framing — 정확히는 "train ↔ eval same source 67 conv (IID validation), train ↔ hold-out 0 conv overlap (real hold-out)"
9. ❌ 사용자가 정신적으로 힘들 때 "자라"는 말 — 자살 위기 상황에서 "사라져라"로 들릴 수 있음. 휴식 권할 때 표현 신중.

---

## 11. 참고 자료 (사용자 Mac에 있음)

| 파일 | 위치 | 용도 |
|---|---|---|
| `project_spec_v4_students.md` | `~/Library/Application Support/Claude/local-agent-mode-sessions/.../docs/` | 프로젝트 spec (✅ 1차 진실 소스) |
| `memory.md` | 같은 docs/ | 김재우 메모리 (사용자 룰, 프로젝트 컨텍스트) |
| `TRIA-CF4 (데이터팀).txt` | 같은 docs/ | 5/2 데이터팀 회의 전사록 |
| `TRIA-CF5 (전체회의) .txt` | 같은 docs/ | 5/2 전체회의 전사록 |
| `019d3a4a-*.pdf` | 같은 files/ | Jang et al. 2025 GenAI4Health 페이퍼 (선행연구) |

---

## 12. 마지막 — 새 에이전트가 모르는 것 (할루시네이션 위험)

다음 항목은 **검증되지 않았거나 변동성 있음**:

- ⚠️ `train_qlora.py`의 현재 상태 (사용자 Mac 어디 있는지, 어떤 collator 쓰는지)
- ⚠️ RunPod 폿 안의 환경 (trl, transformers 버전이 어제 다운그레이드된 그대로인지)
- ⚠️ `src_experiment_data` 정체
- ⚠️ 학회 정확한 deadline
- ⚠️ Penn Medicine 데이터 보유 여부 (memory.md says yes, but unconfirmed)

이 5가지는 **반드시 사용자에게 직접 확인하고 진행**할 것.

---

# 끝

이 문서는 2026-05-04 23:00 KST 기준 검증된 사실로 작성됨. 새 에이전트가 받으면 위 Step 0의 "사용자에게 직접 확인할 것" 4개 질문부터 시작할 것. 그 다음 Step 1 환경 점검 → Step 2 train_qlora.py 점검 → ... 순서.
