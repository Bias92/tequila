ATLAS Handoff 2/12: 3-Component 아키텍처

실시간 동작 루프 (~5초 간격):

발화 도착
  → [1] Agenda LLM (GPU, ~1초)
       입력: [Context]: s1 | s2 | s3
             [Utterance]: [Speaker] text
       출력: {"relevant": bool, "detail_list": [{"type": str, "summary": str}, ...]}
       relevant=false면 → 무시, 다음 발화 대기
  → [2] System Tracker (CPU, ~100ms)
       linking, first-mention, resolution, nesting
       규칙 기반 Python. LLM 호출 없음.
  → [3] Context Manager (CPU, <10ms)
       MLP (396→128→64→1)로 점수 매기고 낮은 것부터 제거
       버퍼 토큰 ≤ B 유지
  → 버퍼 업데이트 → 다음 발화의 [Context]로 피드백

Component 1: Agenda LLM
* 모델: Llama 3.2 3B, GGUF Q4_K_M (~2GB), llama.cpp 추론
* 출력: detail_list 구조 (1 utterance → 여러 type/summary 가능)
* 7가지 type: agenda_item, question, detail, medication, social_history, follow_up, question_unanswered
* Context: pipe-delimited string (s1 | s2 | s3). raw transcript 아님.
* 학습: QLoRA 3-phase (50K→5K→500 examples)

Component 2: System Tracker
* 규칙 기반 Python 코드. 학습 없음.
* Linking: all-MiniLM-L6-v2 임베딩, cosine sim > 0.8
* First-mention: cosine sim > 0.85면 반복
* Resolution: mentioned → discussed → resolved → unresolved
* Visit phase: keyword 기반 (greeting→history→exam→planning→wrap_up)

Component 3: Context Manager / Eviction Policy
* MLP: 396-d input → 128 → 64 → 1 (raw logits, sigmoid at inference)
* 396-d = 384(embedding) + position + recency + speaker + token_count + entity_count + type_onehot(7)
* 학습: hindsight supervision (BERTScore > 0.75 → label=1)
* 5 baselines: FIFO(token-budget), sliding(count-based K), random, oldest-first, lowest-attention(placeholder)
* Budget sweep: B ∈ {256, 512, 1024, 2048, 4096}

중요 변경 (2026.04.09):
* type 6개 → 7개 (question 추가)
* 출력 포맷: single type/summary → detail_list 구조
* feature_dim: 395 → 396 (type onehot 6→7)
