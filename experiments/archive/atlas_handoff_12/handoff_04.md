ATLAS Handoff 4/12: 코드 현황 (2026.04.09 업데이트)

GitHub: https://github.com/Bias92/ATLAS (Private)
Branch: main
최신 커밋: 2bb12c9 "feat: support detail_list format + question type (7 labels)"

파일 6개:

src/config.py — ✅
* threshold, budget, MLP 구조 등 설정
* feature_dim: 396 (384 embedding + 5 scalar + 7 type onehot)
* budget_sweep: [256, 512, 1024, 2048, 4096]

src/harness.py — ✅
* 스트리밍 시뮬레이션 메인 루프
* LLMOutput: detail_list 필드 지원 (1 utterance → 여러 type/summary)
* MockLLM: detail_list 포맷과 legacy 포맷 둘 다 지원
* run(): detail_list에서 여러 BufferItem 생성
* buffer 전체를 LLM에 보여줌

src/baselines.py — ✅
* FIFO: token-budget 기반. 넘칠 때만 oldest 제거.
* Sliding: count-based window (K=10 default). FIFO와 동작이 진짜 다름.
* Random: 랜덤 점수.
* Oldest-first: timestamp 기반. chronological buffer에서는 FIFO와 유사.
* Lowest-attention: ⚠️ PLACEHOLDER. oldest-first fallback.

src/state_tracker.py — ✅
* update()가 detail_list를 순회하며 각 detail 처리
* linking, first-mention, resolution, nesting 모두 detail_list 대응

src/policy.py — ✅
* TYPE_TO_INDEX에 "question": 6 추가
* type_onehot: 6-d → 7-d
* extract_features: 396-d 출력

src/metrics.py — ✅
* ALL_CLASSES에 "question" 추가 (8 classes: 7 types + irrelevant)

아직 placeholder인 것:
1. token budget (len(text)//4) — llama.cpp tokenizer 연결 필요
2. lowest-attention baseline — llama.cpp attention weight 추출 필요
3. FIFO vs oldest 차이 — chronological buffer에서는 동일
