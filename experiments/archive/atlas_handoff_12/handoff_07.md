ATLAS Handoff 7/12: 다음 단계 (할 일 순서)

맥에서 할 수 있는 것:
1. llama.cpp 빌드 (Metal 가속 지원됨)
2. GGUF 모델 다운로드 (Llama 3.2 3B Q4_K_M)
3. real inference 1턴 검증 (MockLLM 대신 진짜 LLM)
4. tokenizer 연결 (len//4 placeholder 해결)

데이터팀이 해야 할 것:
* annotation 50개 확대 (현재 3개)
* --mode format 실행 → llm_format_a.jsonl
* context_manager_labels.py → eviction_policy.jsonl

Jetson 있어야만 가능한 것:
* tegrastats profiling
* 최종 hardware metrics
* 논문에 들어갈 latency/memory 숫자

아직 하면 안 되는 것:
* lowest-attention 포함한 최종 baseline 표 주장
* budget sweep 결과를 논문 결론으로 해석 (tokenizer 미연결)
* "eviction policy superiority 입증됐다" 같은 논문 문구

교수님 확인 필요 사항:
1. 7 labels (question 추가) 승인 여부
2. 모델 변경 (Llama 3.2 3B → MedGemma 4B?) — 미확정
3. Overleaf 논문 초안 공유
4. annotation freeze 일정
