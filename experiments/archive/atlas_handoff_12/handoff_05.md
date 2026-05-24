ATLAS Handoff 5/12: 선행 논문 (Jang et al. AAAI 2025)

"Towards a Real-time Clinical Agenda Setting System for Enhancing Clinical Interactions in Primary Care Visits"
UPenn 연구팀. 16개 simulated 환자-의사 대화로 실시간 agenda setting feasibility 탐색.

핵심 실험 결과:
* Baseline: GPT 3.5 Turbo 최고 ROUGE-L (28.89), Llama 3 최고 BERTScore (86.74)
* Input line 5줄이 최적
* Context size ↑ → Recall ↑ but Precision ↓ (trade-off)
* Growing window (input=5, context=5): Precision 66.7%, Recall 77.8% — ATLAS의 직접적 동기

ATLAS가 이 논문을 확장하는 방법:
* GPT 3.5 / Llama 8B → Llama 3.2 3B (edge)
* 클라우드 → Jetson AGX Orin
* 고정 window → 학습된 eviction policy
* 2가지 type → 7가지 type
* 후처리 없음 → System Tracker
* KV cache 관리 없음 → B-token buffer + eviction
