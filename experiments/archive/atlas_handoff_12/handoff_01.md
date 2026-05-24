ATLAS Handoff 1/12: 프로젝트 개요

ATLAS = Adaptive Token-Level Agenda Setting
병원에서 환자-의사 대화가 실시간으로 진행될 때, AI가 대화를 듣고 핵심 안건을 추적하는 시스템. 특히 환자가 질문했는데 의사가 넘어간 것(question_unanswered)을 잡아내는 게 핵심 가치.
클라우드 서버 없이 NVIDIA Jetson AGX Orin 엣지 디바이스에서 돌리겠다는 게 논문의 contribution 후보.

타겟 학회:
NeurIPS 2026. 메인 트랙이 1순위, 워크숍(GenAI4Health)은 백업. 아직 확정 아님.
* Abstract: ~5/4
* Full paper: ~5/6 (AOE)

선행 연구:
Jang et al. AAAI 2025 GenAI4Health Workshop — 같은 연구팀의 prior paper. GPT 3.5 Turbo / Llama 3.1 8B로 실시간 agenda setting feasibility 탐색. context aggregation (growing/sliding window) 실험.

팀 구성:
- 장국진 (교수님): 논문 작성, 방향 설정, GenAI4Health 워크숍 리뷰어
- 김재우 (본인): 프로젝트 리드, 구현 + 알고리즘 + Jetson 통제. 유일한 시스템 구현자.
- 김형준: annotation 기준 확립, annotation.py/prompt.py 수정, D2N001~003.json 생성
- 이광호: annotation_gemini.py batch mode 완성, 7 labels TYPES 정의
- 손기배: Notion 버전 관리, D2N002_v1 생성, Prompt 결정.pdf

핵심 연구 질문:
"버퍼가 꽉 찼을 때 뭘 버릴지를 학습된 MLP로 결정하면, 단순한 방법(FIFO, 랜덤 등)보다 좋은가?"
이건 아직 가설이지 입증된 사실이 아님. 실험으로 증명해야 함.
