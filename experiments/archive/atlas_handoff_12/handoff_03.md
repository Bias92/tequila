ATLAS Handoff 3/12: 데이터 파이프라인 현황

전체 흐름:
annotated JSON (D2N001~003.json)
  → annotation_gemini.py --mode format (로컬, API 불필요)
  → llm_format_a.jsonl (Agenda LLM 학습 데이터)     ← 아직 안 돌림
  → context_manager_labels.py
  → eviction_policy.jsonl (MLP 학습 데이터)          ← 아직 안 돌림

annotation_gemini.py --mode batch:
* Gemini Flash API를 호출해서 대화 전체를 annotation
* 이광호가 D2N001~003 완료
* 50개 목표 중 3개만 완료된 상태

annotation_gemini.py --mode format:
* API 호출 없음. 로컬 변환.
* annotated JSON을 순서대로 재생하면서 pipe-delimited context를 누적 생성
* 출력: llm_format_a.jsonl
* 아직 실행 안 됨. 데이터팀이 해야 할 일.

현재 annotation 데이터 상태 (2026.04.09 카톡에서 받은 최신):
| 파일 | Utterances | Annotations | detail_list | Types |
| D2N001 | 56 | 27 | ✅ | agenda_item, detail, follow_up, medication, question |
| D2N002 | 85 | 46 | ✅ | agenda_item, detail, follow_up, medication, question |
| D2N003 | 53 | 36 | ✅ | agenda_item, detail, follow_up, medication, question |

주의: Google Drive에는 아직 최신 버전 안 올라감 (4/2 버전이 마지막). 카톡으로 공유된 게 최신.

데이터팀 역할 분담:
* --mode format 실행 → llm_format_a.jsonl 생성 = 데이터팀
* context_manager_labels.py → eviction_policy.jsonl 생성 = 데이터팀
* 학습 데이터 받아서 QLoRA + MLP 학습 = 본인

Google Drive 구조:
ATLAS/aci-bench-2023/
├── Annotation/
├── annotated_gemini/
│   ├── aci/     → D2N001~003.json (3-field) — 4/2 버전
│   └── eval/    → D2N001~003.json (3-field + _eval_only) — 4/2 버전
├── aci-bench-corpus/
│   └── challenge_data/train.csv
├── annotation_gemini.py (이광호, 22KB)
└── annotation_Example (김형준, 23KB)

ATLAS/Data annotation plan v3/
├── prompts.py
├── data_plan_v3.md
├── context_manager_training.md
└── annotation_pipeline.py (교수님 원본, _call() = NotImplementedError)
