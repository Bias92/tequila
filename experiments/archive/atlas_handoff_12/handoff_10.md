ATLAS Handoff 10/12: Evaluation 프레임워크 + prompts.py 현황

Model-level metrics:
| Metric | 구현 상태 |
| ROUGE-1/2/L | ✅ 구현됨 |
| BERTScore (F1) | ✅ 구현됨 |
| Detection P/R/F1 | ✅ 구현됨, 8-class (7 types + irrelevant) |
| Agenda completeness | ✅ 구현됨 |
| Temporal F1 | ❌ 미구현 |

System-level metrics:
| Linking accuracy | ✅ 구현됨 |
| First-mention rate | ✅ 구현됨 |
| Resolution tracking | ✅ 구현됨 |

Hardware metrics:
| Per-turn latency | ✅ time.perf_counter() |
| Peak GPU/memory | ❌ tegrastats 필요 |
| Tokens/sec | ❌ llama.cpp 필요 |
| KV-cache memory | ✅ 공식 구현됨 |

prompts.py 현황 (카톡에서 받은 최신):
* 7 labels: agenda_item, question, medication, follow_up, question_unanswered, detail, social_history
* BATCH_SYSTEM: Clinical Note 기반 annotating, detail_list 출력
* BATCH_EVAL_SYSTEM: 3-field + _eval_only 출력
* REALTIME_SYSTEM: pipe-delimited context + utterance → detail_list
* GENERATE_FROM_CASE/NOTE: synthetic conversation 생성
* VALIDATE_SYSTEM: annotation 품질 검증
* ASR_NOISE_SYSTEM: ASR noise injection

주의: prompts.py에서 "question" type은 spec v4의 원래 6 types에 없던 것.
데이터팀이 추가한 것이며 교수님 승인 여부 미확인.
