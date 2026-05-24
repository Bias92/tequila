ATLAS Handoff 11/12: 진행률 + 남은 작업

현재 단계: skeleton 완성 + 실제 데이터 호환 확인 (~15%)

| 항목 | 상태 | 비중 |
| skeleton 구조 (6파일) | ✅ 완료 | ~15% |
| detail_list + 7 labels 대응 | ✅ 완료 | 위에 포함 |
| LlamaCppLLM 연결 | ❌ 안 함 | ~15% |
| QLoRA fine-tuning (3-phase) | ❌ 안 함 | ~20% |
| MLP 학습 | ❌ 안 함 | ~10% |
| 실험 (5 baselines × 5 budgets + ablation) | ❌ 안 함 | ~15% |
| Hardware profiling (tegrastats) | ❌ 안 함 | ~5% |
| Compression module (optional) | ❌ 안 함 | ~5% |
| 결과 정리 + 논문 figures/tables | ❌ 안 함 | ~15% |

코드에서 각 파일이 하는 일:
* config.py — 설정값 모음
* harness.py — C1→C2→C3 전체 루프 엔진
* baselines.py — 비교 대상 5개 전략 (C3 대안)
* state_tracker.py — Component 2
* policy.py — Component 3 (핵심 contribution)
* metrics.py — 결과 측정 도구

harness.py 실행 시 경유하는 파일:
python3 harness.py --strategy fifo → config.py + harness.py + baselines.py (3개만)
tracker, policy, metrics는 아직 실제 실행에서 사용 안 됨

MockLLM:
* 실제 LLM이 아니라 annotation JSON의 정답을 그대로 리턴
* "LLM이 완벽하게 맞췄다고 가정"하고 나머지 파이프라인 테스트하는 용도
* 나중에 LlamaCppLLM으로 교체 예정
