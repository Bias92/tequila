ATLAS Handoff 6/12: 오늘 한 것 (2026.04.09)

1. 핸드오프 10개 전달 → 새 세션에 프로젝트 컨텍스트 완전 이전
2. skeleton 코드 6파일 직접 확인 (원본 대조 완료)
3. Google Drive 원본 D2N001.json(28 ann, 구버전)으로 harness 실행 성공 (로컬에서 직접)
4. 카톡 채팅 분석 → 데이터팀 진행 상황 파악
5. annotation 포맷 변경 발견:
   - detail_list 구조 (1 utterance → 여러 type/summary)
   - question type 추가 (7 labels)
   - 최신 D2N001: 27 annotations (구버전 28과 다름)
6. skeleton 코드 6파일 수정:
   - harness.py: LLMOutput에 detail_list, MockLLM 두 포맷 지원, run loop 수정
   - config.py: feature_dim 395→396
   - policy.py: question type 추가, onehot 6→7
   - metrics.py: ALL_CLASSES에 question 추가
   - state_tracker.py: update()가 detail_list 순회
7. 최신 D2N001~003(카톡 버전)으로 로컬 검증 완료
8. GitHub push 완료: 2bb12c9
9. ACE 슬라이드 내용 정리 (코워크로 PPTX 생성)

검증 결과 (최신 D2N001, detail_list 포맷):
| Strategy | B=256 |
| FIFO | 16 items, 254 tok |
| Sliding K=10 | 10 items, 212 tok |
| Random | 비결정적 |
| Oldest | FIFO와 동일 |
| B=1024 | 전부 유지 (총 토큰 < 1024) |
