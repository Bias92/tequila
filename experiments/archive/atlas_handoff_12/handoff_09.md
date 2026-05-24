ATLAS Handoff 9/12: 데이터팀 진행 상황 (카톡 분석, 4/3~4/9)

타임라인:
4/3: 김형준 회의 전문+요약 공유, annotation.py/prompt.py 수정 시작
4/6: 김형준 0ann.json 공유, 손기배 Notion 테이블 구축, 이광호 prompt+annotation 수정 후 노션 업로드
4/7: 김형준 D2N001_answer.json(정답지) 공유, 이광호 12개 라인 수정 + 2개 타입 중복 수정
4/8: 프롬프트 비교 회의 예정 조율 (5시), 이광호 prompts.zip 공유
4/9: 3명 프롬프트 비교 후 최종 선택, 김형준 D2N001~003.json 공유, 손기배 D2N002_v1/D2N001_v10 공유, 김형준 annotation 진행상황 4단계 정리, 10개 정답지 생성 목표

핵심 결정사항:
1. Annotation 기준 = Clinical Note 포함 여부
2. 1 utterance → multiple type entries (detail_list 구조)
3. 7 labels: agenda_item, question, medication, follow_up, question_unanswered, detail, social_history
4. 교수님 프롬프트 기반으로 재수정 방향 결정 (이광호)
5. 3명(1=광호, 2=형준, 3=기배)이 각자 프롬프트 만들어서 D2N001 라인별 비교 후 선택

라인별 비교 결과 (Prompt 결정.pdf):
line 4: 1,2 / line 5: 3 / line 8: 1 / line 9: 2 ...
일부 라인: summary 수정, type 수정, 빠져야 할 항목 등 세부 피드백

발표 관련:
김형준이 "@재우 내일 발표 혹시 재우님이 하시는거 맞나요?" (4/9 오후 3:58)
