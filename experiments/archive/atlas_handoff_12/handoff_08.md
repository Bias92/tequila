ATLAS Handoff 8/12: 알려진 실수 + 교훈

이전 세션(~handoff 10)에서 보고된 실수:
1. 타겟 학회를 워크숍으로 단정 → 메인 트랙 1순위, 워크숍 백업
2. CF1 화자 매핑 오류 → 화자1 = 김재우(본인)
3. 교수님을 NeurIPS 메인 리뷰어로 과장 → GenAI4Health 워크숍 리뷰어
4. placeholder를 "해결 완료"라고 표현
5. contribution 후보를 입증된 사실처럼 기술

이번 세션(2026.04.09)에서 발견된 문제:
1. D2N001.json 버전 혼동 — 로컬(16 ann), 이전 업로드(28 ann), 카톡 최신(27 ann) 3개가 서로 달랐음
2. "28 annotations"이라는 숫자가 이전 세션에서 만든 테스트 데이터의 숫자와 혼동됨
3. Google Drive에 최신 annotation이 안 올라가 있음 (4/2 버전이 마지막)
4. "GPT 4라운드 교차검증" — spec v4에 없는 표현. 이전 세션 Claude가 만든 용어. 발표에서 빼거나 "코드 리뷰 완료"로 대체.

교훈:
* 파일 버전 관리가 중요 — Google Drive, 카톡, 로컬이 다 다를 수 있음
* annotation 개수로 버전 구분 가능: 16(구), 28(중간), 27(최신 detail_list)
* Claude가 만든 용어를 spec 근거 없이 발표에 넣으면 안 됨
