ATLAS Handoff 12/12: 정확한 상태 표현 + 금지 표현 + 파일 위치

쓰면 안 되는 표현:
❌ "GPT 4라운드 교차검증 통과" → ✅ "코드 리뷰 완료"
❌ "baseline separation 해결" → ✅ "FIFO vs sliding 분리 완료, FIFO vs oldest는 제한적"
❌ "token budget fix 완료" → ✅ "token budget은 아직 placeholder (len//4)"
❌ "lowest-attention 구현 완료" → ✅ "deferred placeholder"
❌ "learned policy superiority 입증" → ✅ "아직 가설. 실험으로 증명해야 함"
❌ "50배 효율성" → ✅ "아직 측정 안 됨"
❌ "NeurIPS GenAI4Health Workshop" → ✅ "NeurIPS 2026 (메인 or 워크숍, 미확정)"
❌ "교수님이 NeurIPS 리뷰어" → ✅ "GenAI4Health 워크숍 리뷰어"
❌ "395-d" → ✅ "396-d" (question type 추가로 변경됨)

파일 위치:
| 파일 | 위치 |
| project_spec_v4_students.md | 프로젝트 파일 (Claude 프로젝트 내) |
| Jang et al. PDF | 프로젝트 파일 (Claude 프로젝트 내) |
| src/*.py (6개) | GitHub: https://github.com/Bias92/ATLAS |
| D2N001~003.json (최신) | 카톡에서 받은 버전 (Google Drive 미업로드) |
| annotation_gemini.py | Google Drive + 카톡 |
| prompts.py | Google Drive + 카톡 |
| ACE 슬라이드 | 프레젠테이션1.pptx (코워크로 생성) |

새 세션 시작 시 참고:
1. project_spec_v4_students.md를 먼저 읽기
2. 이 핸드오프 12개를 순서대로 참고
3. GitHub repo: https://github.com/Bias92/ATLAS (Private)
4. 코드 수정 시 원본과 대조 후 수정
5. placeholder를 "해결 완료"라고 표현하지 않기
6. 사용자(김재우)는 한국어로 소통
7. D2N001 버전 주의: 16(구) / 28(중간) / 27(최신 detail_list)

ACE 슬라이드 수정 필요:
1. Slide 2: "뼀 버릴지를" → "뭘 버릴지를" (오타)
2. Slide 2: "395d" → "396d"
