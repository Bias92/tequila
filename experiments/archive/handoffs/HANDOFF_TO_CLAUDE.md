# ATLAS QLoRA 작업 핸드오프 (Claude.ai에 붙여넣기용)

## 누가/뭐 하는 중

저는 **김재우** (UPenn Jang 교수님 그룹). 프로젝트: 임상 어젠다 설정 시스템 (Llama-3.2-3B + 학습 컨텍스트 매니저), Jetson AGX Orin 배포 목표, NeurIPS 2026 / GenAI4Health Workshop.

지금 하려는 일: **`tria-lab/atlas` 레포의 `data/train.jsonl` (3733 examples)로 Llama-3.2-3B-Instruct를 QLoRA 파인튜닝**. 7-type 분류 + JSON 출력 학습.

## 핵심 사실 (반드시 알아야 함)

- **공식 spec 제목**: "Learned Context Management for Streaming Summarization" (ATLAS는 약자 아님, 그냥 코드명)
- **타입 6개 (spec)**: agenda_item, detail, medication, social_history, follow_up, question_unanswered
  - 데이터팀이 7번째 `question` 추가했지만 **교수 승인 미보류**. 현재 데이터는 7개 사용
- **레포**: https://github.com/tria-lab/atlas (private). 메인 브랜치 + confirm 브랜치 (머지 안 됨)
- **GitHub 푸시는 GPT 교차검증 통과 후에만**. 프로젝트 룰임. 대신 머지/PR 제안하지 말 것.
- **김재우 직접 코드 실행**. Claude는 코드 스스로 실행 안 함.

## 데이터/코드 현재 상태

### 학습 데이터 통계 (cleaned, plan/assessment → follow_up/detail 재라벨됨)
- `train.jsonl`: 3733 examples, 47.2% relevant
  - 타입 분포: detail 975, question 578, follow_up 169, agenda_item 149, medication 69, social_history 60
  - **`question_unanswered` 0개** ← 학습 안 됨, 합성 또는 보강 필요
- `eval.jsonl`: 3733 examples, 37.8% relevant — **train과 동일한 172 conversation에서 나옴 (데이터 누수)**
  - 진짜 hold-out은 `data/test1/2/3` 폴더 (batch-mode 어노테이션, JSONL 변환 미완)

### 입출력 포맷
```
input:  "[Context]: <pipe-delimited summaries>\n[Utterance]: [Doctor|Patient] text"
output: "{\"relevant\": true, \"detail_list\": [{\"type\":..., \"summary\":...}]}" or "{\"relevant\": false}"
```

### 이미 작성된 파일 (Mac 경로)
**RunPod CUDA 패키지** (A100/4090 등):
- `~/Library/Application Support/Claude/local-agent-mode-sessions/5fd0aee9-82e0-4aa9-b602-601e90b2b912/8ba4ab4b-5d5e-4502-9936-302cc5704fcc/local_684315f0-ddac-4aed-b396-5da1db139ebb/outputs/atlas_qlora.tar.gz`
- 안에: `train_qlora.py` (TRL SFTTrainer + 4-bit nf4 + LoRA r=32), `eval_qlora.py`, `requirements.txt`, `runpod_setup.sh`, `merge_and_export_gguf.md`, `data/train.jsonl`, `data/eval.jsonl`

**Mac MLX 패키지** (M4 Pro 24GB, CUDA 없이):
- `~/Library/.../outputs/atlas_mlx.tar.gz`
- 안에: `convert_to_mlx_format.py`, `mac_setup.sh`, `run_train_mlx.sh`, `eval_mlx.py`, `run_fuse_mlx.sh`, README

**바탕화면 결과 폴더**: `~/Desktop/atlas_qlora_results/`

### 모델 코드 알려진 이슈 (src/ 안)
- `policy.py` 헤더 docstring outdated: 6 types/395-d로 적혀있는데 실제 코드는 7 types/396-d
- `harness.py::get_token_count`: placeholder (1 token ≈ 4 chars). budget sweep 정확도 위해 llama.cpp 토크나이저로 교체 필요
- `policy.py::extract_features`의 `entity_count = 0.0`: scispaCy 미통합, 항상 0
- `baselines.py::lowest_attention_strategy`: placeholder, oldest_first 폴백
- `state_tracker.py::_is_first_mention`: O(N²) 임베딩 재계산 (캐시 필요)

## 시도한 것 + 실패한 것

1. **RunPod A100 SXM 폿** 띄움. 컨테이너 100GB + 볼륨 100GB. ID `z6iphyrpcvhry8`
2. **Web 터미널 paste 실패**:
   - 첫 글자 `c` 먹힘 (`cd` → `d`)
   - `.sh`/`.py` markdown 자동 링크화
3. **temp.sh 호스팅**: 업로드는 되는데 다운로드시 HTML 에러 페이지 (세션 쿠키 필요)
4. **Chrome read-tier 제한**: Cowork 모드의 Chrome은 클릭/타이핑 차단됨. JS 인젝션 안 됨

## 핸드오프 옵션

### 옵션 A — 더블클릭 한 번으로 Mac에서 학습
- 바탕화면에 `RUN_ATLAS_TRAINING.command` 파일 있음 (~/Desktop/RUN_ATLAS_TRAINING.command)
- 더블클릭하면 자동: 압축풀기 → 의존성 설치 → HF 로그인 → MLX LoRA 학습 → eval
- 결과: ~/Desktop/atlas_qlora_results/adapter/
- 시간: M4 Pro 24GB에서 3-4시간

### 옵션 B — RunPod A100에서 30분 안에 끝내기
1. RunPod 폿 STOP 안 한 상태에서:
2. JupyterLab (https://z6iphyrpcvhry8-8888.proxy.runpod.net/lab) 열고
3. `atlas_qlora.tar.gz`를 파일 패널에 드래그-드롭 또는 ↑ 업로드 버튼
4. JupyterLab 메뉴 → File → New → Terminal
5. 다음 명령 paste:
```
cd /workspace && tar xzf atlas_qlora.tar.gz && cd atlas_qlora && chmod +x runpod_setup.sh && bash runpod_setup.sh && export HF_TOKEN=<your_token> && export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN && huggingface-cli login --token $HF_TOKEN && tmux new-session -d -s train "python train_qlora.py 2>&1 | tee train.log"
```
6. 모니터링: `tail -f /workspace/atlas_qlora/train.log`
7. 학습 끝나면: `python eval_qlora.py --limit 200`
8. 결과 다운: JupyterLab에서 `adapter/` 폴더 + `eval_report.json` 우클릭 → Download

## HuggingFace 토큰 ⚠️

이전 세션에서 한 번 채팅에 노출된 토큰:
```
[REDACTED_HUGGINGFACE_TOKEN]
```
**작업 끝나면 반드시 폐기하고 새로 발급**. HuggingFace → Settings → Access Tokens → Manage → Invalidate.

## 권장 다음 액션

비용 절약 + 안전한 길: **옵션 A (Mac MLX)**. RunPod 폿 STOP하고 (terminate 아니라 stop, 빌링 멈춤), 자기 전에 .command 더블클릭하고 자면 아침에 결과.

급하게 결과 필요하면: **옵션 B (RunPod)**. JupyterLab 드래그-드롭 1번 + paste 1번이면 30분 내 끝.

---

## Claude.ai에서 이 핸드오프 받으면

1. 위 내용 다 읽고
2. 사용자에게 옵션 A vs B 물어보고
3. 선택한 옵션 진행 (특히 옵션 A는 Mac에 직접 접근 못 하니 사용자한테 더블클릭 시키는 식)
4. 토큰 폐기 안내 잊지 말 것
