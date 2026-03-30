#  Pill Detection Project  
AI 8기 2팀 — 알약 객체 탐지 & 정보 제공 서비스

<br>

## 협업 일지

모두 같은 노션페이지에 작성하였습니다.
- [링크](https://www.notion.so/32506b12e06280dea5fdf832300228b5?v=32506b12e06280998ab9000c52ef2023)
<br>

## 최종 보고서

- 파일 첨부하기
<br>

## 프로젝트 소개
헬스잇의 AI 엔지니어링 팀은 유저가 본인의 모바일 애플리케이션으로 자신이 복용중인 약 사진을 찍었을 때, 

이미지 인식을 통해 해당 약에 대한 정보를 확인할 수 있는 모델을 만들어야하는 미션

- 프로젝트 기간 | 2026.03.16. 2026.04.02. 
- 평가 지표 | Kaggle mAP@0.75:0.95 
<br>

## 팀 구성 및 역할

| 역할 | 담당자 | 핵심 업무 |
|------|--------|-----------|
| Project Manager | 김기현 | 프로젝트 총괄 관리, 일정 조율 |
| Data Engineer | 한의정 | EDA, 데이터 전처리, 증강 기법 |
| Experimentation Lead | 김범수 | RetinaNet 모델 기반 다양한 실험 |
| Experimentation Lead | 박찬영 | Faster R-CNN 모델 기반 다양한 실험 |
| Experimentation Lead | 유소연 | YOLO 모델 기반 다양한 실험 |

<br>

## 주요 기능

- Object Detection(Faster R-CNN / RetinaNet / YOLO) 모델 성능 극대화를 위한 데이터 전처리 및 증강 파이프라인

1. Data Leakage 원천 차단: Stratified Split을 통해 클래스 분포를 유지하며 검증 데이터의 독립성 보장
2. 소수 클래스 방어: 지능형 Copy-Paste 증강 엔진으로 희귀 알약 객체 강제 확보 (4,095개 → 11,350개)
3. 800x800 규격화 및 무결성: Letterbox 정규화 및 이미지 경계를 벗어나는 BBox(Out-of-bounds) 정밀 클리핑 완료 (이슈 0건)
4. 시력 교정(Feature Enhancement): L-channel CLAHE 전처리를 통한 알약 각인/음각 대비 극대화
5. YOLO-Native 데이터 파이프라인: 2-stage(COCO)에서 1-stage(YOLO)로의 신속한 아키텍처 전환을 위해 Annotation 포맷 자동 변환(JSON to TXT) 및 정규화 엔진 구축
6. 실시간 학습 최적화: Mosaic 및 Mixup 증강이 통합된 고성능 Dataloader를 설계하여 GPU Utilization 극대화 및 학습 대기 시간 최소화
Gemini의 응답
👉 **[데이터 전처리 및 소수 클래스 확충 상세 가이드](./src/preprocessing/README_DATA.md)**

- 내용 추가하기 (실험 내용, 서빙 내용)

---

## 실행 방법

### 1. 프로젝트 다운로드

```bash
git clone https://github.com/wina0901/pill_detection_project.git
cd PILL_DETECTION_PROJECT
```

### 2. 필수 파일 추가

```
data/
├─ merged_annotations_train_final.json
└─ meta.csv

models/yolo/
├─ yolov8s_v2_v3_ft_uf_lr_0p0003_best.pt
└─ yolo11m_v2_v3_ft_uf_lr_0p0005_best.pt
```
(필수 파일은 아래 구글드라이브에서 전달.zip을 받으시면 됩니다!)

[구글 드라이브 링크](https://drive.google.com/drive/folders/1ED5sme7FjaaUazkk6336glpzjrV-bVGY)

---

## Windows 실행

```
start_for_windows.bat
```

## macOS 실행

### 최초 1회
```bash
chmod +x start_for_mac.command
```

### 실행
```bash
./start_for_mac.command
```
---


## 프로젝트 구조

```text
PILL_DETECTION_PROJECT/
├── data/                         # 데이터셋 및 메타 정보
│
├── models/                       # 모델별 가중치 파일
│   ├── fasterrcnn/
│   ├── retinanet/
│   └── yolo/
│
├── notebooks/                    # 노트북 실험 파일
│
├── results/                      # 추론 결과 및 모델별 결과 저장
│
├── serve/                        # 서빙 환경 설치 및 실행 스크립트
│   ├── run_server.py
│   └── setup_serve.py
│
├── src/
│   ├── evaluation/               # 평가 지표 코드
│   ├── inference/                # 최종 추론 코드
│   ├── models/                   # 모델별 실험 코드
│   ├── preprocessing/            # 데이터 전처리 코드
│   └── utils/                    # 공통 유틸리티
│
├── ui/                           # 웹 UI 파일
│   ├── static/
│   └── templates/
│
├── run_preprocessing.py          # 전처리 실행 스크립트
├── server.py                     # FastAPI 서버 진입점
├── start_for_mac.command         # macOS 실행 파일
├── start_for_windows.bat         # Windows 실행 파일
├── requirements.txt              # 학습/실험용 의존성
├── requirements-serve.txt        # 서빙용 의존성
├── README.md
└── .gitignore

---
 
> **notebooks/ vs src/ vs run_preprocessing.py 역할 구분**
> - `notebooks/`: 데이터를 **만드는** 파이프라인 (로직 확인 및 재현용)
> - `src/`: 만들어진 데이터를 **불러와 학습에 사용**하는 모듈 (팀원들은 이것만 import)
> - `run_preprocessing.py`: 전처리 파이프라인을 **한 번에 실행**하는 스크립트
>
> ✅ `get_loaders()`는 전처리 산출물이 없으면 **자동으로 `run_preprocessing.py`를 실행**합니다.
> 단, 구글 드라이브에 원본 데이터(`merged_annotations_train_final.json`)가 반드시 있어야 합니다.