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

## 📌 프로젝트 소개
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

- 내용 추가하기 (전처리 과정 , 실험 내용, 서빙 내용)

---

## ⚙️ 실행 방법

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
 
