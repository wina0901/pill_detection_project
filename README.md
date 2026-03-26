# 💊 Pill Detection Project  
> AI 8기 2팀 — 알약 객체 탐지 & 정보 제공 서비스

---

## 협업 일지

모두 같은 노션페이지에 작성하였습니다.
- 링크 연결하기

---

## 📌 프로젝트 소개

헬스잇의 AI 엔지니어링 팀은 유저가 본인의 모바일 애플리케이션으로 자신이 복용중인 약 사진을 찍었을 때, 
이미지 인식을 통해 해당 약에 대한 정보를 확인할 수 있는 모델을 만들어야하는 미션

- 프로젝트 기간 : 
- 평가 : kaggle mAP@[0.75:0.95] 지표
- 팀 구성
역할	담당자	핵심 업무
Project Manager        김기현	프로젝트 총괄 관리, 일정 조율
Data Engineer          한의정	EDA, 데이터 전처리, 증강 기법
Experimentation Lead   김범수   RetinaNet model을 통한 다양한 실험
Experimentation Lead   박찬영   Faster RCNN을 통한 다양한 실험
Experimentation Lead   유소연   YOLO model을 통한 다양한 실험

---

## 🎯 주요 기능

- 📷 이미지 업로드 및 실시간 추론
- 🟩 Bounding Box 시각화
- 📊 Confidence Score 표시
- 💊 약 이름 + 특징 정보 제공
- ✂️ 탐지된 알약 crop 이미지 제공
- 📱 모바일 접속 (QR 코드 지원)

---

## ⚙️ 실행 방법

### 📦 1. 프로젝트 다운로드

```bash
git clone https://github.com/wina0901/pill_detection_project.git
cd PILL_DETECTION_PROJECT
```

### 📁 2. 필수 파일 추가

#### 🔹 data 폴더

```
data/
├─ merged_annotations_train_final.json
└─ meta.csv
```

#### 🔹 모델 파일

```
models/yolo/
├─ yolov8s_v2_v3_ft_uf_lr_0p0003_best.pt
└─ yolo11m_v2_v3_ft_uf_lr_0p0005_best.pt
```

---

## 💻 Windows 실행

```
start_for_windows.bat
```

---

## 🍎 macOS 실행

### 최초 1회
```bash
chmod +x start_for_mac.command
```

### 실행
```bash
./start_for_mac.command
```

---

## 🌐 접속

```
http://127.0.0.1:8000
```

---

## 📱 모바일 접속

- 같은 Wi-Fi 연결
- 콘솔 QR 코드 스캔

---

## 📊 평가 지표

- mAP@0.75:0.95
- 최종 성능 = 0.99093


---

## 🧩 기술 스택

- Python
- FastAPI
- PyTorch
- Ultralytics YOLO
- HTML / CSS / JS

---

## 📁 프로젝트 구조

```
PILL_DETECTION_PROJECT/
├── data/                   # 데이터셋 
│
├── models/                 # 모델별 가중치
│ ├── fasterrcnn/
│ ├── retinanet/
│ └── yolo/
│
├── notebooks/              
│
├── results/                # 모델별 제출파일 
│ ├── fasterrcnn/
│ ├── retinanet/
│ └── yolo/
│
├── src/
│ ├── evaluation/           # 평가 지표 코드
│ │
│ ├── models/               # 모델별 실험 코드
│ │ ├── fasterrcnn/
│ │ ├── retinanet/
│ │ └── yolo/
│ │
│ ├── preprocessing/         # 데이터 전처리 코드
│ └── utils/
│
├── run_preprocessing.py
├── requirements.txt
├── README.md
└── .gitignore
```

---
 
