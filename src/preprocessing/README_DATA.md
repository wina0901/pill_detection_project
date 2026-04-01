# 💊 HealthEat: Advanced Data Engineering Pipeline
Object Detection(Faster R-CNN / RetinaNet / YOLO) 모델 성능 극대화를 위한 데이터 전처리 및 증강 파이프라인입니다.

## Key Achievements

1. **Data Leakage 원천 차단**: Stratified Split을 통해 클래스 분포를 유지하며 검증 데이터의 독립성 보장
2. **소수 클래스 방어**: 지능형 Copy-Paste 증강 엔진 및 AI허브 데이터 타겟팅 병합으로 희귀 알약 객체 강제 확보 (4,095개 -> 11,350개)
3. **800x800 규격화 및 무결성**: Letterbox 정규화 및 이미지 경계를 벗어나는 BBox(Out-of-bounds) 정밀 클리핑 완료 (이슈 0건)
4. **시력 교정(Feature Enhancement)**: L-channel CLAHE 전처리를 통한 알약 각인/음각 대비 극대화
5. **YOLO-Native 데이터 파이프라인**: 2-stage(COCO)에서 1-stage(YOLO)로의 신속한 아키텍처 전환을 위해 Annotation 포맷 자동 변환(JSON to TXT) 및 정규화 엔진 구축
6. **실시간 학습 최적화**: Mosaic 및 Mixup 증강이 통합된 고성능 Dataloader를 설계하여 GPU Utilization 극대화 및 학습 대기 시간 최소화

---

> **💡 데이터 전처리 결과물 안내**
> - **구글 드라이브** : 'https://drive.google.com/drive/u/1/folders/1x5r-s-LNpesZmdi-Qsh1zpSCzGioWTA1'

---

> **💡 PyTorch 설치 안내**
> - **Colab**: PyTorch가 이미 설치되어 있습니다. 별도 설치 불필요.
> - **Windows (CUDA 12.x)**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
> - **Mac (M1/M2/M3)**: `pip install torch torchvision torchaudio`

---

## Data preprocessing pipeline Repository Structure

```
pill_detection_project/
├── notebooks/              # 데이터 파이프라인 구축 히스토리 (로직 확인 및 재현용)
│   ├── 01_data_eda.ipynb
│   ├── 02_split_and_copy_paste_augmentation.ipynb
│   ├── 03_letterbox_normalization.ipynb
│   ├── 04_clahe_preprocessing.ipynb
│   └── 05_pytorch_dataset_dataloader.ipynb
│
├── src/
│   ├── utils/
│   │   └── eda_tools.py            # EDA 시각화 유틸리티
│   └── preprocessing/              # 모델 학습 시 import하여 사용
│       ├── README_DATA.md             
│       ├── aihub_merge.py          # AI허브 데이터 타겟팅 병합 스크립트      
│       ├── __init__.py
│       ├── augmentation.py         # Copy-Paste 증강 엔진
│       ├── transforms.py           # Letterbox 변환, CLAHE 전처리
│       ├── dataset.py              # OralDrugDataset, DataLoader
│       ├── format_converter.py     # COCO -> YOLO 변환 엔진
│       └── viz_utils.py            # 전처리 결과 시각화
│
├── run_preprocessing.py   # 전처리 파이프라인 전체 실행 스크립트 (최초 1회)
├── requirements.txt
└── README.md
```
---
> **notebooks/ vs src/ vs run_preprocessing.py 역할 구분**
> - `notebooks/`: 데이터를 **만드는** 파이프라인 (로직 확인 및 재현용)
> - `src/`: 만들어진 데이터를 **불러와 학습에 사용**하는 모듈 (팀원들은 이것만 import)
> - `run_preprocessing.py`: 전처리 파이프라인을 **한 번에 실행**하는 스크립트
> ✅ `get_loaders()`는 전처리 산출물이 없으면 **자동으로 `run_preprocessing.py`를 실행**합니다.
> 단, 구글 드라이브에 원본 데이터(`merged_annotations_train_final.json`)가 반드시 있어야 합니다.

---

## ⚠️ Note for Modeling Team (Colab Users)

* 각 노트북 **Cell 0을 실행**하면 구글 드라이브 마운트 및 레포 클론이 자동으로 처리됩니다.
* 데이터는 구글 드라이브 `/MyDrive/data/초급_프로젝트/dataset/` 경로에 준비되어 있습니다.
* Mac 로컬 환경에서는 `../data/`를 자동으로 바라봅니다.

> **구조 설계 원칙**
> - notebooks/: 연구 및 로직 검증을 위한 샌드박스
> - src/: 학습 파이프라인에 직접 포함되는 운영 모듈
> - get_loaders()는 전처리 산출물이 없을 경우 run_preprocessing.py를 자동 호출하여 환경 구축의 편의성을 제공합니다.


## Indexing Consistency Warning
모델 아키텍처별 인덱스 체계가 상이하므로 아래 매핑 가이드를 반드시 준수해야 합니다.
- **Faster R-CNN / RetinaNet**: 1-based index 적용 (0 = background)
- **YOLO**: 0-based index 적용 (format_converter.py를 통해 자동 변환)
- **Submission 생성 시**: 모든 예측값은 다시 category_id (4543, 6192 등) 포맷으로 복원해야 하며, YOLO 결과값은 반드시 +1 오프셋 처리가 필요합니다.

---

```
python

from src.preprocessing.dataset import get_loaders
train_loader, val_loader, orig2model, num_classes, val_json = get_loaders(base_dir=BASE_DIR)
```

### YOLO
- 레이블은 0-based (Faster R-CNN `orig2model`과 1 차이)
- NB05 실행 시 `yolo_labels/` 및 `data.yaml` 자동 생성

```
python
from src.preprocessing.format_converter import run_yolo_conversion
run_yolo_conversion(base_dir=BASE_DIR)
```

### 평가 지표 산출
- YOLO(0-based)와 Faster R-CNN(1-based) 결과 비교 시 인덱스 통일 필요 (YOLO 예측값 +1)

---

## Minority Class Strategy: 14 Selected Species
데이터 무결성 검수(Data Integrity Check)를 통과한 고품질 AI허브 단일 객체 데이터를 병합하여 소수 클래스의 학습 가용성을 확보했습니다. (총 1,938개 Annotation 추가)

| 약품명 (클래스) | 추가 원본 수량 |
| :--- | :---: |
| **트윈스타정** 40/5mg | 150 |
| **써스펜8시간**이알서방정 | 150 |
| **노바스크정** 5mg | 150 |
| **리렉스펜정** 300mg/PTP | 150 |
| **삐콤씨에프정** 618.6mg/병 | 150 |
| **오마코**연질캡슐 | 150 |
| **마도파정** 200 | 150 |
| **맥시부펜이알정** 300mg | 150 |
| **쿠에타핀정** 25mg | 150 |
| **렉사프로정** 15mg | 150 |
| **아질렉트정**(라사길린메실산염) | 150 |
| **메가파워정** 90mg/병 | 108 |
| **비타비백정** 100mg/병 | 108 |
| **에어탈정**(아세클로페낙) | 72 |
| **합계 (Total Added)** | **1,938개** |

---

## Indexing Consistency Warning
모델 아키텍처별 인덱스 체계가 상이하므로 아래 매핑 가이드를 반드시 준수해야 합니다.

- **Faster R-CNN / RetinaNet**: 1-based index 적용 (0 = background)
- **YOLO**: 0-based index 적용 (format_converter.py를 통해 자동 변환)
- **Submission 생성 시**: 모든 예측값은 다시 category_id (4543, 6192 등) 포맷으로 복원해야 하며, YOLO 결과값은 반드시 +1 오프셋 처리가 필요합니다.

