# 💊 HealthEat: Advanced Data Engineering Pipeline
**Lead Architect: 한의정**

Object Detection(Faster R-CNN / RetinaNet / YOLO) 모델 성능 극대화를 위한 데이터 전처리 및 증강 파이프라인입니다.

## 🚀 Key Achievements
1. **Data Leakage 원천 차단**: Stratified Split을 통해 검증 데이터 독립성 100% 보장.
2. **소수 클래스 방어**: 지능형 Copy-Paste 증강 엔진으로 희귀 알약 객체 강제 확보 (4,095개 → 6,199개).
3. **800x800 규격화 및 무결성**: Letterbox 정규화 및 이미지 경계를 벗어나는 BBox(Out-of-bounds) 정밀 클리핑 완료 (이슈 0건).
4. **시력 교정(Feature Enhancement)**: L-channel CLAHE 전처리를 통한 알약 각인/음각 대비 극대화.

---

## 📂 Repository Structure

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
│       ├── __init__.py
│       ├── transforms.py           # Letterbox 변환, CLAHE
│       ├── augmentation.py         # Copy-Paste 증강
│       ├── dataset.py              # OralDrugDataset, DataLoader
│       └── format_converter.py     # COCO → YOLO 변환
│
├── requirements.txt
└── README.md
```

> **notebooks/ vs src/ 역할 구분**
> - `notebooks/`: 데이터를 **만드는** 파이프라인 (1회 실행, 결과물은 Google Drive에 저장)
> - `src/`: 만들어진 데이터를 **불러와 학습에 사용**하는 모듈 (팀원들은 이것만 import)
> - 팀원들은 **노트북을 직접 실행할 필요 없습니다.** 전처리 완료 데이터가 Drive에 준비되어 있습니다.

---

## ⚙️ 데이터 전처리 파이프라인 순서

```
[NB02] Stratified Split (9:1) + Copy-Paste 증강
    → train_augmented_final.json / val.json
        ↓
[NB03] Letterbox 800×800 규격화 + BBox 클리핑
    → train_letterbox.json / val_letterbox.json
    → letterbox_images/train, val/
        ↓
[NB04] L-channel CLAHE (in-place)
    → letterbox_images/train, val/ (덮어쓰기)
        ↓
[NB05] DataLoader 구성 + YOLO 라벨 변환
    → yolo_labels/train, val/ + data.yaml
```

---

## ⚠️ Note for Modeling Team (Colab Users)

* 각 노트북 **Cell 0을 실행**하면 구글 드라이브 마운트 및 레포 클론이 자동으로 처리됩니다.
* 데이터는 구글 드라이브 `/MyDrive/data/초급_프로젝트/dataset/` 경로에 준비되어 있습니다.
* Mac 로컬 환경에서는 `../data/`를 자동으로 바라봅니다.

### Faster R-CNN / RetinaNet
- `num_classes = 74` (background 포함, NB05 실행 시 출력됩니다)
- `orig2model`: category_id → 모델 레이블 매핑 (1-based, 0=background)
- DataLoader에서 `T.Normalize` 미적용 (모델 내부 `GeneralizedRCNNTransform`에서 처리)
- 추론 시각화 시 역정규화 필요: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

```python
from src.preprocessing.dataset import get_loaders

train_loader, val_loader, orig2model, num_classes = get_loaders(base_dir=BASE_DIR)
```

### YOLO
- 레이블은 0-based (Faster R-CNN `orig2model`과 1 차이)
- NB05 실행 시 `yolo_labels/` 및 `data.yaml` 자동 생성

```python
from src.preprocessing.format_converter import run_yolo_conversion

run_yolo_conversion(base_dir=BASE_DIR)
```

### 평가 지표 산출
- YOLO(0-based)와 Faster R-CNN(1-based) 결과 비교 시 인덱스 통일 필요 (YOLO 예측값 +1)
