"""
dataset.py
==========
PyTorch Dataset / DataLoader 인터페이스

【이 파일의 역할】
  모델 학습에 필요한 DataLoader를 만들어주는 파일입니다.
  get_loaders() 함수 하나만 호출하면 학습 준비가 완료됩니다.

【팀원 사용법】
  from src.preprocessing.dataset import get_loaders

  BASE_DIR = '/content/drive/MyDrive/data/초급_프로젝트/dataset'
  train_loader, val_loader, orig2model, num_classes = get_loaders(base_dir=BASE_DIR)

  ✅ 전처리 산출물(train_letterbox.json 등)이 없어도 괜찮습니다.
     get_loaders() 가 자동으로 전처리를 실행한 뒤 DataLoader를 반환합니다.

【포함된 함수/클래스】
  - validate_coco      : 전처리된 JSON 파일의 이상 여부를 검사
  - build_df_from_json : COCO JSON → Pandas DataFrame 변환
  - OralDrugDataset    : Faster R-CNN / RetinaNet 학습용 Dataset 클래스
  - collate_fn         : DataLoader용 배치 묶음 함수
  - get_loaders        : train / val DataLoader 생성 (진입점)

【레이블 규칙 — 중요!】
  Faster R-CNN / RetinaNet : 1-based  (0번은 background로 예약)
  YOLO                     : 0-based  (format_converter.py 참고)
  → 두 방식은 동일한 category_id에 대해 1 차이가 납니다.
"""

import os
import sys
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T  # ✅ v2로 업그레이드 (PyTorch 2.0+)


# ---------------------------------------------------------------------------
# ImageNet 정규화 상수
# ---------------------------------------------------------------------------
# ResNet, EfficientNet 등 ImageNet으로 사전학습된 백본을 사용할 때
# 입력 이미지를 이 값으로 정규화해야 성능이 제대로 나옵니다.
# 단, Faster R-CNN / RetinaNet은 모델 내부(GeneralizedRCNNTransform)에서
# 자동으로 정규화를 수행하므로 DataLoader에서는 적용하지 않습니다.
# 추론 결과를 시각화할 때 이 값으로 역정규화(denormalize)해서 원본 이미지를 복원하세요.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 역정규화 유틸
# ---------------------------------------------------------------------------
def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    정규화된 이미지 텐서를 시각화 가능한 형태로 복원합니다.

    정규화 공식: normalized = (pixel - mean) / std
    역정규화 공식: pixel = normalized * std + mean

    Args:
        tensor : 정규화된 이미지 텐서 [C, H, W]
        mean   : 채널별 평균 (기본값: ImageNet 평균)
        std    : 채널별 표준편차 (기본값: ImageNet 표준편차)

    Returns:
        [0, 1] 범위로 복원된 이미지 텐서

    사용 예시:
        img_tensor = next(iter(train_loader))[0][0]  # 첫 번째 이미지
        img_vis = denormalize(img_tensor)             # 역정규화
        plt.imshow(img_vis.permute(1, 2, 0).numpy()) # 시각화
    """
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)  # 부동소수점 오차로 범위를 벗어날 수 있으므로 클리핑


# ---------------------------------------------------------------------------
# COCO JSON 무결성 검증
# ---------------------------------------------------------------------------
def validate_coco(json_path, target_size=800):
    """
    Letterbox 전처리가 완료된 COCO JSON 파일의 무결성을 검사합니다.

    전처리 후 아래 조건을 모두 만족해야 정상입니다:
      1) 모든 이미지의 width / height 가 target_size(800) x target_size(800)
      2) 모든 BBox의 x, y 좌표가 0 이상 (음수 좌표 없음)
      3) 모든 BBox의 width, height 가 양수 (크기가 0 이하인 박스 없음)
      4) 모든 BBox가 이미지 경계(800x800) 안에 들어옴

    Args:
        json_path   : 검증할 COCO JSON 파일 경로 (train_letterbox.json 등)
        target_size : 기대하는 이미지 크기 (기본 800)

    사용 예시:
        from src.preprocessing.dataset import validate_coco
        validate_coco('/path/to/train_letterbox.json')
    """
    if not os.path.exists(json_path):
        print(f"🚨 파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 이미지 크기 검사: Letterbox 후 모든 이미지가 800x800이어야 함
    wrong_size = [img for img in coco['images']
                  if img['width'] != target_size or img['height'] != target_size]

    # BBox 좌표 이상 여부 검사
    issues = []
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        if x < 0 or y < 0:
            issues.append(('negative_xy', ann['id']))     # 음수 좌표
        if w <= 0 or h <= 0:
            issues.append(('non_positive_wh', ann['id'])) # 크기가 0 이하
        if x + w > target_size or y + h > target_size:
            issues.append(('out_of_bounds', ann['id']))   # 이미지 경계 초과

    print(f"\n[{os.path.basename(json_path)}]")
    print(f"  • 이미지 수        : {len(coco['images'])}장")
    print(f"  • BBox 총 수       : {len(coco['annotations'])}개")
    print(f"  • 규격 이상 이미지  : {len(wrong_size)}장  ← 0이어야 정상")
    print(f"  • BBox 좌표 이슈   : {len(issues)}개     ← 0이어야 정상")


# ---------------------------------------------------------------------------
# COCO JSON → Pandas DataFrame 변환
# ---------------------------------------------------------------------------
def build_df_from_json(json_path, img_dir):
    """
    COCO 형식의 JSON 파일을 읽어 annotation(객체 탐지 정보) 단위의
    Pandas DataFrame으로 변환합니다.

    COCO JSON 구조:
      images      : 이미지 목록 [{id, file_name, width, height}, ...]
      annotations : 객체 탐지 목록 [{id, image_id, category_id, bbox}, ...]
      categories  : 클래스 목록 [{id, name}, ...]

    변환 결과 DataFrame 컬럼:
      image_path  : 이미지 파일 절대 경로
      image_id    : 이미지 식별자 (파일명에서 확장자 제거)
      category_id : 원본 클래스 ID (COCO 기준)
      bbox_x      : BBox 좌상단 x 좌표
      bbox_y      : BBox 좌상단 y 좌표
      bbox_w      : BBox 너비
      bbox_h      : BBox 높이

    ⚠️ 이미지 파일이 img_dir에 실제로 존재하는 것만 DataFrame에 포함됩니다.
       파일이 없으면 해당 annotation은 조용히 스킵됩니다.

    Args:
        json_path : COCO JSON 파일 경로 (train_letterbox.json 등)
        img_dir   : 이미지가 저장된 폴더 경로 (letterbox_images/train 등)

    Returns:
        pd.DataFrame : annotation 단위 데이터프레임
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # image_id → file_name 매핑 딕셔너리 생성 (O(1) 탐색용)
    id_to_fname = {img['id']: img['file_name'] for img in data['images']}

    records = []
    for ann in data['annotations']:
        file_name = id_to_fname.get(ann['image_id'])
        if not file_name:
            continue  # image_id가 images 목록에 없는 경우 스킵

        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue  # 실제 파일이 없는 경우 스킵

        x, y, w, h = ann['bbox']
        records.append({
            'image_path':  img_path,
            'image_id':    os.path.splitext(file_name)[0],  # 확장자 제거
            'category_id': int(ann['category_id']),
            'bbox_x': float(x),
            'bbox_y': float(y),
            'bbox_w': float(w),
            'bbox_h': float(h),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 자동 전처리 실행 (내부 함수)
# ---------------------------------------------------------------------------
def _run_preprocessing_if_needed(base_dir):
    """
    전처리 산출물이 없을 경우 run_preprocessing.py를 자동으로 실행합니다.
    get_loaders() 내부에서 호출되며, 팀원이 직접 호출할 필요는 없습니다.

    【산출물 존재 여부 체크 기준】
      - base_dir/train_letterbox.json
      - base_dir/val_letterbox.json
      - base_dir/letterbox_images/train/

    위 세 가지가 모두 존재하면 전처리를 건너뜁니다.
    하나라도 없으면 run_preprocessing.py를 실행하여 생성합니다.

    【run_preprocessing.py 위치 탐색 방법】
      이 파일(dataset.py)의 위치:  src/preprocessing/dataset.py
      프로젝트 루트 계산:          src/preprocessing/ → src/ → 프로젝트 루트/
      스크립트 경로:               프로젝트 루트/run_preprocessing.py

    Args:
        base_dir : 전처리 산출물이 저장될 데이터 루트 경로
    """
    train_json = os.path.join(base_dir, 'train_letterbox.json')
    val_json   = os.path.join(base_dir, 'val_letterbox.json')
    train_img  = os.path.join(base_dir, 'letterbox_images', 'train')

    # 세 가지 산출물이 모두 있으면 전처리 생략 (최초 1회 이후는 항상 이 경로)
    if os.path.exists(train_json) and os.path.exists(val_json) and os.path.exists(train_img):
        return

    print("⚠️  전처리 산출물이 없습니다. 자동으로 전처리 파이프라인을 실행합니다...")
    print("   (최초 1회 실행이며, 완료 후 다음 실행부터는 이 단계가 생략됩니다)\n")

    # 이 파일(dataset.py)의 절대 경로를 기준으로 프로젝트 루트를 찾습니다.
    # __file__ = .../pill_detection_project/src/preprocessing/dataset.py
    current_dir  = os.path.dirname(os.path.abspath(__file__))    # src/preprocessing/
    project_root = os.path.dirname(os.path.dirname(current_dir)) # 프로젝트 루트

    preprocess_script = os.path.join(project_root, 'run_preprocessing.py')

    if not os.path.exists(preprocess_script):
        raise FileNotFoundError(
            f"🚨 run_preprocessing.py 를 찾을 수 없습니다: {preprocess_script}\n"
            f"   프로젝트 루트에 run_preprocessing.py 가 있는지 확인해주세요."
        )

    # sys.executable : 현재 실행 중인 파이썬 인터프리터 경로
    # Colab, 로컬, 가상환경 등 어떤 환경이든 올바른 파이썬으로 실행됩니다.
    import subprocess
    result = subprocess.run(
        [sys.executable, preprocess_script],
        check=True  # 오류 발생 시 CalledProcessError 예외 발생 후 즉시 중단
    )

    if result.returncode != 0:
        raise RuntimeError("🚨 전처리 파이프라인 실행 중 오류가 발생했습니다.")

    print("\n✅ 전처리 완료! DataLoader 구성을 시작합니다.\n")


# ---------------------------------------------------------------------------
# Dataset 클래스
# ---------------------------------------------------------------------------
class OralDrugDataset(Dataset):
    """
    Faster R-CNN / RetinaNet 학습을 위한 PyTorch Dataset 클래스.

    PyTorch Dataset 클래스는 두 가지 메서드를 반드시 구현해야 합니다:
      __len__     : 전체 데이터 개수 반환 (이미지 단위)
      __getitem__ : 인덱스를 받아 (이미지, 타겟) 쌍 반환

    【반환 형식】
      image  : [C, H, W] float32 텐서 (0~1 범위)
      target : {
          'boxes'    : [N, 4] float32 텐서  — [x1, y1, x2, y2] 형식
          'labels'   : [N]    int64 텐서    — 모델용 레이블 (1-based)
          'image_id' : [1]    int64 텐서    — 배치 내 인덱스
      }
      N = 해당 이미지의 객체(알약) 수

    【BBox 형식 변환】
      COCO 형식         : [x_min, y_min, width, height]
      Faster R-CNN 형식 : [x_min, y_min, x_max, y_max]
      → x_max = x_min + width, y_max = y_min + height 로 변환합니다.

    【레이블 매핑】
      원본 category_id(COCO) → 모델용 label(1-based) 변환은
      get_loaders()에서 생성한 orig2model 딕셔너리를 사용합니다.
      예: {5: 1, 12: 2, 23: 3, ...}  (category_id → 모델 레이블)

    Args:
        df          : build_df_from_json()으로 생성한 DataFrame
        orig2model  : {원본 category_id: 모델용 label} 딕셔너리
        transforms  : 이미지에 적용할 torchvision.transforms (없으면 None)
    """

    def __init__(self, df, orig2model, transforms=None):
        self.df         = df.reset_index(drop=True)  # 인덱스 초기화 (안전하게)
        self.orig2model = orig2model
        self.transforms = transforms

        # 이미지 단위로 묶기 위해 고유 image_id 목록을 미리 추출합니다.
        # DataFrame은 annotation(객체) 단위이므로 한 이미지에 여러 행이 있을 수 있습니다.
        self.image_ids = self.df['image_id'].unique().tolist()

    def __len__(self):
        # DataLoader가 전체 데이터 크기를 알기 위해 호출합니다.
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        인덱스 idx에 해당하는 이미지와 타겟을 반환합니다.

        처리 순서:
          1) image_id 선택
          2) 해당 이미지의 모든 annotation 행 추출 (알약 여러 개)
          3) 이미지 로드 (PIL → RGB)
          4) BBox 좌표 변환 (COCO → Faster R-CNN 형식)
          5) 레이블 변환 (category_id → 모델용 1-based label)
          6) 텐서 변환 및 transforms 적용
        """
        # 1) 이미지 ID 선택
        image_id = self.image_ids[idx]

        # 2) 해당 이미지의 모든 annotation 추출
        #    (한 이미지에 알약이 여러 개일 경우 여러 행이 있음)
        df_img = self.df[self.df['image_id'] == image_id]

        # 3) 이미지 로드 — PIL로 열어서 RGB로 변환
        #    OpenCV는 BGR이지만 torchvision은 RGB를 기대하므로 PIL 사용
        image = Image.open(df_img['image_path'].iloc[0]).convert('RGB')

        # 4-5) BBox 좌표 변환 + 레이블 매핑
        boxes, labels = [], []
        for _, row in df_img.iterrows():
            # COCO: [x_min, y_min, w, h] → Faster R-CNN: [x_min, y_min, x_max, y_max]
            x1, y1 = row['bbox_x'], row['bbox_y']
            x2, y2 = x1 + row['bbox_w'], y1 + row['bbox_h']
            boxes.append([x1, y1, x2, y2])

            # category_id → 모델용 1-based label 변환
            # get() 의 두 번째 인자(1)는 매핑이 없을 때의 기본값 (background 방지용)
            labels.append(self.orig2model.get(int(row['category_id']), 1))

        # 6) 텐서 변환
        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),  # 배치 내 위치 추적용 (loss에는 영향 없음)
        }

        # transforms 적용 (ToImage, ColorJitter, ToDtype 등)
        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    """
    DataLoader의 배치 묶음 함수입니다.

    기본 collate_fn은 텐서들을 하나의 배치 텐서로 쌓으려고 시도하는데,
    객체 탐지 데이터는 이미지마다 객체 수(N)가 달라서 단순히 쌓을 수 없습니다.

    이 함수는 배치를 리스트 형태로 묶어서 이 문제를 해결합니다:
      입력: [(image1, target1), (image2, target2), ...]
      출력: ((image1, image2, ...), (target1, target2, ...))

    DataLoader에 collate_fn=collate_fn 으로 전달해야 합니다.
    """
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# DataLoader 빌더 — 팀원 진입점
# ---------------------------------------------------------------------------
def get_loaders(base_dir, batch_size=2, num_workers=2):
    """
    train / val DataLoader를 한 번에 생성하여 반환합니다.

    【자동 전처리】
    전처리 산출물(train_letterbox.json 등)이 없으면 run_preprocessing.py를
    자동으로 실행하여 산출물을 생성합니다. 최초 1회만 실행되며,
    이후에는 기존 산출물을 그대로 사용합니다.

    【반환값 설명】
      train_loader : 학습용 DataLoader (shuffle=True)
      val_loader   : 검증용 DataLoader (shuffle=False)
      orig2model   : {원본 category_id: 모델 레이블} 딕셔너리
                     추론 결과를 원본 클래스 ID로 되돌릴 때 역방향으로 사용:
                     model2orig = {v: k for k, v in orig2model.items()}
      num_classes  : background(0) 포함 전체 클래스 수
                     Faster R-CNN / RetinaNet 모델 정의 시 그대로 전달하세요.

    【transforms 설명】
      train : ColorJitter(밝기/대비 ±20%) + ToImage + ToDtype
              → 다양한 조명 조건에 강건하게 만들기 위한 증강
      val   : ToImage + ToDtype만 적용 (증강 없이 원본 그대로 평가)

      T.Normalize는 적용하지 않습니다.
      Faster R-CNN / RetinaNet은 모델 내부(GeneralizedRCNNTransform)에서
      자동으로 정규화를 수행하기 때문입니다.

    Args:
        base_dir    : 전처리 산출물이 있는 데이터 루트 경로
                      Colab: '/content/drive/MyDrive/data/초급_프로젝트/dataset'
                      로컬 : '../data'
        batch_size  : 배치 크기 (기본 2, GPU 메모리에 맞게 조정)
        num_workers : 데이터 로딩 병렬 워커 수 (기본 2, Colab은 2 권장)

    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        orig2model   : dict {category_id → model_label}
        num_classes  : int (background 포함)

    사용 예시:
        from src.preprocessing.dataset import get_loaders

        BASE_DIR = '/content/drive/MyDrive/data/초급_프로젝트/dataset'
        train_loader, val_loader, orig2model, num_classes = get_loaders(base_dir=BASE_DIR)

        # 모델 정의 시
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    """
    # ── 전처리 산출물 없으면 자동 실행 ──────────────────────────
    _run_preprocessing_if_needed(base_dir)

    # ── 경로 설정 ────────────────────────────────────────────────
    train_json = os.path.join(base_dir, 'train_letterbox.json')
    val_json   = os.path.join(base_dir, 'val_letterbox.json')
    train_img  = os.path.join(base_dir, 'letterbox_images/train')
    val_img    = os.path.join(base_dir, 'letterbox_images/val')

    # ── JSON → DataFrame 변환 ────────────────────────────────────
    df_train = build_df_from_json(train_json, train_img)
    df_val   = build_df_from_json(val_json,   val_img)

    # ── 레이블 매핑 생성 (0 = background 예약 → 1-based) ─────────
    # COCO의 category_id는 연속적이지 않을 수 있습니다. (예: 1, 5, 12, 23, ...)
    # 모델은 0부터 시작하는 연속된 정수를 기대하므로 매핑이 필요합니다.
    # 0은 background로 예약되어 있으므로 실제 클래스는 1부터 시작합니다.
    # 예: {5: 1, 12: 2, 23: 3, ...}
    unique_cats = sorted(df_train['category_id'].unique())
    orig2model  = {cid: i + 1 for i, cid in enumerate(unique_cats)}
    num_classes = len(unique_cats) + 1  # +1 은 background(0)

    print(f"✅ 고유 클래스 수  : {len(unique_cats)}종")
    print(f"✅ num_classes     : {num_classes}  ← 모델 정의 시 사용")
    print(f"✅ Train: {df_train['image_id'].nunique()}장 / {len(df_train)}개")
    print(f"✅ Val  : {df_val['image_id'].nunique()}장 / {len(df_val)}개")

    # ── transforms 정의 ──────────────────────────────────────────
    # ✅ torchvision.transforms.v2 사용 (PyTorch 2.0+)
    # T.Normalize 미적용 이유: Faster R-CNN / RetinaNet 내부에서 자동 처리됨
    train_transforms = T.Compose([
        T.ToImage(),                                      # PIL → uint8 Image 텐서
        T.ColorJitter(brightness=0.2, contrast=0.2),     # 밝기/대비 랜덤 변화 (증강)
        T.ToDtype(torch.float32, scale=True),             # uint8 → [0, 1] float32
    ])
    val_transforms = T.Compose([
        T.ToImage(),                                      # PIL → uint8 Image 텐서
        T.ToDtype(torch.float32, scale=True),             # 검증은 증강 없이 그대로 평가
    ])

    # ── Dataset / DataLoader 생성 ─────────────────────────────────
    train_ds = OralDrugDataset(df_train, orig2model, train_transforms)
    val_ds   = OralDrugDataset(df_val,   orig2model, val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,              # 학습 시 매 epoch마다 순서를 섞어 과적합 방지
        collate_fn=collate_fn,     # 이미지마다 객체 수가 달라 커스텀 collate 필요
        num_workers=num_workers,
        pin_memory=True,           # GPU로 데이터 전송 속도 향상 (CUDA 환경)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,             # 검증은 항상 같은 순서로 (재현성 보장)
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, orig2model, num_classes
