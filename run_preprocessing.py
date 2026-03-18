"""
run_preprocessing.py
====================
HealthEat 데이터 전처리 파이프라인 전체 실행 스크립트

실행 방법:
    python run_preprocessing.py

파이프라인 순서:
    [Step 1] Stratified Split       → train_raw.json / val.json
    [Step 2] Copy-Paste 증강        → train_augmented_final.json
    [Step 3] Letterbox 800×800 변환 → train_letterbox.json / val_letterbox.json
                                      letterbox_images/train, val/
    [Step 4] CLAHE 대비 강화        → letterbox_images/ in-place 덮어쓰기
    [Step 5] YOLO 라벨 변환         → yolo_labels/train, val/ + data.yaml

사전 조건:
    - data/ 폴더에 merged_annotations_train_final.json 과 원본 이미지 존재
    - requirements.txt 설치 완료

출력 결과물:
    data/
    ├── train_raw.json
    ├── val.json
    ├── train_augmented_final.json
    ├── train_letterbox.json
    ├── val_letterbox.json
    ├── letterbox_images/
    │   ├── train/
    │   └── val/
    ├── yolo_labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
"""

import os
import sys
import json
import random
from collections import defaultdict

# 프로젝트 루트를 sys.path에 추가 (어느 위치에서 실행해도 동작)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.augmentation     import run_copy_paste
from src.preprocessing.transforms       import run_letterbox_pipeline, apply_clahe_to_folder
from src.preprocessing.format_converter import run_yolo_conversion


# ============================================================
# 경로 설정 — 필요 시 수정
# ============================================================
BASE_DIR = os.path.join(PROJECT_ROOT, 'data')


# ============================================================
# Step 1. Stratified Split (9:1)
# ============================================================
def run_stratified_split(base_dir, val_ratio=0.1, random_seed=42):
    """
    원본 COCO JSON을 클래스 비율을 보존하며 Train / Val 로 분할합니다.

    Args:
        base_dir    : merged_annotations_train_final.json 이 있는 데이터 루트
        val_ratio   : Validation 비율 (기본 0.1 = 10%)
        random_seed : 재현성 시드 (기본 42)

    출력:
        base_dir/train_raw.json
        base_dir/val.json
    """
    original_json = os.path.join(base_dir, 'merged_annotations_train_final.json')
    train_out     = os.path.join(base_dir, 'train_raw.json')
    val_out       = os.path.join(base_dir, 'val.json')

    if not os.path.exists(original_json):
        raise FileNotFoundError(f"🚨 원본 JSON 없음: {original_json}")

    print(f"\n{'='*60}")
    print(f"[Step 1] Stratified Split ({int((1-val_ratio)*100)}:{int(val_ratio*100)})")
    print(f"{'='*60}")

    random.seed(random_seed)

    with open(original_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images, annotations, categories = coco['images'], coco['annotations'], coco['categories']

    # 이미지별 대표 클래스(최빈 클래스) 추출
    img_to_cats = defaultdict(list)
    for ann in annotations:
        img_to_cats[ann['image_id']].append(ann['category_id'])

    img_dominant = {
        img_id: max(set(cats), key=cats.count)
        for img_id, cats in img_to_cats.items()
    }

    # 클래스별 이미지 목록 구성 후 분할
    class_to_imgs = defaultdict(list)
    for img_id, label in img_dominant.items():
        class_to_imgs[label].append(img_id)

    train_ids, val_ids = set(), set()

    for label, img_list in class_to_imgs.items():
        random.shuffle(img_list)
        if len(img_list) == 1:
            # 1장뿐인 클래스 → Train으로만 보냄
            train_ids.update(img_list)
        elif len(img_list) < 5:
            # 소수 클래스 → Val에 최소 1장 보장
            val_ids.add(img_list[0])
            train_ids.update(img_list[1:])
        else:
            split_idx = max(1, int(len(img_list) * val_ratio))
            val_ids.update(img_list[:split_idx])
            train_ids.update(img_list[split_idx:])

    # 분할 검증
    val_classes = set(img_dominant[i] for i in val_ids if i in img_dominant)
    missing     = set(img_dominant.values()) - val_classes
    print(f"📊 총 {len(images):,}장 → Train: {len(train_ids):,}장 / Val: {len(val_ids):,}장")
    if missing:
        print(f"⚠️  Val 누락 클래스 {len(missing)}개 (데이터 1장뿐인 클래스 — 정상)")
    else:
        print(f"✅ 전체 {len(categories)}개 클래스 Val 포함 확인")

    # JSON 저장
    train_anns = [a for a in annotations if a['image_id'] in train_ids]
    val_anns   = [a for a in annotations if a['image_id'] in val_ids]

    for path, imgs, anns in [
        (train_out, [i for i in images if i['id'] in train_ids], train_anns),
        (val_out,   [i for i in images if i['id'] in val_ids],   val_anns),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'images': imgs, 'annotations': anns, 'categories': categories},
                      f, ensure_ascii=False)

    print(f"✅ 저장 완료 → {os.path.basename(train_out)} ({len(train_anns):,}개) / "
          f"{os.path.basename(val_out)} ({len(val_anns):,}개)")


# ============================================================
# 메인 파이프라인
# ============================================================
def main():
    print(f"\n{'#'*60}")
    print(f"  HealthEat 데이터 전처리 파이프라인 시작")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"{'#'*60}")

    # ── Step 1. Stratified Split
    run_stratified_split(base_dir=BASE_DIR)

    # ── Step 2. Copy-Paste 증강 (소수 클래스 보강)
    print(f"\n{'='*60}")
    print(f"[Step 2] Copy-Paste 증강")
    print(f"{'='*60}")
    run_copy_paste(base_dir=BASE_DIR, aug_count=500, random_seed=42)

    # ── Step 3. Letterbox 800×800 변환
    print(f"\n{'='*60}")
    print(f"[Step 3] Letterbox 규격화 (800×800)")
    print(f"{'='*60}")
    run_letterbox_pipeline(
        json_path     = os.path.join(BASE_DIR, 'train_augmented_final.json'),
        out_json_path = os.path.join(BASE_DIR, 'train_letterbox.json'),
        img_out_dir   = os.path.join(BASE_DIR, 'letterbox_images/train'),
        base_dir      = BASE_DIR,
        desc          = 'Train Letterbox 변환',
    )
    run_letterbox_pipeline(
        json_path     = os.path.join(BASE_DIR, 'val.json'),
        out_json_path = os.path.join(BASE_DIR, 'val_letterbox.json'),
        img_out_dir   = os.path.join(BASE_DIR, 'letterbox_images/val'),
        base_dir      = BASE_DIR,
        desc          = 'Val Letterbox 변환',
    )

    # ── Step 4. CLAHE 대비 강화 (in-place)
    print(f"\n{'='*60}")
    print(f"[Step 4] L-channel CLAHE 대비 강화")
    print(f"{'='*60}")
    apply_clahe_to_folder(os.path.join(BASE_DIR, 'letterbox_images/train'))
    apply_clahe_to_folder(os.path.join(BASE_DIR, 'letterbox_images/val'))

    # ── Step 5. YOLO 라벨 변환 + data.yaml 생성
    print(f"\n{'='*60}")
    print(f"[Step 5] YOLO 라벨 변환 및 data.yaml 생성")
    print(f"{'='*60}")
    run_yolo_conversion(base_dir=BASE_DIR)

    print(f"\n{'#'*60}")
    print(f"  ✅ 전처리 파이프라인 완료!")
    print(f"  학습 준비 완료 경로: {BASE_DIR}")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
