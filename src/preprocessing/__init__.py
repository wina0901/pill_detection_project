"""
src/preprocessing/__init__.py
==============================
HealthEat 데이터 전처리 패키지  ·  v3

파이프라인 실행 순서
  [NB01] EDA
  [NB02] augmentation.run_stratified_split (수동) +
         augmentation.extract_minority_crops +
         augmentation.run_copy_paste         → train_augmented_final.json
  [NB03] transforms.run_letterbox_pipeline  → letterbox_images/ + *_letterbox.json
  [NB04] transforms.apply_clahe_to_folder   → in-place CLAHE
  [NB05] dataset.get_loaders                → DataLoader

모듈 구성
  augmentation.py  : make_pill_mask, blend_with_mask, check_overlap,
                     extract_minority_crops, run_copy_paste
  transforms.py    : letterbox_with_bbox, run_letterbox_pipeline, apply_clahe_to_folder
  dataset.py       : OralDrugDataset, build_df_from_json, get_loaders,
                     validate_coco, denormalize, collate_fn
  format_converter.py: convert_coco_to_yolo, generate_data_yaml, run_yolo_conversion
  viz_utils.py     : show_samples, show_augmented_samples, show_mask_preview,
                     show_class_distribution, show_letterbox_comparison

v3 변경
  - augmentation.py: 마스크 기반 Copy-Paste (사각형 아티팩트 제거)
  - viz_utils.py   : 노트북 공용 시각화 유틸리티 추가
  - aihub_merge.py 추가 (데이터셋 병합 시 1회 시행, True로 설정 후 실행, 완료 후 False로 변경)
  
"""

from .transforms       import letterbox_with_bbox, run_letterbox_pipeline, apply_clahe_to_folder
from .augmentation     import (make_pill_mask, blend_with_mask, check_overlap,
                                extract_minority_crops, run_copy_paste)
from .dataset          import (OralDrugDataset, build_df_from_json,
                                get_loaders, validate_coco, denormalize,
                                collate_fn, IMAGENET_MEAN, IMAGENET_STD)
from .format_converter import (convert_coco_to_yolo, generate_data_yaml,
                                run_yolo_conversion)
from .viz_utils        import (show_samples, show_augmented_samples, show_mask_preview,
                                show_class_distribution, show_letterbox_comparison)

__all__ = [
    # augmentation (v3)
    'make_pill_mask', 'blend_with_mask', 'check_overlap',
    'extract_minority_crops', 'run_copy_paste',
    # transforms
    'letterbox_with_bbox', 'run_letterbox_pipeline', 'apply_clahe_to_folder',
    # dataset
    'OralDrugDataset', 'build_df_from_json', 'get_loaders',
    'validate_coco', 'denormalize', 'collate_fn',
    'IMAGENET_MEAN', 'IMAGENET_STD',
    # format_converter
    'convert_coco_to_yolo', 'generate_data_yaml', 'run_yolo_conversion',
    # viz_utils
    'show_samples', 'show_augmented_samples', 'show_mask_preview',
    'show_class_distribution', 'show_letterbox_comparison',
]
