from pathlib import Path
from typing import List, Dict, Any
import json
import csv
import uuid

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 프로젝트 경로
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

MODEL_PATHS = [
    PROJECT_ROOT / "models" / "yolo" / "yolov8s_v2_v3_ft_uf_lr_0p0003_best.pt",
    PROJECT_ROOT / "models" / "yolo" / "yolo11m_v2_v3_ft_uf_lr_0p0005_best.pt",
]

TRAIN_JSON = PROJECT_ROOT / "data" / "merged_annotations_train_final.json"
META_CSV = PROJECT_ROOT / "data" / "meta.csv"

OUTPUT_DIR = PROJECT_ROOT / "results"
CROP_DIR = OUTPUT_DIR / "crops"


# =========================================================
# 추론 설정
# =========================================================
MODEL_WEIGHTS = [1.0, 1.0]
PRED_CONF = 0.05
PRED_IOU = 0.70
ENS_NMS_IOU = 0.55
TOPK_PER_IMAGE = 4
IMGSZ = 800

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =========================================================
# 출력 폴더 보장
# =========================================================
def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)


ensure_output_dirs()


# =========================================================
# category 매핑
# model class index(0~N-1) -> 원본 category_id
# =========================================================
def load_category_mapping(train_json_path: Path) -> Dict[int, int]:
    if not train_json_path.exists():
        raise FileNotFoundError(f"TRAIN_JSON 파일이 없습니다: {train_json_path}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        train_coco = json.load(f)

    if "annotations" not in train_coco:
        raise ValueError("TRAIN_JSON에 'annotations' 키가 없습니다.")

    orig_category_ids = sorted({int(a["category_id"]) for a in train_coco["annotations"]})
    model2orig = {i: cid for i, cid in enumerate(orig_category_ids)}
    return model2orig


MODEL2ORIG = load_category_mapping(TRAIN_JSON)


# =========================================================
# meta.csv 로드
# 컬럼:
# category_id,pill_name,feature
# =========================================================
def load_pill_metadata(csv_path: Path) -> Dict[int, Dict[str, str]]:
    if not csv_path.exists():
        return {}

    metadata: Dict[int, Dict[str, str]] = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                category_id = int(str(row.get("category_id", "")).strip())
            except Exception:
                continue

            metadata[category_id] = {
                "pill_name": str(row.get("pill_name", "")).strip(),
                "feature": str(row.get("feature", "")).strip(),
            }

    return metadata


PILL_METADATA = load_pill_metadata(META_CSV)


# =========================================================
# 모델 로드
# =========================================================
def load_models(model_paths: List[Path]) -> List[YOLO]:
    models = []

    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"가중치 파일이 없습니다: {model_path}")

        model = YOLO(str(model_path))
        models.append(model)

    return models


MODELS = load_models(MODEL_PATHS)


# =========================================================
# class-wise NMS
# =========================================================
def classwise_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    keep_indices: List[int] = []

    unique_classes = np.unique(classes)
    for cls_id in unique_classes:
        cls_mask = classes == cls_id
        cls_indices = np.where(cls_mask)[0]

        cls_boxes = torch.tensor(boxes_xyxy[cls_indices], dtype=torch.float32)
        cls_scores = torch.tensor(scores[cls_indices], dtype=torch.float32)

        kept = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend(cls_indices[kept.cpu().numpy()].tolist())

    return keep_indices


# =========================================================
# 단일 모델 1장 추론
# =========================================================
def predict_single_model(
    model: YOLO,
    image_path: str,
    pred_conf: float = PRED_CONF,
    pred_iou: float = PRED_IOU,
    imgsz: int = IMGSZ,
) -> Dict[str, np.ndarray]:
    result = model.predict(
        source=image_path,
        conf=pred_conf,
        iou=pred_iou,
        imgsz=imgsz,
        verbose=False,
        save=False,
        device=DEVICE,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "classes": np.empty((0,), dtype=np.int32),
        }

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
    }


# =========================================================
# 예측 결과에 메타 정보 붙이기
# =========================================================
def enrich_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []

    for pred in predictions:
        category_id = int(pred["category_id"])
        meta = PILL_METADATA.get(category_id, {})

        display_name = meta.get("pill_name") or f"ID {category_id}"
        feature = meta.get("feature") or "특징 정보 없음"

        item = dict(pred)
        item["display_name"] = display_name
        item["feature"] = feature
        enriched.append(item)

    return enriched


# =========================================================
# 앙상블 추론
# =========================================================
def predict_ensemble(image_path: str) -> List[Dict[str, Any]]:
    image_path = str(image_path)

    all_boxes = []
    all_scores = []
    all_classes = []

    for model, model_weight in zip(MODELS, MODEL_WEIGHTS):
        pred = predict_single_model(model, image_path)

        if len(pred["boxes"]) == 0:
            continue

        all_boxes.append(pred["boxes"])
        all_scores.append(pred["scores"] * model_weight)
        all_classes.append(pred["classes"])

    if len(all_boxes) == 0:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    keep = classwise_nms(
        boxes_xyxy=boxes,
        scores=scores,
        classes=classes,
        iou_threshold=ENS_NMS_IOU,
    )

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    order = np.argsort(scores)[::-1][:TOPK_PER_IMAGE]
    boxes = boxes[order]
    scores = scores[order]
    classes = classes[order]

    results: List[Dict[str, Any]] = []

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.tolist()

        orig_cat = int(MODEL2ORIG[int(cls_id)])
        competition_category_id = orig_cat + 1

        results.append({
            "category_id": competition_category_id,
            "score": round(float(score), 6),
            "bbox_xyxy": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ],
            "bbox_xywh": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2 - x1), 2),
                round(float(y2 - y1), 2),
            ],
        })

    return enrich_predictions(results)


# =========================================================
# 폰트 로드
# =========================================================
def get_font(font_size: int = 24):
    font_candidates = [
        "malgunbd.ttf",
        "malgun.ttf",
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue

    return ImageFont.load_default()


# =========================================================
# 결과 이미지 저장
# =========================================================
def draw_predictions(
    image_path: str,
    predictions: List[Dict[str, Any]],
    save_path: str,
) -> str:
    ensure_output_dirs()

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size
    box_width = max(3, int(min(width, height) * 0.006))
    font_size = max(18, int(min(width, height) * 0.028))
    font = get_font(font_size)

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox_xyxy"]
        display_name = pred.get("display_name", f'ID {pred["category_id"]}')
        score = pred["score"]

        label = f"{display_name} | {score:.2f}"

        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

        draw.rectangle(
            [x1_i, y1_i, x2_i, y2_i],
            outline=(0, 255, 0),
            width=box_width
        )

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pad_x = 10
        pad_y = 6

        rect_x1 = x1_i
        rect_y1 = y1_i - text_h - (pad_y * 2) - 6
        rect_x2 = x1_i + text_w + (pad_x * 2)
        rect_y2 = y1_i

        if rect_y1 < 0:
            rect_y1 = y1_i
            rect_y2 = y1_i + text_h + (pad_y * 2) + 6

        draw.rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2],
            fill=(0, 255, 0)
        )

        text_x = rect_x1 + pad_x
        text_y = rect_y1 + pad_y
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    return save_path


# =========================================================
# bbox crop 저장
# =========================================================
def save_detection_crops(
    image_path: str,
    predictions: List[Dict[str, Any]],
    crop_dir: Path = CROP_DIR,
) -> List[Dict[str, Any]]:
    ensure_output_dirs()
    crop_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    saved = []

    for idx, pred in enumerate(predictions, start=1):
        x1, y1, x2, y2 = pred["bbox_xyxy"]

        x1_i = max(0, int(x1))
        y1_i = max(0, int(y1))
        x2_i = min(image.width, int(x2))
        y2_i = min(image.height, int(y2))

        # 잘못된 bbox 방지
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        crop = image.crop((x1_i, y1_i, x2_i, y2_i))

        crop_name = f"crop_{uuid.uuid4().hex}_{idx}.jpg"
        crop_path = crop_dir / crop_name

        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(str(crop_path), format="JPEG")

        item = dict(pred)
        item["crop_filename"] = crop_name
        saved.append(item)

    return saved


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("TRAIN_JSON:", TRAIN_JSON)
    print("META_CSV:", META_CSV)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("CROP_DIR:", CROP_DIR)
    print("DEVICE:", DEVICE)
    print("모델 로드 완료:", len(MODELS))
    print("메타 로드 수:", len(PILL_METADATA))