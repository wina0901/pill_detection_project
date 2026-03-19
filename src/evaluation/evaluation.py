import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================================================
# category_id가 원본 COCO 기준으로 맞춰진 리스트
# =========================================================

def _normalize_predictions(predictions, model2orig=None):
    normalized = []

    for p in predictions:
        pred_cat = int(p["category_id"])

        if model2orig is not None:
            if pred_cat not in model2orig:
                # 매핑 안 되는 예측은 평가에서 제외
                continue
            pred_cat = int(model2orig[pred_cat])

        normalized.append({
            "image_id": int(p["image_id"]),
            "category_id": pred_cat,
            "bbox_xyxy": [
                float(p["bbox_xyxy"][0]),
                float(p["bbox_xyxy"][1]),
                float(p["bbox_xyxy"][2]),
                float(p["bbox_xyxy"][3]),
            ],
            "score": float(p["score"]),
        })

    return normalized

# =========================================================
# COCO GT에 존재하지 않는 image_id / category_id를 가진 예측 제거
# =========================================================

def _filter_predictions_for_coco(coco_gt, predictions):

    valid_image_ids = set(coco_gt.getImgIds())
    valid_cat_ids = set(coco_gt.getCatIds())

    filtered = []
    for p in predictions:
        if int(p["image_id"]) not in valid_image_ids:
            continue
        if int(p["category_id"]) not in valid_cat_ids:
            continue
        filtered.append(p)

    return filtered

# =========================================================
# bbox 유틸
# =========================================================
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def compute_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# =========================================================
# prediction -> COCO result 변환
# =========================================================
def convert_predictions_to_coco_results(predictions):

    coco_results = []

    for p in predictions:
        coco_results.append({
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": xyxy_to_xywh(p["bbox_xyxy"]),
            "score": float(p["score"])
        })
    return coco_results


# =========================================================
# GT 로드 (precision / recall 계산용)
# =========================================================
def load_gt_from_coco_json(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_json = json.load(f)

    gt_by_image = defaultdict(list)
    ann_id = 0

    for ann in gt_json["annotations"]:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        x, y, w, h = ann["bbox"]

        gt_by_image[image_id].append({
            "ann_id": ann_id,
            "category_id": category_id,
            "bbox_xyxy": [x, y, x + w, y + h]
        })
        ann_id += 1

    return gt_by_image


# =========================================================
# precision / recall 계산
# =========================================================
def compute_precision_recall_from_predictions(
    gt_json_path,
    predictions,
    conf_threshold=0.25,
    iou_threshold=0.5,
    model2orig=None
):
    gt_by_image = load_gt_from_coco_json(gt_json_path)

    # 모델 label -> 원본 category_id 변환
    predictions = _normalize_predictions(predictions, model2orig=model2orig)

    filtered_preds = [p for p in predictions if p["score"] >= conf_threshold]
    filtered_preds = sorted(filtered_preds, key=lambda x: x["score"], reverse=True)

    matched = set()
    tp = 0
    fp = 0
    total_gt = sum(len(v) for v in gt_by_image.values())

    for pred in filtered_preds:
        image_id = int(pred["image_id"])
        pred_cat = int(pred["category_id"])
        pred_box = pred["bbox_xyxy"]

        gt_candidates = gt_by_image.get(image_id, [])

        best_iou = 0.0
        best_gt_key = None

        for gt in gt_candidates:
            if int(gt["category_id"]) != pred_cat:
                continue

            gt_key = (image_id, gt["ann_id"])
            if gt_key in matched:
                continue

            iou = compute_iou_xyxy(pred_box, gt["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_key = gt_key

        if best_iou >= iou_threshold and best_gt_key is not None:
            tp += 1
            matched.add(best_gt_key)
        else:
            fp += 1

    fn = total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0

    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold
    }

# =========================================================
# COCO 공식 mAP 0.75 : 0.95 정의
# =========================================================
def compute_map_75_to_95(coco_eval):
 
    precision = coco_eval.eval["precision"]

    if precision is None or precision.size == 0:
        return 0.0

    iou_thrs = coco_eval.params.iouThrs
    valid_t = np.where(iou_thrs >= 0.75)[0]

    if len(valid_t) == 0:
        return 0.0

    # area=all -> index 0
    # maxDets=100 -> 마지막 index
    selected = precision[valid_t, :, :, 0, -1]

    # 유효하지 않은 값은 -1
    selected = selected[selected > -1]

    if selected.size == 0:
        return 0.0

    return float(np.mean(selected))

# =========================================================
# COCO 공식 mAP 계산
# =========================================================
def compute_coco_map(gt_json_path, predictions, temp_json_path="temp_eval.json", model2orig=None):
    coco_gt = COCO(gt_json_path)

    # 1) 모델 label -> 원본 COCO category_id 변환
    predictions = _normalize_predictions(predictions, model2orig=model2orig)

    # 2) GT에 없는 image_id / category_id 제거
    predictions = _filter_predictions_for_coco(coco_gt, predictions)

    # 3) 빈 예측이면 안전하게 0점 반환
    if len(predictions) == 0:
        return {
            "mAP@50": 0.0,
            "mAP@75:95": 0.0,
            "coco_eval": None
        }

    coco_results = convert_predictions_to_coco_results(predictions)

    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_results, f, ensure_ascii=False)

    try:
        coco_dt = coco_gt.loadRes(temp_json_path)
    except AssertionError as e:
        raise ValueError(
            "COCO 평가용 prediction 포맷이 GT와 맞지 않습니다. "
            "image_id 또는 category_id 매핑을 확인하세요."
        ) from e

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted({int(p["image_id"]) for p in predictions})

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCO stats[1] = AP@0.50
    # mAP@75:95 는 coco_eval.eval["precision"]에서 직접 계산
    map_50 = float(coco_eval.stats[1])
    map_75_95 = compute_map_75_to_95(coco_eval)

    return {
        "mAP@50": round(map_50, 6),
        "mAP@75:95": round(map_75_95, 6),
        "coco_eval": coco_eval
    }


# =========================================================
# 최종 통합 평가 함수
# =========================================================
def evaluate_all(
    gt_json_path,
    predictions,
    conf_threshold=0.25,
    pr_iou_threshold=0.5,
    temp_json_path="temp_eval.json",
    model2orig=None
):
    map_result = compute_coco_map(
        gt_json_path=gt_json_path,
        predictions=predictions,
        temp_json_path=temp_json_path,
        model2orig=model2orig
    )

    pr_result = compute_precision_recall_from_predictions(
        gt_json_path=gt_json_path,
        predictions=predictions,
        conf_threshold=conf_threshold,
        iou_threshold=pr_iou_threshold,
        model2orig=model2orig
    )

    result = {
        "mAP@50": map_result["mAP@50"],
        "mAP@75:95": map_result["mAP@75:95"],
        "precision": pr_result["precision"],
        "recall": pr_result["recall"],
        "details": {
            "tp": pr_result["tp"],
            "fp": pr_result["fp"],
            "fn": pr_result["fn"],
            "conf_threshold": pr_result["conf_threshold"],
            "pr_iou_threshold": pr_result["iou_threshold"]
        }
    }
    return result


# =========================================================
# 학습 로그 저장 유틸
# =========================================================
def init_history():
    """
    공통 history 구조
    """
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "mAP@50": [],
        "mAP@75:95": [],
        "precision": [],
        "recall": []
    }


def update_history(
    history,
    epoch,
    train_loss=None,
    val_loss=None,
    metrics=None
):
    history["epoch"].append(epoch)
    history["train_loss"].append(None if train_loss is None else float(train_loss))
    history["val_loss"].append(None if val_loss is None else float(val_loss))

    if metrics is None:
        history["mAP@50"].append(None)
        history["mAP@75:95"].append(None)
        history["precision"].append(None)
        history["recall"].append(None)
    else:
        history["mAP@50"].append(float(metrics.get("mAP@50", 0)))
        history["mAP@75:95"].append(float(metrics.get("mAP@75:95", 0)))
        history["precision"].append(float(metrics.get("precision", 0)))
        history["recall"].append(float(metrics.get("recall", 0)))


def save_history(history, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_history(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 시각화 함수
# =========================================================
def _safe_plot(ax, x, y, label):
    y_clean = [np.nan if v is None else v for v in y]
    ax.plot(x, y_clean, marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_training_history(history, title_prefix="Model"):
    epochs = history["epoch"]

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    _safe_plot(ax1, epochs, history["train_loss"], "train_loss")
    _safe_plot(ax1, epochs, history["val_loss"], "val_loss")
    ax1.set_title(f"{title_prefix} Loss")

    ax2 = fig.add_subplot(2, 2, 2)
    _safe_plot(ax2, epochs, history["mAP@50"], "mAP@50")
    _safe_plot(ax2, epochs, history["mAP@75:95"], "mAP@75:95")
    ax2.set_title(f"{title_prefix} mAP")

    ax3 = fig.add_subplot(2, 2, 3)
    _safe_plot(ax3, epochs, history["precision"], "precision")
    ax3.set_title(f"{title_prefix} Precision")

    ax4 = fig.add_subplot(2, 2, 4)
    _safe_plot(ax4, epochs, history["recall"], "recall")
    ax4.set_title(f"{title_prefix} Recall")

    plt.tight_layout()
    plt.show()

# 여러 모델 결과 비교
def plot_compare_histories(histories, labels, metric_key="mAP@75:95", title=None):
    """
    histories: [history1, history2, ...]
    labels: ["YOLO", "FasterRCNN", "RetinaNet"]
    """
    plt.figure(figsize=(8, 5))

    for history, label in zip(histories, labels):
        x = history["epoch"]
        y = [np.nan if v is None else v for v in history.get(metric_key, [])]
        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric_key)
    plt.title(title if title else f"Comparison: {metric_key}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# 결과 변환 (YOLO)
def convert_yolo_results(results, image_ids, model2orig=None):
    predictions = []

    for result, image_id in zip(results, image_ids):
        boxes = result.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)

        for box, score, label in zip(xyxy, scores, labels):
            if model2orig is not None:
                if int(label) not in model2orig:
                    continue
                category_id = int(model2orig[int(label)])
            else:
                category_id = int(label)

            predictions.append({
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score)
            })

    return predictions

# 결과 변환 (Faster R-CNN / RetinaNet)
def convert_torchvision_outputs(outputs, image_ids, model2orig=None):
    predictions = []

    for output, image_id in zip(outputs, image_ids):
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy().astype(int)

        for box, score, label in zip(boxes, scores, labels):
            if model2orig is not None:
                if int(label) not in model2orig:
                    continue
                category_id = int(model2orig[int(label)])
            else:
                category_id = int(label)

            predictions.append({
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score)
            })

    return predictions