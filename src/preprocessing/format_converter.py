"""
format_converter.py
===================
COCO JSON → YOLO txt 포맷 변환기 + data.yaml 자동 생성

🚨 레이블 인덱싱 주의
  YOLO  : 0-based  (이 모듈에서 생성하는 cat2yolo)
  Faster R-CNN / RetinaNet : 1-based  (dataset.py의 orig2model)
  둘은 동일 category_id에 대해 1 차이가 납니다.
"""

import os
import json
from collections import defaultdict
from tqdm.auto import tqdm

def convert_coco_to_yolo(json_path, output_label_dir, cat2yolo=None):
    """COCO JSON을 YOLO txt로 변환하여 지정된 폴더에 저장"""
    if not os.path.exists(json_path):
        print(f"🚨 [에러] 파일을 찾을 수 없습니다: {json_path}")
        return None
        
    os.makedirs(output_label_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 카테고리 매핑 (Train 실행 시 생성, Val 실행 시 재사용)
    if cat2yolo is None:
        cat2yolo = {cat['id']: i for i, cat in enumerate(coco['categories'])}

    images = {img['id']: img for img in coco['images']}
    anns_by_img = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    print(f"📝 {os.path.basename(json_path)} 변환 중...")
    for img_id, anns in tqdm(anns_by_img.items()):
        img = images[img_id]
        h, w = img['height'], img['width']
        
        # 확장자 제외한 파일명으로 .txt 생성
        txt_name = os.path.splitext(img['file_name'])[0] + '.txt'
        txt_path = os.path.join(output_label_dir, txt_name)

        with open(txt_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in cat2yolo: continue
                
                yolo_id = cat2yolo[cat_id]
                x, y, bw, bh = ann['bbox']
                
                # YOLO 정규화 좌표 계산 (0~1)
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    
    return cat2yolo

def generate_data_yaml(base_dir, cat2yolo, json_path, label_root):
    """분리된 구조를 반영한 data.yaml 생성"""
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    class_names = [id_to_name[cid] for cid in sorted(cat2yolo, key=cat2yolo.get)]

    yaml_content = f"""# YOLO data.yaml
path: {base_dir}
train: letterbox_images/train
val:   letterbox_images/val
# 라벨 폴더가 labels가 아닐 경우 명시적으로 지정하거나, 
# YOLOv8 규칙에 따라 'yolo_labels'를 인식하게 합니다.
names: {class_names}
nc: {len(class_names)}
"""
    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"✅ data.yaml 생성 완료: {yaml_path}")

def run_yolo_conversion(base_dir):
    """최종 실행 함수"""
    # 🚨 NB03에서 생성한 '800x800' 결과물을 읽어야 합니다!
    train_json = os.path.join(base_dir, 'train_letterbox.json')
    val_json   = os.path.join(base_dir, 'val_letterbox.json')
    
    # 따로 생성될 라벨 폴더 경로
    label_root = os.path.join(base_dir, 'yolo_labels')
    train_label_dir = os.path.join(label_root, 'train')
    val_label_dir   = os.path.join(label_root, 'val')

    # 1. Train 변환
    cat2yolo = convert_coco_to_yolo(train_json, train_label_dir)
    # 2. Val 변환 (Train의 클래스 ID 유지)
    if cat2yolo:
        convert_coco_to_yolo(val_json, val_label_dir, cat2yolo=cat2yolo)
        # 3. YAML 생성
        generate_data_yaml(base_dir, cat2yolo, train_json, label_root)