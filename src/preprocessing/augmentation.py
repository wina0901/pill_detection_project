"""
augmentation.py  ·  v4
======================
Copy-Paste 증강 엔진 (GrabCut 마스크 기반)

변경 이력
  v3 : HSV inRange + Flood Fill → 흰색/민트색 알약에서 C자 구멍 문제 잔존
  v4 : GrabCut으로 교체 → 배경색과 유사한 알약도 정확하게 분리

함수 목록
  make_pill_mask        : GrabCut 기반 알약 픽셀 마스크 생성
  blend_with_mask       : 마스크 기반 픽셀 단위 합성 (사각형 경계선 없음)
  check_overlap         : AABB 충돌 감지
  extract_minority_crops: 소수 클래스 스티커 추출 + crop_meta.csv
  run_copy_paste        : 전체 증강 파이프라인 실행

파이프라인 순서
  [Step 1] extract_minority_crops → crops_minority/ + crop_meta.csv
  [Step 2] run_copy_paste         → train_augmented_final.json + 합성 이미지
"""

import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import random
import copy
from tqdm.auto import tqdm
from collections import defaultdict


# ============================================================
# 마스크 생성 + 합성
# ============================================================

def make_pill_mask(crop_img):
    """
    GrabCut을 이용해 알약 픽셀만 남기는 float32 마스크를 반환합니다.

    처리 순서:
      1) crop 중앙 80% 영역을 전경(알약) 초기 rect로 지정
      2) GrabCut 5회 반복 → 전경/배경 픽셀 분류
      3) 전경(GC_FGD, GC_PR_FGD) 픽셀만 추출
      4) 모폴로지 Close로 내부 구멍 제거
      5) 가장 큰 contour만 유지 (배경 잡음 제거)
      6) Gaussian Blur로 엣지 페더링 (자연스러운 경계)

    [HSV Flood Fill 방식 대비 장점]
      - 배경색과 유사한 흰색/민트색 알약도 분리 가능
      - 각인 음각에 의한 C자 구멍 발생 없음
      - 색상 범위 수동 튜닝 불필요
    [단점]
      - crop당 처리 시간이 HSV 방식보다 느림 (약 0.1~0.3초)
      - GrabCut 실패 시 Fail-safe로 전체 마스크 반환

    Args:
        crop_img : BBox 영역을 crop한 BGR 이미지 (np.ndarray)

    Returns:
        mask_f   : [0.0~1.0] float32 2D 마스크 (1=알약, 0=배경)
    """
    h, w = crop_img.shape[:2]

    # GrabCut 내부 상태 모델 초기화
    # bgd/fgd_model은 GMM(가우시안 혼합 모델) 파라미터 저장용 배열
    mask      = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 1) 초기 rect: 중앙 80% 영역을 전경으로 지정
    #    margin을 10%로 설정하여 알약 경계가 잘리지 않도록 여유 확보
    margin_y = max(2, int(h * 0.10))
    margin_x = max(2, int(w * 0.10))
    rect = (margin_x, margin_y,
            w - margin_x * 2,
            h - margin_y * 2)

    try:
        # 2) GrabCut 실행 (5회 반복 → 전경/배경 경계 정밀화)
        #    GC_INIT_WITH_RECT: rect 기반 초기화
        cv2.grabCut(crop_img, mask, rect,
                    bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # GrabCut 실패 시 (너무 작은 crop 등) 전체를 알약으로 반환
        return np.ones((h, w), dtype=np.float32)

    # 3) 전경 픽셀 추출
    #    GC_FGD(1)    : 확실한 전경
    #    GC_PR_FGD(3) : 전경으로 추정되는 픽셀
    #    GC_BGD(0)    : 확실한 배경  → 제외
    #    GC_PR_BGD(2) : 배경으로 추정 → 제외
    pill_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # 4) 내부 구멍 완전 제거 (Connected Component 방식)
    #    원리: 배경(0)을 연결 요소로 분리 → 이미지 경계에 닿은 것만 진짜 배경
    #          경계에 안 닿은 배경 영역 = 알약 내부 구멍 → 255로 채움
    #    flood fill보다 안정적 (C자 틈새로 새는 문제 없음)
    pill_inv = cv2.bitwise_not(pill_mask)  # 배경=255, 알약=0으로 반전

    n_labels, labels = cv2.connectedComponents(pill_inv)

    # 이미지 경계 픽셀의 label 집합 추출 → 외부 배경 label
    h_m, w_m  = pill_inv.shape
    border_px = np.concatenate([
        labels[0, :], labels[-1, :],   # 상하 경계
        labels[:, 0], labels[:, -1],   # 좌우 경계
    ])
    bg_labels = set(border_px.tolist()) - {0}  # 0=알약(배경에서 반전됨)

    # 경계와 연결된 label만 진짜 배경 → 나머지는 내부 구멍 → 알약으로 처리
    bg_mask_cc = np.isin(labels, list(bg_labels)).astype(np.uint8) * 255
    pill_mask  = cv2.bitwise_not(bg_mask_cc)  # 최종 알약 마스크

    # 모폴로지 Close로 잔여 노이즈 정리
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)


    # 5) 가장 큰 contour만 유지 → 배경 잡음 제거
    #    전체 면적의 5% 미만이면 유효하지 않은 contour로 판단해 스킵
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask_clean = np.zeros_like(pill_mask)
        largest    = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > h * w * 0.05:
            cv2.drawContours(mask_clean, [largest], -1, 255, cv2.FILLED)
            pill_mask = mask_clean

    # 6) float32 변환 후 Gaussian Blur → 엣지 페더링
    #    blur 커널 크기는 이미지 크기에 비례 (홀수 보장)
    mask_f = pill_mask.astype(np.float32) / 255.0
    ksize  = max(5, (min(h, w) // 15) * 2 + 1)
    mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)

    return mask_f


def blend_with_mask(bg_img, crop_img, mask_f, x, y):
    """
    마스크를 이용해 알약 픽셀만 배경에 합성합니다 (in-place).

    Args:
        bg_img   : 배경 이미지 (수정됨)
        crop_img : 알약 crop 이미지
        mask_f   : [0~1] float32 마스크
        x, y     : 배경에서 붙일 좌상단 좌표
    """
    ch, cw   = crop_img.shape[:2]
    mask3    = np.stack([mask_f, mask_f, mask_f], axis=-1)
    bg_patch = bg_img[y:y+ch, x:x+cw].astype(np.float32)
    sticker  = crop_img.astype(np.float32)
    blended  = sticker * mask3 + bg_patch * (1.0 - mask3)
    bg_img[y:y+ch, x:x+cw] = blended.astype(np.uint8)


# ============================================================
# 충돌 감지
# ============================================================

def check_overlap(new_box, existing_boxes, min_dist=15):
    """
    새 박스가 기존 박스들과 겹치는지 AABB 방식으로 검사합니다.

    Returns:
        True  : 겹침 → 사용 불가
        False : 안전 → 사용 가능
    """
    nx, ny, nw, nh = new_box
    for ex, ey, ew, eh in existing_boxes:
        if not (nx + nw + min_dist < ex or
                nx > ex + ew + min_dist or
                ny + nh + min_dist < ey or
                ny > ey + eh + min_dist):
            return True
    return False

# ============================================================
# Step 1: 소수 클래스 스티커 추출
# ============================================================

def extract_minority_crops(base_dir, threshold=50):
    """
    train_raw.json에서 소수 클래스(threshold 미만) 알약을 crop 추출합니다.

    처리 순서:
      1) train_raw.json에서 클래스별 객체 수 집계
      2) threshold 미만인 소수 클래스 목록 추출
      3) 각 소수 클래스의 모든 annotation에서 BBox crop
      4) make_pill_mask()로 마스크 유효성 검증
         → pill_ratio 0.35 미만 또는 0.85 초과인 crop 제외
           (배경 제거 실패 또는 알약-배경 색상 혼동)
      5) 유효한 crop을 crops_minority/{클래스명}/ 에 저장
      6) crop_meta.csv 생성 (run_copy_paste에서 참조)

    [pill_ratio 기준]
      < 0.35 : 배경 제거 실패 (파란 알약, 흰 알약 등 배경색과 유사)
      0.35 ~ 0.92 : 정상 범위 → 저장
      > 0.92 : 알약이 배경을 대부분 차지 (이상 crop)

    Args:
        base_dir  : train_raw.json이 있는 데이터 루트 경로
        threshold : 소수 클래스 기준 (기본 50개 미만)

    출력:
        base_dir/crops_minority/{클래스명}/*.png : 클래스별 스티커 이미지
        base_dir/crops_minority/crop_meta.csv    : 스티커 메타데이터
                                                   (run_copy_paste에서 참조)
    """
    json_path     = os.path.join(base_dir, 'train_raw.json')
    crop_save_dir = os.path.join(base_dir, 'crops_minority')
    os.makedirs(crop_save_dir, exist_ok=True)

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"train_raw.json 없음: {json_path}\n"
            f"Stratified Split을 먼저 실행하세요."
        )

    print(f"\n{'='*60}")
    print(f"[Step 1] 소수 클래스 스티커 추출 (기준: {threshold}개 미만)")
    print(f"{'='*60}")

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images_df      = pd.DataFrame(coco_data['images'])
    annotations_df = pd.DataFrame(coco_data['annotations'])
    categories_df  = pd.DataFrame(coco_data['categories'])

    # category_id → 클래스명 매핑
    cat_dict = dict(zip(categories_df['id'], categories_df['name']))
    annotations_df['class_name'] = annotations_df['category_id'].map(cat_dict)

    # threshold 미만인 소수 클래스만 추출
    class_counts     = annotations_df['class_name'].value_counts()
    minority_classes = class_counts[class_counts < threshold].index.tolist()
    print(f"증강 대상 소수 클래스 : {len(minority_classes)}종")
    print(f"전체 클래스 중 비율   : {len(minority_classes)}/{len(class_counts)}")

    # 이미지 파일 경로 해시맵 (O(1) 탐색)
    # 한글 경로 및 대소문자 확장자(.JPG) 모두 처리
    all_files = (glob.glob(os.path.join(base_dir, '**', '*.png'), recursive=True) +
                 glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True) +
                 glob.glob(os.path.join(base_dir, '**', '*.JPG'), recursive=True))
    path_map = {os.path.basename(f): f for f in all_files}

    crop_metadata = []
    skipped       = 0
    PAD           = 4  # BBox 주변 여유 픽셀 (알약 경계 포함 + 마스크 품질 향상)

    for cls_name in tqdm(minority_classes, desc='스티커 추출'):
        # 파일명에 사용 불가한 특수문자 치환
        safe_name = str(cls_name).replace('/', '_').replace(' ', '_').replace(':', '_')
        cls_dir   = os.path.join(crop_save_dir, safe_name)
        os.makedirs(cls_dir, exist_ok=True)

        cls_annos = annotations_df[annotations_df['class_name'] == cls_name]

        for _, anno in cls_annos.iterrows():
            # 해당 annotation의 이미지 정보 조회
            img_info = images_df[images_df['id'] == anno['image_id']]
            if img_info.empty:
                continue

            f_name = os.path.basename(img_info.iloc[0]['file_name'])
            if f_name not in path_map:
                continue

            # 한글/네트워크 드라이브 경로 대응을 위해 np.fromfile 사용
            img = cv2.imdecode(np.fromfile(path_map[f_name], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # BBox에 PAD 적용 + 이미지 경계 클리핑
            x, y, w, h = map(int, anno['bbox'])
            ih, iw     = img.shape[:2]
            x1 = max(0, x - PAD);      y1 = max(0, y - PAD)
            x2 = min(iw, x + w + PAD); y2 = min(ih, y + h + PAD)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 마스크 유효성 검증
            # pill_ratio가 정상 범위를 벗어나면 저장하지 않고 스킵
            mask       = make_pill_mask(crop)
            pill_ratio = float(mask.mean())
            if pill_ratio < 0.35 or pill_ratio > 0.85:
                skipped += 1
                continue

            # PNG로 저장 (무손실 — crop 품질 보존)
            save_name = f"{safe_name}_{int(anno['id'])}.png"
            save_path = os.path.join(cls_dir, save_name)
            ok, enc   = cv2.imencode('.png', crop)
            if ok:
                with open(save_path, 'w+b') as f:
                    enc.tofile(f)

            crop_metadata.append({
                'class_name':  cls_name,
                'category_id': anno['category_id'],
                'crop_path':   save_path,
                'width':       x2 - x1,
                'height':      y2 - y1,
                'pill_ratio':  round(pill_ratio, 4),
            })

    print(f"\n✅ 스티커 추출 완료 : {len(crop_metadata):,}개")
    print(f"   마스크 실패 제외  : {skipped}개")

    if crop_metadata:
        meta_df = pd.DataFrame(crop_metadata)
        meta_df.to_csv(
            os.path.join(crop_save_dir, 'crop_meta.csv'),
            index=False, encoding='utf-8-sig'
        )
        print(f"   pill_ratio 평균  : {meta_df['pill_ratio'].mean():.3f}")
        print(f"   pill_ratio 분포  : min={meta_df['pill_ratio'].min():.3f} "
              f"/ max={meta_df['pill_ratio'].max():.3f}")
    else:
        print("⚠️  추출된 스티커가 없습니다. 경로 및 JSON을 확인하세요.")
        


def generate_plain_backgrounds(base_dir, n=200):
    """
    원본 이미지에서 알약 없는 배경 영역만 추출하여
    순수 배경 이미지를 생성합니다.
    BBox 영역을 마스킹하고 inpaint로 채워서 알약 흔적 제거.
    """
    import cv2, numpy as np, json, os, glob, random
    
    with open(os.path.join(base_dir, 'train_raw.json')) as f:
        coco = json.load(f)
    
    ann_map = defaultdict(list)
    for ann in coco['annotations']:
        ann_map[ann['image_id']].append(ann['bbox'])
    
    all_files = {os.path.basename(f): f
                 for f in glob.glob(os.path.join(base_dir, '**', '*'), recursive=True)
                 if os.path.isfile(f)}
    
    bg_dir = os.path.join(base_dir, 'plain_backgrounds')
    os.makedirs(bg_dir, exist_ok=True)
    
    imgs = random.sample(coco['images'], min(n, len(coco['images'])))
    saved = 0
    
    for img_info in imgs:
        fname = os.path.basename(img_info['file_name'])
        if fname not in all_files:
            continue
        
        img = cv2.imdecode(np.fromfile(all_files[fname], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        # BBox 영역을 마스킹
        mask = np.zeros(img.shape[:2], np.uint8)
        for x, y, w, h in ann_map[img_info['id']]:
            x1, y1 = max(0, int(x)-10), max(0, int(y)-10)
            x2, y2 = min(img.shape[1], int(x+w)+10), min(img.shape[0], int(y+h)+10)
            mask[y1:y2, x1:x2] = 255
        
        # inpaint으로 알약 영역을 배경색으로 채우기
        bg = cv2.inpaint(img, mask, 15, cv2.INPAINT_TELEA)
        
        save_path = os.path.join(bg_dir, f'bg_{saved:04d}.jpg')
        ok, enc = cv2.imencode('.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            with open(save_path, 'w+b') as f:
                enc.tofile(f)
            saved += 1
    
    print(f"✅ 순수 배경 {saved}장 생성 완료 → {bg_dir}")
    return bg_dir



# ============================================================
# Step 2: Copy-Paste 증강 실행
# ============================================================

def run_copy_paste(base_dir, aug_count=500, random_seed=42):
    """
    GrabCut 마스크 기반 Copy-Paste 증강 실행.

    처리 순서:
      1) train_raw.json + crop_meta.csv 로드
      2) 배경 이미지 랜덤 선택
      3) 소수 클래스 crop 랜덤 선택 + 랜덤 스케일 적용 (0.7~1.2배)
      4) make_pill_mask()로 마스크 재계산 (스케일 변경 반영)
      5) check_overlap()으로 빈 자리 탐색 (최대 50회)
      6) blend_with_mask()로 알약 픽셀만 배경에 합성
      7) 합성 이미지 저장 + COCO JSON 갱신

    v4 변경:
      - make_pill_mask()를 GrabCut 기반으로 교체
        → 흰색/민트색 알약 C자 구멍 문제 해결

    Args:
        base_dir    : 데이터셋 루트 경로
        aug_count   : 생성할 합성 이미지 수 (기본 500)
        random_seed : 재현성 시드 (기본 42)

    사전 조건:
        base_dir/train_raw.json
        base_dir/crops_minority/crop_meta.csv  ← extract_minority_crops() 결과

    출력:
        base_dir/train_augmented_images/       : 합성 이미지
        base_dir/train_augmented_final.json    : 증강된 COCO JSON
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    train_json_path = os.path.join(base_dir, 'train_raw.json')
    meta_path       = os.path.join(base_dir, 'crops_minority', 'crop_meta.csv')
    aug_img_dir     = os.path.join(base_dir, 'train_augmented_images')
    aug_json_path   = os.path.join(base_dir, 'train_augmented_final.json')
    os.makedirs(aug_img_dir, exist_ok=True)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"crop_meta.csv 없음: {meta_path}\n"
            f"extract_minority_crops() 를 먼저 실행하세요."
        )

    with open(train_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    aug_coco  = copy.deepcopy(coco_data)
    crop_meta = pd.read_csv(meta_path)

    all_files = (glob.glob(os.path.join(base_dir, '**', '*.png'), recursive=True) +
                 glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True) +
                 glob.glob(os.path.join(base_dir, '**', '*.JPG'), recursive=True))
    path_map  = {os.path.basename(f): f for f in all_files}

    max_img_id    = max((img['id'] for img in aug_coco['images']),      default=0)
    max_anno_id   = max((ann['id'] for ann in aug_coco['annotations']), default=0)
    bg_candidates = list(coco_data['images'])

    print(f"\n{'='*60}")
    print(f"[Step 2] Copy-Paste 증강  (목표: {aug_count}장)")
    print(f"{'='*60}")

    for _ in tqdm(range(aug_count), desc='Copy-Paste 증강'):
        bg_info = random.choice(bg_candidates)
        f_name  = os.path.basename(bg_info['file_name'])
        if f_name not in path_map:
            continue

        bg_img = cv2.imdecode(np.fromfile(path_map[f_name], np.uint8), cv2.IMREAD_COLOR)
        if bg_img is None:
            continue

        bg_h, bg_w     = bg_img.shape[:2]
        existing_boxes = [
            ann['bbox'] for ann in aug_coco['annotations']
            if ann['image_id'] == bg_info['id']
        ]

        num_pastes = random.randint(1, 4)
        pastes     = crop_meta.sample(num_pastes, replace=True)
        new_anns   = []
        success    = False

        for _, row in pastes.iterrows():
            crop_img = cv2.imdecode(np.fromfile(row['crop_path'], np.uint8), cv2.IMREAD_COLOR)
            if crop_img is None:
                continue

            # 랜덤 스케일 적용
            scale = random.uniform(0.7, 1.2)
            cw    = max(20, int(int(row['width'])  * scale))
            ch    = max(20, int(int(row['height']) * scale))
            if cw != int(row['width']) or ch != int(row['height']):
                crop_img = cv2.resize(crop_img, (cw, ch), interpolation=cv2.INTER_LINEAR)

            # 스케일 후 마스크 재계산
            mask = make_pill_mask(crop_img)

            # 빈 자리 탐색 (최대 50회)
            safe_x = safe_y = -1
            for _ in range(50):
                max_x = bg_w - cw - 10
                max_y = bg_h - ch - 10
                if max_x < 10 or max_y < 10:
                    break
                rx, ry = random.randint(10, max_x), random.randint(10, max_y)
                if not check_overlap((rx, ry, cw, ch), existing_boxes, min_dist=30):
                    safe_x, safe_y = rx, ry
                    break

            if safe_x == -1 or safe_x + cw > bg_w or safe_y + ch > bg_h:
                continue

            # 마스크 기반 합성
            blend_with_mask(bg_img, crop_img, mask, safe_x, safe_y)
            existing_boxes.append([safe_x, safe_y, cw, ch])

            max_anno_id += 1
            new_anns.append({
                'id':           max_anno_id,
                'image_id':     max_img_id + 1,
                'category_id':  int(row['category_id']),
                'bbox':         [safe_x, safe_y, cw, ch],
                'area':         float(cw * ch),
                'iscrowd':      0,
                'segmentation': [],
            })
            success = True

        if not success:
            continue

        max_img_id   += 1
        new_filename  = f'aug_cp_{max_img_id:06d}.jpg'
        ok, enc       = cv2.imencode('.jpg', bg_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            with open(os.path.join(aug_img_dir, new_filename), 'w+b') as f:
                enc.tofile(f)

        aug_coco['images'].append({
            'id':        max_img_id,
            'file_name': new_filename,
            'width':     bg_w,
            'height':    bg_h,
            'source_file': f_name,  
        })

        for ann in coco_data['annotations']:
            if ann['image_id'] == bg_info['id']:
                max_anno_id       += 1
                cloned             = copy.deepcopy(ann)
                cloned['id']       = max_anno_id
                cloned['image_id'] = max_img_id
                aug_coco['annotations'].append(cloned)

        aug_coco['annotations'].extend(new_anns)

    with open(aug_json_path, 'w', encoding='utf-8') as f:
        json.dump(aug_coco, f, ensure_ascii=False)

    n_orig = len(coco_data['annotations'])
    n_aug  = len(aug_coco['annotations'])
    print(f"\n✅ Copy-Paste 증강 완료")
    print(f"   이미지: {len(coco_data['images']):,}장 → {len(aug_coco['images']):,}장")
    print(f"   객체  : {n_orig:,}개 → {n_aug:,}개  (+{n_aug - n_orig:,}개)")
    print(f"   저장  : {aug_json_path}")
