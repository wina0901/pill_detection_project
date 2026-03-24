"""
viz_utils.py
============
노트북 공용 시각화 유틸리티

모든 NB에서 import해서 사용합니다.
  from src.preprocessing.viz_utils import show_samples, show_augmented_samples, show_mask_preview
"""

import os
import glob
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict


def show_samples(img_dir, json_path=None, n=20, title="Sample Images",
                 show_bbox=True, figsize_per=4):
    """
    폴더에서 n장 랜덤 샘플링하여 BBox와 함께 시각화합니다.

    Args:
        img_dir    : 이미지 폴더 경로
        json_path  : COCO JSON 경로 (None이면 BBox 없이 표시)
        n          : 표시할 이미지 수 (기본 20)
        title      : 제목
        show_bbox  : BBox 표시 여부
        figsize_per: 이미지 한 장당 크기 (인치)
    """
    imgs = (glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.png')))
    if not imgs:
        print(f"⚠️  이미지 없음: {img_dir}")
        return

    sample = random.sample(imgs, min(n, len(imgs)))
    cols   = 5
    rows   = (len(sample) + cols - 1) // cols

    # BBox 로딩
    bbox_map = defaultdict(list)
    cat_map  = {}
    if json_path and os.path.exists(json_path) and show_bbox:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        cat_map     = {c['id']: c['name'] for c in coco['categories']}
        fname_to_id = {os.path.basename(img['file_name']): img['id']
                       for img in coco['images']}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            bbox_map[img_id].append((ann['bbox'], ann['category_id']))
        # fname → anns 재매핑
        fname_anns = {}
        for fname, iid in fname_to_id.items():
            fname_anns[fname] = bbox_map[iid]
        bbox_map = fname_anns

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per, rows * figsize_per))
    axes      = np.array(axes).flatten()

    colors = plt.cm.Set1.colors

    for i, img_path in enumerate(sample):
        img_arr = np.fromfile(img_path, np.uint8)
        img     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            axes[i].axis('off')
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)

        fname = os.path.basename(img_path)
        if fname in bbox_map:
            for (x, y, w, h), cat_id in bbox_map[fname]:
                color = colors[cat_id % len(colors)]
                rect  = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1.5, edgecolor=color, facecolor='none'
                )
                axes[i].add_patch(rect)
                label = cat_map.get(cat_id, str(cat_id))
                axes[i].text(x, y - 2, label[:8], fontsize=5,
                             color='white',
                             bbox=dict(boxstyle='round,pad=0.1', fc=color, alpha=0.7))

        axes[i].set_title(fname[:18], fontsize=7)
        axes[i].axis('off')

    for i in range(len(sample), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"{title}  ({len(sample)}/{len(imgs)}장)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    print(f"✅ {title}: {len(imgs)}장 중 {len(sample)}장 표시")


def show_augmented_samples(aug_img_dir, aug_json_path, n=20):
    """
    Copy-Paste 증강 결과 이미지를 BBox와 함께 시각화합니다.
    aug_cp_ 로 시작하는 이미지만 보여줍니다.
    """
    imgs = glob.glob(os.path.join(aug_img_dir, 'aug_cp_*.jpg'))
    if not imgs:
        print(f"⚠️  증강 이미지 없음: {aug_img_dir}")
        return

    print(f"✅ 생성된 aug_cp 이미지: {len(imgs)}장")
    show_samples(aug_img_dir, aug_json_path, n=n,
                 title="Copy-Paste 증강 결과 (aug_cp_*)")


def show_mask_preview(crop_dir, n=10):
    """
    extract_minority_crops 결과로 생성된 스티커 crop에
    make_pill_mask를 적용한 결과를 시각화합니다.

    3열 구성:
      [1열] 원본 crop + 클래스명 (폴더명에서 추출)
      [2열] 마스크 흑백 + avg 값 (필터 기준 밖이면 빨간색)
      [3열] 마스크 적용 결과 (알약 픽셀만 남긴 결과)

    마스크 품질 판단 기준 (extract_minority_crops 필터와 동일):
      avg < 0.35  → 배경 제거 실패 → 빨간색 (자동 스킵됨)
      0.35~0.85   → 정상 범위     → 검정색
      avg > 0.85  → 이상 crop     → 빨간색 (자동 스킵됨)

    Args:
        crop_dir : crops_minority/ 폴더 경로
        n        : 랜덤 샘플링할 crop 수 (기본 10)
    """
    # 내부에서 import (노트북 환경 호환)
    try:
        from src.preprocessing.augmentation import make_pill_mask
    except ImportError:
        from augmentation import make_pill_mask

    # Mac 한글 폰트 설정 (AppleGothic)
    # 없으면 클래스명이 □□□로 깨져서 출력됨
    plt.rcParams['font.family']        = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    crops = glob.glob(os.path.join(crop_dir, '**', '*.png'), recursive=True)
    if not crops:
        print(f"⚠️  crop 없음: {crop_dir}")
        return

    sample = random.sample(crops, min(n, len(crops)))

    fig, axes = plt.subplots(len(sample), 3,
                             figsize=(12, len(sample) * 3))
    # n=1일 때 axes가 1D로 반환되므로 2D 리스트로 통일
    if len(sample) == 1:
        axes = [axes]

    for i, crop_path in enumerate(sample):
        img = cv2.imdecode(np.fromfile(crop_path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # 마스크 생성 및 알약 픽셀만 추출
        mask   = make_pill_mask(img)
        mask3  = np.stack([mask, mask, mask], axis=-1)              # (H,W) → (H,W,3)
        masked = (img.astype(np.float32) * mask3).astype(np.uint8)  # 알약 픽셀만 남김

        # 클래스명: crops_minority/{클래스명}/{파일}.png 구조에서 상위 폴더명 추출
        cls_name = os.path.basename(os.path.dirname(crop_path))

        # [1열] 원본 crop + 클래스명
        axes[i][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(cls_name, fontsize=8)

        # [2열] 마스크 흑백 + avg 값
        # 필터 기준(0.35~0.85) 밖이면 빨간색으로 경고 표시
        avg   = float(mask.mean())
        color = 'red' if avg < 0.35 or avg > 0.85 else 'black'
        axes[i][1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(f'마스크 (avg={avg:.2f})', fontsize=8, color=color)

        # [3열] 마스크 적용 결과
        axes[i][2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        axes[i][2].set_title('마스크 적용', fontsize=8)

        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle("알약 마스크 추출 결과", fontsize=13)
    plt.tight_layout()
    plt.show()
    print(f"✅ {len(sample)}개 crop 시각화 완료  (빨간 avg = 스킵 대상)")


def show_class_distribution(json_path, title="클래스 분포", top_n=30):
    """
    COCO JSON에서 클래스별 객체 수를 막대그래프로 시각화합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    cat_map = {c['id']: c['name'] for c in coco['categories']}
    counts  = defaultdict(int)
    for ann in coco['annotations']:
        counts[cat_map.get(ann['category_id'], str(ann['category_id']))] += 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names  = [x[0] for x in sorted_items[:top_n]]
    values = [x[1] for x in sorted_items[:top_n]]

    plt.figure(figsize=(16, 5))
    plt.bar(range(len(names)), values, color='steelblue', edgecolor='white')
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=8)
    plt.ylabel('객체 수')
    plt.title(f"{title}  (상위 {top_n}개 클래스)")

    if values:
        plt.axhline(y=min(values), color='red',   linestyle='--', alpha=0.5,
                    label=f'최소 {min(values)}')
        plt.axhline(y=50,          color='orange', linestyle='--', alpha=0.5,
                    label='소수 클래스 기준 (50)')
        plt.legend()

    plt.tight_layout()
    plt.show()
    print(f"총 클래스: {len(counts)}개 | 총 객체: {sum(counts.values()):,}개")
    print(f"최소: {min(counts.values())}개  최대: {max(counts.values())}개  "
          f"평균: {sum(counts.values()) / len(counts):.1f}개")


def show_letterbox_comparison(orig_img_dir, lb_img_dir, orig_json,
                               lb_json, n=5):
    """
    Letterbox 적용 전후 비교를 시각화합니다.
    왼쪽: 원본 | 오른쪽: Letterbox 적용 결과

    Args:
        orig_img_dir : 원본 이미지 폴더
        lb_img_dir   : Letterbox 결과 폴더
        orig_json    : 원본 COCO JSON
        lb_json      : Letterbox 결과 COCO JSON
        n            : 비교할 쌍의 수
    """
    lb_imgs = glob.glob(os.path.join(lb_img_dir, '*.jpg'))[:n]
    if not lb_imgs:
        print(f"⚠️  Letterbox 이미지 없음: {lb_img_dir}")
        return

    with open(orig_json, 'r', encoding='utf-8') as f:
        orig_coco = json.load(f)
    with open(lb_json, 'r', encoding='utf-8') as f:
        lb_coco = json.load(f)

    orig_id_to_fname = {img['id']: img['file_name'] for img in orig_coco['images']}

    lb_anns   = defaultdict(list)
    orig_anns = defaultdict(list)
    for ann in lb_coco['annotations']:
        lb_anns[ann['image_id']].append(ann['bbox'])
    for ann in orig_coco['annotations']:
        orig_anns[ann['image_id']].append(ann['bbox'])

    all_orig = {os.path.basename(f): f
                for f in glob.glob(os.path.join(orig_img_dir, '**', '*'), recursive=True)
                if os.path.isfile(f)}

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 5))
    if n == 1:
        axes = [axes]

    for i, lb_path in enumerate(lb_imgs[:n]):
        lb_fname = os.path.basename(lb_path)
        # lb_xxxxxx.jpg → image id
        img_id   = int(lb_fname.replace('lb_', '').replace('.jpg', ''))

        # [우측] Letterbox 결과
        lb_img = cv2.imdecode(np.fromfile(lb_path, np.uint8), cv2.IMREAD_COLOR)
        if lb_img is not None:
            axes[i][1].imshow(cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB))
            for x, y, w, h in lb_anns[img_id]:
                axes[i][1].add_patch(patches.Rectangle(
                    (x, y), w, h, lw=1.5, edgecolor='lime', facecolor='none'))
            axes[i][1].set_title(f'Letterbox 800×800\n{lb_fname}', fontsize=9)

        # [좌측] 원본
        orig_fname = os.path.basename(orig_id_to_fname.get(img_id, ''))
        if orig_fname in all_orig:
            orig_img = cv2.imdecode(np.fromfile(all_orig[orig_fname], np.uint8),
                                    cv2.IMREAD_COLOR)
            if orig_img is not None:
                oh, ow = orig_img.shape[:2]
                axes[i][0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                for x, y, w, h in orig_anns[img_id]:
                    axes[i][0].add_patch(patches.Rectangle(
                        (x, y), w, h, lw=1.5, edgecolor='red', facecolor='none'))
                axes[i][0].set_title(f'원본 ({ow}×{oh})\n{orig_fname}', fontsize=9)

        for ax in axes[i]:
            ax.axis('off')

    plt.suptitle("Letterbox 변환 전후 비교", fontsize=13)
    plt.tight_layout()
    plt.show()


def show_aug_vs_original(aug_img_dir, aug_json_path, orig_img_dir, n=5):
    """
    증강 이미지(우)와 원본 배경 이미지(좌)를 나란히 비교합니다.
    겹쳐보이는 알약이 원본에 있던 건지, 새로 합성된 건지 확인용.
    """
    plt.rcParams['font.family']        = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    with open(aug_json_path, 'r', encoding='utf-8') as f:
        aug_coco = json.load(f)

    # source_file 있는 이미지만 (aug_cp_ 이미지)
    aug_imgs = [img for img in aug_coco['images']
                if 'source_file' in img and img['file_name'].startswith('aug_cp_')]
    if not aug_imgs:
        print("⚠️  source_file 정보 없음 — run_copy_paste()를 다시 돌려야 해요.")
        return

    cat_map  = {c['id']: c['name'] for c in aug_coco['categories']}
    ann_map  = defaultdict(list)
    for ann in aug_coco['annotations']:
        ann_map[ann['image_id']].append(ann)

    # 원본 이미지 경로 해시맵
    all_orig = {os.path.basename(f): f
                for f in glob.glob(os.path.join(orig_img_dir, '**', '*'), recursive=True)
                if os.path.isfile(f)}

    sample = random.sample(aug_imgs, min(n, len(aug_imgs)))
    colors = plt.cm.Set1.colors

    fig, axes = plt.subplots(n, 2, figsize=(14, n * 6))
    if n == 1:
        axes = [axes]

    for i, img_info in enumerate(sample):
        aug_path  = os.path.join(aug_img_dir, img_info['file_name'])
        orig_name = img_info['source_file']

        # [좌] 원본 배경
        if orig_name in all_orig:
            orig_img = cv2.imdecode(np.fromfile(all_orig[orig_name], np.uint8), cv2.IMREAD_COLOR)
            if orig_img is not None:
                axes[i][0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f'원본 배경\n{orig_name}', fontsize=9)
        axes[i][0].axis('off')

        # [우] 증강 결과 + BBox
        aug_img = cv2.imdecode(np.fromfile(aug_path, np.uint8), cv2.IMREAD_COLOR)
        if aug_img is not None:
            axes[i][1].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            for ann in ann_map[img_info['id']]:
                x, y, w, h = ann['bbox']
                color = colors[ann['category_id'] % len(colors)]
                axes[i][1].add_patch(patches.Rectangle(
                    (x, y), w, h, lw=1.5, edgecolor=color, facecolor='none'))
                name = cat_map.get(ann['category_id'], '')
                axes[i][1].text(x, y-2, name[:8], fontsize=5, color='white',
                                bbox=dict(fc=color, alpha=0.7, pad=0.1))
        axes[i][1].set_title(f'증강 결과\n{img_info["file_name"]}', fontsize=9)
        axes[i][1].axis('off')

    plt.suptitle("원본 배경 vs 증강 결과 비교", fontsize=13)
    plt.tight_layout()
    plt.show()