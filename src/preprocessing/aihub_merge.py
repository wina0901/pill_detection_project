"""
aihub_merge.py
==============
AI허브 경구약제 단일 데이터를 우리 팀 merged_annotations_train_final.json에 병합합니다.
병합 후 NB02를 재실행하면 train_raw.json / val.json 이 자동으로 재생성됩니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[왜 백업본을 읽어서 merged에 저장하는가?]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  이 스크립트는 항상 원본(1489장) 기준으로 새로 쌓습니다.
  이미 AI허브 데이터가 섞인 현재 merged에 또 append하면
  기존 클래스가 중복 누적되기 때문입니다.

  INPUT  : merged_annotations_train_final_backup1.json  ← 원본 1489장
  OUTPUT : merged_annotations_train_final.json          ← 1489 + 22클래스 병합본
  NEXT   : NB02 재실행 → train_raw.json / val.json 재생성

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[중요] category_id 3단계 변환 흐름
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AI허브 코드  →  -1  →  우리 팀 category_id  →  +1  →  캐글 제출값

  [기존 4개 클래스 - TL_27/30/45/69]
     4543                     4542                        4543  (에어탈정)
     6192                     6191                        6192  (삐콤씨에프정)
    16688                    16687                       16688  (오마코연질캡슐)
    31863                    31862                       31863  (아질렉트정)

  [신규 18개 클래스]
  TL_7
    12420                    12419                       12420  (자이프렉사정 2.5mg)
    13395                    13394                       13395  (써스펜8시간이알서방정)
    12081                    12080                       12081  (리렉스펜정 300mg/PTP)
  TL_10
    20014                    20013                       20014  (마도파정200)
    22362                    22361                       22362  (맥시부펜이알정 300mg)
    19861                    19860                       19861  (노바스크정 5mg)
    20877                    20876                       20877  (엑스포지정 5/160mg)
  TL_13
    29871                    29870                       29871  (렉사프로정 15mg)
    29451                    29450                       29451  (레일라정)
    29345                    29344                       29345  (비모보정 500/20mg)
    27926                    27925                       27926  (울트라셋이알서방정)
    27733                    27732                       27733  (트윈스타정 40/5mg)
  TL_15
    33880                    33879                       33880  (글리틴정)
    33208                    33207                       33208  (에스원엠프정 20mg)
    33009                    33008                       33009  (신바로정)
  TL_53
    23203                    23202                       23203  (쿠에타핀정 25mg)
    22627                    22626                       22627  (메가파워정 90mg/병)
    23223                    23222                       23223  (비타비백정 100mg/병)

  ※ 캐글 제출값이 AI허브 코드와 우연히 같아 보이지만,
     반드시 train JSON category_id 기준으로만 처리합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[중요] 모델별 내부 인덱스 변환 (추론 시 참고)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  YOLO:
    학습: category_id → sorted 후 0-based index (format_converter.py)
    추론: yolo2cat[cls_idx] → category_id 복원 → +1 → 캐글 제출

  Faster R-CNN / RetinaNet:
    학습: category_id → 1-based index (0=background 예약, dataset.py)
    추론: model2orig[label] → category_id 복원 → +1 → 캐글 제출

  ※ 앙상블 시 서브미션 CSV의 category_id 기준으로 통일

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[중요] 이미지 품질 우선순위 샘플링
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  파일명 구조: K-XXXXXX_?_[back]_[dir]_[light]_la_lo_size.png
  split('_') 결과: ['K-XXXXXX', '?', back, dir, light, ...]
                      [0]       [1]  [2]  [3]  [4]

    [2] 배경:  0=검은색, 1=파랑색, 2=연회색(밝은배경)
    [3] 앞/뒤: 0=앞면, 1=뒷면  ← 뒷면은 모든 순위에서 제외
    [4] 조명:  0=전구색, 1=주광색, 2=밝은조명

  우선순위 (PER_CLASS_LIMIT개 채울 때 순서):
    1순위: [2]=2, [3]=0, [4]=2  → 밝은배경 + 앞면 + 밝은조명  (최우선)
    2순위: [2]=*, [3]=0, [4]=2  → 앞면 + 밝은조명, 배경 무관
    3순위: [2]=*, [3]=0, [4]=1  → 앞면 + 주광색, 배경 무관
    4순위: [2]=2, [3]=0, [4]=0  → 밝은배경 + 앞면 + 전구색
    제외:  [3]=1               → 뒷면은 무조건 제외

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[중요] 폴더 없을 때 동작
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  라벨링(TL) 또는 원천(TS) 폴더가 없으면 해당 폴더를 스킵합니다.
  확보된 클래스만 병합하고, 없는 클래스는 조용히 넘어갑니다.
  DRY_RUN=True 출력에서 실제로 파싱된 클래스 목록을 확인하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
처리 순서
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1) INPUT_JSON(원본 1489장) 로드  ← 실제로는 4번에서 일어남
  2) AI허브 라벨링 JSON 파싱       ← 실제로는 1번에서 일어남
  3) 클래스당 PER_CLASS_LIMIT개를 우선순위 순으로 샘플링
  4) image_id / ann_id 원본 최대값 + 1부터 재부여
  5) 이미지 파일을 train_images/ 로 복사
  6) OUTPUT_JSON(merged_annotations_train_final.json)으로 저장
     - 저장 전 OUTPUT_JSON 자동 백업 (번호 자동 증가)
  7) NB02 재실행하여 train_raw.json / val.json 재생성

사용법 (Colab):
  # Step 1: 통계만 확인 (DRY_RUN = True, 기본값)
  !python aihub_merge.py

  # Step 2: DRY_RUN = False 로 바꾼 후 실제 실행
  !python aihub_merge.py
"""

import os
import json
import glob
import random
import shutil
from collections import defaultdict, Counter
from tqdm.auto import tqdm


# ──────────────────────────────────────────────────────────────
# AI허브 코드 → 우리 팀 category_id 매핑 (AI허브 코드 - 1 = 우리 팀 ID)
#
# 총 22개 클래스 (기존 4개 + 신규 18개)
# 라벨링/원천 폴더가 없는 클래스는 파싱 시 자동 스킵됩니다.
# ──────────────────────────────────────────────────────────────
AIHUB_TO_OURS = {
    # ── 기존 4개 클래스 (TL_27/30/45/69) ──────────────────────
     4543:  4542,  # 에어탈정(아세클로페낙)
     6192:  6191,  # 삐콤씨에프정 618.6mg/병
    16688: 16687,  # 오마코연질캡슐(오메가-3-산에틸에스테르90)
    31863: 31862,  # 아질렉트정(라사길린메실산염)

    # ── 신규 18개 클래스 ───────────────────────────────────────
    # TL_7
    12420: 12419,  # 자이프렉사정 2.5mg
    13395: 13394,  # 써스펜8시간이알서방정
    12081: 12080,  # 리렉스펜정 300mg/PTP
    # TL_10
    20014: 20013,  # 마도파정200
    22362: 22361,  # 맥시부펜이알정 300mg
    19861: 19860,  # 노바스크정 5mg
    20877: 20876,  # 엑스포지정 5/160mg
    # TL_13
    29871: 29870,  # 렉사프로정 15mg
    29451: 29450,  # 레일라정
    29345: 29344,  # 비모보정 500/20mg
    27926: 27925,  # 울트라셋이알서방정
    27733: 27732,  # 트윈스타정 40/5mg
    # TL_15
    33880: 33879,  # 글리틴정(콜린알포세레이트)
    33208: 33207,  # 에스원엠프정 20mg
    33009: 33008,  # 신바로정
    # TL_53
    23203: 23202,  # 쿠에타핀정 25mg
    22627: 22626,  # 메가파워정 90mg/병
    23223: 23222,  # 비타비백정 100mg/병
}
TARGET_AIHUB_CODES = set(AIHUB_TO_OURS.keys())

# 클래스당 최대 추가 이미지 수
PER_CLASS_LIMIT = 150

# AI허브 코드 기준 이름 (로그 출력용)
NAME_MAP_CODE = {
     4543: '에어탈정(아세클로페낙)',
     6192: '삐콤씨에프정 618.6mg/병',
    16688: '오마코연질캡슐',
    31863: '아질렉트정(라사길린메실산염)',
    12420: '자이프렉사정 2.5mg',
    13395: '써스펜8시간이알서방정',
    12081: '리렉스펜정 300mg/PTP',
    20014: '마도파정200',
    22362: '맥시부펜이알정 300mg',
    19861: '노바스크정 5mg',
    20877: '엑스포지정 5/160mg',
    29871: '렉사프로정 15mg',
    29451: '레일라정',
    29345: '비모보정 500/20mg',
    27926: '울트라셋이알서방정',
    27733: '트윈스타정 40/5mg',
    33880: '글리틴정(콜린알포세레이트)',
    33208: '에스원엠프정 20mg',
    33009: '신바로정',
    23203: '쿠에타핀정 25mg',
    22627: '메가파워정 90mg/병',
    23223: '비타비백정 100mg/병',
}

# 우리 팀 category_id 기준 이름 (결과 요약 출력용)
NAME_MAP_CAT = {
     4542: '에어탈정(아세클로페낙)',
     6191: '삐콤씨에프정 618.6mg/병',
    16687: '오마코연질캡슐',
    31862: '아질렉트정(라사길린메실산염)',
    12419: '자이프렉사정 2.5mg',
    13394: '써스펜8시간이알서방정',
    12080: '리렉스펜정 300mg/PTP',
    20013: '마도파정200',
    22361: '맥시부펜이알정 300mg',
    19860: '노바스크정 5mg',
    20876: '엑스포지정 5/160mg',
    29870: '렉사프로정 15mg',
    29450: '레일라정',
    29344: '비모보정 500/20mg',
    27925: '울트라셋이알서방정',
    27732: '트윈스타정 40/5mg',
    33879: '글리틴정(콜린알포세레이트)',
    33207: '에스원엠프정 20mg',
    33008: '신바로정',
    23202: '쿠에타핀정 25mg',
    22626: '메가파워정 90mg/병',
    23222: '비타비백정 100mg/병',
}


# ──────────────────────────────────────────────────────────────
# 파일명 파싱 및 우선순위 계산
#
# AI허브 파일명 구조:
#   K-023203_0_0_0_0_60_000_200.png
#   split('_') → ['K-023203', '0', '0', '0', '0', '60', '000', '200']
#                    [0]      [1]  [2]  [3]  [4]
#                              ?  back  dir light
#
# TL JSON 샘플로 검증한 매핑:
#   [2] 배경(back_color): 0=검은색, 1=파랑색, 2=연회색(밝은배경)
#   [3] 앞/뒤(drug_dir):  0=앞면,   1=뒷면
#   [4] 조명(light_color):0=전구색, 1=주광색, 2=밝은조명
# ──────────────────────────────────────────────────────────────
def get_priority(file_name: str) -> int | None:
    """
    파일명에서 배경/앞뒤/조명을 추출하여 이미지 품질 우선순위를 반환합니다.

    Returns:
        1 : 밝은배경([2]=2) + 앞면([3]=0) + 밝은조명([4]=2)  ← 최우선
        2 : 앞면([3]=0) + 밝은조명([4]=2), 배경 무관
        3 : 앞면([3]=0) + 주광색([4]=1), 배경 무관
        4 : 밝은배경([2]=2) + 앞면([3]=0) + 전구색([4]=0)
        None: 뒷면([3]=1) 또는 파싱 실패 → 제외
    """
    try:
        stem  = os.path.splitext(os.path.basename(file_name))[0]
        parts = stem.split('_')
        back   = int(parts[2])  # 배경:  0=검정, 1=파랑, 2=연회색
        direcc = int(parts[3])  # 앞/뒤: 0=앞면, 1=뒷면
        light  = int(parts[4])  # 조명:  0=전구색, 1=주광색, 2=밝은조명
    except (IndexError, ValueError):
        return None

    if direcc == 1:                    # 뒷면 → 무조건 제외
        return None
    if back == 2 and light == 2:
        return 1                       # 1순위: 밝은배경 + 밝은조명
    if light == 2:
        return 2                       # 2순위: 밝은조명 (배경 무관)
    if light == 1:
        return 3                       # 3순위: 주광색 (배경 무관)
    if back == 2 and light == 0:
        return 4                       # 4순위: 밝은배경 + 전구색
    return None                        # 나머지 → 제외


# ──────────────────────────────────────────────────────────────
# AI허브 라벨링 폴더 파싱
# ──────────────────────────────────────────────────────────────
def parse_aihub_label_dir(label_dir: str) -> list:
    """
    AI허브 단일 라벨링 폴더를 파싱하여 레코드 리스트를 반환합니다.
    폴더가 존재하지 않으면 빈 리스트를 반환합니다 (graceful skip).

    폴더 구조:
      TL_N_단일/
        K-XXXXXX_json/
          K-XXXXXX_?_back_dir_light_la_lo_size.json  ← 이미지 1장당 JSON 1개

    Returns:
        list of dict:
            aihub_code : AI허브 약품 코드 (int)
            file_name  : 이미지 파일명 (str)
            width      : 이미지 너비 (int)
            height     : 이미지 높이 (int)
            bboxes     : [[x,y,w,h], ...] (list)
            priority   : 1~4 품질 우선순위 (int)
    """
    # 폴더 없으면 graceful skip (다운로드 안 된 TL 폴더 대응)
    if not os.path.exists(label_dir):
        print(f"  ⚠️  라벨링 폴더 없음 (스킵): {os.path.basename(label_dir)}")
        return []

    records = []

    # K-XXXXXX_json 형태의 서브폴더만 수집
    drug_folders = [
        d for d in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, d)) and d.endswith('_json')
    ]

    for drug_folder in tqdm(drug_folders, desc=f'파싱: {os.path.basename(label_dir)}'):
        # 폴더명 K-XXXXXX_json 에서 숫자(AI허브 코드) 추출
        code_str = drug_folder.replace('K-', '').replace('_json', '').split('_')[0]
        try:
            aihub_code = int(code_str)
        except ValueError:
            continue

        # 22개 타겟 클래스가 아니면 스킵
        if aihub_code not in TARGET_AIHUB_CODES:
            continue

        drug_path  = os.path.join(label_dir, drug_folder)
        json_files = glob.glob(os.path.join(drug_path, '*.json'))

        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            if not data.get('images'):
                continue

            img_info  = data['images'][0]
            file_name = img_info.get('file_name') or img_info.get('imgfile', '')
            if not file_name:
                continue

            # 품질 우선순위 계산 (뒷면이면 None → 스킵)
            priority = get_priority(file_name)
            if priority is None:
                continue

            bboxes = [ann['bbox'] for ann in data.get('annotations', []) if ann.get('bbox')]
            if not bboxes:
                continue

            records.append({
                'aihub_code': aihub_code,
                'file_name':  os.path.basename(file_name),
                'width':      img_info.get('width', 976),
                'height':     img_info.get('height', 1280),
                'bboxes':     bboxes,
                'priority':   priority,
            })

    print(f"  → {len(records)}개 레코드 파싱 완료 (뒷면 제외)")
    return records


# ──────────────────────────────────────────────────────────────
# 우선순위 기반 샘플링
# ──────────────────────────────────────────────────────────────
def priority_sample(records: list, limit: int) -> list:
    """
    우선순위(1→4) 순서로 limit개를 채웁니다.
    같은 우선순위 내에서는 랜덤 셔플 (seed는 main()에서 한 번만 고정).

    Args:
        records : 동일 클래스의 레코드 리스트 (priority 필드 포함)
        limit   : 최대 선택 개수 (PER_CLASS_LIMIT)

    Returns:
        선택된 레코드 리스트 (len <= limit)
    """
    by_priority = defaultdict(list)
    for r in records:
        by_priority[r['priority']].append(r)

    sampled = []
    for p in sorted(by_priority.keys()):      # 1순위 → 2순위 → 3순위 → 4순위
        if len(sampled) >= limit:
            break
        bucket = by_priority[p][:]
        random.shuffle(bucket)
        need = limit - len(sampled)
        sampled.extend(bucket[:need])

    return sampled


# ──────────────────────────────────────────────────────────────
# 백업 파일명 자동 생성 (번호 증가)
# ──────────────────────────────────────────────────────────────
def get_backup_path(json_path: str) -> str:
    """
    OUTPUT_JSON을 덮어쓰기 전에 기존 파일을 백업합니다.
    이미 백업이 있으면 번호를 올려서 새 경로를 반환합니다.
      예: _backup_1.json → _backup_2.json → _backup_3.json ...
    """
    base = json_path.replace('.json', '')
    n = 1
    while True:
        candidate = f"{base}_backup_{n}.json"
        if not os.path.exists(candidate):
            return candidate
        n += 1


# ──────────────────────────────────────────────────────────────
# 핵심 병합 함수
# ──────────────────────────────────────────────────────────────
def merge_into_json(input_json_path, output_json_path, all_records,
                    img_src_dirs, img_dst_dir, dry_run=False):
    """
    INPUT_JSON(원본)을 읽어 AI허브 레코드를 병합한 뒤 OUTPUT_JSON으로 저장합니다.

    - INPUT  = 원본 1489장 백업본 (중복 누적 방지)
    - OUTPUT = merged_annotations_train_final.json (병합 결과)
    - OUTPUT 저장 전 OUTPUT 파일을 자동 백업합니다.

    원천 이미지(TS) 폴더가 없는 클래스는 img_not_found로 스킵됩니다.
    """
    print(f"\n{'='*60}")
    print(f"INPUT  : {os.path.basename(input_json_path)}")
    print(f"OUTPUT : {os.path.basename(output_json_path)}")
    print(f"{'='*60}")

    # ── 원본 JSON 로드
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    max_image_id = max((img['id'] for img in coco['images']),      default=0)
    max_ann_id   = max((ann['id'] for ann in coco['annotations']), default=0)

    print(f"\n  원본 이미지 수:        {len(coco['images'])}장")
    print(f"  원본 annotation 수:    {len(coco['annotations'])}개")
    print(f"  원본 최대 image_id:    {max_image_id}")
    print(f"  원본 최대 ann_id:      {max_ann_id}")

    # ── 카테고리 검증: 병합할 클래스가 원본 categories에 있는지 확인
    # 없는 클래스는 경고만 출력하고 계속 진행 (graceful)
    existing_cats  = {c['id']: c['name'] for c in coco['categories']}
    our_cat_ids    = set(AIHUB_TO_OURS.values())
    missing_cats   = our_cat_ids - set(existing_cats.keys())
    available_cats = our_cat_ids - missing_cats

    if missing_cats:
        missing_names = [NAME_MAP_CAT.get(c, str(c)) for c in sorted(missing_cats)]
        print(f"\n  ⚠️  categories에 없는 클래스 {len(missing_cats)}개 (병합 스킵):")
        for cid, name in zip(sorted(missing_cats), missing_names):
            print(f"       category_id {cid}: {name}")
        print(f"  ✓  병합 가능한 클래스: {len(available_cats)}개")
    else:
        print(f"\n  ✓ 22개 클래스 모두 categories에 존재")

    # ── 기존 train_images 파일명 인덱싱 (중복 이미지 복사 방지)
    existing_fnames = set()
    if os.path.exists(img_dst_dir):
        existing_fnames = set(os.listdir(img_dst_dir))
    print(f"\n  기존 train_images 파일 수: {len(existing_fnames)}개")

    # ── 원천 이미지 인덱싱 (파일명 → 전체 경로 매핑)
    # 없는 TS 폴더는 자동 스킵 (원천 이미지 없는 클래스는 img_not_found 처리)
    print(f"  원천 이미지 인덱싱 중...")
    img_path_map = {}
    for src_dir in img_src_dirs:
        if not os.path.exists(src_dir):
            print(f"  ⚠️  원천 폴더 없음 (스킵): {os.path.basename(src_dir)}")
            continue
        for ext in ['*.png', '*.jpg', '*.JPG']:
            for fp in glob.glob(os.path.join(src_dir, '**', ext), recursive=True):
                img_path_map[os.path.basename(fp)] = fp
    print(f"  → {len(img_path_map)}개 이미지 인덱싱 완료")

    # ── 파일명 중복 사전 경고
    overlap = set(img_path_map.keys()) & existing_fnames
    if overlap:
        print(f"\n  ⚠️  파일명 중복 {len(overlap)}개 → 기존 파일 보존 (복사 스킵)")
    else:
        print(f"  ✓ 파일명 중복 없음")

    # ── 병합 처리
    stats = defaultdict(int)
    new_images, new_anns = [], []
    img_id = max_image_id + 1
    ann_id = max_ann_id + 1

    os.makedirs(img_dst_dir, exist_ok=True)

    for rec in tqdm(all_records, desc='병합'):
        fname   = rec['file_name']
        our_cat = AIHUB_TO_OURS[rec['aihub_code']]

        # categories에 없는 클래스면 스킵
        if our_cat not in existing_cats:
            stats['cat_missing'] += 1
            continue

        # 원천 이미지가 없으면 스킵
        if fname not in img_path_map:
            stats['img_not_found'] += 1
            continue

        # 이미지 복사 (중복이면 스킵)
        dst_path = os.path.join(img_dst_dir, fname)
        if not dry_run:
            if not os.path.exists(dst_path):
                shutil.copy2(img_path_map[fname], dst_path)
            else:
                stats['img_skipped_dup'] += 1

        new_images.append({
            'id':        img_id,
            'file_name': fname,
            'width':     rec['width'],
            'height':    rec['height'],
        })

        for bbox in rec['bboxes']:
            x, y, w, h = bbox
            new_anns.append({
                'id':           ann_id,
                'image_id':     img_id,
                'category_id':  our_cat,
                'bbox':         [float(x), float(y), float(w), float(h)],
                'area':         float(w * h),
                'iscrowd':      0,
                'segmentation': [],
            })
            ann_id += 1
            stats[f'ann_{our_cat}'] += 1

        img_id += 1
        stats['images_added'] += 1

    # ── 결과 요약
    print(f"\n  병합 결과:")
    print(f"    추가된 이미지:          {stats['images_added']}장")
    print(f"    이미지 없음 스킵:       {stats['img_not_found']}개")
    print(f"    카테고리 없음 스킵:     {stats['cat_missing']}개")
    print(f"    파일명 중복 스킵:       {stats['img_skipped_dup']}개")
    print(f"    추가된 annotation (클래스별):")
    for cat_id, name in sorted(NAME_MAP_CAT.items()):
        cnt = stats[f'ann_{cat_id}']
        if cnt > 0:
            print(f"      category {cat_id} ({name}): {cnt}개")
    print(f"    총 추가 annotation:     {len(new_anns)}개")

    if dry_run:
        print(f"\n  [dry_run=True] 파일 변경 없음.")
        print(f"  확인 완료 후 DRY_RUN = False 로 바꾸고 재실행하세요.")
        return

    # ── OUTPUT_JSON 백업 (덮어쓰기 전, 번호 자동 증가)
    if os.path.exists(output_json_path):
        backup_path = get_backup_path(output_json_path)
        shutil.copy2(output_json_path, backup_path)
        print(f"\n  백업 저장: {os.path.basename(backup_path)}")

    # ── OUTPUT_JSON 저장
    coco['images']      += new_images
    coco['annotations'] += new_anns

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\n  ✅ 저장 완료: {os.path.basename(output_json_path)}")
    print(f"     최종 이미지 수:     {len(coco['images'])}장")
    print(f"     최종 annotation 수: {len(coco['annotations'])}개")
    print(f"\n  ➡️  다음 단계: NB02 재실행 → train_raw.json / val.json 재생성")


# ══════════════════════════════════════════════════════════════
# 경로 설정 (여기만 수정하세요)
# ══════════════════════════════════════════════════════════════
BASE       = '/content/drive/MyDrive/data/초급_프로젝트/dataset'
AIHUB_BASE = f'{BASE}/data_add/aihub/train'

# INPUT: 원본 1489장 백업본 (AI허브 데이터가 섞이기 전 깨끗한 원본)
# → 항상 이 파일을 기준으로 읽어서 중복 누적을 방지합니다.
INPUT_JSON = f'{BASE}/merged_annotations_train_final_backup1.json'

# OUTPUT: 병합 결과를 저장할 파일 (NB02의 입력으로 사용됨)
# → 저장 전 자동 백업됩니다 (_backup_N.json)
OUTPUT_JSON = f'{BASE}/merged_annotations_train_final.json'

# 병합된 이미지 파일을 복사할 목적지
IMG_DST_DIR = f'{BASE}/train_images'

# AI허브 라벨링 폴더 목록
# → 폴더가 없어도 오류 없이 스킵됩니다 (graceful)
LABEL_DIRS = [
    # 기존 4개 클래스
    f'{AIHUB_BASE}/라벨링/TL_27_단일',
    f'{AIHUB_BASE}/라벨링/TL_30_단일',
    f'{AIHUB_BASE}/라벨링/TL_45_단일',
    f'{AIHUB_BASE}/라벨링/TL_69_단일',
    # 신규 18개 클래스
    f'{AIHUB_BASE}/라벨링/TL_7_단일',
    f'{AIHUB_BASE}/라벨링/TL_10_단일',
    f'{AIHUB_BASE}/라벨링/TL_13_단일',
    f'{AIHUB_BASE}/라벨링/TL_15_단일',
    f'{AIHUB_BASE}/라벨링/TL_53_단일',
]

# AI허브 원천 이미지 폴더 목록
# → 폴더가 없어도 오류 없이 스킵됩니다 (graceful)
IMG_SRC_DIRS = [
    # 기존 4개 클래스
    f'{AIHUB_BASE}/원천/TS_27_단일',
    f'{AIHUB_BASE}/원천/TS_30_단일',
    f'{AIHUB_BASE}/원천/TS_45_단일',
    f'{AIHUB_BASE}/원천/TS_69_단일',
    # 신규 18개 클래스
    f'{AIHUB_BASE}/원천/TS_7_단일',
    f'{AIHUB_BASE}/원천/TS_10_단일',
    f'{AIHUB_BASE}/원천/TS_13_단일',
    f'{AIHUB_BASE}/원천/TS_15_단일',
    f'{AIHUB_BASE}/원천/TS_53_단일',
]

# ✅ True : 통계만 출력, 파일 변경 없음 (먼저 이걸로 확인!)
# ✅ False: 실제 병합 실행
DRY_RUN = True


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    # 재현성 보장: 전체 샘플링에 걸쳐 seed를 한 번만 고정
    random.seed(42)

    # ── 1) 라벨링 파싱 (22개 클래스 시도, 없는 폴더 자동 스킵)
    all_records = []
    for label_dir in LABEL_DIRS:
        records = parse_aihub_label_dir(label_dir)
        all_records.extend(records)

    print(f"\n전체 파싱 레코드: {len(all_records)}개 (뒷면 제외)")

    if not all_records:
        print("⚠️  파싱된 레코드 없음. 경로와 폴더 구조를 확인하세요.")
        return

    # ── 2) 클래스별 파싱 통계 출력 (우선순위 분포 포함)
    by_code = defaultdict(list)
    for r in all_records:
        by_code[r['aihub_code']].append(r)

    print(f"\n클래스별 파싱 결과 (확보된 {len(by_code)}개 클래스):")
    for code, recs in sorted(by_code.items()):
        our_cat  = AIHUB_TO_OURS.get(code, '?')
        p_counts = Counter(r['priority'] for r in recs)
        p_str    = ', '.join(f"{p}순위:{p_counts[p]}개" for p in sorted(p_counts))
        print(f"  K-{code:06d} ({NAME_MAP_CODE.get(code,'?')}) → cat {our_cat}: 총 {len(recs)}개 [{p_str}]")

    # 확보 못 한 클래스 출력
    missing_codes = TARGET_AIHUB_CODES - set(by_code.keys())
    if missing_codes:
        print(f"\n  ⚠️  라벨링 데이터 없는 클래스 {len(missing_codes)}개 (스킵):")
        for code in sorted(missing_codes):
            print(f"       K-{code:06d} ({NAME_MAP_CODE.get(code,'?')})")

    # ── 3) 우선순위 기반 샘플링 (클래스당 PER_CLASS_LIMIT개)
    print(f"\n우선순위 샘플링 (PER_CLASS_LIMIT={PER_CLASS_LIMIT}):")
    sampled_records = []
    for code, recs in sorted(by_code.items()):
        sampled  = priority_sample(recs, PER_CLASS_LIMIT)
        our_cat  = AIHUB_TO_OURS.get(code, '?')
        p_counts = Counter(r['priority'] for r in sampled)
        p_str    = ', '.join(f"{p}순위:{p_counts[p]}개" for p in sorted(p_counts))
        print(f"  K-{code:06d} ({NAME_MAP_CODE.get(code,'?')}) → {len(recs)}개 중 {len(sampled)}개 선택 [{p_str}]")
        sampled_records.extend(sampled)

    print(f"\n최종 병합 대상: {len(sampled_records)}개 레코드")

    # ── 4) INPUT → OUTPUT 병합 실행
    if not os.path.exists(INPUT_JSON):
        print(f"\n🚨 INPUT_JSON 없음: {INPUT_JSON}")
        print("   merged_annotations_train_final_backup1.json 경로를 확인하세요.")
        return

    merge_into_json(
        input_json_path  = INPUT_JSON,
        output_json_path = OUTPUT_JSON,
        all_records      = sampled_records,
        img_src_dirs     = IMG_SRC_DIRS,
        img_dst_dir      = IMG_DST_DIR,
        dry_run          = DRY_RUN,
    )

    if not DRY_RUN:
        print(f"\n{'='*60}")
        print(f"🎉 병합 완료!")
        print(f"   다음 단계: NB02 재실행 → train_raw.json / val.json 재생성")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
