# RetinaNet 기반 의약품 객체 검출 실험 정리

## 1. 개요

본 저장소는 RetinaNet을 기반으로 의약품 이미지를 탐지하기 위한 실험을 정리한 작업물이다.  
저장소에 포함된 노트북들은 단순한 단일 학습 스크립트가 아니라, 데이터 점검부터 학습, 검증, 후처리, 제출 파일 생성, 시각화, 실험 비교까지 단계적으로 확장된 기록으로 구성되어 있다.

실행 관점에서는 `RetinaNet_best.ipynb`를 최종 실행용 노트북으로 보면 되고, `RetinaNet_research_style_v1.ipynb`는 여러 실험을 체계적으로 관리하기 위해 다시 정리한 통합 연구 노트북으로 이해하면 된다.  
`RetinaNet_v1.ipynb`부터 `RetinaNet_v10.ipynb`까지는 성능 개선 과정을 남긴 버전 히스토리다.

## 2. 작업 범위

이 노트북 묶음에서 다루는 범위는 아래와 같다.

- RetinaNet 기반 객체 검출 학습
- COCO 형식 annotation 사용
- letterbox 전처리 기반 학습 및 추론
- validation 기준 성능 평가
- threshold / NMS sweep
- submission CSV 생성
- 클래스별 AP 분석 및 예측 시각화
- 실험 비교, staged fine-tuning, 재현성 확인

## 3. 주요 노트북 안내

| 파일 | 역할 |
| --- | --- |
| `RetinaNet_best.ipynb` | 현재 기준 최종 실행용 노트북. 학습, 평가, threshold sweep, 추론, 시각화, 결과 저장까지 한 번에 수행한다. |
| `RetinaNet_research_style_v1.ipynb` | 데이터 QC, dataset A/B 비교, RetinaNet 설정 비교, staged fine-tuning, postprocess sweep, 재현 실험을 한 곳에서 관리하는 통합 노트북이다. |
| `RetinaNet_v1.ipynb` ~ `RetinaNet_v8.ipynb` | 실험 과정을 단계적으로 확장한 중간 버전이다. |
| `RetinaNet_v9(best).ipynb` | threshold sweep, class-agnostic NMS, 클래스별 AP 분석까지 포함된 고성능 후보 버전이다. |
| `RetinaNet_v10.ipynb` | v9 흐름을 단순화한 파생 버전이다. |

처음 프로젝트를 파악할 때는 `RetinaNet_research_style_v1.ipynb`로 전체 구조를 보고, 실제 재현이나 제출 파일 생성은 `RetinaNet_best.ipynb`로 진행하는 방식이 가장 무난하다.

## 4. 실행 환경

노트북은 Google Colab + Google Drive 환경을 전제로 작성되어 있다.  
코드 안에서 `google.colab`, `/content`, `/content/drive/MyDrive/...` 경로를 직접 사용하므로 로컬 환경에서 실행하려면 경로와 일부 셀 구성을 먼저 조정해야 한다.

주요 의존성은 다음과 같다.

- PyTorch / Torchvision
- pandas, numpy, matplotlib
- Pillow
- pycocotools
- optuna 일부 실험에서 사용
- requests 알림 셀에서 사용

노트북에 포함된 기본 설치 흐름은 아래와 같다.

```bash
git clone https://github.com/wina0901/pill_detection_project.git
cd pill_detection_project

pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118
pip install -r requirements.txt --no-deps
```

## 5. 데이터 구성

노트북은 COCO 형식 annotation과 letterbox 이미지셋을 기준으로 동작한다.  
`RetinaNet_research_style_v1.ipynb`에서는 두 가지 데이터셋 변형을 관리한다.

- `NO_CP`: 기본 증강 버전
- `CP`: `new_aug` 기반 대체 버전

통합 노트북에서 기본 선택값은 `NO_CP`로 설정되어 있다.

예상하는 데이터 구조는 다음과 같다.

```text
dataset/
├── train_letterbox_aug_v1.json
├── val_letterbox_aug_v1.json
├── letterbox_images_aug_v1/
│   ├── train/
│   └── val/
├── new_aug/
│   ├── train_letterbox.json
│   ├── val_letterbox.json
│   └── letterbox_images/
│       ├── train/
│       └── val/
└── test_images/
```

실제 경로는 노트북의 경로 설정 셀에서 관리하므로, 환경에 맞게 그 부분만 먼저 맞춰두면 된다.

## 6. 학습 및 추론 흐름

`RetinaNet_best.ipynb` 기준 전체 흐름은 아래 순서로 진행된다.

1. Google Drive 마운트
2. 저장소 clone 및 패키지 설치
3. 데이터 경로 설정과 DataLoader 구성
4. RetinaNet 모델 생성 및 학습
5. validation 성능 평가와 history 저장
6. threshold sweep으로 제출형 후처리 파라미터 탐색
7. `test_images` 대상 추론 및 submission CSV 생성
8. 예측 시각화
9. 가중치, history, CSV를 Google Drive에 저장

통합 연구 노트북(`RetinaNet_research_style_v1.ipynb`)은 위 흐름을 더 잘게 쪼개어, 데이터 QC부터 실험군 비교와 재현성 확인까지 한 번에 다룰 수 있게 정리되어 있다.  
필요한 블록만 실행할 수 있도록 `RUN_FLAGS`로 각 실험 단계를 제어하도록 되어 있다.

## 7. 현재 기준 핵심 설정

최종 계열 노트북을 기준으로 보면 현재 주력 설정은 다음과 같다.

| 항목 | 설정 |
| --- | --- |
| Backbone | `retinanet_resnet50_fpn` |
| 입력 크기 | `800` |
| 전처리 | letterbox |
| Batch size | `4` |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Scheduler | 2 epoch warmup + cosine annealing |
| Focal loss | `alpha=0.25`, `gamma=3.0` |
| Early stopping | `patience=10` |
| 내부 추론 설정 | `score_thresh=0.001`, `nms_thresh=0.60`, `detections_per_img=300`, `topk_candidates=1000` |
| 제출 후처리 | `score_threshold=0.20`, `top_k_per_image=4`, `class-agnostic NMS IoU=0.50` |

## 8. 참고 성능

아래 수치는 `RetinaNet_best.ipynb`에 저장된 실행 출력 기준이다.

- 학습 중 최고 raw `mAP@75:95`: `0.8890`  
  - 기록 시점: epoch 19
- threshold sweep 기준 최적 제출형 설정  
  - `score_threshold=0.20`
  - `top_k_per_image=4`
  - `class-agnostic NMS IoU=0.50`
- 위 설정에서 validation 결과
  - `mAP@75:95 = 0.917853`
  - `mAP@50 = 0.926690`
  - `precision = 0.850208`
  - `recall = 0.985531`

즉, 최종 노트북은 학습 성능만 확인하는 수준에서 끝나지 않고, 제출 형식에 맞춘 후처리까지 포함해 실제 사용 기준의 성능을 따로 점검하도록 구성되어 있다.

## 9. 버전별 변경 사항

| 버전 | 핵심 변경 |
| --- | --- |
| `RetinaNet_v1.ipynb` | 기본 RetinaNet 학습 루프, StepLR 기반 스케줄링, 평가 및 시각화 구성 |
| `RetinaNet_v2.ipynb` | Optuna를 이용한 learning rate 탐색 실험 추가 |
| `RetinaNet_v3.ipynb` | custom classification head와 focal loss 적용, test 추론 및 submission CSV 생성 추가 |
| `RetinaNet_v4.ipynb` | best model 선정 기준을 `val_loss`에서 `mAP@75:95`로 변경 |
| `RetinaNet_v5.ipynb` | StepLR을 warmup + cosine annealing으로 교체 |
| `RetinaNet_v6.ipynb` | focal loss `gamma=3.0` 적용 |
| `RetinaNet_v7.ipynb` | backbone / head 분리 learning rate, precision tie-break, 제출용 category offset 보정 |
| `RetinaNet_v8.ipynb` | focal loss `alpha=0.75` 실험 |
| `RetinaNet_v9(best).ipynb` | threshold sweep, class-agnostic NMS, 클래스별 AP 분석 추가 |
| `RetinaNet_v10.ipynb` | 후처리와 추론 흐름을 단순화한 파생 버전 |
| `RetinaNet_best.ipynb` | v10 정리본에 v9의 후처리와 분석 요소를 다시 반영한 최종 실행본 |
| `RetinaNet_research_style_v1.ipynb` | screening, dataset 비교, ablation, staged fine-tuning, repro를 한 노트북에서 관리하도록 재구성한 통합본 |

## 10. 생성 산출물

최종 계열 노트북에서 기본적으로 생성하거나 저장하는 파일은 다음과 같다.

- `retinanet_oral_drug.pth`
- `history_retinanet.json`
- `retinanet_test_submission.csv`

통합 연구 노트북은 별도로 다음 위치를 기준으로 결과를 정리한다.

- `models/retinanet/`
- `results/retinanet/`
- `master_results.csv`

제출 CSV는 아래 컬럼 구조를 사용한다.

```text
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

## 11. 권장 사용 방법

빠르게 결과를 재현하려면 `RetinaNet_best.ipynb`만 순서대로 실행하면 된다.  
실험 조건을 비교하거나 구조적으로 정리된 로그를 남기고 싶다면 `RetinaNet_research_style_v1.ipynb`를 사용하는 편이 낫다.

정리하면 다음과 같이 구분해서 쓰면 된다.

- **최종 학습 / 제출 파일 생성**: `RetinaNet_best.ipynb`
- **실험 설계 / 비교 / 재현성 점검**: `RetinaNet_research_style_v1.ipynb`
- **변경 이력 확인**: `RetinaNet_v1.ipynb` ~ `RetinaNet_v10.ipynb`

## 12. 유의 사항

1. 노트북은 Colab 전용 셀 구성과 경로를 사용하므로, 로컬 실행 시 경로 수정이 선행되어야 한다.
2. `src.preprocessing`, `src.evaluation` 모듈이 저장소에 준비되어 있어야 정상 동작한다.
3. 데이터는 COCO 형식과 letterbox 전처리를 전제로 한다.
4. 노트북 말미의 외부 알림 셀은 공개 저장소에 그대로 두기보다, 개인 환경에 맞게 비활성화하거나 별도 설정으로 분리하는 편이 안전하다.

---

실무적으로는 `RetinaNet_best.ipynb`를 기준 실행본으로 두고, `RetinaNet_research_style_v1.ipynb`를 실험 관리 문서처럼 병행하는 구성이 가장 자연스럽다.
