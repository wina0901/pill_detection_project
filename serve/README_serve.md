#  Serving

## 1. Overview
학습이 완료된 YOLO 기반 객체 탐지 모델을 활용하여,  
이미지를 업로드하면 알약 객체를 탐지하고 결과를 시각적으로 확인할 수 있는  
추론 및 서비스이다.

단순 모델 실행이 아닌 다음을 포함한다:

- FastAPI 기반 API 서버
- 웹 UI 기반 추론 환경
- 결과 시각화 및 crop 이미지 생성
- 메타데이터(알약 이름/특징) 결합
- 실행 환경 자동화 (setup & run)

---

## 2. System Architecture

- FastAPI Server
- Web UI (Jinja2 Templates)
- YOLO Ensemble Inference Module
- Visualization & Crop Module
- Metadata Mapping Module
- Setup Automation Script
- OS-specific Run Scripts

---

## 3. Key Features

### 3.1 FastAPI 기반 추론 서버
- REST API 형태로 객체 탐지 기능 제공
- `/predict-ui` → 이미지 업로드 후 즉시 추론
- `/health` → 서버 상태 확인

---

### 3.2 Web UI 기반 실시간 추론
- Jinja2 템플릿 기반 웹 인터페이스 제공
- 이미지 업로드 후 결과 즉시 확인
- 결과 이미지 및 crop 이미지 제공

---

### 3.3 Ensemble Inference Pipeline
- 여러 YOLO 모델 동시 로드
- 모델별 결과 통합
- class-wise NMS 적용
- 상위 결과만 선택

---

### 3.4 Category Mapping & Metadata
- model class index → 원본 category_id 변환
- `meta.csv` 기반 정보 제공

포함 정보:
- pill_name (알약 이름)
- feature (특징)

→ 단순 detection이 아닌 설명 가능한 결과 제공

---

### 3.5 Visualization & Crop Generation
- bbox 포함 결과 이미지 생성
- 객체별 crop 이미지 자동 저장

각 detection 포함 정보:
- bbox
- score
- label
- crop 이미지 URL

---

### 3.6 File Handling & Resource Management
- 업로드 이미지 → temp 폴더 저장
- 추론 수행 후 자동 삭제
- 지원 포맷:
  - png, jpg, jpeg, bmp, webp

---

### 3.7 Setup & Run Automation

#### setup_serve.py
- venv 자동 생성
- PyTorch 및 패키지 설치
- 디렉토리 자동 생성

#### run_server.py
- venv 자동 감지 및 재실행
- 서버 실행 후 브라우저 자동 오픈
- QR 코드 생성 (모바일 접속 지원)

---

### 3.8 Cross Platform Support
- Windows: `.bat`
- Mac: `.command`

→ OS 관계없이 실행 가능

---

### 3.9 Device Auto Selection
- GPU 사용 가능 시 CUDA 사용
- 미지원 시 CPU fallback

---

## 4. Workflow

```text
1. 사용자 이미지 업로드
2. 파일 검증
3. temp 저장
4. YOLO 앙상블 추론
5. NMS 적용
6. category 매핑
7. 메타데이터 결합
8. 결과 이미지 생성
9. crop 이미지 생성
10. JSON 응답 반환
11. temp 파일 삭제