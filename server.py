from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.inference.inference import (
    predict_ensemble,
    draw_predictions,
    save_detection_crops,
    OUTPUT_DIR,
    ensure_output_dirs,
)


app = FastAPI(
    title="Pill Detection Ensemble Server",
    description="YOLO 앙상블 기반 알약 객체 탐지 API",
    version="3.2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ensure_output_dirs()

UI_DIR = PROJECT_ROOT / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

if not TEMPLATES_DIR.exists():
    raise RuntimeError(f"템플릿 폴더가 없습니다: {TEMPLATES_DIR}")

if not STATIC_DIR.exists():
    raise RuntimeError(f"static 폴더가 없습니다: {STATIC_DIR}")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(OUTPUT_DIR)), name="results")

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "temp_dir": str(TEMP_DIR),
        "results_dir": str(OUTPUT_DIR),
    }


@app.post("/predict-ui")
async def predict_ui(file: UploadFile = File(...)):
    ensure_output_dirs()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="업로드된 파일 이름이 없습니다.")

    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식입니다. 지원 형식: {sorted(ALLOWED_SUFFIXES)}"
        )

    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = TEMP_DIR / temp_name

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="빈 파일이 업로드되었습니다.")

        predictions = predict_ensemble(str(temp_path))

        result_filename = f"pred_{uuid.uuid4().hex}.jpg"
        result_path = OUTPUT_DIR / result_filename

        draw_predictions(
            image_path=str(temp_path),
            predictions=predictions,
            save_path=str(result_path),
        )

        detections = save_detection_crops(
            image_path=str(temp_path),
            predictions=predictions,
        )

        response_detections = []
        for det in detections:
            response_detections.append({
                "category_id": det["category_id"],
                "display_name": det.get("display_name", ""),
                "score": det["score"],
                "feature": det.get("feature", ""),
                "bbox_xyxy": det["bbox_xyxy"],
                "bbox_xywh": det["bbox_xywh"],
                "crop_url": f"/results/crops/{det['crop_filename']}",
            })

        return JSONResponse({
            "filename": file.filename,
            "num_predictions": len(predictions),
            "result_image_url": f"/results/{result_filename}",
            "detections": response_detections,
        })

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {str(e)}")

    finally:
        try:
            file.file.close()
        except Exception:
            pass

        if temp_path.exists():
            temp_path.unlink(missing_ok=True)