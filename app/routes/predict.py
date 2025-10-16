from fastapi import APIRouter, File, UploadFile
from app.models.yolov8_model import model_predict
import shutil
import uuid
import os

router = APIRouter()

UPLOAD_DIR = "app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model prediction
    results = model_predict(file_path)

    # Delete image after prediction
    os.remove(file_path)

    return {"predicted_objects": results}
