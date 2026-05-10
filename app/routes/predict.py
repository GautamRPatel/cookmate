from fastapi import APIRouter, File, UploadFile
from app.models.recipe_engine import RecipeEngine
from app.models.yolov8_model import model_predict
import shutil
import uuid
import os
from functools import lru_cache
from fastapi import HTTPException

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "IndianFoodDatasetCSV.csv")

UPLOAD_DIR = os.path.join(BASE_DIR, "app", "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def get_recipe_engine():
    return RecipeEngine(DATASET_PATH)


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        # Save uploaded image temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model prediction
        vegetables = model_predict(file_path)

        print("Vegetables" , vegetables)

        if not vegetables:
            raise HTTPException(
                status_code=400,
                detail="Invalid image, Please upload image related to vegetables"
            )

        return {
            "detected_vegetables": vegetables,
            "top_3_recipes": get_recipe_engine().get_best_recipes(vegetables)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
   
