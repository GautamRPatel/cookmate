from ultralytics import YOLO
import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "best.pt")


@lru_cache(maxsize=1)
def get_model():
    return YOLO(MODEL_PATH)

def model_predict(image_path: str):
    model = get_model()
    results = model(image_path)
    result = results[0]  

    # class ID
    class_ids = result.boxes.cls.tolist()

    # Convert class IDs to class names
    class_names = [model.names[int(cls_id)] for cls_id in class_ids]

    return list(set(class_names))

