from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("app\models\best.pt")

def model_predict(image_path: str):
    results = model(image_path)
    result = results[0]  

    # class ID
    class_ids = result.boxes.cls.tolist()

    # Convert class IDs to class names
    class_names = [model.names[int(cls_id)] for cls_id in class_ids]

    return list(set(class_names))

