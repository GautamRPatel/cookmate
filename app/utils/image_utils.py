from PIL import Image

def verify_image(file_path: str):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False
