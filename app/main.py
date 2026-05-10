from fastapi import FastAPI

from app.routes.predict import router as predict_router

app = FastAPI(title="Vegetable Detection API")
app.include_router(predict_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Veg Detection API is running"}
