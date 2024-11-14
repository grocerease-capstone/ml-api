from fastapi import FastAPI
from ml.product_classification import predict_classification, standardize_product_name
from models.classification import PredictRequest

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def handle_post_predict(request: PredictRequest):
    return predict_classification(request.product_names)
