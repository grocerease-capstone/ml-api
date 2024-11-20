from fastapi import FastAPI, responses
from ml.product_classification import predict_classification, standardize_product_name
from models.classification import PredictRequest


app = FastAPI()


@app.get("/")
async def root():
    """
    Root endpoint that returns a simple HTML page with a welcome message.
    """
    return responses.HTMLResponse(
        """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>ML API</title>
                </head>
                <body>
                    <h1>Welcome to the ML API</h1>
                    <p>Visit the <a href="/docs">API documentation</a> for more information.</p>
                </body>
            </html>
        """
    )


@app.post("/predict")
async def handle_post_predict(request: PredictRequest):
    return predict_classification(request.product_names)
