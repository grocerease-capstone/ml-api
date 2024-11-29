import io
import os.path
from typing import List

import tensorflow as tf
import easyocr
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, responses, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
import pickle

from ml.product_classification import predict_classification
from models.classification import PredictRequest

app = FastAPI()

trained_models_dir = os.path.join("trained_models")

nlp_encoder = pickle.load(
    open(os.path.join(trained_models_dir, "nlp", "encoder.pickle"), "rb")
)
nlp_vectorizer = pickle.load(
    open(os.path.join(trained_models_dir, "nlp", "vectorizer.pickle"), "rb")
)
nlp_model = tf.keras.models.load_model(
    os.path.join(trained_models_dir, "nlp", "model.keras")
)
object_detection_model = YOLO(
    os.path.join(trained_models_dir, "object_detection", "model.pt")
)
ocr = easyocr.Reader(["en", "id"])


@app.get("/")
async def handle_root():
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


@app.post(
    path="/v1/receipt",
)
async def handle_receipt_detection(file: UploadFile):
    image_content = await file.read()
    receipt = Image.open(io.BytesIO(image_content))

    results = object_detection_model.predict(source=receipt)

    class ProductItem(BaseModel):
        name: str
        price: str
        amount: int

    products: List[ProductItem] = list[ProductItem]()

    for result in results:
        result.show()

        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            tolerance = 10
            x1, y1, x2, y2 = box.astype(int)

            cropped_image = receipt.crop(
                (
                    x1 - tolerance,
                    y1 - tolerance,
                    x2 + tolerance,
                    y2 + tolerance,
                )
            )

            # cropped_image.show()

            np_cropped_image = np.array(cropped_image)
            ocr_results = ocr.readtext(np_cropped_image, detail=1)

            # ocr_result = [text for text in ocr_results if len(text) > 0]
            # ocr_result_text: str = ' '.join(ocr_result)
            #
            # products.append(
            #     ProductItem(
            #         name=ocr_result_text,
            #         price="dummy",
            #         amount=10,
            #     )
            # )

            #
            for ocr_result_item in ocr_results:
                bounding_box, text, confidence = ocr_result_item

                x1, y1 = map(int, bounding_box[0])
                x2, y2 = map(int, bounding_box[2])

                result_with_ocr = ImageDraw.Draw(cropped_image)
                result_with_ocr.rectangle([x1, y1, x2, y2], outline="red", width=2)
                result_with_ocr.text(
                    (x1, y1),
                    text,
                    fill="red",
                )

            cropped_image.show()

    return {"products": products}


@tf.keras.utils.register_keras_serializable()
def standardize_product_name(product_name: str) -> tf.strings:
    product_name = tf.strings.lower(product_name)
    # product_name = tf.strings.regex_replace(product_name, r'[^a-z\s]', '')
    product_name = tf.strings.regex_replace(
        product_name, r"\b\d+(\.\d+)?x\d+(\.\d+)?(g|ml|kg|lt)\b", ""
    )
    product_name = tf.strings.regex_replace(
        product_name, r"\b\d+(\.\d+)?(g|ml|kg|lt)\b", ""
    )
    product_name = tf.strings.regex_replace(product_name, r"\b\d+(g|ml|kg|lt)\b", "")
    product_name = tf.strings.regex_replace(product_name, r"\b\d+\'s\b", "")
    product_name = tf.strings.regex_replace(product_name, r"[^a-z\s]", "")
    product_name = tf.strings.strip(product_name)
    return product_name
