import io
import os.path
import pickle
import tensorflow as tf
import easyocr
import numpy as np

from contextlib import asynccontextmanager
from typing import List
from PIL import Image
from fastapi import FastAPI, responses, UploadFile
from ultralytics import YOLO
from models import (
    HealthCheckResponse,
    ProductItemDetail,
    ProductItem,
    ScanReceiptResponse,
)


@tf.keras.utils.register_keras_serializable()
def standardize_product_name(product_name: str) -> tf.strings:
    product_name = tf.strings.lower(product_name)
    product_name = tf.strings.regex_replace(
        product_name, r"\b\d+(\.\d+)?x\d+(\.\d+)?(g|ml|kg|lt)\b", ""
    )
    product_name = tf.strings.regex_replace(
        product_name, r"\b\d+(\.\d+)?(g|ml|kg|lt)\b", ""
    )
    product_name = tf.strings.regex_replace(product_name, r"\b\d+(g|ml|kg|lt)\b", "")
    product_name = tf.strings.regex_replace(product_name, r"\b\d+\'s\b", "")
    product_name = tf.strings.regex_replace(product_name, r"[^a-z\s]", "")
    product_name = tf.strings.regex_replace(product_name, r"\s{2,}", " ")
    product_name = tf.strings.strip(product_name)

    return product_name


tf.keras.utils.get_custom_objects()[
    "standardize_product_name"
] = standardize_product_name


trained_models_dir = os.path.join("trained_models")
object_detection_labels = [
    "product_item",
    "product_item_discount",
    "product_item_voucher",
]
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load all ml models
    ml_models["nlp_encoder"] = pickle.load(
        open(os.path.join(trained_models_dir, "nlp", "v2", "encoder.pickle"), "rb")
    )
    ml_models["nlp_vectorizer"] = pickle.load(
        open(os.path.join(trained_models_dir, "nlp", "v2", "vectorizer.pickle"), "rb")
    )
    ml_models["nlp_model"] = tf.keras.models.load_model(
        os.path.join(trained_models_dir, "nlp", "v2", "model.keras")
    )
    ml_models["object_detection_model"] = YOLO(
        os.path.join(trained_models_dir, "object_detection", "model.pt")
    )
    ml_models["ocr"] = easyocr.Reader(["en", "id"])

    yield

    # clean up ml models
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


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


@app.get(
    path="/health",
    responses={200: {"model": HealthCheckResponse}},
)
async def handle_health_check():
    return HealthCheckResponse(status="healthy")


@app.post(
    path="/v1/receipt",
    responses={200: {"model": ScanReceiptResponse}},
)
async def handle_receipt_detection(file: UploadFile):
    image_content = await file.read()
    receipt = Image.open(io.BytesIO(image_content))

    results = ml_models["object_detection_model"].predict(source=receipt)

    scanned_products: List[ProductItem] = list[ProductItem]()

    for result in results:
        # result.show()

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

            label_index = int(label)

            if label_index == 0:
                # product item
                width, height = cropped_image.size

                # membagi gambar menjadi dua bagian, kiri dan kanan
                # kiri untuk nama product
                # kanan untuk jumlah dan harga
                left_image = cropped_image.crop((0, 0, width // 2, height))
                right_image = cropped_image.crop((width // 2, 0, width, height))

                # mendapatkan nama product
                product_name_ocr_results = ml_models["ocr"].readtext(
                    np.array(left_image), detail=0
                )
                product_name_result = "".join(product_name_ocr_results)

                # mendapatkan prediksi kategori yang sesuai
                standardized_product_name = standardize_product_name(
                    product_name_result
                )
                vectorized_product_name = ml_models["nlp_vectorizer"](
                    [standardized_product_name]
                )

                nlp_prediction = ml_models["nlp_model"].predict(vectorized_product_name)
                nlp_predicted_index = np.argmax(nlp_prediction)
                nlp_max_probability = nlp_prediction[0][nlp_predicted_index]

                if nlp_max_probability < 0.8:
                    nlp_predicted_label = "lainnya"
                else:
                    nlp_predicted_label = ml_models["nlp_encoder"].get_vocabulary()[
                        nlp_predicted_index
                    ]

                # mendapatkan jumlah dan harga
                right_width, right_height = right_image.size
                right_image_part1 = right_image.crop(
                    (0, 0, right_width // 3, right_height)
                )
                right_image_part2 = right_image.crop(
                    (right_width // 3, 0, 2 * right_width // 3, right_height)
                )
                right_image_part3 = right_image.crop(
                    (2 * right_width // 3, 0, right_width, right_height)
                )
                product_amount_ocr_results = ml_models["ocr"].readtext(
                    np.array(right_image_part1), detail=0
                )
                product_price_ocr_results = ml_models["ocr"].readtext(
                    np.array(right_image_part2), detail=0
                )
                product_total_price_ocr_results = ml_models["ocr"].readtext(
                    np.array(right_image_part3), detail=0
                )

                try:
                    product_amount_result = int(
                        "".join(
                            filter(str.isdigit, "".join(product_amount_ocr_results))
                        )
                    )
                except ValueError:
                    product_amount_result = 0

                try:
                    product_price_result = int(
                        "".join(filter(str.isdigit, "".join(product_price_ocr_results)))
                    )
                except ValueError:
                    product_price_result = 0

                try:
                    product_total_price_result = int(
                        "".join(
                            filter(
                                str.isdigit, "".join(product_total_price_ocr_results)
                            )
                        )
                    )
                except ValueError:
                    product_total_price_result = 0

                scanned_products.append(
                    ProductItem(
                        name=product_name_result,
                        amount=product_amount_result,
                        price=product_price_result,
                        total_price=product_total_price_result,
                        detail=ProductItemDetail(
                            type=object_detection_labels[label_index],
                            category=nlp_predicted_label,
                            category_probability=nlp_max_probability,
                        ),
                    )
                )

            elif label_index == 1:
                # product item discount
                pass

            else:
                # product item voucher
                pass

            # np_cropped_image = np.array(cropped_image)
            # ocr_results = ocr.readtext(np_cropped_image, detail=1)

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
            # for ocr_result_item in ocr_results:
            #     bounding_box, text, confidence = ocr_result_item

            #     x1, y1 = map(int, bounding_box[0])
            #     x2, y2 = map(int, bounding_box[2])

            #     result_with_ocr = ImageDraw.Draw(cropped_image)
            #     result_with_ocr.rectangle([x1, y1, x2, y2], outline="red", width=2)
            #     result_with_ocr.text(
            #         (x1, y1),
            #         text,
            #         fill="red",
            #     )

            # cropped_image.show()

    return ScanReceiptResponse(
        products=scanned_products,
    )
