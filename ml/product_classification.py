import tensorflow as tf
import os
import pickle
import numpy as np
from typing import List
from models.classification import PredictedClassificationItem


def predict_classification(
    product_names: List[str],
) -> List[PredictedClassificationItem]:
    result: List[PredictedClassificationItem] = []
    model = tf.keras.models.load_model(os.path.join("tf_models", "model.keras"))

    with open(os.path.join("tf_models", "vectorizer.pickle"), "rb") as file:
        vectorizer = pickle.load(file)
        print(vectorizer)

    with open(os.path.join("tf_models", "category_encoder.pickle"), "rb") as file:
        category_encoder = pickle.load(file)
        print(category_encoder)

    for product_name in product_names:
        standardized_product_name = standardize_product_name(product_name)
        vectorized_product_name = vectorizer([standardized_product_name])

        prediction = model.predict(vectorized_product_name)
        predicted_category_index = np.argmax(prediction)
        predicted_category = category_encoder.get_vocabulary()[predicted_category_index]

        result.append(
            PredictedClassificationItem(
                product_name=product_name,
                category=predicted_category,
            )
        )

    return result


@tf.keras.utils.register_keras_serializable()
def standardize_product_name(product_name: str) -> tf.strings:
    product_name = tf.strings.lower(product_name)
    # product_name = tf.strings.regex_replace(product_name, r'\d+([xgmlkgcm]*)', '')

    return product_name
