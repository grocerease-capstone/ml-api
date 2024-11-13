from fastapi import FastAPI
import tensorflow as tf
import os
import pickle
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    model = tf.keras.models.load_model(os.path.join('model', 'model.keras'))

    with open(os.path.join('model', 'vectorizer.pickle'), 'rb') as file:
        vectorizer = pickle.load(file)
        print(vectorizer)

    with open(os.path.join('model', 'category_encoder.pickle'), 'rb') as file:
        category_encoder = pickle.load(file)
        print(category_encoder)

    standardized_product_name = standardize_product_name(name)
    print(standardized_product_name)
    vectorized_product_name = vectorizer([standardized_product_name])

    prediction = model.predict(vectorized_product_name)
    print(prediction)
    predicted_category_index = np.argmax(prediction)
    predicted_category = category_encoder.get_vocabulary()[predicted_category_index]

    print(f"predicted category for '{name} is: {predicted_category}'")

    return {"message": f"product: {name}, category: {predicted_category}"}

@tf.keras.utils.register_keras_serializable()
def standardize_product_name(product_name: str):
    product_name = tf.strings.lower(product_name)
    # product_name = tf.strings.regex_replace(product_name, r'\d+([xgmlkgcm]*)', '')

    return product_name