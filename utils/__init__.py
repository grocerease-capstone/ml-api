import tensorflow as tf


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
