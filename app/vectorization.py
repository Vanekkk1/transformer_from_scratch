import tensorflow as tf
from pathlib import Path

url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = tf.keras.utils.get_file(
    "spa-eng.zip", origin=url, cache_dir="datasets", extract=True
)
text = (Path(path).with_name("spa-eng") / "spa.txt").read_text()
import numpy as np

text = text.replace("¡", "").replace("¿", "")
pairs = [line.split("\t") for line in text.splitlines()]
np.random.seed(42)  # extra code – ensures reproducibility on CPU
np.random.shuffle(pairs)
sentences_en, sentences_es = zip(*pairs)  # separates the pairs into 2 lists


# Define your TextVectorization layer
vocab_size = 1000  # Adjust as necessary
max_length = 50    # Adjust as necessary
# Define your TextVectorization layer
text_vectorization_spain = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size, output_sequence_length=max_length
)

# Adapt the layer
text_vectorization_spain.adapt([f"startofseq {s} endofseq" for s in sentences_es])

# Wrap the layer in a tf.keras.Model
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
outputs = text_vectorization_spain(inputs)
model = tf.keras.Model(inputs, outputs)

model.save('text_vectorization_spain.keras')
