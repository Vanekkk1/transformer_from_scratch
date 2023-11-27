# model_utils.py
import tensorflow as tf
import numpy as np

@tf.keras.saving.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed_size must be even"
        p, i = np.meshgrid(np.arange(max_length), 2 * np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_length, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True

    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]

def load_model(model_path):
    return tf.keras.models.load_model(
        model_path, custom_objects={"PositionalEncoding": PositionalEncoding}
    )

def load_text_vectorization_layer(path):
    loaded_model = tf.keras.models.load_model(path)
    return loaded_model.layers[-1] 

text_vectorization_spain = load_text_vectorization_layer("app/text_vectorization_spain.keras")

MAX_LENGTH = 50

def beam_search_tf(model, sentence_en, beam_width=3):
    sentence_en = tf.constant([sentence_en])
    start_seq = tf.constant(["startofseq"])

    vocab = text_vectorization_spain.get_vocabulary()  # Call this once outside the loop
    vocab_size = len(vocab)

    # Initial prediction
    y_proba = model.predict([sentence_en, start_seq])[0, 0]
    top_k = tf.math.top_k(y_proba, k=beam_width)

    top_translations = [
        (tf.math.log(word_proba), vocab[word_id])
        for word_proba, word_id in zip(top_k.values.numpy(), top_k.indices.numpy())
    ]

    for idx in range(1, MAX_LENGTH):
        candidates = []
        # Prepare batch inputs for all candidates
        batch_sentence_en = tf.constant([sentence_en.numpy()[0]] * beam_width)
        batch_start_seq = tf.constant([f"startofseq {tr[1]}" for tr in top_translations])

        # Batch prediction
        y_proba_batch = model.predict([batch_sentence_en, batch_start_seq])

        for i, (log_proba, translation) in enumerate(top_translations):
            y_proba = y_proba_batch[i, idx]
            for word_id in range(vocab_size):
                word = vocab[word_id]
                new_log_proba = log_proba + tf.math.log(y_proba[word_id])
                candidates.append((new_log_proba, f"{translation} {word}"))

        top_translations = sorted(candidates, reverse=True)[:beam_width]

        if all([tr.endswith("endofseq") for _, tr in top_translations]):
            return top_translations[0][1].replace("endofseq", "").strip()

    return top_translations[0][1].replace("endofseq", "").strip()

def translate(model, sentence_en):
    translation = ""
    for word_idx in range(MAX_LENGTH):
        X = np.array([sentence_en])  # encoder input
        X_dec = np.array(["startofseq " + translation])  # decoder input
        y_proba = model.predict((X, X_dec))[0, word_idx]  # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = text_vectorization_spain.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()