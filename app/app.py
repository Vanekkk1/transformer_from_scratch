import streamlit as st
from model_utils import load_model, translate

# Load your model
model = load_model('eng_esp_model.keras')

# Sidebar for model limitations
st.sidebar.title("Model Limitations")
st.sidebar.write("""
This model understands only 1000 words. In some cases, you might see [UNK] instead of actual words 
if they are not in the model's vocabulary.
It supports only up to 50 words in a sentence.
""")

# Main page title
st.title('ENG-ESP Translation with Transformer Model')

# User input for translation
user_input = st.text_input("Enter English text to translate:")

# Display translation in a text area instead of plain page
if st.button('Translate'):
    result = translate(model, user_input)
    st.text_area('Spanish Translation:', result, height=50)

# GitHub link
st.markdown("Built by [Vanekkk1](https://github.com/Vanekkk1)")


# Dropdown for source code
st.title("Source codes for implementation and inference")
with st.expander("Transformer Source Code"):
    st.code("""
# %%
import tensorflow as tf
from pathlib import Path

# %% [markdown]
# # Preprocessing

# %%
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

# %%
tf.keras.backend.clear_session()
vocab_size = 1000
max_lenght = 50
text_vectorization_eng = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size, output_sequence_length=max_lenght
)
text_vectorization_spain = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size, output_sequence_length=max_lenght
)

text_vectorization_eng.adapt(sentences_en)
text_vectorization_spain.adapt([f"startofseq {s} endofseq" for s in sentences_es])

# %%
X_train = tf.constant(sentences_en[:105000])
X_valid = tf.constant(sentences_en[105000:])

# Convert generator expressions to lists and then to tensors
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:105000]])
X_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[105000:]])

# Now vectorize the modified sentences
y_train = text_vectorization_spain([f"{s} endofseq" for s in sentences_es[:105000]])
y_valid = text_vectorization_spain([f"{s} endofseq" for s in sentences_es[105000:]])

# %% [markdown]
# # Model

# %%
embed_size = 128
num_stacks = 2
num_heads_per_stack = 8
dropout_rate = 0.1
n_units = embed_size


# Define the model inputs
encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)


# Apply the TextVectorization layer
encoder_input_vector = text_vectorization_eng(encoder_inputs)
decoder_input_vector = text_vectorization_spain(decoder_inputs)

# Define the shared embedding layer
encoder_embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_size, mask_zero=True
)
decoder_embedding_layer = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_size, mask_zero=True
)

encoder_embedding = encoder_embedding_layer(encoder_input_vector)
decoder_embedding = decoder_embedding_layer(decoder_input_vector)


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


# Define and apply the positional encoding layer
# positional_encoding_layer = keras_nlp.layers.SinePositionEncoding()
pos_embed_layer = PositionalEncoding(max_lenght, embed_size)

encoder_in = pos_embed_layer(encoder_embedding)
decoder_in = pos_embed_layer(decoder_embedding)

# %% [markdown]
# # Encoder&Decoder

# %%
# Encoder
Z = encoder_in
encoder_pad_mask = tf.math.not_equal(encoder_input_vector, 0)[:, tf.newaxis]
for _ in range(num_stacks):
    skip = Z
    attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads_per_stack, key_dim=embed_size, dropout=dropout_rate
    )
    Z = attn_layer(Z, value=Z, attention_mask=encoder_pad_mask)
    Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))
    skip = Z
    Z = tf.keras.layers.Dense(n_units, activation="relu")(Z)
    Z = tf.keras.layers.Dense(embed_size)(Z)
    Z = tf.keras.layers.Dropout(dropout_rate)(Z)
    Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))

# Decoder
encoder_outputs = Z
Z = decoder_in
decoder_pad_mask = tf.math.not_equal(decoder_input_vector, 0)[:, tf.newaxis]
batch_max_len_dec = tf.shape(decoder_embedding)[1]
causal_mask = tf.linalg.band_part(  # creates a lower triangular matrix
    tf.ones((batch_max_len_dec, batch_max_len_dec), tf.bool), -1, 0
)
for _ in range(num_stacks):
    skip = Z
    attn_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads_per_stack, key_dim=embed_size, dropout=dropout_rate
    )
    Z = attn_layer(Z, value=Z, attention_mask=decoder_pad_mask & causal_mask)
    Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))
    skip = Z
    cross_attentin_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads_per_stack, key_dim=embed_size, dropout=dropout_rate
    )
    # key and value from encoder compared to decoder query
    Z = cross_attentin_layer(Z, value=encoder_outputs, attention_mask=encoder_pad_mask)
    Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))
    skip = Z
    Z = tf.keras.layers.Dense(n_units, activation="relu")(Z)
    Z = tf.keras.layers.Dense(embed_size)(Z)
    Z = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([Z, skip]))

Y_proba = tf.keras.layers.Dense(vocab_size, activation="softmax")(Z)

# %%
model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba])
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
)
early_st = tf.keras.callbacks.EarlyStopping(
    patience=3, monitor="val_accuracy", restore_best_weights=True
)
model.fit(
    (X_train, X_train_dec),
    y_train,
    epochs=15,
    validation_data=((X_valid, X_valid_dec), y_valid),
    callbacks=[early_st],
)
model.summary()

# %%
model.save("eng_esp_model.keras")
model.save_weights("eng_esp_weights")
""")

with st.expander("Inference Source Code"):
    st.code("""
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
""")