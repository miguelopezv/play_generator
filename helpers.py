import numpy as np
from tensorflow import keras


def encode(char2idx, text):
    return np.array([char2idx[c] for c in text])


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(rnn_units,
                          # return_sequences=True,
                          stateful=True,
                          # recurrent_initializer='glorot_uniform'
                          ),
        keras.layers.Dense(vocab_size)
    ])


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
