import numpy as np
import tensorflow as tf

from constants import SHAKESPEARE_URL, BATCH_SIZE, EMBEDDING_DIM, RNN_UNITS, BUFFER_SIZE
from helpers import encode, split_input_target, build_model

tf.config.set_visible_devices([], 'GPU')

res = input('We are going to train our model with a default play, if ok press ENTER otherwise write \'custom\': ')
while res not in ['', 'custom']:
    res = input('wrong answer! please leave empty for default, or write \'custom\':')

path_to_file = tf.keras.utils.get_file('shakespeare.txt', SHAKESPEARE_URL) if res == '' else 'ownFile'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = {char: i for i, char in enumerate(vocab)}
idx2npArr = np.array(vocab)

text_as_int = encode(char2idx, text)

# divide dataset into batches
seq_length = 100
text_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = text_dataset.batch(seq_length + 1, drop_remainder=True)

# Transform each batch into input and target
dataset = sequences.map(split_input_target)

VOCAB_SIZE = len(vocab)

# Create batches to train the model
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
