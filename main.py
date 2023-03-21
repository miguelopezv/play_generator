import numpy as np
import tensorflow as tf

from constants import SHAKESPEARE_URL, BATCH_SIZE, EMBEDDING_DIM, RNN_UNITS, BUFFER_SIZE, SEQ_LENGTH
from helpers import encode, split_input_target, build_model, loss, checkpoint_callback

res = input('We are going to train our model with a default play, if ok press ENTER otherwise write \'custom\': ')
while res not in ['', 'custom']:
    res = input('wrong answer! please leave empty for default, or write \'custom\':')

path_to_file = tf.keras.utils.get_file('shakespeare.txt', SHAKESPEARE_URL) if res == '' else 'ownFile'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = encode(char2idx, text)

# divide dataset into batches
text_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = text_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

# Transform each batch into input and target
dataset = sequences.map(split_input_target)

# Create batches to train the model
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
model = build_model(len(vocab), EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.compile(optimizer='adam', loss=loss)

# Comment this line after the model has been trained
history = model.fit(data, epochs=1, callbacks=[checkpoint_callback])

# With model already trained, use this to recreate the model expecting 1 input
# model = build_model(len(vocab), EMBEDDING_DIM, RNN_UNITS, batch_size=1)
# model.load_weights(tf.train.latest_checkpoint(CHECKPOINTS_DIR))
# model.build(tf.TensorShape([1, None]))
#
# inp = input("Type a starting string: ")
# print(generate_text(model, inp, char2idx, idx2char))
