import numpy as np
import tensorflow as tf

shakespeare_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

res = input('We are going to train our model with a default play, if ok press ENTER otherwise write \'custom\': ')
while res not in ['', 'custom']:
    res = input('wrong answer! please leave empty for default, or write \'custom\':')

path_to_file = tf.keras.utils.get_file('shakespeare.txt', shakespeare_url) if res == '' else 'ownFile'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = {char: i for i, char in enumerate(vocab)}
idx2npArr = np.array(vocab)


def encode(raw_text):
    return np.array([char2idx[c] for c in raw_text])


text_as_int = encode(text)
