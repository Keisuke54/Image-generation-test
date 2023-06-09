import numpy as np
import tensorflow as tf

# input texts (prompts)
prompts = ['' , '']

# ouput images 


# model
Sequential = tf.keras.models.Sequential

# layers 
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense


# converting texts into sequence 
Tokenizer = tf.keras.preprocessing.text.Tokenizer

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
tokenizer = Tokenizer(char_level=True, lower=True)

tokenizer.fit_on_texts(prompts)
vocab_size = len(tokenizer.word_index) + 1

