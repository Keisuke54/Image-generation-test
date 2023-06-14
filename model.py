# Code run on Kaggle due to '.DS_Store' issue when accessing directory 
# https://www.kaggle.com/code/keisukenakamura54/txttoimage-test-sequence

import numpy as np
import tensorflow as tf
from PIL import Image

# input texts (prompts)
prompts = ['lanscape with tree and grass land', 
           'lanscape with sun and shore', 
           'landscape with sun and photographer on cliff', 
           'landscape with boat and sun', 
           'landscape with river', 
           'landscape with tree and sky', 
           'landscape with tree, glassland, and sky', 
           'landscape with bridge and river', 
           'landscape with swing and ocean', 
           'landscape with bay and lighthouse']

# ouput images 
import os
from os import listdir
dir = 'Your Directory'
imgFolder = os.listdir(dir)

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

max_length = 50

txtSequence = []

for x in range(len(prompts)):
    sequences = tokenizer.texts_to_sequences(prompts)[x]
    
    if len(sequences) <= max_length:
        for i in range(max_length - len(sequences)):
            sequences.append(0)
            
    txtSequence.append(sequences)
    
txtSequence = np.array(txtSequence)

# converting images into sequence 
imgSequence = []

image_size = (224, 224)

for image_path in imgFolder:
    timage = Image.open(os.path.join(dir, image_path))
    image = image.resize(image_size)
    # convert the image to grayscale
    gray_image = image.convert('L')
    pixel_array = np.array(gray_image)
    pixel_sequence = pixel_array.flatten() / 255.0
    imgSequence.append(pixel_sequence)

# splitting data into training and validation sets
split_ratio = 0.8 
split_indexTxt = int(len(txtSequence) * split_ratio)
split_indexImg = int(len(imgSequence) * split_ratio)

train_text = txtSequence[:split_indexTxt]
train_images = np.array(imgSequence[:split_indexImg])

val_text = txtSequence[split_indexTxt:]
val_images = np.array(imgSequence[split_indexImg:])

# model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(image_size[0] * image_size[1], activation='softmax')
    # times 3 for RGB, color images 
    #Dense(image_size[0] * image_size[1] * 3, activation='softmax')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_text, train_images, epochs=100, batch_size=32, validation_data=(val_text, val_images))

# testing the model 
import matplotlib.pyplot as plt

text_sequence = "landscape with tree, sun, and ocean"

encoded_sequence = tokenizer.texts_to_sequences([text_sequence])
encoded_sequence = pad_sequences(encoded_sequence, maxlen=max_length, padding='post')

# Generate the image
predicted_pixels = model.predict(encoded_sequence)
decoded_pixels = predicted_pixels.reshape((224, 224))
# when RGB
#decoded_pixels = predicted_pixels.reshape((224, 224, 3))
decoded_pixels *= 255.0

print(decoded_pixels)

plt.imshow(decoded_pixels)
plt.axis('off')
plt.show()