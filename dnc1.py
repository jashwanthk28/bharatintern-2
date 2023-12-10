# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.engine.sequential import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
# Setting the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Creating the image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
encodings = ['utf-8', 'latin-1']
for encoding in encodings:
    try:
        train_generator = datagen.flow_from_directory('C:\\Users\\HP\\Desktop\\bharatintern\\data', target_size=(150, 150), batch_size=32, class_mode='binary', subset='training')
        break
    except UnicodeDecodeError:
        print(f"Failed to decode using {encoding} encoding. Trying the next one.")

# Loading the validation dataset
for encoding in encodings:
    try:
        validation_generator = datagen.flow_from_directory('C:\\Users\\HP\\Desktop\\bharatintern\\data', target_size=(150, 150), batch_size=32, class_mode='binary', subset='validation')
        break
    except UnicodeDecodeError:
        print(f"Failed to decode using {encoding} encoding. Trying the next one.")

# Creating the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit
try:
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size

except AttributeError as e:
    print(f'AttributeError: {e}')

# Evaluating the model

try:
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

except AttributeError as e:
    print(f'AttributeError: {e}')
