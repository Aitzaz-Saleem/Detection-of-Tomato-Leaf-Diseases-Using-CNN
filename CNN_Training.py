# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:11:28 2022

@author: aitza
"""

import numpy as np
import os, requests, cv2, random
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Set TensorFlow to only use the first GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Using {len(logical_gpus)} GPU(s): {logical_gpus}")
else:
    print("No GPU found. Switching to CPU mode.")


dataset_dir = r'D:\Extra\Tomato_Diseases\train'

# Create an instance of the ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Load the training dataset using the ImageDataGenerator
train_dataset = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=42
)

# Load the validation dataset using the ImageDataGenerator
validation_dataset = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=42
)

# Print the number of classes in the dataset
num_classes = len(train_dataset.class_indices)
print("Number of classes:", num_classes)

# Print the names of the classes
class_names = list(train_dataset.class_indices.keys())
print("Class names:", class_names)

# Create a CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D (32, kernel_size = (3,3), activation='relu', input_shape = [224, 224,3])),
model.add(MaxPooling2D(pool_size = (2, 2))),

model.add(Conv2D (64, (3,3), activation='relu')),
model.add(MaxPooling2D((2, 2))),

model.add(Conv2D (64, (3,3), activation='relu')),
model.add(MaxPooling2D((2, 2))),

model.add(Conv2D (64, (3, 3), activation='relu')), 
model.add(MaxPooling2D((2, 2))),

model.add(Conv2D (64, (3, 3), activation='relu')), 
model.add(MaxPooling2D((2, 2))),

model.add(Conv2D(64, (3, 3), activation='relu')), 
model.add(MaxPooling2D((2, 2))),

model.add(Flatten()),

model.add(Dense(64,activation='relu'))

model.add(Dense(10,activation='softmax'))


# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.categorical_crossentropy,  
    metrics=['accuracy'],
    run_eagerly=None
)
print(model.summary())

# Define the checkpoint callback to save the model with the minimum loss
checkpoint_callback = ModelCheckpoint('model.h5', 
                                      monitor='val_loss', 
                                      save_best_only=True)

# Define early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=10,          
    verbose=1,            
    restore_best_weights=True  
)

# Define the CSVLogger callback to log training metrics to a CSV file
csv_logger = CSVLogger('training_log.csv')

# Train the model with the checkpoint callback
history = model.fit(train_dataset,
                    batch_size=32,
                    epochs=100, 
                    validation_data=validation_dataset, 
                    callbacks=[checkpoint_callback, csv_logger, early_stopping_callback])


# Plot accuracy vs. epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.savefig('accuracy_vs_epochs.png')  

# Plot loss vs. epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.savefig('loss_vs_epochs.png')  
plt.show()
