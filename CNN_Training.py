# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:11:28 2022

@author: aitza
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger


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

# Create a CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

# Add fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.categorical_crossentropy(),
              metrics=['accuracy'],
              run_eagerly=None)
print(model.summary())

# Define the checkpoint callback to save the model with the minimum loss
checkpoint_callback = ModelCheckpoint('model_min_loss.h5', 
                                      monitor='val_loss', 
                                      save_best_only=True)

# Define the CSVLogger callback to log training metrics to a CSV file
csv_logger = CSVLogger('training_log.csv')

# Train the model with the checkpoint callback
model.fit(train_dataset,
          batch_size=32,
          epochs=100, 
          validation_data=validation_dataset, 
          callbacks=[checkpoint_callback, csv_logger])