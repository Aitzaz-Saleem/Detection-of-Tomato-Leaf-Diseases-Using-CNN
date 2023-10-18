# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:43:49 2023

@author: aitza
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


test_dir = r'D:\Extra\Tomato_Diseases\test'

# Create an instance of the ImageDataGenerator for testing
test_data_generator = ImageDataGenerator(rescale=1.0/255)

# Load the test dataset using the ImageDataGenerator
test_dataset = test_data_generator.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print('Test Accuracy: {:.2f}%'.format(accuracy*100))
print('Test Loss: {:.2f}%'.format(loss*100))
