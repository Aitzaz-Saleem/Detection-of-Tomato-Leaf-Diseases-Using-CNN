# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:43:49 2023

@author: aitza
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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
model = tf.keras.models.load_model('model_min_loss.h5')

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Get the true labels for the test dataset
true_labels = test_dataset.classes

# Generate predictions for the test dataset
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions, axis=1)

# Print classification report
class_names = list(test_dataset.class_indices.keys())
print('\nClassification Report:')
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Print confusion matrix
print('\nConfusion Matrix:')
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

# Calculate and print AUC score
auc_score = roc_auc_score(true_labels, predictions, multi_class='ovr')
print('\nAUC Score:', auc_score)
