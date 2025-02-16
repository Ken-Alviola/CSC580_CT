#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Display 5 original sample images before preprocessing
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i])
    ax.set_title("Original")
    ax.axis('off')
plt.show()

# Define a preprocessing function using OpenCV
def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (optional for OpenCV)
        img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Normalize to range [0,1]
        processed_images.append(img)
    return np.array(processed_images, dtype=np.float32)

# Apply preprocessing to training and test data
x_train_processed = preprocess_images(x_train)
x_test_processed = preprocess_images(x_test)

# Display 5 sample images after preprocessing
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(x_train_processed[i])
    ax.set_title("Processed")
    ax.axis('off')
plt.show()

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the network 
history = model.fit(x_train_processed, y_train, epochs=10, batch_size=64)

# Evaluate the network on the test set
test_loss, test_acc = model.evaluate(x_test_processed, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Use the trained network to predict on test data
predictions = model.predict(x_test_processed[:10])

# Display 10 predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))  # 2 rows, 5 columns
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for i, ax in enumerate(axes.flat):  # Loop over 10 images
    ax.imshow(cv2.cvtColor(x_test_processed[i], cv2.COLOR_BGR2RGB))  # Convert back to RGB for display
    ax.set_title(f"Pred: {class_names[np.argmax(predictions[i])]}")
    ax.axis('off')

plt.tight_layout()
plt.show()


