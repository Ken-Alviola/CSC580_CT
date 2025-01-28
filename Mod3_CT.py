#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(101)
tf.random.set_seed(101)

# Generating random linear data
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

# Normalize the data
x_mean, x_std = np.mean(x), np.std(x)
y_mean, y_std = np.mean(y), np.std(y)

x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

n = len(x)  # Number of data points


# Step 1: Plot the training data
plt.scatter(x, y, label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Step 2: Create placeholders for X and Y 
X = tf.constant(x, dtype=tf.float32)
Y = tf.constant(y, dtype=tf.float32)

# Step 3: Initialize trainable variables for weights and bias
W = tf.Variable(np.random.randn() * 0.01, name="weight", dtype=tf.float32)
b = tf.Variable(np.random.randn() * 0.01, name="bias", dtype=tf.float32)

# Step 4: Define hyperparameters
learning_rate = 0.01  
training_epochs = 1000

# Step 5: Define the hypothesis, cost function, and optimizer
def linear_model(X):
    return W * X + b

def cost_function(X, Y):
    predictions = linear_model(X)
    return tf.reduce_mean(tf.square(predictions - Y))

optimizer = tf.optimizers.SGD(learning_rate)

# Step 6: Training process
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        cost = cost_function(X, Y)
    gradients = tape.gradient(cost, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Cost: {cost.numpy()}, Weight: {W.numpy()}, Bias: {b.numpy()}")


# Final results
print("\nTraining complete!")
print(f"Final Cost: {cost.numpy()}")
print(f"Weight: {W.numpy()}")
print(f"Bias: {b.numpy()}")

# Step 7: Plot the fitted line
plt.scatter(x, y, label="Training Data")
plt.plot(x, linear_model(X).numpy(), color="red", label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()






