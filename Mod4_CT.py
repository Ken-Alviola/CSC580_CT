#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: Generate synthetic data
N = 100

# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=0.1 * np.eye(2), size=(N // 2,))
y_zeros = np.zeros((N // 2,))

# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(mean=np.array((1, 1)), cov=0.1 * np.eye(2), size=(N // 2,))
y_ones = np.ones((N // 2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])

# Step 2: Plot x_zeros and x_ones on the same graph
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label="Class 0", alpha=0.7)
plt.scatter(x_ones[:, 0], x_ones[:, 1], label="Class 1", alpha=0.7)
plt.legend()
plt.title("Synthetic Data")
plt.show()

# Step 3: Generate a TensorFlow graph
class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.W = tf.Variable(tf.random.normal((2, 1)), trainable=True)
        self.b = tf.Variable(tf.random.normal((1,)), trainable=True)

    def call(self, x):
        logits = tf.squeeze(tf.matmul(x, self.W) + self.b)
        probs = tf.sigmoid(logits)
        return logits, probs

# Initialize model
model = LogisticRegression()

# Define loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Prepare dataset
x_tensor = tf.convert_to_tensor(x_np, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_np, dtype=tf.float32)

# Step 4: Train the model
epochs = 1000
batch_size = N
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits, probs = model(x_tensor)
        loss = loss_fn(y_tensor, logits)

    gradients = tape.gradient(loss, [model.W, model.b])
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Get trained weights
trained_W = model.W.numpy()
trained_b = model.b.numpy()
print(f"Trained weights: {trained_W}, Trained bias: {trained_b}")

# Step 5: Make predictions
_, predicted_probs = model(x_tensor)
predicted_classes = tf.round(predicted_probs).numpy()

# Plot predictions
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label="Class 0 (True)", alpha=0.7)
plt.scatter(x_ones[:, 0], x_ones[:, 1], label="Class 1 (True)", alpha=0.7)

# Highlight predicted class 1
plt.scatter(x_np[predicted_classes.flatten() == 1, 0],
            x_np[predicted_classes.flatten() == 1, 1],
            label="Predicted Class 1", alpha=0.5, marker='x', color='red')

# Highlight predicted class 0
plt.scatter(x_np[predicted_classes.flatten() == 0, 0],
            x_np[predicted_classes.flatten() == 0, 1],
            label="Predicted Class 0", alpha=0.5, marker='x', color='black')

plt.legend()
plt.title("Predicted Outputs")
plt.show()

# Compute accuracy
correct_predictions = np.sum(predicted_classes.flatten() == y_np)
accuracy = correct_predictions / len(y_np)

print(f"Model Accuracy: {accuracy:.2%}")

#Generate more challenging synthetic data
N = 100

# Zeros form a Gaussian centered at (-0.2, -0.2) with smaller spread
x_zeros = np.random.multivariate_normal(mean=np.array((-0.20, -0.20)), cov=0.05 * np.eye(2), size=(N // 2,))
y_zeros = np.zeros((N // 2,))

# Ones form a Gaussian centered at (0.2, 0.2) with smaller spread
x_ones = np.random.multivariate_normal(mean=np.array((0.20, 0.20)), cov=0.05 * np.eye(2), size=(N // 2,))
y_ones = np.ones((N // 2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])


# In[21]:


# Plot x_zeros and x_ones on the same graph
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label="Class 0", alpha=0.7)
plt.scatter(x_ones[:, 0], x_ones[:, 1], label="Class 1", alpha=0.7)
plt.legend()
plt.title("Synthetic Data")
plt.show()

# Prepare dataset
x_tensor = tf.convert_to_tensor(x_np, dtype=tf.float32)

# Make predictions
_, predicted_probs = model(x_tensor)
predicted_classes = tf.round(predicted_probs).numpy()

# Plot predictions
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label="Class 0 (True)", alpha=0.7)
plt.scatter(x_ones[:, 0], x_ones[:, 1], label="Class 1 (True)", alpha=0.7)

# Highlight predicted class 1
plt.scatter(x_np[predicted_classes.flatten() == 1, 0],
            x_np[predicted_classes.flatten() == 1, 1],
            label="Predicted Class 1", alpha=0.5, marker='x', color='red')

# Highlight predicted class 0
plt.scatter(x_np[predicted_classes.flatten() == 0, 0],
            x_np[predicted_classes.flatten() == 0, 1],
            label="Predicted Class 0", alpha=0.5, marker='x', color='black')

plt.legend()
plt.title("Predicted Outputs")
plt.show()


# Compute accuracy
correct_predictions = np.sum(predicted_classes.flatten() == y_np)
accuracy = correct_predictions / len(y_np)

print(f"Model Accuracy: {accuracy:.2%}")




