#Program to implement MNIST Dataset learning of digits 0 – 9.

# Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load the MNIST dataset
# The dataset contains 60,000 training images and 10,000 test images
# Each image is 28x28 pixels (grayscale)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess the data
# Normalize pixel values from range [0,255] to [0,1] for better learning
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images from 28x28 into a vector of size 784
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# 3. Build the Neural Network model
# Sequential model: input layer → hidden layer → output layer
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)), 
    layers.Dense(64, activation='relu'),                      
    layers.Dense(10, activation='softmax')                    # Output layer (10 digits)
])

# 4. Compile the model
# - optimizer: adjusts weights during training
# - loss: measures prediction error
# - metrics: used to evaluate performance
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model
# The model learns patterns from the training data
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 6. Evaluate the model
# Test accuracy on unseen data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

# 7. Make predictions
# Predict the class of the first test image
predictions = model.predict(x_test)

# Display predicted digit vs actual digit
import numpy as np
predicted_digit = np.argmax(predictions[0])
print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])