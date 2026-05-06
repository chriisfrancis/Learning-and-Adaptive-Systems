import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -----------------------------------
# Load MNIST dataset (digits 0–9)
# -----------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images (28x28 → 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -----------------------------------
# Build Multilayer Perceptron Model
# -----------------------------------
model = Sequential()

# Input + Hidden Layer 1
model.add(Dense(128, activation='relu', input_shape=(784,)))

# Hidden Layer 2 (feature learning)
model.add(Dense(64, activation='relu'))

# Output Layer (10 classes)
model.add(Dense(10, activation='softmax'))

# -----------------------------------
# Compile Model
# -----------------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------------
# Train Model
# -----------------------------------
model.fit(x_train, y_train, epochs=5, batch_size=32)

# -----------------------------------
# Evaluate Model
# -----------------------------------
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)
