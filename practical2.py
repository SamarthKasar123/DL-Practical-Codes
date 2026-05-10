import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape dataset for CNN
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Build CNN model
model = models.Sequential()

# Convolution layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

# Pooling layer
model.add(layers.MaxPooling2D((2,2)))

# Second convolution layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# Second pooling layer
model.add(layers.MaxPooling2D((2,2)))

# Flatten layer
model.add(layers.Flatten())

# Dense layer
model.add(layers.Dense(64, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_images,
    train_labels,
    epochs=5
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_acc)
