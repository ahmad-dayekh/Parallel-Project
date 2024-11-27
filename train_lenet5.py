import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
print("Loading and preprocessing the MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Expand dimensions to include the channel dimension (grayscale images)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Resize images to 32x32 pixels (LeNet-5 expects 32x32 input)
x_train = tf.image.resize(x_train, [32, 32]).numpy()
x_test = tf.image.resize(x_test, [32, 32]).numpy()

# Normalize the images to [0, 1] range
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data preprocessing completed.")

# Define the LeNet-5 architecture
print("Defining the LeNet-5 model...")
model = models.Sequential()

# C1: Convolutional Layer (6 filters, 5x5 kernel, tanh activation)
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1)))

# S2: Average Pooling Layer (2x2 pool size, stride 2)
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))

# C3: Convolutional Layer (16 filters, 5x5 kernel, tanh activation)
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))

# S4: Average Pooling Layer (2x2 pool size, stride 2)
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))

# C5: Convolutional Layer (120 filters, 5x5 kernel, tanh activation)
model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), activation='tanh'))

# Flatten the output from the convolutional layers
model.add(layers.Flatten())

# F6: Fully Connected Layer (84 units, tanh activation)
model.add(layers.Dense(units=84, activation='tanh'))

# Output Layer: Fully Connected Layer (10 units for 10 classes, softmax activation)
model.add(layers.Dense(units=10, activation='softmax'))

# Display the model architecture
model.summary()

# Compile the model with optimizer, loss function, and metrics
print("Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Starting training...")
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
print("Training finished.")

# Evaluate the model on the test dataset
print("Evaluating the model on the test dataset...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuracy on 10000 test images: {test_acc * 100:.2f}%')

# Save the trained model weights
print("Saving the model weights...")
model.save_weights('lenet5_mnist.weights.h5')
print('Model weights saved to lenet5_mnist.weights.h5')

# Create the weights directory if it doesn't exist
if not os.path.exists('weights'):
    os.makedirs('weights')

# Extract and save the weights for use in C code
print('Extracting and saving weights for C code...')
for layer in model.layers:
    # Check if the layer has weights (some layers like pooling don't have weights)
    if len(layer.get_weights()) > 0:
        weights = layer.get_weights()  # This returns a list: [weights, biases] or just [weights]
        layer_name = layer.name

        # Save weights
        weight = weights[0]
        weight_flat = weight.flatten()
        weight_filename = os.path.join('weights', f'{layer_name}_weight.txt')  # Save in /weights/
        np.savetxt(weight_filename, weight_flat, fmt='%.6f')
        print(f'Saved weights of {layer_name} to {weight_filename}')

        # Check if biases exist
        if len(weights) > 1:
            bias = weights[1]
            bias_flat = bias.flatten()
            bias_filename = os.path.join('weights', f'{layer_name}_bias.txt')  # Save in /weights/
            np.savetxt(bias_filename, bias_flat, fmt='%.6f')
            print(f'Saved biases of {layer_name} to {bias_filename}')

print('All weights have been saved to the /weights/ directory.')

# Plot training history
print("Generating training summary plots...")
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('weights/training_summary.png')
plt.show()

print("Training summary saved as 'weights/training_summary.png'.")
