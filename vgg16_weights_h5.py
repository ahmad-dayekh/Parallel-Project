from keras.applications import VGG16
import os

# Load the VGG16 model pretrained on ImageNet
model = VGG16(weights='imagenet')

# Save weights to an .h5 file
model.save_weights("vgg16_model.weights.h5")
print(f"Weights and biases saved")