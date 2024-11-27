import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf

# Create a directory to save the output files
output_dir = 'test_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the MNIST dataset
print("Loading the MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize a dictionary to store one image per digit
digit_images = {}

# Combine training and test sets for a broader selection
x_total = np.concatenate((x_train, x_test))
y_total = np.concatenate((y_train, y_test))

# Loop through the dataset to find one image per digit
print("Selecting one image for each digit (0-9)...")
for i in range(len(y_total)):
    label = y_total[i]
    if label not in digit_images:
        # Pad the image to 32x32
        image = np.pad(x_total[i], ((2,2), (2,2)), 'constant')
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        digit_images[label] = image
    if len(digit_images) == 10:
        break

# Sort the digits for consistent ordering
sorted_digits = sorted(digit_images.keys())

# Create a grid image containing all digits
print("Creating a grid image of all digits...")
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for ax, digit in zip(axes, sorted_digits):
    image = digit_images[digit]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{digit}')
    ax.axis('off')

plt.tight_layout()
grid_image_path = os.path.join(output_dir, 'digits_grid.png')
plt.savefig(grid_image_path)
plt.close(fig)
print(f"Grid image saved to {grid_image_path}")

# Save the raw pixel data for each digit
print("Saving raw pixel data for each digit...")
for digit in sorted_digits:
    image = digit_images[digit]
    # Convert to uint8 for raw file
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Save as raw file
    data_filename = f'digit_{digit}.raw'
    data_filepath = os.path.join(output_dir, data_filename)
    image_uint8.tofile(data_filepath)
    print(f'Saved raw pixel data of digit {digit} to {data_filename}')

# Print the dimensions of saved images for verification
print("\nVerification of image dimensions:")
for digit in sorted_digits:
    image = digit_images[digit]
    print(f"Digit {digit} image shape: {image.shape}")

# Print extra verification info
print("\nVerification of pixel value ranges:")
for digit in sorted_digits:
    image = digit_images[digit]
    print(f"Digit {digit} - Min: {image.min():.3f}, Max: {image.max():.3f}")