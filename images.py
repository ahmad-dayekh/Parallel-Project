import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from scipy import ndimage
from skimage import transform as skimage_transform
from skimage.util import random_noise
from skimage import exposure
import random

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
        image = np.pad(x_total[i], ((2, 2), (2, 2)), 'constant')
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

# Function to apply transformations
def apply_transformations(image):
    transformed = image.copy()

    # Random rotation between -20 and 20 degrees
    angle = random.uniform(-20, 20)
    transformed = ndimage.rotate(transformed, angle, reshape=False, mode='constant', cval=0.0)

    # Random translation
    shift_x = random.uniform(-5, 5)
    shift_y = random.uniform(-5, 5)
    transformed = ndimage.shift(transformed, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    # Random zoom
    zoom = random.uniform(0.9, 1.1)
    h, w = transformed.shape
    transformed = ndimage.zoom(transformed, zoom)
    # Crop or pad to get back to original size
    crop_h = transformed.shape[0] - h
    crop_w = transformed.shape[1] - w
    if crop_h > 0:
        start_h = crop_h // 2
        transformed = transformed[start_h:start_h + h, :]
    else:
        pad_h = -crop_h // 2
        transformed = np.pad(transformed, ((pad_h, pad_h), (0, 0)), 'constant')
    if crop_w > 0:
        start_w = crop_w // 2
        transformed = transformed[:, start_w:start_w + w]
    else:
        pad_w = -crop_w // 2
        transformed = np.pad(transformed, ((0, 0), (pad_w, pad_w)), 'constant')

    # Random shearing
    shear = random.uniform(-0.3, 0.3)
    afine_tf = skimage_transform.AffineTransform(shear=shear)
    transformed = skimage_transform.warp(transformed, inverse_map=afine_tf, mode='constant', cval=0.0)

    # Elastic deformation
    transformed = elastic_transform(transformed, alpha=random.uniform(30, 36), sigma=random.uniform(5, 6))

    # **Rescale intensity to ensure non-negative values**
    transformed = exposure.rescale_intensity(transformed, out_range=(0, 1))

    # Brightness and contrast adjustments
    gamma = random.uniform(0.8, 1.2)
    gain = random.uniform(0.9, 1.1)
    transformed = exposure.adjust_gamma(transformed, gamma=gamma, gain=gain)

    # Add Gaussian blur
    sigma_blur = random.uniform(0, 1)
    transformed = ndimage.gaussian_filter(transformed, sigma=sigma_blur)

    # Add random noise
    transformed = random_noise(transformed, mode='s&p', amount=0.02)

    # Random inversion
    if random.choice([True, False]):
        transformed = 1.0 - transformed

    # Random occlusion
    if random.choice([True, False]):
        h, w = transformed.shape
        occlusion_size = random.randint(5, 15)
        x_start = random.randint(0, w - occlusion_size)
        y_start = random.randint(0, h - occlusion_size)
        transformed[y_start:y_start + occlusion_size, x_start:x_start + occlusion_size] = 0.0

    # Random erasing
    if random.choice([True, False]):
        h, w = transformed.shape
        eraser_size = random.randint(5, 15)
        x_start = random.randint(0, w - eraser_size)
        y_start = random.randint(0, h - eraser_size)
        transformed[y_start:y_start + eraser_size, x_start:x_start + eraser_size] = random.uniform(0.0, 1.0)

    # Clip values to [0, 1]
    transformed = np.clip(transformed, 0, 1)

    return transformed

# Function to perform elastic deformation
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images."""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).flatten(), (x + dx).flatten()

    distorted_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted_image

# Create multiple synthetic images per digit
print("\nCreating multiple synthetic images per digit...")
num_synthetic_per_digit = 5  # Number of synthetic images per digit
synthetic_images = {}
for digit in sorted_digits:
    image = digit_images[digit]
    synthetic_images[digit] = []
    for idx in range(num_synthetic_per_digit):
        transformed_image = apply_transformations(image)
        synthetic_images[digit].append(transformed_image)

# Save the synthetic images as raw files
print("Saving synthetic images...")
for digit in sorted_digits:
    for idx, image in enumerate(synthetic_images[digit]):
        # Convert to uint8 for raw file
        image_uint8 = (image * 255).astype(np.uint8)

        # Save as raw file
        data_filename = f'synthetic_digit_{digit}_{idx}.raw'
        data_filepath = os.path.join(output_dir, data_filename)
        image_uint8.tofile(data_filepath)
        print(f'Saved synthetic image {idx} of digit {digit} to {data_filename}')

# Create grid images containing all synthetic digits
print("Creating grid images of synthetic digits...")
for digit in sorted_digits:
    fig, axes = plt.subplots(1, num_synthetic_per_digit, figsize=(15, 2))
    for ax, image in zip(axes, synthetic_images[digit]):
        ax.imshow(image, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    synthetic_grid_image_path = os.path.join(output_dir, f'synthetic_digits_grid_{digit}.png')
    plt.savefig(synthetic_grid_image_path)
    plt.close(fig)
    print(f"Synthetic grid image for digit {digit} saved to {synthetic_grid_image_path}")

# Additionally, create a single grid image showing one synthetic image per digit
print("Creating a grid image of one synthetic image per digit...")
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for ax, digit in zip(axes, sorted_digits):
    # Use the first synthetic image for each digit
    image = synthetic_images[digit][0]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{digit}')
    ax.axis('off')

plt.tight_layout()
synthetic_grid_image_path = os.path.join(output_dir, 'synthetic_digits_grid.png')
plt.savefig(synthetic_grid_image_path)
plt.close(fig)
print(f"Synthetic grid image saved to {synthetic_grid_image_path}")