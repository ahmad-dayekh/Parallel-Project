import numpy as np
from keras.applications import VGG16

np.random.seed(2016)

def build_vgg16_model(weights_path=None):
    """
    Builds the VGG-16 model architecture and loads weights.

    Args:
        weights_path (str): Path to the pre-trained weights file.

    Returns:
        VGG-16 model with loaded weights.
    """
    # Use Keras's built-in VGG16 model for consistency and compatibility
    model = VGG16(weights=None)  # Initialize model without weights
    if weights_path:
        model.load_weights(weights_path)  # Load pre-trained weights
    return model


def save_model_weights_to_file(model, output_file):
    """
    Saves the weights and biases of a model to a text file.

    Args:
        model: Keras model with weights.
        output_file (str): File path for saving weights and biases.
    """
    with open(output_file, "w") as file:
        for layer_idx, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights:  # Only process layers with weights
                for weight_array in weights:
                    print(f"Layer {layer_idx} - Weight shape: {weight_array.shape}")
                    file.write(' '.join(map(str, weight_array.flatten())) + "\n")


# Set file paths
weights_file_path = 'vgg16_model.weights.h5'  # Replace with the correct weights file path
output_text_file_path = 'vgg16_weights_output.txt'  # Desired output file path for weights

# Build model, print summary, and save weights
vgg16_model = build_vgg16_model(weights_file_path)
print(vgg16_model.summary())
save_model_weights_to_file(vgg16_model, output_text_file_path)
print(f"Weights saved to {output_text_file_path}")
