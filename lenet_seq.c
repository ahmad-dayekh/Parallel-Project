#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Network architecture parameters
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_CHANNELS 1

// Layer dimensions
#define CONV1_FILTERS 6
#define CONV1_SIZE 5
#define CONV1_OUTPUT 28  // (32 - 5 + 1)

#define POOL1_SIZE 2
#define POOL1_OUTPUT 14  // (28 / 2)

#define CONV2_FILTERS 16
#define CONV2_SIZE 5
#define CONV2_OUTPUT 10  // (14 - 5 + 1)

#define POOL2_SIZE 2
#define POOL2_OUTPUT 5   // (10 / 2)

#define CONV3_FILTERS 120
#define CONV3_SIZE 5
#define CONV3_OUTPUT 1   // (5 - 5 + 1)

#define FC1_SIZE 84
#define OUTPUT_SIZE 10

// Structure to hold the network weights and biases
typedef struct {
    float* conv1_weights;
    float* conv1_bias;
    float* conv2_weights;
    float* conv2_bias;
    float* conv3_weights;
    float* conv3_bias;
    float* fc1_weights;
    float* fc1_bias;
    float* fc2_weights;
    float* fc2_bias;
} NetworkWeights;

// Helper function to load weights from file
float* load_weights(const char* filename, int size) {
    float* weights = (float*)malloc(size * sizeof(float));
    if (!weights) {
        printf("Failed to allocate memory for weights\n");
        exit(1);
    }

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "weights/%s", filename);
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        printf("Error opening file: %s\n", filepath);
        free(weights);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(fp, "%f", &weights[i]) != 1) {
            printf("Error reading weight at index %d from %s\n", i, filepath);
            fclose(fp);
            free(weights);
            exit(1);
        }
    }

    fclose(fp);
    return weights;
}

// Rearrange convolutional weights to match expected ordering
void rearrange_conv_weights(float* weights_in, float* weights_out,
                            int kernel_size, int input_channels, int output_channels) {
    for (int o = 0; o < output_channels; o++) {
        for (int i = 0; i < input_channels; i++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int idx_in = kh * kernel_size * input_channels * output_channels +
                                 kw * input_channels * output_channels +
                                 i * output_channels + o;
                    int idx_out = o * input_channels * kernel_size * kernel_size +
                                  i * kernel_size * kernel_size +
                                  kh * kernel_size + kw;
                    weights_out[idx_out] = weights_in[idx_in];
                }
            }
        }
    }
}

// Activation function (tanh)
float tanh_activate(float x) {
    return tanh(x);
}

// Convolution operation
void convolution(float* input, int input_width, int input_height, int input_channels,
                 float* weights, float* bias, int kernel_size, int num_filters,
                 float* output, int output_width, int output_height) {
    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float sum = 0.0f;
                for (int c = 0; c < input_channels; c++) {
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            int in_i = i + ki;
                            int in_j = j + kj;
                            int input_idx = c * input_height * input_width +
                                            in_i * input_width + in_j;
                            int weight_idx = f * input_channels * kernel_size * kernel_size +
                                             c * kernel_size * kernel_size +
                                             ki * kernel_size + kj;
                            float input_val = input[input_idx];
                            float weight = weights[weight_idx];
                            sum += input_val * weight;
                        }
                    }
                }
                sum += bias[f];
                output[f * output_height * output_width + i * output_width + j] = tanh_activate(sum);
            }
        }
    }
}

// Average pooling operation
void avg_pooling(float* input, int input_width, int input_height, int num_channels,
                 int pool_size, float* output, int output_width, int output_height) {
    float scale = 1.0f / (pool_size * pool_size);

    for (int c = 0; c < num_channels; c++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float sum = 0.0f;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_i = i * pool_size + pi;
                        int in_j = j * pool_size + pj;
                        sum += input[c * input_height * input_width +
                                     in_i * input_width + in_j];
                    }
                }
                output[c * output_height * output_width + i * output_width + j] = sum * scale;
            }
        }
    }
}

// Softmax activation for output layer
void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }

    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Dense layer operation
void dense(float* input, float* weights, float* bias,
           int input_size, int output_size, float* output, int is_output) {
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + i];  // Note the index order
        }
        sum += bias[i];
        output[i] = is_output ? sum : tanh_activate(sum);
    }

    if (is_output) {
        softmax(output, output_size);
    }
}

// Initialize network weights
NetworkWeights init_network() {
    NetworkWeights weights;

    // Load convolution layer 1 weights and rearrange
    int conv1_weights_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE;
    float* conv1_weights_raw = load_weights("conv2d_weight.txt", conv1_weights_size);
    weights.conv1_weights = (float*)malloc(conv1_weights_size * sizeof(float));
    rearrange_conv_weights(conv1_weights_raw, weights.conv1_weights,
                           CONV1_SIZE, INPUT_CHANNELS, CONV1_FILTERS);
    free(conv1_weights_raw);

    weights.conv1_bias = load_weights("conv2d_bias.txt", CONV1_FILTERS);

    // Load convolution layer 2 weights and rearrange
    int conv2_weights_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE;
    float* conv2_weights_raw = load_weights("conv2d_1_weight.txt", conv2_weights_size);
    weights.conv2_weights = (float*)malloc(conv2_weights_size * sizeof(float));
    rearrange_conv_weights(conv2_weights_raw, weights.conv2_weights,
                           CONV2_SIZE, CONV1_FILTERS, CONV2_FILTERS);
    free(conv2_weights_raw);

    weights.conv2_bias = load_weights("conv2d_1_bias.txt", CONV2_FILTERS);

    // Load convolution layer 3 weights and rearrange
    int conv3_weights_size = CONV3_FILTERS * CONV2_FILTERS * CONV3_SIZE * CONV3_SIZE;
    float* conv3_weights_raw = load_weights("conv2d_2_weight.txt", conv3_weights_size);
    weights.conv3_weights = (float*)malloc(conv3_weights_size * sizeof(float));
    rearrange_conv_weights(conv3_weights_raw, weights.conv3_weights,
                           CONV3_SIZE, CONV2_FILTERS, CONV3_FILTERS);
    free(conv3_weights_raw);

    weights.conv3_bias = load_weights("conv2d_2_bias.txt", CONV3_FILTERS);

    // Load dense layer weights
    int fc1_weights_size = CONV3_FILTERS * FC1_SIZE;
    weights.fc1_weights = load_weights("dense_weight.txt", fc1_weights_size);
    weights.fc1_bias = load_weights("dense_bias.txt", FC1_SIZE);

    int fc2_weights_size = FC1_SIZE * OUTPUT_SIZE;
    weights.fc2_weights = load_weights("dense_1_weight.txt", fc2_weights_size);
    weights.fc2_bias = load_weights("dense_1_bias.txt", OUTPUT_SIZE);

    return weights;
}

// Free network weights
void free_network(NetworkWeights* weights) {
    free(weights->conv1_weights);
    free(weights->conv1_bias);
    free(weights->conv2_weights);
    free(weights->conv2_bias);
    free(weights->conv3_weights);
    free(weights->conv3_bias);
    free(weights->fc1_weights);
    free(weights->fc1_bias);
    free(weights->fc2_weights);
    free(weights->fc2_bias);
}

// Forward pass through the network
void forward(float* input, float* output, NetworkWeights* weights) {
    clock_t start_total = clock();
    clock_t start, end;
    double cpu_time_used;

    // Allocate memory for intermediate results
    float* conv1_output = (float*)malloc(CONV1_FILTERS * CONV1_OUTPUT * CONV1_OUTPUT * sizeof(float));
    float* pool1_output = (float*)malloc(CONV1_FILTERS * POOL1_OUTPUT * POOL1_OUTPUT * sizeof(float));
    float* conv2_output = (float*)malloc(CONV2_FILTERS * CONV2_OUTPUT * CONV2_OUTPUT * sizeof(float));
    float* pool2_output = (float*)malloc(CONV2_FILTERS * POOL2_OUTPUT * POOL2_OUTPUT * sizeof(float));
    float* conv3_output = (float*)malloc(CONV3_FILTERS * CONV3_OUTPUT * CONV3_OUTPUT * sizeof(float));
    float* fc1_output = (float*)malloc(FC1_SIZE * sizeof(float));

    // Conv1 layer
    start = clock();
    convolution(input, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS,
                weights->conv1_weights, weights->conv1_bias,
                CONV1_SIZE, CONV1_FILTERS,
                conv1_output, CONV1_OUTPUT, CONV1_OUTPUT);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Conv1 layer time: %.7f seconds\n", cpu_time_used);

    // Pool1 layer
    start = clock();
    avg_pooling(conv1_output, CONV1_OUTPUT, CONV1_OUTPUT, CONV1_FILTERS,
                POOL1_SIZE, pool1_output, POOL1_OUTPUT, POOL1_OUTPUT);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Pool1 layer time: %.7f seconds\n", cpu_time_used);

    // Conv2 layer
    start = clock();
    convolution(pool1_output, POOL1_OUTPUT, POOL1_OUTPUT, CONV1_FILTERS,
                weights->conv2_weights, weights->conv2_bias,
                CONV2_SIZE, CONV2_FILTERS,
                conv2_output, CONV2_OUTPUT, CONV2_OUTPUT);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Conv2 layer time: %.7f seconds\n", cpu_time_used);

    // Pool2 layer
    start = clock();
    avg_pooling(conv2_output, CONV2_OUTPUT, CONV2_OUTPUT, CONV2_FILTERS,
                POOL2_SIZE, pool2_output, POOL2_OUTPUT, POOL2_OUTPUT);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Pool2 layer time: %.7f seconds\n", cpu_time_used);

    // Conv3 layer
    start = clock();
    convolution(pool2_output, POOL2_OUTPUT, POOL2_OUTPUT, CONV2_FILTERS,
                weights->conv3_weights, weights->conv3_bias,
                CONV3_SIZE, CONV3_FILTERS,
                conv3_output, CONV3_OUTPUT, CONV3_OUTPUT);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Conv3 layer time: %.7f seconds\n", cpu_time_used);

    // FC1 layer
    start = clock();
    dense(conv3_output, weights->fc1_weights, weights->fc1_bias,
          CONV3_FILTERS, FC1_SIZE, fc1_output, 0);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("FC1 layer time: %.7f seconds\n", cpu_time_used);

    // FC2 layer (output)
    start = clock();
    dense(fc1_output, weights->fc2_weights, weights->fc2_bias,
          FC1_SIZE, OUTPUT_SIZE, output, 1);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("FC2 layer time: %.7f seconds\n", cpu_time_used);

    // Total time
    clock_t end_total = clock();
    double total_time = ((double) (end_total - start_total)) / CLOCKS_PER_SEC;
    printf("\nTotal execution time: %.7f seconds\n", total_time);

    // Free intermediate results
    free(conv1_output);
    free(pool1_output);
    free(conv2_output);
    free(pool2_output);
    free(conv3_output);
    free(fc1_output);
}

// Read image from file
float* read_image(const char* filename) {
    float* image = (float*)malloc(INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening image file: %s\n", filename);
        free(image);
        return NULL;
    }

    unsigned char pixel;
    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; i++) {
        if (fread(&pixel, 1, 1, fp) != 1) {
            printf("Error reading image data\n");
            fclose(fp);
            free(image);
            return NULL;
        }
        image[i] = (float)pixel / 255.0f;
    }

    fclose(fp);
    return image;
}

int main() {
    // Initialize network
    NetworkWeights weights = init_network();

    // Read input image
    float* input = read_image("test_images/synthetic_digit_0_1.raw");
    if (!input) {
        printf("Failed to read input image\n");
        free_network(&weights);
        return 1;
    }

    // Allocate output buffer
    float* output = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Perform forward pass
    forward(input, output, &weights);

    // Find and print the predicted digit
    int predicted_class = 0;
    float max_prob = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    printf("Predicted digit: %d (confidence: %.2f%%)\n",
           predicted_class, max_prob * 100);

    // Print all probabilities
    printf("\nClass probabilities:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Digit %d: %.2f%%\n", i, output[i] * 100);
    }

    // Cleanup
    free(input);
    free(output);
    free_network(&weights);

    return 0;
}
