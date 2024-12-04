#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (cudaSuccess != err) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

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
__device__ float tanh_activate(float x) {
    return tanhf(x);
}

// Convolution operation
__global__ void convolution_cuda(float* input, int input_width, int input_height, int input_channels,
                                 float* weights, float* bias, int kernel_size, int num_filters,
                                 float* output, int output_width, int output_height) {
    int f = blockIdx.z * blockDim.z + threadIdx.z; // Filter index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Output row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output column

    if (f < num_filters && i < output_height && j < output_width) {
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

// Average pooling operation
__global__ void avg_pooling_cuda(float* input, int input_width, int input_height, int num_channels,
                                 int pool_size, float* output, int output_width, int output_height) {
    int c = blockIdx.z * blockDim.z + threadIdx.z; // Channel index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Output row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output column

    if (c < num_channels && i < output_height && j < output_width) {
        float sum = 0.0f;
        for (int pi = 0; pi < pool_size; pi++) {
            for (int pj = 0; pj < pool_size; pj++) {
                int in_i = i * pool_size + pi;
                int in_j = j * pool_size + pj;
                sum += input[c * input_height * input_width +
                             in_i * input_width + in_j];
            }
        }
        float scale = 1.0f / (pool_size * pool_size);
        output[c * output_height * output_width + i * output_width + j] = sum * scale;
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
__global__ void dense_cuda(float* input, float* weights, float* bias,
                           int input_size, int output_size, float* output, int is_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Output neuron index

    if (i < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + i];  // Note the index order
        }
        sum += bias[i];
        output[i] = is_output ? sum : tanh_activate(sum);
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
    // Timing variables
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_layer, stop_layer;
    float milliseconds = 0;

    // Create CUDA events
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_layer);
    cudaEventCreate(&stop_layer);

    cudaEventRecord(start_total);

    // Device pointers
    float *d_input, *d_conv1_output, *d_pool1_output, *d_conv2_output, *d_pool2_output;
    float *d_conv3_output, *d_fc1_output, *d_output;
    float *d_conv1_weights, *d_conv1_bias, *d_conv2_weights, *d_conv2_bias;
    float *d_conv3_weights, *d_conv3_bias, *d_fc1_weights, *d_fc1_bias, *d_fc2_weights, *d_fc2_bias;

    // Allocate and copy data to device
    CUDA_CHECK(cudaMalloc((void**)&d_input, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate and copy weights and biases to device
    int conv1_weights_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_conv1_weights, conv1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv1_weights, weights->conv1_weights, conv1_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_conv1_bias, CONV1_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv1_bias, weights->conv1_bias, CONV1_FILTERS * sizeof(float), cudaMemcpyHostToDevice));

    int conv2_weights_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_conv2_weights, conv2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv2_weights, weights->conv2_weights, conv2_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_conv2_bias, CONV2_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv2_bias, weights->conv2_bias, CONV2_FILTERS * sizeof(float), cudaMemcpyHostToDevice));

    int conv3_weights_size = CONV3_FILTERS * CONV2_FILTERS * CONV3_SIZE * CONV3_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_conv3_weights, conv3_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv3_weights, weights->conv3_weights, conv3_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_conv3_bias, CONV3_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_conv3_bias, weights->conv3_bias, CONV3_FILTERS * sizeof(float), cudaMemcpyHostToDevice));

    int fc1_weights_size = CONV3_FILTERS * FC1_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_fc1_weights, fc1_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc1_weights, weights->fc1_weights, fc1_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_fc1_bias, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc1_bias, weights->fc1_bias, FC1_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int fc2_weights_size = FC1_SIZE * OUTPUT_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_fc2_weights, fc2_weights_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc2_weights, weights->fc2_weights, fc2_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_fc2_bias, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fc2_bias, weights->fc2_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for intermediate outputs
    int conv1_output_size = CONV1_FILTERS * CONV1_OUTPUT * CONV1_OUTPUT;
    CUDA_CHECK(cudaMalloc((void**)&d_conv1_output, conv1_output_size * sizeof(float)));

    int pool1_output_size = CONV1_FILTERS * POOL1_OUTPUT * POOL1_OUTPUT;
    CUDA_CHECK(cudaMalloc((void**)&d_pool1_output, pool1_output_size * sizeof(float)));

    int conv2_output_size = CONV2_FILTERS * CONV2_OUTPUT * CONV2_OUTPUT;
    CUDA_CHECK(cudaMalloc((void**)&d_conv2_output, conv2_output_size * sizeof(float)));

    int pool2_output_size = CONV2_FILTERS * POOL2_OUTPUT * POOL2_OUTPUT;
    CUDA_CHECK(cudaMalloc((void**)&d_pool2_output, pool2_output_size * sizeof(float)));

    int conv3_output_size = CONV3_FILTERS * CONV3_OUTPUT * CONV3_OUTPUT;
    CUDA_CHECK(cudaMalloc((void**)&d_conv3_output, conv3_output_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_fc1_output, FC1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));

    // Conv1 layer
    cudaEventRecord(start_layer);
    dim3 blockDimConv(16, 16, 1);
    dim3 gridDimConv((CONV1_OUTPUT + blockDimConv.x - 1) / blockDimConv.x,
                     (CONV1_OUTPUT + blockDimConv.y - 1) / blockDimConv.y,
                     (CONV1_FILTERS + blockDimConv.z - 1) / blockDimConv.z);

    convolution_cuda<<<gridDimConv, blockDimConv>>>(
        d_input, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS,
        d_conv1_weights, d_conv1_bias, CONV1_SIZE, CONV1_FILTERS,
        d_conv1_output, CONV1_OUTPUT, CONV1_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("Conv1 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // Pool1 layer
    cudaEventRecord(start_layer);
    dim3 blockDimPool(16, 16, 1);
    dim3 gridDimPool((POOL1_OUTPUT + blockDimPool.x - 1) / blockDimPool.x,
                     (POOL1_OUTPUT + blockDimPool.y - 1) / blockDimPool.y,
                     (CONV1_FILTERS + blockDimPool.z - 1) / blockDimPool.z);

    avg_pooling_cuda<<<gridDimPool, blockDimPool>>>(
        d_conv1_output, CONV1_OUTPUT, CONV1_OUTPUT, CONV1_FILTERS,
        POOL1_SIZE, d_pool1_output, POOL1_OUTPUT, POOL1_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("Pool1 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // Conv2 layer
    cudaEventRecord(start_layer);
    dim3 gridDimConv2((CONV2_OUTPUT + blockDimConv.x - 1) / blockDimConv.x,
                      (CONV2_OUTPUT + blockDimConv.y - 1) / blockDimConv.y,
                      (CONV2_FILTERS + blockDimConv.z - 1) / blockDimConv.z);

    convolution_cuda<<<gridDimConv2, blockDimConv>>>(
        d_pool1_output, POOL1_OUTPUT, POOL1_OUTPUT, CONV1_FILTERS,
        d_conv2_weights, d_conv2_bias, CONV2_SIZE, CONV2_FILTERS,
        d_conv2_output, CONV2_OUTPUT, CONV2_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("Conv2 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // Pool2 layer
    cudaEventRecord(start_layer);
    dim3 gridDimPool2((POOL2_OUTPUT + blockDimPool.x - 1) / blockDimPool.x,
                      (POOL2_OUTPUT + blockDimPool.y - 1) / blockDimPool.y,
                      (CONV2_FILTERS + blockDimPool.z - 1) / blockDimPool.z);

    avg_pooling_cuda<<<gridDimPool2, blockDimPool>>>(
        d_conv2_output, CONV2_OUTPUT, CONV2_OUTPUT, CONV2_FILTERS,
        POOL2_SIZE, d_pool2_output, POOL2_OUTPUT, POOL2_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("Pool2 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // Conv3 layer
    cudaEventRecord(start_layer);
    dim3 gridDimConv3(1, 1, (CONV3_FILTERS + blockDimConv.z - 1) / blockDimConv.z);

    convolution_cuda<<<gridDimConv3, blockDimConv>>>(
        d_pool2_output, POOL2_OUTPUT, POOL2_OUTPUT, CONV2_FILTERS,
        d_conv3_weights, d_conv3_bias, CONV3_SIZE, CONV3_FILTERS,
        d_conv3_output, CONV3_OUTPUT, CONV3_OUTPUT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("Conv3 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // FC1 layer
    cudaEventRecord(start_layer);
    int blockSizeDense = 256;
    int gridSizeDense = (FC1_SIZE + blockSizeDense - 1) / blockSizeDense;

    dense_cuda<<<gridSizeDense, blockSizeDense>>>(
        d_conv3_output, d_fc1_weights, d_fc1_bias,
        CONV3_FILTERS, FC1_SIZE, d_fc1_output, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("FC1 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // FC2 layer (output)
    cudaEventRecord(start_layer);
    gridSizeDense = (OUTPUT_SIZE + blockSizeDense - 1) / blockSizeDense;

    dense_cuda<<<gridSizeDense, blockSizeDense>>>(
        d_fc1_output, d_fc2_weights, d_fc2_bias,
        FC1_SIZE, OUTPUT_SIZE, d_output, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_layer);
    cudaEventSynchronize(stop_layer);
    cudaEventElapsedTime(&milliseconds, start_layer, stop_layer);
    printf("FC2 layer time: %.7f seconds\n", milliseconds / 1000.0);

    // Total time
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&milliseconds, start_total, stop_total);
    printf("\nTotal execution time: %.7f seconds\n", milliseconds / 1000.0);

    // Copy final output back to host
    float* fc2_output_host = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    CUDA_CHECK(cudaMemcpy(fc2_output_host, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Apply softmax on host
    softmax(fc2_output_host, OUTPUT_SIZE);

    // Copy softmax output to the output parameter
    memcpy(output, fc2_output_host, OUTPUT_SIZE * sizeof(float));

    // Free host memory
    free(fc2_output_host);

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_conv1_weights));
    CUDA_CHECK(cudaFree(d_conv1_bias));
    CUDA_CHECK(cudaFree(d_conv2_weights));
    CUDA_CHECK(cudaFree(d_conv2_bias));
    CUDA_CHECK(cudaFree(d_conv3_weights));
    CUDA_CHECK(cudaFree(d_conv3_bias));
    CUDA_CHECK(cudaFree(d_fc1_weights));
    CUDA_CHECK(cudaFree(d_fc1_bias));
    CUDA_CHECK(cudaFree(d_fc2_weights));
    CUDA_CHECK(cudaFree(d_fc2_bias));
    CUDA_CHECK(cudaFree(d_conv1_output));
    CUDA_CHECK(cudaFree(d_pool1_output));
    CUDA_CHECK(cudaFree(d_conv2_output));
    CUDA_CHECK(cudaFree(d_pool2_output));
    CUDA_CHECK(cudaFree(d_conv3_output));
    CUDA_CHECK(cudaFree(d_fc1_output));
    CUDA_CHECK(cudaFree(d_output));

    // Destroy CUDA events
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_layer);
    cudaEventDestroy(stop_layer);
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
    float* input = read_image("test_images/digit_6.raw");
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