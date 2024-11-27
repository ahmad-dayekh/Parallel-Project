#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

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

// Global MPI variables
int world_rank;
int world_size;

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

// Rearrange convolutional weights
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

// Parallel convolution operation with MPI and timing
void convolution_mpi(float* input, int input_width, int input_height, int input_channels,
                    float* weights, float* bias, int kernel_size, int num_filters,
                    float* output, int output_width, int output_height) {
    
    double conv_start = MPI_Wtime();
    
    // Calculate work distribution
    int filters_per_process = (num_filters + world_size - 1) / world_size;  // Round up division
    int start_filter = world_rank * filters_per_process;
    int end_filter = start_filter + filters_per_process;
    if (end_filter > num_filters) end_filter = num_filters;
    int local_num_filters = end_filter - start_filter;

    // Calculate buffer sizes
    int local_output_size = local_num_filters * output_width * output_height;
    int max_local_output_size = filters_per_process * output_width * output_height;

    // Allocate local output buffer
    float* local_output = (float*)calloc(local_output_size, sizeof(float));
    if (!local_output) {
        printf("Rank %d: Failed to allocate local output buffer\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Perform convolution for assigned filters
    for (int f = 0; f < local_num_filters; f++) {
        int global_f = start_filter + f;
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
                            int weight_idx = global_f * input_channels * kernel_size * kernel_size +
                                           c * kernel_size * kernel_size +
                                           ki * kernel_size + kj;
                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
                sum += bias[global_f];
                local_output[f * output_height * output_width + 
                           i * output_width + j] = tanh_activate(sum);
            }
        }
    }

    // Gather results from all processes
    MPI_Barrier(MPI_COMM_WORLD);

    int* recvcounts = (int*)malloc(world_size * sizeof(int));
    int* displs = (int*)malloc(world_size * sizeof(int));
    
    // Calculate receive counts and displacements
    for (int i = 0; i < world_size; i++) {
        int start = i * filters_per_process;
        int end = start + filters_per_process;
        if (end > num_filters) end = num_filters;
        recvcounts[i] = (end - start) * output_width * output_height;
        displs[i] = start * output_width * output_height;
    }

    MPI_Allgatherv(local_output, local_output_size, MPI_FLOAT,
                   output, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);

    double conv_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Convolution time: %.7f seconds\n", conv_end - conv_start);
    }

    free(local_output);
    free(recvcounts);
    free(displs);
}
// Average pooling operation with timing
void avg_pooling(float* input, int input_width, int input_height, int num_channels,
                int pool_size, float* output, int output_width, int output_height) {
    double pool_start = MPI_Wtime();
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
                output[c * output_height * output_width + 
                      i * output_width + j] = sum * scale;
            }
        }
    }

    double pool_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Pooling time: %.7f seconds\n", pool_end - pool_start);
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

// Dense layer operation with timing
void dense(float* input, float* weights, float* bias,
          int input_size, int output_size, float* output, int is_output) {
    double dense_start = MPI_Wtime();

    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + i];
        }
        sum += bias[i];
        output[i] = is_output ? sum : tanh_activate(sum);
    }

    if (is_output) {
        softmax(output, output_size);
    }

    double dense_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Dense layer time: %.7f seconds\n", dense_end - dense_start);
    }
}

// Initialize network weights
NetworkWeights init_network() {
    double init_start = MPI_Wtime();
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
    weights.fc1_weights = load_weights("dense_weight.txt", CONV3_FILTERS * FC1_SIZE);
    weights.fc1_bias = load_weights("dense_bias.txt", FC1_SIZE);
    weights.fc2_weights = load_weights("dense_1_weight.txt", FC1_SIZE * OUTPUT_SIZE);
    weights.fc2_bias = load_weights("dense_1_bias.txt", OUTPUT_SIZE);

    double init_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Network initialization time: %.7f seconds\n", init_end - init_start);
    }

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

// Forward pass through the network with MPI parallelization and timing
void forward_mpi(float* input, float* output, NetworkWeights* weights) {
    double total_start = MPI_Wtime();

    // Allocate memory on all processes
    int input_size = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
    float* local_input = NULL;
    
    if (world_rank == 0) {
        local_input = input;
    } else {
        local_input = (float*)malloc(input_size * sizeof(float));
        if (!local_input) {
            printf("Rank %d: Failed to allocate local input buffer\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }

    // Broadcast weights to all processes first
    MPI_Bcast(weights->conv1_weights, CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE, 
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->conv1_bias, CONV1_FILTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->conv2_weights, CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE,
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->conv2_bias, CONV2_FILTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->conv3_weights, CONV3_FILTERS * CONV2_FILTERS * CONV3_SIZE * CONV3_SIZE,
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->conv3_bias, CONV3_FILTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->fc1_weights, CONV3_FILTERS * FC1_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->fc1_bias, FC1_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->fc2_weights, FC1_SIZE * OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->fc2_bias, OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast input data
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("Broadcasting input data...\n");
    }
    MPI_Bcast(local_input, input_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Allocate intermediate buffers with error checking
    float* conv1_output = (float*)calloc(CONV1_FILTERS * CONV1_OUTPUT * CONV1_OUTPUT, sizeof(float));
    float* pool1_output = (float*)calloc(CONV1_FILTERS * POOL1_OUTPUT * POOL1_OUTPUT, sizeof(float));
    float* conv2_output = (float*)calloc(CONV2_FILTERS * CONV2_OUTPUT * CONV2_OUTPUT, sizeof(float));
    float* pool2_output = (float*)calloc(CONV2_FILTERS * POOL2_OUTPUT * POOL2_OUTPUT, sizeof(float));
    float* conv3_output = (float*)calloc(CONV3_FILTERS * CONV3_OUTPUT * CONV3_OUTPUT, sizeof(float));
    float* fc1_output = (float*)calloc(FC1_SIZE, sizeof(float));

    if (!conv1_output || !pool1_output || !conv2_output || 
        !pool2_output || !conv3_output || !fc1_output) {
        printf("Rank %d: Failed to allocate intermediate buffers\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Conv1 layer
    if (world_rank == 0) printf("\nStarting Conv1 layer...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    convolution_mpi(local_input, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS,
                   weights->conv1_weights, weights->conv1_bias,
                   CONV1_SIZE, CONV1_FILTERS,
                   conv1_output, CONV1_OUTPUT, CONV1_OUTPUT);

    // Pool1 layer
    if (world_rank == 0) printf("\nStarting Pool1 layer...\n");
    if (world_rank == 0) {
        avg_pooling(conv1_output, CONV1_OUTPUT, CONV1_OUTPUT, CONV1_FILTERS,
                   POOL1_SIZE, pool1_output, POOL1_OUTPUT, POOL1_OUTPUT);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(pool1_output, CONV1_FILTERS * POOL1_OUTPUT * POOL1_OUTPUT,
              MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Conv2 layer
    if (world_rank == 0) printf("\nStarting Conv2 layer...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    convolution_mpi(pool1_output, POOL1_OUTPUT, POOL1_OUTPUT, CONV1_FILTERS,
                   weights->conv2_weights, weights->conv2_bias,
                   CONV2_SIZE, CONV2_FILTERS,
                   conv2_output, CONV2_OUTPUT, CONV2_OUTPUT);

    // Pool2 layer
    if (world_rank == 0) printf("\nStarting Pool2 layer...\n");
    if (world_rank == 0) {
        avg_pooling(conv2_output, CONV2_OUTPUT, CONV2_OUTPUT, CONV2_FILTERS,
                   POOL2_SIZE, pool2_output, POOL2_OUTPUT, POOL2_OUTPUT);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(pool2_output, CONV2_FILTERS * POOL2_OUTPUT * POOL2_OUTPUT,
              MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Conv3 layer
    if (world_rank == 0) printf("\nStarting Conv3 layer...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    convolution_mpi(pool2_output, POOL2_OUTPUT, POOL2_OUTPUT, CONV2_FILTERS,
                   weights->conv3_weights, weights->conv3_bias,
                   CONV3_SIZE, CONV3_FILTERS,
                   conv3_output, CONV3_OUTPUT, CONV3_OUTPUT);

    // FC1 layer
    if (world_rank == 0) printf("\nStarting FC1 layer...\n");
    if (world_rank == 0) {
        dense(conv3_output, weights->fc1_weights, weights->fc1_bias,
              CONV3_FILTERS, FC1_SIZE, fc1_output, 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(fc1_output, FC1_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // FC2 layer (output)
    if (world_rank == 0) printf("\nStarting FC2 layer...\n");
    if (world_rank == 0) {
        dense(fc1_output, weights->fc2_weights, weights->fc2_bias,
              FC1_SIZE, OUTPUT_SIZE, output, 1);
    }
    MPI_Bcast(output, OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double total_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("\nTotal network execution time: %.7f seconds\n", total_end - total_start);
    }

    // Cleanup
    if (world_rank != 0) {
        free(local_input);
    }
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
    if (!image) {
        printf("Failed to allocate memory for image\n");
        return NULL;
    }

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Verify minimum number of processes
    if (world_size < 2) {
        if (world_rank == 0) {
            printf("This program requires at least 2 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    NetworkWeights weights;
    float* input = NULL;
    float* output = NULL;

    // Initialize network weights on all processes
    if (world_rank == 0) {
        printf("Initializing network...\n");
    }
    weights = init_network();
    MPI_Barrier(MPI_COMM_WORLD);

    // Root process reads input
    if (world_rank == 0) {
        printf("Reading input image...\n");
        input = read_image("test_images/digit_6.raw");
        if (!input) {
            printf("Failed to read input image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Allocate output buffer on all processes
    output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    if (!output) {
        printf("Rank %d: Failed to allocate output buffer\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Synchronize before starting computation
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) printf("Starting forward pass...\n");
    forward_mpi(input, output, &weights);

    // Print results only on root process
    if (world_rank == 0) {
        int predicted_class = 0;
        float max_prob = output[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (output[i] > max_prob) {
                max_prob = output[i];
                predicted_class = i;
            }
        }

        printf("\nPredicted digit: %d (confidence: %.2f%%)\n",
               predicted_class, max_prob * 100);

        printf("\nClass probabilities:\n");
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            printf("Digit %d: %.2f%%\n", i, output[i] * 100);
        }

    }

    // Cleanup
    if (world_rank == 0) {
        free(input);
    }
    free(output);
    free_network(&weights);

    MPI_Finalize();
    return 0;
}