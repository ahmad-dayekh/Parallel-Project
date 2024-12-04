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
        printf("Rank %d: Failed to allocate memory for weights\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "weights/%s", filename);
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        printf("Rank %d: Error opening file: %s\n", world_rank, filepath);
        free(weights);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(fp, "%f", &weights[i]) != 1) {
            printf("Rank %d: Error reading weight at index %d from %s\n", world_rank, i, filepath);
            fclose(fp);
            free(weights);
            MPI_Abort(MPI_COMM_WORLD, 1);
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

// Parallel convolution operation with MPI
void convolution_mpi(float* input, int input_width, int input_height, int input_channels,
                    float* weights, float* bias, int kernel_size, int num_filters,
                    float* output, int output_width, int output_height) {

    // Updated work distribution
    int filters_per_process = num_filters / world_size + (world_rank < (num_filters % world_size) ? 1 : 0);
    int start_filter = (world_rank * (num_filters / world_size)) + 
                       (world_rank < (num_filters % world_size) ? world_rank : (num_filters % world_size));
    int local_num_filters = filters_per_process;

    // Allocate local output buffer only if there are filters to process
    float* local_output = NULL;
    if (local_num_filters > 0) {
        local_output = (float*)calloc(local_num_filters * output_width * output_height, sizeof(float));
        if (!local_output) {
            printf("Rank %d: Failed to allocate local output buffer\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Perform convolution only if there are filters assigned
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

    // Prepare for gathering
    int* recvcounts = NULL;
    int* displs = NULL;

    if (world_rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        if (!recvcounts || !displs) {
            printf("Rank %d: Failed to allocate recvcounts or displs\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            int proc_filters = num_filters / world_size + (i < (num_filters % world_size) ? 1 : 0);
            recvcounts[i] = proc_filters * output_width * output_height;
            displs[i] = displacement;
            displacement += recvcounts[i];
        }
    }

    // Gather all local outputs to the root process
    MPI_Gatherv(local_output, (local_num_filters > 0) ? local_num_filters * output_width * output_height : 0, 
                MPI_FLOAT,
                (world_rank == 0) ? output : NULL, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Free allocated memory
    if (local_output) free(local_output);
    if (world_rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

// Parallel average pooling operation with MPI
void pooling_mpi(float* input, int input_width, int input_height, int num_channels,
                int pool_size, float* output, int output_width, int output_height) {

    // Updated work distribution
    int channels_per_process = num_channels / world_size + (world_rank < (num_channels % world_size) ? 1 : 0);
    int start_channel = (world_rank * (num_channels / world_size)) + 
                        (world_rank < (num_channels % world_size) ? world_rank : (num_channels % world_size));
    int local_num_channels = channels_per_process;

    // Allocate local output buffer only if there are channels to process
    float* local_output = NULL;
    if (local_num_channels > 0) {
        local_output = (float*)calloc(local_num_channels * output_width * output_height, sizeof(float));
        if (!local_output) {
            printf("Rank %d: Failed to allocate local pooling output buffer\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    float scale = 1.0f / (pool_size * pool_size);

    // Perform pooling only if there are channels assigned
    for (int c = 0; c < local_num_channels; c++) {
        int global_c = start_channel + c;
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                float sum = 0.0f;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_i = i * pool_size + pi;
                        int in_j = j * pool_size + pj;
                        int input_idx = global_c * input_height * input_width +
                                        in_i * input_width + in_j;
                        sum += input[input_idx];
                    }
                }
                local_output[c * output_width * output_height +
                            i * output_width + j] = sum * scale;
            }
        }
    }

    // Prepare for gathering
    int* recvcounts = NULL;
    int* displs = NULL;

    if (world_rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        if (!recvcounts || !displs) {
            printf("Rank %d: Failed to allocate recvcounts or displs\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            int proc_channels = num_channels / world_size + (i < (num_channels % world_size) ? 1 : 0);
            recvcounts[i] = proc_channels * output_width * output_height;
            displs[i] = displacement;
            displacement += recvcounts[i];
        }
    }

    // Gather all local pooling outputs to the root process
    MPI_Gatherv(local_output, (local_num_channels > 0) ? local_num_channels * output_width * output_height : 0, 
                MPI_FLOAT,
                (world_rank == 0) ? output : NULL, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Free allocated memory
    if (local_output) free(local_output);
    if (world_rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

// Softmax activation for output layer
void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
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

// Parallel Dense layer operation with MPI
void dense_mpi(float* input, float* weights, float* bias,
               int input_size, int output_size, float* output, int is_output) {

    // Updated work distribution
    int outputs_per_process = output_size / world_size + (world_rank < (output_size % world_size) ? 1 : 0);
    int start_output = (world_rank * (output_size / world_size)) + 
                       (world_rank < (output_size % world_size) ? world_rank : (output_size % world_size));
    int local_output_size = outputs_per_process;

    // Allocate local output buffer only if there are outputs to compute
    float* local_output = NULL;
    if (local_output_size > 0) {
        local_output = (float*)malloc(local_output_size * sizeof(float));
        if (!local_output) {
            printf("Rank %d: Failed to allocate local output buffer for dense layer\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Compute local outputs only if there are outputs assigned
    for (int i = 0; i < local_output_size; i++) {
        int neuron = start_output + i;
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + neuron];
        }
        sum += bias[neuron];
        local_output[i] = is_output ? sum : tanh_activate(sum);
    }

    // Prepare for gathering
    int* recvcounts = NULL;
    int* displs = NULL;

    if (world_rank == 0) {
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        if (!recvcounts || !displs) {
            printf("Rank %d: Failed to allocate recvcounts or displs\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            int proc_outputs = output_size / world_size + (i < (output_size % world_size) ? 1 : 0);
            recvcounts[i] = proc_outputs;
            displs[i] = displacement;
            displacement += recvcounts[i];
        }
    }

    // Gather partial outputs
    MPI_Gatherv(local_output, (local_output_size > 0) ? local_output_size : 0, MPI_FLOAT,
                (world_rank == 0) ? output : NULL, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // If this is the output layer and on root, perform softmax
    if (is_output && world_rank == 0) {
        softmax(output, output_size);
    }

    // Free allocated memory
    if (local_output) free(local_output);
    if (world_rank == 0) {
        free(recvcounts);
        free(displs);
    }
}

// Initialize network weights with optimized broadcasting
NetworkWeights init_network() {
    NetworkWeights weights;

    // Allocate memory for all weight arrays on all processes
    weights.conv1_weights = (float*)malloc(CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE * sizeof(float));
    weights.conv1_bias = (float*)malloc(CONV1_FILTERS * sizeof(float));

    weights.conv2_weights = (float*)malloc(CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE * sizeof(float));
    weights.conv2_bias = (float*)malloc(CONV2_FILTERS * sizeof(float));

    weights.conv3_weights = (float*)malloc(CONV3_FILTERS * CONV2_FILTERS * CONV3_SIZE * CONV3_SIZE * sizeof(float));
    weights.conv3_bias = (float*)malloc(CONV3_FILTERS * sizeof(float));

    weights.fc1_weights = (float*)malloc(CONV3_FILTERS * FC1_SIZE * sizeof(float));
    weights.fc1_bias = (float*)malloc(FC1_SIZE * sizeof(float));

    weights.fc2_weights = (float*)malloc(FC1_SIZE * OUTPUT_SIZE * sizeof(float));
    weights.fc2_bias = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    if (!weights.conv1_weights || !weights.conv1_bias ||
        !weights.conv2_weights || !weights.conv2_bias ||
        !weights.conv3_weights || !weights.conv3_bias ||
        !weights.fc1_weights || !weights.fc1_bias ||
        !weights.fc2_weights || !weights.fc2_bias) {
        printf("Rank %d: Failed to allocate memory for network weights\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0) {
        // Load convolution layer 1 weights and rearrange
        int conv1_weights_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE;
        float* conv1_weights_raw = load_weights("conv2d_weight.txt", conv1_weights_size);
        rearrange_conv_weights(conv1_weights_raw, weights.conv1_weights,
                              CONV1_SIZE, INPUT_CHANNELS, CONV1_FILTERS);
        free(conv1_weights_raw);

        float* temp_conv1_bias = load_weights("conv2d_bias.txt", CONV1_FILTERS);
        memcpy(weights.conv1_bias, temp_conv1_bias, CONV1_FILTERS * sizeof(float));
        free(temp_conv1_bias);

        // Load convolution layer 2 weights and rearrange
        int conv2_weights_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE;
        float* conv2_weights_raw = load_weights("conv2d_1_weight.txt", conv2_weights_size);
        rearrange_conv_weights(conv2_weights_raw, weights.conv2_weights,
                              CONV2_SIZE, CONV1_FILTERS, CONV2_FILTERS);
        free(conv2_weights_raw);

        float* temp_conv2_bias = load_weights("conv2d_1_bias.txt", CONV2_FILTERS);
        memcpy(weights.conv2_bias, temp_conv2_bias, CONV2_FILTERS * sizeof(float));
        free(temp_conv2_bias);

        // Load convolution layer 3 weights and rearrange
        int conv3_weights_size = CONV3_FILTERS * CONV2_FILTERS * CONV3_SIZE * CONV3_SIZE;
        float* conv3_weights_raw = load_weights("conv2d_2_weight.txt", conv3_weights_size);
        rearrange_conv_weights(conv3_weights_raw, weights.conv3_weights,
                              CONV3_SIZE, CONV2_FILTERS, CONV3_FILTERS);
        free(conv3_weights_raw);

        float* temp_conv3_bias = load_weights("conv2d_2_bias.txt", CONV3_FILTERS);
        memcpy(weights.conv3_bias, temp_conv3_bias, CONV3_FILTERS * sizeof(float));
        free(temp_conv3_bias);

        // Load dense layer weights
        float* temp_fc1_weights = load_weights("dense_weight.txt", CONV3_FILTERS * FC1_SIZE);
        memcpy(weights.fc1_weights, temp_fc1_weights, CONV3_FILTERS * FC1_SIZE * sizeof(float));
        free(temp_fc1_weights);

        float* temp_fc1_bias = load_weights("dense_bias.txt", FC1_SIZE);
        memcpy(weights.fc1_bias, temp_fc1_bias, FC1_SIZE * sizeof(float));
        free(temp_fc1_bias);

        float* temp_fc2_weights = load_weights("dense_1_weight.txt", FC1_SIZE * OUTPUT_SIZE);
        memcpy(weights.fc2_weights, temp_fc2_weights, FC1_SIZE * OUTPUT_SIZE * sizeof(float));
        free(temp_fc2_weights);

        float* temp_fc2_bias = load_weights("dense_1_bias.txt", OUTPUT_SIZE);
        memcpy(weights.fc2_bias, temp_fc2_bias, OUTPUT_SIZE * sizeof(float));
        free(temp_fc2_bias);
    }

    // Broadcast all weights and biases from root to all processes
    MPI_Bcast(weights.conv1_weights, CONV1_FILTERS * INPUT_CHANNELS * CONV1_SIZE * CONV1_SIZE, 
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights.conv1_bias, CONV1_FILTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(weights.conv2_weights, CONV2_FILTERS * CONV1_FILTERS * CONV2_SIZE * CONV2_SIZE,
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights.conv2_bias, CONV2_FILTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(weights.conv3_weights, CONV3_FILTERS * CONV2_FILTERS * CONV3_SIZE * CONV3_SIZE,
              MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights.conv3_bias, CONV3_FILTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(weights.fc1_weights, CONV3_FILTERS * FC1_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights.fc1_bias, FC1_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(weights.fc2_weights, FC1_SIZE * OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights.fc2_bias, OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);


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

// Forward pass through the network with MPI parallelization
void forward_mpi(float* input, float* output, NetworkWeights* weights) {
    double total_start = MPI_Wtime();

    // Allocate intermediate buffers on all processes
    float* conv1_output = (float*)malloc(CONV1_FILTERS * CONV1_OUTPUT * CONV1_OUTPUT * sizeof(float));
    float* pool1_output = (float*)malloc(CONV1_FILTERS * POOL1_OUTPUT * POOL1_OUTPUT * sizeof(float));
    float* conv2_output = (float*)malloc(CONV2_FILTERS * CONV2_OUTPUT * CONV2_OUTPUT * sizeof(float));
    float* pool2_output = (float*)malloc(CONV2_FILTERS * POOL2_OUTPUT * POOL2_OUTPUT * sizeof(float));
    float* conv3_output = (float*)malloc(CONV3_FILTERS * CONV3_OUTPUT * CONV3_OUTPUT * sizeof(float));
    float* fc1_output = (float*)malloc(FC1_SIZE * sizeof(float));

    if (!conv1_output || !pool1_output || !conv2_output || 
        !pool2_output || !conv3_output || !fc1_output) {
        printf("Rank %d: Failed to allocate intermediate buffers\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Broadcast input data
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(input, INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Conv1 layer
    double conv1_start = MPI_Wtime();
    convolution_mpi(input, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS,
                   weights->conv1_weights, weights->conv1_bias,
                   CONV1_SIZE, CONV1_FILTERS,
                   conv1_output, CONV1_OUTPUT, CONV1_OUTPUT);
    double conv1_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Conv1 layer time: %.7f seconds\n", conv1_end - conv1_start);
    }

    // Broadcast conv1_output to all processes
    MPI_Bcast(conv1_output, CONV1_FILTERS * CONV1_OUTPUT * CONV1_OUTPUT, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Pool1 layer
    double pool1_start = MPI_Wtime();
    pooling_mpi(conv1_output, CONV1_OUTPUT, CONV1_OUTPUT, CONV1_FILTERS,
               POOL1_SIZE, pool1_output, POOL1_OUTPUT, POOL1_OUTPUT);
    double pool1_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Pool1 layer time: %.7f seconds\n", pool1_end - pool1_start);
    }

    // Broadcast pool1_output to all processes
    MPI_Bcast(pool1_output, CONV1_FILTERS * POOL1_OUTPUT * POOL1_OUTPUT, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Conv2 layer
    double conv2_start = MPI_Wtime();
    convolution_mpi(pool1_output, POOL1_OUTPUT, POOL1_OUTPUT, CONV1_FILTERS,
                   weights->conv2_weights, weights->conv2_bias,
                   CONV2_SIZE, CONV2_FILTERS,
                   conv2_output, CONV2_OUTPUT, CONV2_OUTPUT);
    double conv2_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Conv2 layer time: %.7f seconds\n", conv2_end - conv2_start);
    }

    // Broadcast conv2_output to all processes
    MPI_Bcast(conv2_output, CONV2_FILTERS * CONV2_OUTPUT * CONV2_OUTPUT, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Pool2 layer
    double pool2_start = MPI_Wtime();
    pooling_mpi(conv2_output, CONV2_OUTPUT, CONV2_OUTPUT, CONV2_FILTERS,
               POOL2_SIZE, pool2_output, POOL2_OUTPUT, POOL2_OUTPUT);
    double pool2_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Pool2 layer time: %.7f seconds\n", pool2_end - pool2_start);
    }

    // Broadcast pool2_output to all processes
    MPI_Bcast(pool2_output, CONV2_FILTERS * POOL2_OUTPUT * POOL2_OUTPUT, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Conv3 layer
    double conv3_start = MPI_Wtime();
    convolution_mpi(pool2_output, POOL2_OUTPUT, POOL2_OUTPUT, CONV2_FILTERS,
                   weights->conv3_weights, weights->conv3_bias,
                   CONV3_SIZE, CONV3_FILTERS,
                   conv3_output, CONV3_OUTPUT, CONV3_OUTPUT);
    double conv3_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Conv3 layer time: %.7f seconds\n", conv3_end - conv3_start);
    }

    // Broadcast conv3_output to all processes
    MPI_Bcast(conv3_output, CONV3_FILTERS * CONV3_OUTPUT * CONV3_OUTPUT, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // FC1 layer
    double fc1_start = MPI_Wtime();
    dense_mpi(conv3_output, weights->fc1_weights, weights->fc1_bias,
             CONV3_FILTERS, FC1_SIZE, fc1_output, 0);
    double fc1_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("FC1 layer time: %.7f seconds\n", fc1_end - fc1_start);
    }

    // Broadcast fc1_output to all processes
    MPI_Bcast(fc1_output, FC1_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // FC2 layer (output)
    double fc2_start = MPI_Wtime();
    dense_mpi(fc1_output, weights->fc2_weights, weights->fc2_bias,
             FC1_SIZE, OUTPUT_SIZE, output, 1);
    double fc2_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("FC2 layer time: %.7f seconds\n", fc2_end - fc2_start);
    }

    // Broadcast output to all processes
    MPI_Bcast(output, OUTPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double total_end = MPI_Wtime();
    if (world_rank == 0) {
        printf("Total network execution time: %.7f seconds\n", total_end - total_start);
    }

    // Cleanup
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
        printf("Rank %d: Failed to allocate memory for image\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Rank %d: Error opening image file: %s\n", world_rank, filename);
        free(image);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned char pixel;
    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; i++) {
        if (fread(&pixel, 1, 1, fp) != 1) {
            printf("Rank %d: Error reading image data at index %d\n", world_rank, i);
            fclose(fp);
            free(image);
            MPI_Abort(MPI_COMM_WORLD, 1);
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
    weights = init_network();
    MPI_Barrier(MPI_COMM_WORLD);

    // Root process reads input
    if (world_rank == 0) {
        input = read_image("test_images/digit_6.raw");
    } else {
        // Non-root processes allocate memory for input
        input = (float*)malloc(INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float));
        if (!input) {
            printf("Rank %d: Failed to allocate input buffer\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Allocate output buffer on all processes
    output = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    if (!output) {
        printf("Rank %d: Failed to allocate output buffer\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Synchronize before starting computation
    MPI_Barrier(MPI_COMM_WORLD);

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
    } else {
        free(input);
    }
    free(output);
    free_network(&weights);

    MPI_Finalize();
    return 0;
}
