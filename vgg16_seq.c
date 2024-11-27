#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

// Define constants
#define SIZE 224
#define CONV_SIZE 3
#define NUM_CONV_LAYERS 13
#define NUM_DENSE_LAYERS 3

// Convolution layer configurations
int cshape[NUM_CONV_LAYERS][4] = {
    {64, 3, CONV_SIZE, CONV_SIZE},
    {64, 64, CONV_SIZE, CONV_SIZE},
    {128, 64, CONV_SIZE, CONV_SIZE},
    {128, 128, CONV_SIZE, CONV_SIZE},
    {256, 128, CONV_SIZE, CONV_SIZE},
    {256, 256, CONV_SIZE, CONV_SIZE},
    {256, 256, CONV_SIZE, CONV_SIZE},
    {512, 256, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE}
};

// Dense layer configurations
int dshape[NUM_DENSE_LAYERS][2] = {
    {25088, 4096},
    {4096, 4096},
    {4096, 1000}
};

// Global variables for image and weights
float image[3][SIZE][SIZE];
float *****wc; // Convolutional weights
float **bc;    // Convolutional biases
float ***wd;   // Dense weights
float **bd;    // Dense biases

// Intermediate memory blocks
float mem_block1[512][SIZE][SIZE];
float mem_block2[512][SIZE][SIZE];
float mem_block1_dense[512 * 7 * 7];
float mem_block2_dense[1000]; // Adjusted size to match output layer (1000 classes)

void reset_mem_block(float mem[512][SIZE][SIZE], int channels, int size) {
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                mem[c][i][j] = 0.0f;
            }
        }
    }
}

void reset_mem_block_dense(float *mem, int size) {
    for (int i = 0; i < size; i++) {
        mem[i] = 0.0f;
    }
}

void init_memory() {
    // Allocate convolutional weights and biases
    wc = (float *****)malloc(NUM_CONV_LAYERS * sizeof(float ****));
    bc = (float **)malloc(NUM_CONV_LAYERS * sizeof(float *));
    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        int out_channels = cshape[l][0];
        int in_channels = cshape[l][1];
        wc[l] = (float ****)malloc(out_channels * sizeof(float ***));
        for (int i = 0; i < out_channels; i++) {
            wc[l][i] = (float ***)malloc(in_channels * sizeof(float **));
            for (int j = 0; j < in_channels; j++) {
                wc[l][i][j] = (float **)malloc(CONV_SIZE * sizeof(float *));
                for (int k = 0; k < CONV_SIZE; k++) {
                    wc[l][i][j][k] = (float *)malloc(CONV_SIZE * sizeof(float));
                }
            }
        }
        bc[l] = (float *)malloc(out_channels * sizeof(float));
    }

    // Allocate dense weights and biases
    wd = (float ***)malloc(NUM_DENSE_LAYERS * sizeof(float **));
    bd = (float **)malloc(NUM_DENSE_LAYERS * sizeof(float *));
    for (int l = 0; l < NUM_DENSE_LAYERS; l++) {
        int in_features = dshape[l][0];
        int out_features = dshape[l][1];
        wd[l] = (float **)malloc(in_features * sizeof(float *));
        for (int i = 0; i < in_features; i++) {
            wd[l][i] = (float *)malloc(out_features * sizeof(float));
        }
        bd[l] = (float *)malloc(out_features * sizeof(float));
    }

    // Initialize intermediate memory blocks
    reset_mem_block(mem_block1, 512, SIZE);
    reset_mem_block(mem_block2, 512, SIZE);
    reset_mem_block_dense(mem_block1_dense, 512 * 7 * 7);
    reset_mem_block_dense(mem_block2_dense, 1000);
}

void free_memory() {
    // Free convolutional weights and biases
    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        int out_channels = cshape[l][0];
        int in_channels = cshape[l][1];
        for (int i = 0; i < out_channels; i++) {
            for (int j = 0; j < in_channels; j++) {
                for (int k = 0; k < CONV_SIZE; k++) {
                    free(wc[l][i][j][k]);
                }
                free(wc[l][i][j]);
            }
            free(wc[l][i]);
        }
        free(wc[l]);
        free(bc[l]);
    }
    free(wc);
    free(bc);

    // Free dense weights and biases
    for (int l = 0; l < NUM_DENSE_LAYERS; l++) {
        int in_features = dshape[l][0];
        for (int i = 0; i < in_features; i++) {
            free(wd[l][i]);
        }
        free(wd[l]);
        free(bd[l]);
    }
    free(wd);
    free(bd);
}

void read_weights(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening weights file %s\n", filename);
        exit(1);
    }
    float val;

    // Read convolutional weights and biases
    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        printf("Reading conv layer %d weights\n", l + 1);
        int out_channels = cshape[l][0];
        int in_channels = cshape[l][1];
        for (int i = 0; i < out_channels; i++) {
            for (int j = 0; j < in_channels; j++) {
                for (int k = 0; k < CONV_SIZE; k++) {
                    for (int m = 0; m < CONV_SIZE; m++) {
                        fscanf(file, "%f", &val);
                        wc[l][i][j][k][m] = val;
                    }
                }
            }
        }
        // Read biases
        for (int i = 0; i < out_channels; i++) {
            fscanf(file, "%f", &val);
            bc[l][i] = val;
        }
    }

    // Read dense weights and biases
    for (int l = 0; l < NUM_DENSE_LAYERS; l++) {
        printf("Reading dense layer %d weights\n", l + 1);
        int in_features = dshape[l][0];
        int out_features = dshape[l][1];
        for (int i = 0; i < in_features; i++) {
            for (int j = 0; j < out_features; j++) {
                fscanf(file, "%f", &val);
                wd[l][i][j] = val;
            }
        }
        // Read biases
        for (int i = 0; i < out_features; i++) {
            fscanf(file, "%f", &val);
            bd[l][i] = val;
        }
    }
    fclose(file);
}

void read_image(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening image file %s\n", filename);
        exit(1);
    }
    float val;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int c = 0; c < 3; c++) {
                fscanf(file, "%f", &val);
                image[c][i][j] = val;
            }
        }
    }
    fclose(file);
}

void normalize_image() {
    float mean[3] = {103.939, 116.779, 123.68};
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                image[c][i][j] -= mean[c];
            }
        }
    }
}

void convolution_3_x_3(float input[SIZE][SIZE], float kernel[CONV_SIZE][CONV_SIZE], float output[SIZE][SIZE], int size) {
    float padded_input[SIZE + 2][SIZE + 2] = {0.0f};

    // Zero-padding
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            padded_input[i + 1][j + 1] = input[i][j];
        }
    }

    // Convolution
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < CONV_SIZE; ki++) {
                for (int kj = 0; kj < CONV_SIZE; kj++) {
                    sum += padded_input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] += sum;
        }
    }
}

void add_bias_and_relu(float output[SIZE][SIZE], float bias, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            output[i][j] += bias;
            if (output[i][j] < 0)
                output[i][j] = 0.0f;
        }
    }
}

float max_of_4(float a, float b, float c, float d) {
    float max1 = (a > b) ? a : b;
    float max2 = (c > d) ? c : d;
    return (max1 > max2) ? max1 : max2;
}

void maxpooling(float input[SIZE][SIZE], float output[SIZE / 2][SIZE / 2], int size) {
    for (int i = 0; i < size; i += 2) {
        for (int j = 0; j < size; j += 2) {
            output[i / 2][j / 2] = max_of_4(
                input[i][j],
                input[i + 1][j],
                input[i][j + 1],
                input[i + 1][j + 1]);
        }
    }
}

void flatten(float input[512][7][7], float *output) {
    int idx = 0;
    for (int c = 0; c < 512; c++) {
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                output[idx++] = input[c][i][j];
            }
        }
    }
}

void dense(float *input, float **weights, float *output, int in_size, int out_size) {
    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum += input[j] * weights[j][i];
        }
        output[i] = sum;
    }
}

void add_bias_and_relu_flatten(float *output, float *bias, int size, int relu) {
    for (int i = 0; i < size; i++) {
        output[i] += bias[i];
        if (relu && output[i] < 0)
            output[i] = 0.0f;
    }
}

void softmax(float *output, int size) {
    float max_val = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val)
            max_val = output[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void get_VGG16_predict() {
    int cur_size = SIZE; // Start with the original image size (224)

    // Initialize intermediate memory blocks
    reset_mem_block(mem_block1, 512, SIZE);
    reset_mem_block(mem_block2, 512, SIZE);

    // Layer 1: Conv1_1 (Input channels: 3, Output channels: 64)
    for (int oc = 0; oc < 64; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            convolution_3_x_3(image[ic], wc[0][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[0][oc], cur_size);
    }

    // Layer 2: Conv1_2 (Input channels: 64, Output channels: 64)
    for (int oc = 0; oc < 64; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            convolution_3_x_3(mem_block1[ic], wc[1][oc][ic], mem_block2[oc], cur_size);
        }
        add_bias_and_relu(mem_block2[oc], bc[1][oc], cur_size);
    }

    // Layer 3: MaxPooling1
    for (int c = 0; c < 64; c++) {
        maxpooling(mem_block2[c], mem_block1[c], cur_size);
    }
    cur_size /= 2; // 112

    // Layer 4: Conv2_1 (Input channels: 64, Output channels: 128)
    for (int oc = 0; oc < 128; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            convolution_3_x_3(mem_block1[ic], wc[2][oc][ic], mem_block2[oc], cur_size);
        }
        add_bias_and_relu(mem_block2[oc], bc[2][oc], cur_size);
    }

    // Layer 5: Conv2_2 (Input channels: 128, Output channels: 128)
    for (int oc = 0; oc < 128; oc++) {
        for (int ic = 0; ic < 128; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[3][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[3][oc], cur_size);
    }

    // Layer 6: MaxPooling2
    for (int c = 0; c < 128; c++) {
        maxpooling(mem_block1[c], mem_block2[c], cur_size);
    }
    cur_size /= 2; // 56

    // Layer 7: Conv3_1 (Input channels: 128, Output channels: 256)
    for (int oc = 0; oc < 256; oc++) {
        for (int ic = 0; ic < 128; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[4][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[4][oc], cur_size);
    }

    // Layer 8: Conv3_2 (Input channels: 256, Output channels: 256)
    for (int oc = 0; oc < 256; oc++) {
        for (int ic = 0; ic < 256; ic++) {
            convolution_3_x_3(mem_block1[ic], wc[5][oc][ic], mem_block2[oc], cur_size);
        }
        add_bias_and_relu(mem_block2[oc], bc[5][oc], cur_size);
    }

    // Layer 9: Conv3_3 (Input channels: 256, Output channels: 256)
    for (int oc = 0; oc < 256; oc++) {
        for (int ic = 0; ic < 256; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[6][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[6][oc], cur_size);
    }

    // Layer 10: MaxPooling3
    for (int c = 0; c < 256; c++) {
        maxpooling(mem_block1[c], mem_block2[c], cur_size);
    }
    cur_size /= 2; // 28

    // Layer 11: Conv4_1 (Input channels: 256, Output channels: 512)
    for (int oc = 0; oc < 512; oc++) {
        for (int ic = 0; ic < 256; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[7][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[7][oc], cur_size);
    }

    // Layer 12: Conv4_2 (Input channels: 512, Output channels: 512)
    for (int oc = 0; oc < 512; oc++) {
        for (int ic = 0; ic < 512; ic++) {
            convolution_3_x_3(mem_block1[ic], wc[8][oc][ic], mem_block2[oc], cur_size);
        }
        add_bias_and_relu(mem_block2[oc], bc[8][oc], cur_size);
    }

    // Layer 13: Conv4_3 (Input channels: 512, Output channels: 512)
    for (int oc = 0; oc < 512; oc++) {
        for (int ic = 0; ic < 512; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[9][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[9][oc], cur_size);
    }

    // Layer 14: MaxPooling4
    for (int c = 0; c < 512; c++) {
        maxpooling(mem_block1[c], mem_block2[c], cur_size);
    }
    cur_size /= 2; // 14

    // Layer 15: Conv5_1 (Input channels: 512, Output channels: 512)
    for (int oc = 0; oc < 512; oc++) {
        for (int ic = 0; ic < 512; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[10][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[10][oc], cur_size);
    }

    // Layer 16: Conv5_2 (Input channels: 512, Output channels: 512)
    for (int oc = 0; oc < 512; oc++) {
        for (int ic = 0; ic < 512; ic++) {
            convolution_3_x_3(mem_block1[ic], wc[11][oc][ic], mem_block2[oc], cur_size);
        }
        add_bias_and_relu(mem_block2[oc], bc[11][oc], cur_size);
    }

    // Layer 17: Conv5_3 (Input channels: 512, Output channels: 512)
    for (int oc = 0; oc < 512; oc++) {
        for (int ic = 0; ic < 512; ic++) {
            convolution_3_x_3(mem_block2[ic], wc[12][oc][ic], mem_block1[oc], cur_size);
        }
        add_bias_and_relu(mem_block1[oc], bc[12][oc], cur_size);
    }

    // Layer 18: MaxPooling5
    for (int c = 0; c < 512; c++) {
        maxpooling(mem_block1[c], mem_block2[c], cur_size);
    }
    cur_size /= 2; // 7

    // Flatten the output of the last pooling layer
    flatten(mem_block2, mem_block1_dense);

    // Fully Connected Layer 1: FC6
    dense(mem_block1_dense, wd[0], mem_block2_dense, dshape[0][0], dshape[0][1]);
    add_bias_and_relu_flatten(mem_block2_dense, bd[0], dshape[0][1], 1);

    // Fully Connected Layer 2: FC7
    dense(mem_block2_dense, wd[1], mem_block1_dense, dshape[1][0], dshape[1][1]);
    add_bias_and_relu_flatten(mem_block1_dense, bd[1], dshape[1][1], 1);

    // Fully Connected Layer 3: FC8
    dense(mem_block1_dense, wd[2], mem_block2_dense, dshape[2][0], dshape[2][1]);
    add_bias_and_relu_flatten(mem_block2_dense, bd[2], dshape[2][1], 0);

    // Softmax to get probabilities
    softmax(mem_block2_dense, dshape[2][1]);
}

void output_top_5_predictions() {
    int top_k = 5;
    int indices[1000];
    for (int i = 0; i < 1000; i++) {
        indices[i] = i;
    }

    // Sort the probabilities and indices
    for (int i = 0; i < 1000 - 1; i++) {
        for (int j = i + 1; j < 1000; j++) {
            if (mem_block2_dense[i] < mem_block2_dense[j]) {
                // Swap probabilities
                float temp_prob = mem_block2_dense[i];
                mem_block2_dense[i] = mem_block2_dense[j];
                mem_block2_dense[j] = temp_prob;
                // Swap indices
                int temp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = temp_idx;
            }
        }
    }

    // Print top 5 predictions
    printf("Top %d predictions:\n", top_k);
    for (int i = 0; i < top_k; i++) {
        printf("Class %d: Probability %.6f\n", indices[i], mem_block2_dense[i]);
    }
}

char *trimwhitespace(char *str) {
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str))
        str++;

    if (*str == 0) // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end))
        end--;

    // Write new null terminator
    *(end + 1) = 0;

    return str;
}

int main() {
    char *weights_file = "vgg16_weights_output.txt"; 
    char *image_file = "tv.txt";

    // Initialize memory and load weights
    init_memory();
    read_weights(weights_file);

    // Process the single image
    printf("Processing image: %s\n", image_file);
    read_image(image_file);
    normalize_image();
    get_VGG16_predict();
    output_top_5_predictions(); // Print the top 5 predictions

    // Free allocated memory
    free_memory();
    return 0;
}