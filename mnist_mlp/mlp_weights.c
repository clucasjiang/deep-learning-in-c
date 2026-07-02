#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlp_weights.h"
#include "mlp.h"
#include "stored_weights.h"

#define STORED_WEIGHTS_SOURCE_PATH "mnist_mlp/stored_weights.c" // change to stored_weights_pytorch for params trained by pytorch
#define STORED_WEIGHTS_HEADER_PATH "mnist_mlp/stored_weights.h"

float linear_relu_stack_0_weight[HIDDEN_SIZE][784];
float linear_relu_stack_0_bias[HIDDEN_SIZE];

float linear_relu_stack_2_weight[HIDDEN_SIZE][HIDDEN_SIZE];
float linear_relu_stack_2_bias[HIDDEN_SIZE];

float linear_relu_stack_4_weight[10][HIDDEN_SIZE];
float linear_relu_stack_4_bias[10];

void init_linear(int input_features, int output_features, 
    float weights[output_features][input_features], 
    float bias[output_features]) {
    float max = 0.1; // initialize bias and weights to be between -0.1 and +0.1
    float min = -0.1;
    for (int i=0; i<output_features; i++) {
        bias[i] = ((float) rand() / (float) RAND_MAX) * (max - min) + min;
        for (int j=0; j<input_features; j++) {
            weights[i][j] = ((float) rand() / (float) RAND_MAX) * (max - min) + min;
        }
    } 
}

void init_mlp_weights(unsigned int seed){
    srand(seed);
    init_linear(784, HIDDEN_SIZE, linear_relu_stack_0_weight, linear_relu_stack_0_bias);
    init_linear(HIDDEN_SIZE, HIDDEN_SIZE, linear_relu_stack_2_weight, linear_relu_stack_2_bias);
    init_linear(HIDDEN_SIZE, 10, linear_relu_stack_4_weight, linear_relu_stack_4_bias);
    
}

void load_stored_weights(void) {
    memcpy(linear_relu_stack_0_weight, stored_linear_relu_stack_0_weight, sizeof(linear_relu_stack_0_weight));
    memcpy(linear_relu_stack_0_bias, stored_linear_relu_stack_0_bias, sizeof(linear_relu_stack_0_bias));
    memcpy(linear_relu_stack_2_weight, stored_linear_relu_stack_2_weight, sizeof(linear_relu_stack_2_weight));
    memcpy(linear_relu_stack_2_bias, stored_linear_relu_stack_2_bias, sizeof(linear_relu_stack_2_bias));
    memcpy(linear_relu_stack_4_weight, stored_linear_relu_stack_4_weight, sizeof(linear_relu_stack_4_weight));
    memcpy(linear_relu_stack_4_bias, stored_linear_relu_stack_4_bias, sizeof(linear_relu_stack_4_bias));
}

static void write_float(FILE *file, float value) {
    fprintf(file, "%.9gf", value);
}

static void write_vector(FILE *file, const char *name, int length, const float *values) {
    fprintf(file, "const float %s[%d] = {\n", name, length);
    for (int i=0; i<length; i++) {
        if (i % 8 == 0) {
            fprintf(file, "    ");
        }
        write_float(file, values[i]);
        if (i != length - 1) {
            fprintf(file, ", ");
        }
        if (i % 8 == 7 || i == length - 1) {
            fprintf(file, "\n");
        }
    }
    fprintf(file, "};\n\n");
}

static void write_matrix(FILE *file, const char *name, int rows, int cols, const float *values) {
    fprintf(file, "const float %s[%d][%d] = {\n", name, rows, cols);
    for (int row=0; row<rows; row++) {
        fprintf(file, "    {\n");
        for (int col=0; col<cols; col++) {
            if (col % 8 == 0) {
                fprintf(file, "        ");
            }
            write_float(file, values[row * cols + col]);
            if (col != cols - 1) {
                fprintf(file, ", ");
            }
            if (col % 8 == 7 || col == cols - 1) {
                fprintf(file, "\n");
            }
        }
        fprintf(file, "    }");
        if (row != rows - 1) {
            fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "};\n\n");
}

static int write_stored_weights_header(const char *header_path) {
    FILE *file = fopen(header_path, "w");
    if (file == NULL) {
        perror(header_path);
        return 0;
    }

    fprintf(file, "#ifndef STORED_WEIGHTS_H\n");
    fprintf(file, "#define STORED_WEIGHTS_H\n\n");
    fprintf(file, "#include \"mlp.h\"\n\n");
    fprintf(file, "extern const float stored_linear_relu_stack_0_weight[HIDDEN_SIZE][784];\n");
    fprintf(file, "extern const float stored_linear_relu_stack_0_bias[HIDDEN_SIZE];\n\n");
    fprintf(file, "extern const float stored_linear_relu_stack_2_weight[HIDDEN_SIZE][HIDDEN_SIZE];\n");
    fprintf(file, "extern const float stored_linear_relu_stack_2_bias[HIDDEN_SIZE];\n\n");
    fprintf(file, "extern const float stored_linear_relu_stack_4_weight[10][HIDDEN_SIZE];\n");
    fprintf(file, "extern const float stored_linear_relu_stack_4_bias[10];\n\n");
    fprintf(file, "#endif\n");

    if (fclose(file) != 0) {
        perror(header_path);
        return 0;
    }
    return 1;
}

int save_mlp_weights(const char *source_path, const char *header_path) {
    if (!write_stored_weights_header(header_path)) {
        return 0;
    }

    FILE *file = fopen(source_path, "w");
    if (file == NULL) {
        perror(source_path);
        return 0;
    }

    fprintf(file, "#include \"stored_weights.h\"\n\n");
    write_matrix(
        file,
        "stored_linear_relu_stack_0_weight",
        HIDDEN_SIZE,
        784,
        &linear_relu_stack_0_weight[0][0]
    );
    write_vector(file, "stored_linear_relu_stack_0_bias", HIDDEN_SIZE, linear_relu_stack_0_bias);
    write_matrix(
        file,
        "stored_linear_relu_stack_2_weight",
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        &linear_relu_stack_2_weight[0][0]
    );
    write_vector(file, "stored_linear_relu_stack_2_bias", HIDDEN_SIZE, linear_relu_stack_2_bias);
    write_matrix(
        file,
        "stored_linear_relu_stack_4_weight",
        10,
        HIDDEN_SIZE,
        &linear_relu_stack_4_weight[0][0]
    );
    write_vector(file, "stored_linear_relu_stack_4_bias", 10, linear_relu_stack_4_bias);

    if (fclose(file) != 0) {
        perror(source_path);
        return 0;
    }
    return 1;
}

int save_trained_weights(void) {
    return save_mlp_weights(STORED_WEIGHTS_SOURCE_PATH, STORED_WEIGHTS_HEADER_PATH);
}
