#include <math.h>
#include <stdlib.h>
#include "mlp_weights.h"
#include "mlp.h"

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