#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mlp_weights.h"
#include "read_data.h"

void linear_layer(
    int in_features, 
    int out_features, 
    const float weights[out_features][in_features],
    const float bias[out_features],
    const float input[in_features],
    float output[out_features]) {
    
    for (int i=0; i<out_features; i++) {
        float sum = 0;
        for (int j=0; j<in_features; j++) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum + bias[i];
    }
}

void relu(int in_features, float input[in_features]) {
    for (int i=0; i<in_features; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

void softmax(int out_features, float output[out_features]) {
    float max_logit = output[0];
    for (int i=1; i<out_features; i++) {
        if (output[i] > max_logit) {
            max_logit = output[i];
        }
    }

    float sum = 0;
    for (int i=0; i<out_features; i++) {
        output[i] = expf(output[i] - max_logit);
        sum += output[i];
    }
    for (int i=0; i<out_features; i++) {
        output[i] = output[i] / sum;
    }
}

void run_model(float input[], float output[]) {
    float hidden_1[512];
    float hidden_2[512];
    
    linear_layer(784, 512, linear_relu_stack_0_weight, linear_relu_stack_0_bias, input, hidden_1);
    relu(512, hidden_1);
    linear_layer(512, 512, linear_relu_stack_2_weight, linear_relu_stack_2_bias, hidden_1, hidden_2);
    relu(512, hidden_2);
    linear_layer(512, 10, linear_relu_stack_4_weight, linear_relu_stack_4_bias, hidden_2, output);
    softmax(10, output);
}
