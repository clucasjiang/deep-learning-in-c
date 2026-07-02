#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mlp_weights.h"
#include "read_data.h"
#include "mlp.h"

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

void relu(int in_features, float input[in_features], float output[in_features]) {
    for (int i=0; i<in_features; i++) {
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            output[i] = input[i];
        }
    }
}

void softmax(float logits[10], float probs[10]) {
    float max_logit = logits[0];
    for (int i=1; i<10; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum = 0;
    for (int i=0; i<10; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i=0; i<10; i++) {
        probs[i] = probs[i] / sum;
    }
}

void forward(float input[], ForwardCache *cache) {
    linear_layer(784, HIDDEN_SIZE, linear_relu_stack_0_weight, linear_relu_stack_0_bias, input, cache->z1);
    relu(HIDDEN_SIZE, cache->z1, cache->a1);
    linear_layer(HIDDEN_SIZE, HIDDEN_SIZE, linear_relu_stack_2_weight, linear_relu_stack_2_bias, cache->a1, cache->z2);
    relu(HIDDEN_SIZE, cache->z2, cache->a2);
    linear_layer(HIDDEN_SIZE, 10, linear_relu_stack_4_weight, linear_relu_stack_4_bias, cache->a2, cache->logits);
    softmax(cache->logits, cache->probs);
}

float find_mse_cost(float output[10], int label) {
    float cost = 0;
    for (int i=0; i<10; i++) {
        if (i != label) {
            cost += output[i] * output[i];
        } else {
            cost += (output[i] - 1.0f) * (output[i] - 1.0f);
        }
    }
    return cost / 10.0f;
}

void train_model(void) {
    init_mlp_weights(1234);

    
    save_trained_weights();
}
