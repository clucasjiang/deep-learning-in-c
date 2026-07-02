#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "mlp_weights.h"
#include "read_image.h"
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


float find_cross_entropy_cost(float probs[10], int label) {
    return -logf(probs[label] + 1e-7f);
}

void backward(float input[784], int label, ForwardCache *cache, Gradients *grads) {
    float dlogits[10];
    for (int i=0; i<10; i++) {
        dlogits[i] = cache->probs[i] - (i==label ? 1 : 0);
    }
    for (int i=0; i<10; i++) {
        for (int j=0; j<HIDDEN_SIZE; j++) {
            grads->w4[i][j] += dlogits[i] * cache->a2[j];  
        }
        grads->b4[i] += dlogits[i];
    }
}

void train_model(void) {
    init_mlp_weights(1234);
    Gradients grad = {0};

    save_trained_weights();
}
