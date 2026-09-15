#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
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
    // Softmax and cross entropy give this gradient for the output layer.
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

    // Send the gradient back through the output weights and the second ReLU.
    float dz2[HIDDEN_SIZE] = {0};
    for (int i=0; i<10; i++) {
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dz2[j] += linear_relu_stack_4_weight[i][j] * dlogits[i];
        }
    }
    for (int i=0; i<HIDDEN_SIZE; i++) {
        if (cache->z2[i] <= 0) {
            dz2[i] = 0;
        }
        for (int j=0; j<HIDDEN_SIZE; j++) {
            grads->w2[i][j] += dz2[i] * cache->a1[j];
        }
        grads->b2[i] += dz2[i];
    }

    // Do the same for the first hidden layer.
    float dz1[HIDDEN_SIZE] = {0};
    for (int i=0; i<HIDDEN_SIZE; i++) {
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dz1[j] += linear_relu_stack_2_weight[i][j] * dz2[i];
        }
    }
    for (int i=0; i<HIDDEN_SIZE; i++) {
        if (cache->z1[i] <= 0) {
            dz1[i] = 0;
        }
        for (int j=0; j<784; j++) {
            grads->w0[i][j] += dz1[i] * input[j];
        }
        grads->b0[i] += dz1[i];
    }
}

void update_weights(Gradients *grads, float learning_rate, int batch_count) {
    // Average the gradients, then take a small step to lower the loss.
    float step = learning_rate / batch_count;
    for (int i=0; i<HIDDEN_SIZE; i++) {
        for (int j=0; j<784; j++) {
            linear_relu_stack_0_weight[i][j] -= step * grads->w0[i][j];
        }
        linear_relu_stack_0_bias[i] -= step * grads->b0[i];
        for (int j=0; j<HIDDEN_SIZE; j++) {
            linear_relu_stack_2_weight[i][j] -= step * grads->w2[i][j];
        }
        linear_relu_stack_2_bias[i] -= step * grads->b2[i];
    }
    for (int i=0; i<10; i++) {
        for (int j=0; j<HIDDEN_SIZE; j++) {
            linear_relu_stack_4_weight[i][j] -= step * grads->w4[i][j];
        }
        linear_relu_stack_4_bias[i] -= step * grads->b4[i];
    }
}

int train_model(void) {
    int epochs = 20;
    float learning_rate = 0.01f;
    int train_count = 60000;
    int order[60000];
    // Keep the large gradient arrays off the stack.
    static Gradients grads;
    ForwardCache cache;
    float input[784];
    init_mlp_weights(1234);

    for (int i=0; i<train_count; i++) {
        order[i] = i;
    }
    for (int epoch=0; epoch<epochs; epoch++) {
        // Mix up the image order before each pass through the training set.
        for (int i=train_count-1; i>0; i--) {
            int j = rand() % (i + 1);
            int temp = order[i];
            order[i] = order[j];
            order[j] = temp;
        }

        float total_loss = 0;
        for (int start=0; start<train_count; start+=BATCH_SIZE) {
            memset(&grads, 0, sizeof(grads));
            int batch_count = BATCH_SIZE;
            if (start + batch_count > train_count) {
                batch_count = train_count - start;
            }
            for (int i=0; i<batch_count; i++) {
                int index = order[start + i];
                read_image(input, 1, index);
                int label = read_label(1, index);
                forward(input, &cache);
                total_loss += find_cross_entropy_cost(cache.probs, label);
                backward(input, label, &cache, &grads);
            }
            update_weights(&grads, learning_rate, batch_count);
        }
        printf("Epoch %d/%d: training loss = %.4f\n", epoch + 1, epochs, total_loss / train_count);
        fflush(stdout);
    }

    if (!save_trained_weights()) {
        return 1;
    }
    printf("Saved weights to mnist_mlp/stored_weights.c. Run make eval to rebuild and evaluate.\n");
    return 0;
}
