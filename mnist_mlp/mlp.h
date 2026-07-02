#ifndef MLP_H
#define MLP_H

#define BATCH_SIZE 50

#define HIDDEN_SIZE 512

typedef struct {
    float z1[HIDDEN_SIZE];
    float a1[HIDDEN_SIZE];
    float z2[HIDDEN_SIZE];
    float a2[HIDDEN_SIZE];
    float logits[10];
    float probs[10];
} ForwardCache;

typedef struct {
    float w0[HIDDEN_SIZE][784];
    float b0[HIDDEN_SIZE];
    float w2[HIDDEN_SIZE][HIDDEN_SIZE];
    float b2[HIDDEN_SIZE];
    float w4[10][HIDDEN_SIZE];
    float b4[10];
} Gradients;

void forward(float input[], ForwardCache *cache);
void train_model(void);
void softmax(float logits[10], float probs[10]);

#endif
