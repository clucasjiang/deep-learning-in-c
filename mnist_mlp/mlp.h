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

void forward(float input[], ForwardCache *cache);
void train_model(void);
void softmax(float logits[10], float probs[10]);

#endif
