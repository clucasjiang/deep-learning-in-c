#ifndef MLP_WEIGHTS_H
#define MLP_WEIGHTS_H

#include "mlp.h"

extern float linear_relu_stack_0_weight[HIDDEN_SIZE][784];
extern float linear_relu_stack_0_bias[HIDDEN_SIZE];

extern float linear_relu_stack_2_weight[HIDDEN_SIZE][HIDDEN_SIZE];
extern float linear_relu_stack_2_bias[HIDDEN_SIZE];

extern float linear_relu_stack_4_weight[10][HIDDEN_SIZE];
extern float linear_relu_stack_4_bias[10];

void init_mlp_weights(unsigned int seed);
void load_stored_weights(void);
int save_trained_weights(void);
int save_mlp_weights(const char *source_path, const char *header_path);

#endif
