#ifndef MLP_WEIGHTS
#define MLP_WEIGHTS
extern float linear_relu_stack_0_weight[512][784];
extern float linear_relu_stack_0_bias[512];

extern float linear_relu_stack_2_weight[512][512];
extern float linear_relu_stack_2_bias[512];

extern float linear_relu_stack_4_weight[10][512];
extern float linear_relu_stack_4_bias[10];

void init_mlp_weights(unsigned int seed);

#endif
