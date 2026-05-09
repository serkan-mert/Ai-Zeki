#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

typedef enum {
    OPT_SGD,
    OPT_ADAM,
    OPT_RMSPROP
} OptimizerType;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    Tensor** m;
    Tensor** v;
    int num_params;
} AdamParams;

typedef struct {
    OptimizerType type;
    float learning_rate;
    void* params;
} Optimizer;

Optimizer* optimizer_create(OptimizerType type, float lr);
void optimizer_free(Optimizer* opt);
void optimizer_init_param(Optimizer* opt, Tensor* param);
void optimizer_step(Optimizer* opt, Tensor* grad, Tensor* param, int param_idx);

#endif
