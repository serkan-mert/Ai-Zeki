#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int dim;
    float momentum;
    float epsilon;
    Tensor* gamma;
    Tensor* beta;
    Tensor* running_mean;
    Tensor* running_var;
    Tensor* input_cache;
    Tensor* normalized_cache;
    Tensor* std_cache;
    int training;
} BatchNormParams;

void batchnorm_forward(Layer* self, Tensor* input);
void batchnorm_backward(Layer* self, Tensor* grad_output);
void batchnorm_free(Layer* self);

Layer* batchnorm_create(int dim, const char* name);

#endif
