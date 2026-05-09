#ifndef DENSE_H
#define DENSE_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int input_dim;
    int output_dim;
    float learning_rate;
    Tensor* input_cache;
    Tensor* weight_grad;
    Tensor* bias_grad;
} DenseParams;

void dense_forward(Layer* self, Tensor* input);
void dense_backward(Layer* self, Tensor* grad_output);
void dense_free(Layer* self);
void dense_init_weights(Layer* self);

Layer* dense_create(int input_dim, int output_dim, const char* name);

#endif
