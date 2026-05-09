#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include "layer.h"

typedef enum {
    ACT_RELU,
    ACT_SIGMOID,
    ACT_TANH,
    ACT_SOFTMAX
} ActivationType;

typedef struct {
    ActivationType type;
    Tensor* input_cache;
} ActivationParams;

void activation_forward(Layer* self, Tensor* input);
void activation_backward(Layer* self, Tensor* grad_output);
void activation_free(Layer* self);

Layer* activation_create(ActivationType type, const char* name);

#endif
