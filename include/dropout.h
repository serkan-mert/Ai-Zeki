#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    float rate;
    char* mask;
    int training;
} DropoutParams;

void dropout_forward(Layer* self, Tensor* input);
void dropout_backward(Layer* self, Tensor* grad_output);
void dropout_free(Layer* self);

Layer* dropout_create(float rate, const char* name);

#endif
