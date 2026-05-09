#ifndef FLATTEN_H
#define FLATTEN_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int* original_shape;
    int original_ndim;
} FlattenParams;

void flatten_forward(Layer* self, Tensor* input);
void flatten_backward(Layer* self, Tensor* grad_output);
void flatten_free(Layer* self);

Layer* flatten_create(const char* name);

#endif
