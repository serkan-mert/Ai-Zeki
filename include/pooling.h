#ifndef POOLING_H
#define POOLING_H

#include "tensor.h"
#include "layer.h"

typedef enum {
    POOL_MAX,
    POOL_AVG
} PoolType;

typedef struct {
    PoolType type;
    int kernel_size;
    int stride;
    int input_h, input_w, input_c;
    Tensor* indices_cache;
} PoolingParams;

void pooling_forward(Layer* self, Tensor* input);
void pooling_backward(Layer* self, Tensor* grad_output);
void pooling_free(Layer* self);

Layer* maxpool2d_create(int kernel_size, int stride, const char* name);

#endif
