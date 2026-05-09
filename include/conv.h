#ifndef CONV_H
#define CONV_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    Tensor* kernels;
    Tensor* biases;
    Tensor* kernel_grad;
    Tensor* bias_grad;
    Tensor* input_cache;
    int input_h, input_w;
} ConvParams;

void conv_forward(Layer* self, Tensor* input);
void conv_backward(Layer* self, Tensor* grad_output);
void conv_free(Layer* self);

Layer* conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, const char* name);

#endif
