#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

typedef enum {
    LAYER_DENSE,
    LAYER_ACTIVATION,
    LAYER_DROPOUT,
    LAYER_INPUT,
    LAYER_OUTPUT,
    LAYER_CONV2D,
    LAYER_LSTM,
    LAYER_GRU,
    LAYER_FLATTEN,
    LAYER_BATCHNORM = 100,
    LAYER_ATTENTION = 101,
    LAYER_TRANSFORMER = 102
} LayerType;

typedef struct Layer Layer;

struct Layer {
    LayerType type;
    char name[64];
    Tensor* weights;
    Tensor* biases;
    Tensor* output;
    Tensor* grad_input;
    void* params;
    Layer* prev;
    Layer* next;
    void (*forward)(Layer* self, Tensor* input);
    void (*backward)(Layer* self, Tensor* grad_output);
    void (*free)(Layer* self);
};

Layer* layer_create(LayerType type, const char* name);
void layer_free(Layer* layer);
void layer_connect(Layer* prev, Layer* next);
void layer_forward_pass(Layer* input_layer, Tensor* input);
void layer_backward_pass(Layer* output_layer, Tensor* grad_output);

#endif
