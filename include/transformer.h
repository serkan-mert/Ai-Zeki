#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int embed_dim;
    int num_heads;
    int ff_dim;
    Layer* attn_layer;
    Layer* ff_dense1;
    Layer* ff_activation;
    Layer* ff_dense2;
    Layer* ln1;
    Layer* ln2;
} TransformerBlockParams;

void transformer_block_forward(Layer* self, Tensor* input);
void transformer_block_backward(Layer* self, Tensor* grad_output);
void transformer_free(Layer* self);

Layer* transformer_block_create(int embed_dim, int num_heads, int ff_dim, const char* name);

#endif
