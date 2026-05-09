#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int embed_dim;
    Tensor* Wq;
    Tensor* Wk;
    Tensor* Wv;
    Tensor* Wo;
    Tensor* input_cache;
    Tensor* q_cache;
    Tensor* k_cache;
    Tensor* v_cache;
    Tensor* attn_weights;
    int seq_len;
} AttentionParams;

void attention_forward(Layer* self, Tensor* input);
void attention_backward(Layer* self, Tensor* grad_output);
void attention_free(Layer* self);

Layer* attention_create(int embed_dim, const char* name);

#endif
