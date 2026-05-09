#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transformer.h"
#include "attention.h"
#include "dense.h"
#include "activation.h"
#include "layer.h"

void transformer_block_forward(Layer* self, Tensor* input) {
    TransformerBlockParams* p = (TransformerBlockParams*)self->params;

    if (self->output) tensor_free(self->output);

    if (p->attn_layer) {
        p->attn_layer->forward(p->attn_layer, input);
        Tensor* attn_out = tensor_copy(p->attn_layer->output);
        for (int i = 0; i < input->size; i++) {
            attn_out->data[i] += input->data[i];
        }

        Tensor* curr = tensor_copy(attn_out);

        if (p->ln1) {
            p->ln1->forward(p->ln1, curr);
            tensor_free(curr);
            curr = tensor_copy(p->ln1->output);
        }

        if (p->ff_dense1) {
            p->ff_dense1->forward(p->ff_dense1, curr);
            tensor_free(curr);
            if (p->ff_activation) {
                p->ff_activation->forward(p->ff_activation, p->ff_dense1->output);
                curr = tensor_copy(p->ff_activation->output);
            } else {
                curr = tensor_copy(p->ff_dense1->output);
            }

            if (p->ff_dense2) {
                p->ff_dense2->forward(p->ff_dense2, curr);
                int residual_size = p->ff_dense2->output->size < attn_out->size ? p->ff_dense2->output->size : attn_out->size;
                for (int i = 0; i < residual_size; i++) {
                    p->ff_dense2->output->data[i] += attn_out->data[i];
                }

                if (p->ln2) {
                    p->ln2->forward(p->ln2, p->ff_dense2->output);
                    self->output = tensor_copy(p->ln2->output);
                } else {
                    self->output = tensor_copy(p->ff_dense2->output);
                }
                tensor_free(curr);
                curr = NULL;
            } else {
                self->output = curr;
                curr = NULL;
            }
        } else {
            self->output = curr;
            curr = NULL;
        }
        tensor_free(attn_out);
    } else {
        self->output = tensor_copy(input);
    }
}

void transformer_block_backward(Layer* self, Tensor* grad_output) {
    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_copy(grad_output);
}

void transformer_free(Layer* self) {
    if (self->params) {
        TransformerBlockParams* p = (TransformerBlockParams*)self->params;
        if (p->attn_layer) layer_free(p->attn_layer);
        if (p->ff_dense1) layer_free(p->ff_dense1);
        if (p->ff_activation) layer_free(p->ff_activation);
        if (p->ff_dense2) layer_free(p->ff_dense2);
        if (p->ln1) layer_free(p->ln1);
        if (p->ln2) layer_free(p->ln2);
        free(p);
    }
}

Layer* transformer_block_create(int embed_dim, int num_heads, int ff_dim, const char* name) {
    if (embed_dim <= 0 || num_heads <= 0 || ff_dim <= 0) return NULL;
    Layer* l = layer_create(LAYER_TRANSFORMER, name);
    l->forward = transformer_block_forward;
    l->backward = transformer_block_backward;
    l->free = transformer_free;

    TransformerBlockParams* p = (TransformerBlockParams*)malloc(sizeof(TransformerBlockParams));
    p->embed_dim = embed_dim;
    p->num_heads = num_heads;
    p->ff_dim = ff_dim;

    p->attn_layer = attention_create(embed_dim, "attn");
    p->ff_dense1 = dense_create(embed_dim, ff_dim, "ff1");
    p->ff_activation = activation_create(ACT_RELU, "ff_act");
    p->ff_dense2 = dense_create(ff_dim, embed_dim, "ff2");
    p->ln1 = NULL;
    p->ln2 = NULL;

    if (!p->attn_layer || !p->ff_dense1 || !p->ff_dense2) {
        free(p);
        layer_free(l);
        return NULL;
    }

    l->params = p;

    return l;
}
