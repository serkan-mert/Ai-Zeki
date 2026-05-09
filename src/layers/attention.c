#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "attention.h"

void attention_forward(Layer* self, Tensor* input) {
    AttentionParams* p = (AttentionParams*)self->params;
    int batch = input->shape[0];
    int seq_len = input->shape[1];
    int embed_dim = p->embed_dim;

    p->seq_len = seq_len;
    p->input_cache = tensor_copy(input);

    if (self->output) tensor_free(self->output);
    self->output = tensor_create((int[]){batch, seq_len, embed_dim}, 3);

    if (p->q_cache) tensor_free(p->q_cache);
    if (p->k_cache) tensor_free(p->k_cache);
    if (p->v_cache) tensor_free(p->v_cache);
    if (p->attn_weights) tensor_free(p->attn_weights);

    p->q_cache = tensor_create((int[]){batch, seq_len, embed_dim}, 3);
    p->k_cache = tensor_create((int[]){batch, seq_len, embed_dim}, 3);
    p->v_cache = tensor_create((int[]){batch, seq_len, embed_dim}, 3);
    p->attn_weights = tensor_create((int[]){batch, seq_len, seq_len}, 3);

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < embed_dim; d++) {
                float q_val = 0, k_val = 0, v_val = 0;
                for (int k = 0; k < embed_dim; k++) {
                    q_val += input->data[b * seq_len * embed_dim + i * embed_dim + k] * p->Wq->data[d * embed_dim + k];
                    k_val += input->data[b * seq_len * embed_dim + i * embed_dim + k] * p->Wk->data[d * embed_dim + k];
                    v_val += input->data[b * seq_len * embed_dim + i * embed_dim + k] * p->Wv->data[d * embed_dim + k];
                }
                p->q_cache->data[b * seq_len * embed_dim + i * embed_dim + d] = q_val;
                p->k_cache->data[b * seq_len * embed_dim + i * embed_dim + d] = k_val;
                p->v_cache->data[b * seq_len * embed_dim + i * embed_dim + d] = v_val;
            }
        }

        for (int i = 0; i < seq_len; i++) {
            float sum = 0;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0;
                for (int d = 0; d < embed_dim; d++) {
                    dot += p->q_cache->data[b * seq_len * embed_dim + i * embed_dim + d] *
                           p->k_cache->data[b * seq_len * embed_dim + j * embed_dim + d];
                }
                float attn = dot / sqrtf(embed_dim);
                p->attn_weights->data[b * seq_len * seq_len + i * seq_len + j] = attn;
                sum += attn;
            }
            for (int j = 0; j < seq_len; j++) {
                p->attn_weights->data[b * seq_len * seq_len + i * seq_len + j] /= (sum + 1e-10f);
            }
        }

        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < embed_dim; d++) {
                float out = 0;
                for (int j = 0; j < seq_len; j++) {
                    out += p->attn_weights->data[b * seq_len * seq_len + i * seq_len + j] *
                           p->v_cache->data[b * seq_len * embed_dim + j * embed_dim + d];
                }
                self->output->data[b * seq_len * embed_dim + i * embed_dim + d] = out;
            }
        }
    }
}

void attention_backward(Layer* self, Tensor* grad_output) {
    AttentionParams* p = (AttentionParams*)self->params;
    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_create(p->input_cache->shape, p->input_cache->ndim);
    tensor_fill(self->grad_input, 0.0f);
}

void attention_free(Layer* self) {
    if (self->params) {
        AttentionParams* p = (AttentionParams*)self->params;
        tensor_free(p->Wq); tensor_free(p->Wk); tensor_free(p->Wv); tensor_free(p->Wo);
        tensor_free(p->input_cache); tensor_free(p->q_cache); tensor_free(p->k_cache);
        tensor_free(p->v_cache); tensor_free(p->attn_weights);
        free(p);
    }
}

Layer* attention_create(int embed_dim, const char* name) {
    if (embed_dim <= 0) return NULL;
    Layer* l = layer_create(LAYER_ATTENTION, name);
    l->forward = attention_forward;
    l->backward = attention_backward;
    l->free = attention_free;

    AttentionParams* p = (AttentionParams*)malloc(sizeof(AttentionParams));
    p->embed_dim = embed_dim;

    p->Wq = tensor_create((int[]){embed_dim, embed_dim}, 2);
    p->Wk = tensor_create((int[]){embed_dim, embed_dim}, 2);
    p->Wv = tensor_create((int[]){embed_dim, embed_dim}, 2);
    p->Wo = tensor_create((int[]){embed_dim, embed_dim}, 2);

    if (!p->Wq || !p->Wk || !p->Wv || !p->Wo) {
        free(p);
        layer_free(l);
        return NULL;
    }

    float scale = sqrtf(2.0f / embed_dim);
    for (int i = 0; i < p->Wq->size; i++) {
        p->Wq->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wk->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wv->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wo->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }

    p->input_cache = NULL;
    p->q_cache = NULL;
    p->k_cache = NULL;
    p->v_cache = NULL;
    p->attn_weights = NULL;
    p->seq_len = 0;

    l->params = p;
    l->weights = p->Wq;

    return l;
}
