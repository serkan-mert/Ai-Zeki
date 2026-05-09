#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lstm.h"
#include "layer.h"
#include "activation.h"

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

void lstm_cell_forward(Layer* self, Tensor* input) {
    LSTMCellParams* p = (LSTMCellParams*)self->params;
    int batch = input->shape[0];
    int input_dim = p->input_dim;
    int hidden_dim = p->hidden_dim;

    if (!p->h_prev) {
        int h_shape[] = {batch, hidden_dim};
        p->h_prev = tensor_create(h_shape, 2);
        p->c_prev = tensor_create(h_shape, 2);
        tensor_fill(p->h_prev, 0.0f);
        tensor_fill(p->c_prev, 0.0f);
    }

    if (self->output) tensor_free(self->output);
    int out_shape[] = {batch, hidden_dim};
    self->output = tensor_create(out_shape, 2);

    if (p->h_next) tensor_free(p->h_next);
    if (p->c_next) tensor_free(p->c_next);
    p->h_next = tensor_create(out_shape, 2);
    p->c_next = tensor_create(out_shape, 2);

    p->input_cache = tensor_copy(input);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < hidden_dim; j++) {
            float f_gate = sigmoid_f(
                p->Wf->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f +
                p->bf->data[j]);
            float i_gate = sigmoid_f(
                p->Wi->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f +
                p->bi->data[j]);
            float c_hat = tanhf(
                p->Wc->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f +
                p->bc->data[j]);
            float o_gate = sigmoid_f(
                p->Wo->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f +
                p->bo->data[j]);

            p->c_next->data[b * hidden_dim + j] =
                f_gate * p->c_prev->data[b * hidden_dim + j] + i_gate * c_hat;
            p->h_next->data[b * hidden_dim + j] =
                o_gate * tanhf(p->c_next->data[b * hidden_dim + j]);
        }
    }

    for (int i = 0; i < p->h_next->size; i++) {
        self->output->data[i] = p->h_next->data[i];
    }

    Tensor* temp = p->h_prev;
    p->h_prev = p->h_next;
    p->h_next = temp;

    temp = p->c_prev;
    p->c_prev = p->c_next;
    p->c_next = temp;
}

void lstm_cell_backward(Layer* self, Tensor* grad_output) {
    LSTMCellParams* p = (LSTMCellParams*)self->params;
    if (self->grad_input) tensor_free(self->grad_input);
    int grad_shape[] = {grad_output->shape[0], p->input_dim};
    self->grad_input = tensor_create(grad_shape, 2);
    tensor_fill(self->grad_input, 0.0f);

    int batch = grad_output->shape[0];
    int h_dim = p->hidden_dim;
    int total_dim = p->input_dim + h_dim;

    if (!p->f_cache) p->f_cache = tensor_create((int[]){batch, h_dim}, 2);
    if (!p->i_cache) p->i_cache = tensor_create((int[]){batch, h_dim}, 2);
    if (!p->c_hat_cache) p->c_hat_cache = tensor_create((int[]){batch, h_dim}, 2);
    if (!p->o_cache) p->o_cache = tensor_create((int[]){batch, h_dim}, 2);

    int w_shape[] = {h_dim, total_dim};
    if (!p->Wf_grad) p->Wf_grad = tensor_create(w_shape, 2);
    if (!p->Wi_grad) p->Wi_grad = tensor_create(w_shape, 2);
    if (!p->Wc_grad) p->Wc_grad = tensor_create(w_shape, 2);
    if (!p->Wo_grad) p->Wo_grad = tensor_create(w_shape, 2);
    int b_shape[] = {h_dim};
    if (!p->bf_grad) p->bf_grad = tensor_create(b_shape, 1);
    if (!p->bi_grad) p->bi_grad = tensor_create(b_shape, 1);
    if (!p->bc_grad) p->bc_grad = tensor_create(b_shape, 1);
    if (!p->bo_grad) p->bo_grad = tensor_create(b_shape, 1);

    tensor_fill(p->Wf_grad, 0.0f);
    tensor_fill(p->Wi_grad, 0.0f);
    tensor_fill(p->Wc_grad, 0.0f);
    tensor_fill(p->Wo_grad, 0.0f);
    tensor_fill(p->bf_grad, 0.0f);
    tensor_fill(p->bi_grad, 0.0f);
    tensor_fill(p->bc_grad, 0.0f);
    tensor_fill(p->bo_grad, 0.0f);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < h_dim; j++) {
            float dh = grad_output->data[b * h_dim + j];
            float c_val = p->c_next->data[b * h_dim + j];
            float h_prev = p->h_prev->data[b * h_dim + j];
            float c_prev = p->c_prev->data[b * h_dim + j];

            float o_gate = p->o_cache->data[b * h_dim + j];
            float tanh_c = tanhf(c_val);

            float dg_c = dh * o_gate * (1 - tanh_c * tanh_c);

            float f_gate = p->f_cache->data[b * h_dim + j];
            float i_gate = p->i_cache->data[b * h_dim + j];
            float c_hat = p->c_hat_cache->data[b * h_dim + j];

            float dg_f = dg_c * c_prev;
            float dg_i = dg_c * c_hat;
            float dg_c_hat = dg_c * i_gate;
            float dg_o = dh * tanh_c;

            dg_f *= f_gate * (1 - f_gate);
            dg_i *= i_gate * (1 - i_gate);
            dg_c_hat *= (1 - c_hat * c_hat);
            dg_o *= o_gate * (1 - o_gate);

            for (int k = 0; k < total_dim; k++) {
                float x_or_h = (k < p->input_dim) ?
                    p->input_cache->data[b * p->input_dim + k] :
                    h_prev;
                p->Wf_grad->data[j * total_dim + k] += dg_f * x_or_h;
                p->Wi_grad->data[j * total_dim + k] += dg_i * x_or_h;
                p->Wc_grad->data[j * total_dim + k] += dg_c_hat * x_or_h;
                p->Wo_grad->data[j * total_dim + k] += dg_o * x_or_h;

                if (k < p->input_dim) {
                    self->grad_input->data[b * p->input_dim + k] +=
                        dg_f * p->Wf->data[j * total_dim + k] +
                        dg_i * p->Wi->data[j * total_dim + k] +
                        dg_c_hat * p->Wc->data[j * total_dim + k] +
                        dg_o * p->Wo->data[j * total_dim + k];
                }
            }

            p->bf_grad->data[j] += dg_f;
            p->bi_grad->data[j] += dg_i;
            p->bc_grad->data[j] += dg_c_hat;
            p->bo_grad->data[j] += dg_o;
        }
    }
}

void lstm_free(Layer* self) {
    if (self->params) {
        LSTMCellParams* p = (LSTMCellParams*)self->params;
        tensor_free(p->Wf); tensor_free(p->Wi); tensor_free(p->Wc); tensor_free(p->Wo);
        tensor_free(p->bf); tensor_free(p->bi); tensor_free(p->bc); tensor_free(p->bo);
        tensor_free(p->Wf_grad); tensor_free(p->Wi_grad); tensor_free(p->Wc_grad); tensor_free(p->Wo_grad);
        tensor_free(p->bf_grad); tensor_free(p->bi_grad); tensor_free(p->bc_grad); tensor_free(p->bo_grad);
        tensor_free(p->h_prev); tensor_free(p->c_prev);
        tensor_free(p->h_next); tensor_free(p->c_next);
        tensor_free(p->input_cache);
        tensor_free(p->f_cache); tensor_free(p->i_cache);
        tensor_free(p->c_hat_cache); tensor_free(p->o_cache);
        free(p);
    }
}

Layer* lstm_cell_create(int input_dim, int hidden_dim, const char* name) {
    Layer* l = layer_create(LAYER_LSTM, name);
    l->forward = lstm_cell_forward;
    l->backward = lstm_cell_backward;
    l->free = lstm_free;

    LSTMCellParams* p = (LSTMCellParams*)calloc(1, sizeof(LSTMCellParams));
    p->input_dim = input_dim;
    p->hidden_dim = hidden_dim;
    p->learning_rate = 0.01f;

    int w_shape[] = {hidden_dim, input_dim + hidden_dim};
    p->Wf = tensor_create(w_shape, 2);
    p->Wi = tensor_create(w_shape, 2);
    p->Wc = tensor_create(w_shape, 2);
    p->Wo = tensor_create(w_shape, 2);

    int b_shape[] = {hidden_dim};
    p->bf = tensor_create(b_shape, 1);
    p->bi = tensor_create(b_shape, 1);
    p->bc = tensor_create(b_shape, 1);
    p->bo = tensor_create(b_shape, 1);

    float scale = sqrtf(2.0f / (input_dim + hidden_dim));
    for (int i = 0; i < p->Wf->size; i++) {
        p->Wf->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wi->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wc->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wo->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }

    p->h_prev = NULL;
    p->c_prev = NULL;
    p->h_next = NULL;
    p->c_next = NULL;
    p->input_cache = NULL;
    p->f_cache = NULL;
    p->i_cache = NULL;
    p->c_hat_cache = NULL;
    p->o_cache = NULL;
    p->batch_size = 0;
    p->seq_len = 0;

    l->params = p;
    l->weights = p->Wf;
    l->biases = p->bf;

    return l;
}
