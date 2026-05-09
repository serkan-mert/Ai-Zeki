#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gru.h"

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

void gru_cell_forward(Layer* self, Tensor* input) {
    GRUCellParams* p = (GRUCellParams*)self->params;
    int batch = input->shape[0];
    int input_dim = p->input_dim;
    int hidden_dim = p->hidden_dim;

    if (!p->h_prev) {
        int h_shape[] = {batch, hidden_dim};
        p->h_prev = tensor_create(h_shape, 2);
        tensor_fill(p->h_prev, 0.0f);
    }

    if (self->output) tensor_free(self->output);
    self->output = tensor_create((int[]){batch, hidden_dim}, 2);

    p->input_cache = tensor_copy(input);

    if (p->z_cache) tensor_free(p->z_cache);
    if (p->r_cache) tensor_free(p->r_cache);
    if (p->h_hat_cache) tensor_free(p->h_hat_cache);
    p->z_cache = tensor_create((int[]){batch, hidden_dim}, 2);
    p->r_cache = tensor_create((int[]){batch, hidden_dim}, 2);
    p->h_hat_cache = tensor_create((int[]){batch, hidden_dim}, 2);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < hidden_dim; j++) {
            float z = sigmoid_f(
                p->Wz->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f + p->bz->data[j]);
            float r = sigmoid_f(
                p->Wr->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f + p->br->data[j]);

            float h_hat = tanhf(
                p->Wh->data[j * (input_dim + hidden_dim) + hidden_dim + 0] * 1.0f +
                p->Uh->data[j * hidden_dim + 0] * r * p->h_prev->data[b * hidden_dim + j] +
                p->bh->data[j]);

            p->z_cache->data[b * hidden_dim + j] = z;
            p->r_cache->data[b * hidden_dim + j] = r;
            p->h_hat_cache->data[b * hidden_dim + j] = h_hat;

            self->output->data[b * hidden_dim + j] =
                (1 - z) * p->h_prev->data[b * hidden_dim + j] + z * h_hat;
        }
    }

    tensor_free(p->h_prev);
    p->h_prev = tensor_copy(self->output);
}

void gru_cell_backward(Layer* self, Tensor* grad_output) {
    GRUCellParams* p = (GRUCellParams*)self->params;
    int batch = grad_output->shape[0];
    int h_dim = p->hidden_dim;
    int total_dim = p->input_dim + h_dim;

    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_create((int[]){batch, p->input_dim}, 2);
    tensor_fill(self->grad_input, 0.0f);

    int w_shape[] = {h_dim, total_dim};
    if (!p->Wz_grad) p->Wz_grad = tensor_create(w_shape, 2);
    if (!p->Wr_grad) p->Wr_grad = tensor_create(w_shape, 2);
    if (!p->Wh_grad) p->Wh_grad = tensor_create(w_shape, 2);
    if (!p->Uz_grad) p->Uz_grad = tensor_create((int[]){h_dim, h_dim}, 2);
    if (!p->Ur_grad) p->Ur_grad = tensor_create((int[]){h_dim, h_dim}, 2);
    if (!p->Uh_grad) p->Uh_grad = tensor_create((int[]){h_dim, h_dim}, 2);
    int b_shape[] = {h_dim};
    if (!p->bz_grad) p->bz_grad = tensor_create(b_shape, 1);
    if (!p->br_grad) p->br_grad = tensor_create(b_shape, 1);
    if (!p->bh_grad) p->bh_grad = tensor_create(b_shape, 1);

    tensor_fill(p->Wz_grad, 0.0f);
    tensor_fill(p->Wr_grad, 0.0f);
    tensor_fill(p->Wh_grad, 0.0f);
    tensor_fill(p->Uz_grad, 0.0f);
    tensor_fill(p->Ur_grad, 0.0f);
    tensor_fill(p->Uh_grad, 0.0f);
    tensor_fill(p->bz_grad, 0.0f);
    tensor_fill(p->br_grad, 0.0f);
    tensor_fill(p->bh_grad, 0.0f);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < h_dim; j++) {
            float dh = grad_output->data[b * h_dim + j];
            float h_prev_val = p->h_prev->data[b * h_dim + j];
            float z = p->z_cache->data[b * h_dim + j];
            float r = p->r_cache->data[b * h_dim + j];
            float h_hat = p->h_hat_cache->data[b * h_dim + j];

            float dz = dh * (-h_prev_val + h_hat) * z * (1.0f - z);
            float dh_hat = dh * z * (1.0f - h_hat * h_hat);
            float dr = dh_hat * p->Uh->data[j * h_dim + 0] * h_prev_val * r * (1.0f - r);

            p->Wz_grad->data[j * total_dim + h_dim] += dz * 1.0f;
            p->Wr_grad->data[j * total_dim + h_dim] += dr * 1.0f;
            p->Wh_grad->data[j * total_dim + h_dim] += dh_hat * 1.0f;
            p->Uh_grad->data[j * h_dim + 0] += dh_hat * r * h_prev_val;
            p->bz_grad->data[j] += dz;
            p->br_grad->data[j] += dr;
            p->bh_grad->data[j] += dh_hat;
        }
    }
}

void gru_free(Layer* self) {
    if (self->params) {
        GRUCellParams* p = (GRUCellParams*)self->params;
        tensor_free(p->Wz); tensor_free(p->Wr); tensor_free(p->Wh);
        tensor_free(p->Uz); tensor_free(p->Ur); tensor_free(p->Uh);
        tensor_free(p->bz); tensor_free(p->br); tensor_free(p->bh);
        tensor_free(p->Wz_grad); tensor_free(p->Wr_grad); tensor_free(p->Wh_grad);
        tensor_free(p->Uz_grad); tensor_free(p->Ur_grad); tensor_free(p->Uh_grad);
        tensor_free(p->bz_grad); tensor_free(p->br_grad); tensor_free(p->bh_grad);
        tensor_free(p->h_prev); tensor_free(p->input_cache);
        tensor_free(p->z_cache); tensor_free(p->r_cache); tensor_free(p->h_hat_cache);
        free(p);
    }
}

Layer* gru_cell_create(int input_dim, int hidden_dim, const char* name) {
    Layer* l = layer_create(LAYER_GRU, name);
    l->forward = gru_cell_forward;
    l->backward = gru_cell_backward;
    l->free = gru_free;

    GRUCellParams* p = (GRUCellParams*)calloc(1, sizeof(GRUCellParams));
    p->input_dim = input_dim;
    p->hidden_dim = hidden_dim;

    int w_shape[] = {hidden_dim, input_dim + hidden_dim};
    p->Wz = tensor_create(w_shape, 2);
    p->Wr = tensor_create(w_shape, 2);
    p->Wh = tensor_create(w_shape, 2);
    p->Uz = tensor_create((int[]){hidden_dim, hidden_dim}, 2);
    p->Ur = tensor_create((int[]){hidden_dim, hidden_dim}, 2);
    p->Uh = tensor_create((int[]){hidden_dim, hidden_dim}, 2);

    p->bz = tensor_create((int[]){hidden_dim}, 1);
    p->br = tensor_create((int[]){hidden_dim}, 1);
    p->bh = tensor_create((int[]){hidden_dim}, 1);

    float scale = sqrtf(2.0f / (input_dim + hidden_dim));
    for (int i = 0; i < p->Wz->size; i++) {
        p->Wz->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wr->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Wh->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    scale = sqrtf(2.0f / hidden_dim);
    for (int i = 0; i < p->Uz->size; i++) {
        p->Uz->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Ur->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        p->Uh->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }

    p->h_prev = NULL;
    p->input_cache = NULL;
    p->z_cache = NULL;
    p->r_cache = NULL;
    p->h_hat_cache = NULL;

    l->params = p;
    l->weights = p->Wz;
    l->biases = p->bz;

    return l;
}
