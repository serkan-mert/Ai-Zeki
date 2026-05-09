#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "dense.h"
#include "layer.h"

void dense_forward(Layer* self, Tensor* input) {
    DenseParams* p = (DenseParams*)self->params;
    if (!p || !self->weights || !self->biases) {
        if (self->output) tensor_free(self->output);
        self->output = tensor_copy(input);
        return;
    }
    int batch = input->shape[0];
    int in_dim = p->input_dim;
    int out_dim = p->output_dim;

    int out_shape[] = {batch, out_dim};
    if (self->output) tensor_free(self->output);
    self->output = tensor_create(out_shape, 2);

    p->input_cache = tensor_copy(input);

    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = self->biases->data[o];
            for (int i = 0; i < in_dim; i++) {
                sum += input->data[b * in_dim + i] * self->weights->data[o * in_dim + i];
            }
            self->output->data[b * out_dim + o] = sum;
        }
    }
}

void dense_backward(Layer* self, Tensor* grad_output) {
    DenseParams* p = (DenseParams*)self->params;
    int batch = grad_output->shape[0];
    int in_dim = p->input_dim;
    int out_dim = p->output_dim;

    int grad_shape[] = {batch, in_dim};
    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_create(grad_shape, 2);
    tensor_fill(self->grad_input, 0.0f);

    int wgrad_shape[] = {out_dim, in_dim};
    if (!p->weight_grad) p->weight_grad = tensor_create(wgrad_shape, 2);
    tensor_fill(p->weight_grad, 0.0f);

    if (!p->bias_grad) p->bias_grad = tensor_create(&out_dim, 1);
    tensor_fill(p->bias_grad, 0.0f);

    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            float grad = grad_output->data[b * out_dim + o];
            p->bias_grad->data[o] += grad;
            for (int i = 0; i < in_dim; i++) {
                self->grad_input->data[b * in_dim + i] += grad * self->weights->data[o * in_dim + i];
                p->weight_grad->data[o * in_dim + i] += grad * p->input_cache->data[b * in_dim + i];
            }
        }
    }
}

void dense_free(Layer* self) {
    if (self->params) {
        DenseParams* p = (DenseParams*)self->params;
        tensor_free(p->input_cache);
        tensor_free(p->weight_grad);
        tensor_free(p->bias_grad);
        free(p);
    }
    tensor_free(self->weights);
    tensor_free(self->biases);
}

void dense_init_weights(Layer* self) {
    DenseParams* p = (DenseParams*)self->params;
    float scale = sqrtf(2.0f / p->input_dim);
    for (int i = 0; i < self->weights->size; i++) {
        self->weights->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    tensor_fill(self->biases, 0.0f);
}

Layer* dense_create(int input_dim, int output_dim, const char* name) {
    Layer* l = layer_create(LAYER_DENSE, name);
    l->forward = dense_forward;
    l->backward = dense_backward;
    l->free = dense_free;

    DenseParams* p = (DenseParams*)calloc(1, sizeof(DenseParams));
    p->input_dim = input_dim;
    p->output_dim = output_dim;
    p->learning_rate = 0.01f;
    l->params = p;

    int w_shape[] = {output_dim, input_dim};
    l->weights = tensor_create(w_shape, 2);
    int b_shape[] = {output_dim};
    l->biases = tensor_create(b_shape, 1);
    if (!l->weights || !l->biases) {
        layer_free(l);
        return NULL;
    }
    dense_init_weights(l);

    return l;
}
