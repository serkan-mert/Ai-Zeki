#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "activation.h"
#include "layer.h"

static float relu(float x) { return x > 0 ? x : 0; }
static float relu_derivative(float x) { return x > 0 ? 1 : 0; }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float sigmoid_derivative(float y) { return y * (1.0f - y); }
static float tanh_derivative(float y) { return 1.0f - y * y; }

void activation_forward(Layer* self, Tensor* input) {
    ActivationParams* p = (ActivationParams*)self->params;
    if (self->output) tensor_free(self->output);
    self->output = tensor_copy(input);
    p->input_cache = tensor_copy(input);

    if (p->type == ACT_SOFTMAX) {
        int batch = input->shape[0];
        int dim = input->shape[1];
        for (int b = 0; b < batch; b++) {
            float max_val = input->data[b * dim];
            for (int i = 1; i < dim; i++) {
                if (input->data[b * dim + i] > max_val) max_val = input->data[b * dim + i];
            }
            float sum = 0.0f;
            for (int i = 0; i < dim; i++) {
                self->output->data[b * dim + i] = expf(input->data[b * dim + i] - max_val);
                sum += self->output->data[b * dim + i];
            }
            if (sum < 1e-10f) sum = 1e-10f;
            for (int i = 0; i < dim; i++) {
                self->output->data[b * dim + i] /= sum;
                if (self->output->data[b * dim + i] < 1e-10f) self->output->data[b * dim + i] = 1e-10f;
                if (self->output->data[b * dim + i] > 1.0f - 1e-10f) self->output->data[b * dim + i] = 1.0f - 1e-10f;
            }
        }
    } else {
        for (int i = 0; i < input->size; i++) {
            switch (p->type) {
                case ACT_RELU: self->output->data[i] = relu(input->data[i]); break;
                case ACT_SIGMOID: self->output->data[i] = sigmoid(input->data[i]); break;
                case ACT_TANH: self->output->data[i] = tanhf(input->data[i]); break;
                case ACT_SOFTMAX: break;
            }
        }
    }
}

void activation_backward(Layer* self, Tensor* grad_output) {
    ActivationParams* p = (ActivationParams*)self->params;
    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_copy(grad_output);

    if (p->type == ACT_SOFTMAX) {
        int batch = grad_output->shape[0];
        int dim = grad_output->shape[1];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    if (i == j) {
                        sum += grad_output->data[b * dim + j] * self->output->data[b * dim + i] * (1 - self->output->data[b * dim + i]);
                    } else {
                        sum -= grad_output->data[b * dim + j] * self->output->data[b * dim + i] * self->output->data[b * dim + j];
                    }
                }
                self->grad_input->data[b * dim + i] = sum;
            }
        }
    } else {
        for (int i = 0; i < grad_output->size; i++) {
            float output_val = self->output->data[i];
            switch (p->type) {
                case ACT_RELU: self->grad_input->data[i] *= relu_derivative(p->input_cache->data[i]); break;
                case ACT_SIGMOID: self->grad_input->data[i] *= sigmoid_derivative(output_val); break;
                case ACT_TANH: self->grad_input->data[i] *= tanh_derivative(output_val); break;
                case ACT_SOFTMAX: break;
            }
        }
    }
}

void activation_free(Layer* self) {
    if (self->params) {
        ActivationParams* p = (ActivationParams*)self->params;
        tensor_free(p->input_cache);
        free(p);
    }
}

Layer* activation_create(ActivationType type, const char* name) {
    Layer* l = layer_create(LAYER_ACTIVATION, name);
    l->forward = activation_forward;
    l->backward = activation_backward;
    l->free = activation_free;

    ActivationParams* p = (ActivationParams*)malloc(sizeof(ActivationParams));
    p->type = type;
    p->input_cache = NULL;
    l->params = p;

    return l;
}
