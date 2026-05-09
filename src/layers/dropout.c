#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dropout.h"
#include "layer.h"

void dropout_forward(Layer* self, Tensor* input) {
    DropoutParams* p = (DropoutParams*)self->params;
    if (self->output) tensor_free(self->output);
    self->output = tensor_copy(input);

    if (p->training) {
        if (p->mask) free(p->mask);
        p->mask = (char*)malloc(input->size * sizeof(char));
        float scale = 1.0f / (1.0f - p->rate);
        for (int i = 0; i < input->size; i++) {
            if ((float)rand() / RAND_MAX < p->rate) {
                p->mask[i] = 0;
                self->output->data[i] = 0;
            } else {
                p->mask[i] = 1;
                self->output->data[i] *= scale;
            }
        }
    }
}

void dropout_backward(Layer* self, Tensor* grad_output) {
    DropoutParams* p = (DropoutParams*)self->params;
    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_copy(grad_output);

    if (p->training && p->mask) {
        float scale = 1.0f / (1.0f - p->rate);
        for (int i = 0; i < grad_output->size; i++) {
            self->grad_input->data[i] *= p->mask[i] * scale;
        }
    }
}

void dropout_free(Layer* self) {
    if (self->params) {
        DropoutParams* p = (DropoutParams*)self->params;
        if (p->mask) free(p->mask);
        free(p);
    }
}

Layer* dropout_create(float rate, const char* name) {
    Layer* l = layer_create(LAYER_DROPOUT, name);
    l->forward = dropout_forward;
    l->backward = dropout_backward;
    l->free = dropout_free;

    DropoutParams* p = (DropoutParams*)malloc(sizeof(DropoutParams));
    p->rate = rate;
    p->mask = NULL;
    p->training = 1;
    l->params = p;

    return l;
}
