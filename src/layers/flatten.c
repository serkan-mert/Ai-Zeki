#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "flatten.h"

void flatten_forward(Layer* self, Tensor* input) {
    FlattenParams* p = (FlattenParams*)self->params;

    p->original_ndim = input->ndim;
    if (p->original_shape) free(p->original_shape);
    p->original_shape = (int*)malloc(input->ndim * sizeof(int));
    memcpy(p->original_shape, input->shape, input->ndim * sizeof(int));

    int flat_size = 1;
    for (int i = 1; i < input->ndim; i++) flat_size *= input->shape[i];

    if (self->output) tensor_free(self->output);
    self->output = tensor_create((int[]){input->shape[0], flat_size}, 2);
    memcpy(self->output->data, input->data, input->size * sizeof(float));
}

void flatten_backward(Layer* self, Tensor* grad_output) {
    FlattenParams* p = (FlattenParams*)self->params;
    if (self->grad_input) tensor_free(self->grad_input);
    self->grad_input = tensor_create(p->original_shape, p->original_ndim);
    memcpy(self->grad_input->data, grad_output->data, grad_output->size * sizeof(float));
}

void flatten_free(Layer* self) {
    if (self->params) {
        FlattenParams* p = (FlattenParams*)self->params;
        if (p->original_shape) free(p->original_shape);
        free(p);
    }
}

Layer* flatten_create(const char* name) {
    Layer* l = layer_create(LAYER_FLATTEN, name);
    l->forward = flatten_forward;
    l->backward = flatten_backward;
    l->free = flatten_free;

    FlattenParams* p = (FlattenParams*)malloc(sizeof(FlattenParams));
    p->original_shape = NULL;
    p->original_ndim = 0;
    l->params = p;

    return l;
}
