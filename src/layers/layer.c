#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer.h"

Layer* layer_create(LayerType type, const char* name) {
    Layer* l = (Layer*)malloc(sizeof(Layer));
    l->type = type;
    strncpy(l->name, name, 63);
    l->name[63] = '\0';
    l->weights = NULL;
    l->biases = NULL;
    l->output = NULL;
    l->grad_input = NULL;
    l->params = NULL;
    l->prev = NULL;
    l->next = NULL;
    l->forward = NULL;
    l->backward = NULL;
    l->free = NULL;
    return l;
}

void layer_free(Layer* layer) {
    if (!layer) return;
    if (layer->free) layer->free(layer);
    tensor_free(layer->output);
    tensor_free(layer->grad_input);
    free(layer);
}

void layer_connect(Layer* prev, Layer* next) {
    prev->next = next;
    next->prev = prev;
}

void layer_forward_pass(Layer* input_layer, Tensor* input) {
    Layer* curr = input_layer;
    Tensor* inp = input;
    while (curr) {
        if (curr->forward) {
            curr->forward(curr, inp);
            inp = curr->output;
        }
        curr = curr->next;
    }
}

void layer_backward_pass(Layer* output_layer, Tensor* grad_output) {
    Layer* curr = output_layer;
    Tensor* grad = grad_output;
    while (curr) {
        if (curr->backward) curr->backward(curr, grad);
        grad = curr->grad_input;
        curr = curr->prev;
    }
}
