#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "batchnorm.h"
#include "layer.h"

static int spatial_dims(Tensor* t) {
    int s = 1;
    for (int i = 2; i < t->ndim; i++) s *= t->shape[i];
    return s;
}

void batchnorm_forward(Layer* self, Tensor* input) {
    BatchNormParams* p = (BatchNormParams*)self->params;
    int batch = input->shape[0];
    int dim = p->dim;
    int spatial = spatial_dims(input);

    if (self->output) tensor_free(self->output);
    int* out_shape = (int*)malloc(input->ndim * sizeof(int));
    memcpy(out_shape, input->shape, input->ndim * sizeof(int));
    self->output = tensor_create(out_shape, input->ndim);
    free(out_shape);

    p->input_cache = tensor_copy(input);

    if (p->training) {
        Tensor* mean = tensor_create((int[]){dim}, 1);
        Tensor* var = tensor_create((int[]){dim}, 1);

        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < spatial; s++)
                    sum += input->data[b * dim * spatial + d * spatial + s];
            mean->data[d] = sum / (batch * spatial);
        }

        for (int d = 0; d < dim; d++) {
            float sum = 0;
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < spatial; s++) {
                    float diff = input->data[b * dim * spatial + d * spatial + s] - mean->data[d];
                    sum += diff * diff;
                }
            var->data[d] = sum / (batch * spatial);
        }

        for (int d = 0; d < dim; d++) {
            p->running_mean->data[d] = p->momentum * p->running_mean->data[d] + (1 - p->momentum) * mean->data[d];
            p->running_var->data[d] = p->momentum * p->running_var->data[d] + (1 - p->momentum) * var->data[d];
        }

        if (p->normalized_cache) tensor_free(p->normalized_cache);
        p->normalized_cache = tensor_copy(input);
        if (p->std_cache) tensor_free(p->std_cache);
        p->std_cache = tensor_create((int[]){dim}, 1);
        for (int d = 0; d < dim; d++)
            p->std_cache->data[d] = sqrtf(var->data[d] + p->epsilon);

        for (int b = 0; b < batch; b++)
            for (int d = 0; d < dim; d++)
                for (int s = 0; s < spatial; s++) {
                    float norm = (input->data[b * dim * spatial + d * spatial + s] - mean->data[d]) / p->std_cache->data[d];
                    p->normalized_cache->data[b * dim * spatial + d * spatial + s] = norm;
                    self->output->data[b * dim * spatial + d * spatial + s] = p->gamma->data[d] * norm + p->beta->data[d];
                }

        tensor_free(mean);
        tensor_free(var);
    } else {
        if (p->normalized_cache) tensor_free(p->normalized_cache);
        p->normalized_cache = tensor_copy(input);
        if (p->std_cache) tensor_free(p->std_cache);
        p->std_cache = tensor_create((int[]){dim}, 1);
        for (int d = 0; d < dim; d++)
            p->std_cache->data[d] = sqrtf(p->running_var->data[d] + p->epsilon);

        for (int b = 0; b < batch; b++)
            for (int d = 0; d < dim; d++)
                for (int s = 0; s < spatial; s++) {
                    float norm = (input->data[b * dim * spatial + d * spatial + s] - p->running_mean->data[d]) / p->std_cache->data[d];
                    p->normalized_cache->data[b * dim * spatial + d * spatial + s] = norm;
                    self->output->data[b * dim * spatial + d * spatial + s] = p->gamma->data[d] * norm + p->beta->data[d];
                }
    }
}

void batchnorm_backward(Layer* self, Tensor* grad_output) {
    BatchNormParams* p = (BatchNormParams*)self->params;
    int batch = grad_output->shape[0];
    int dim = p->dim;
    int spatial = spatial_dims(grad_output);

    if (self->grad_input) tensor_free(self->grad_input);
    int* grad_shape = (int*)malloc(grad_output->ndim * sizeof(int));
    memcpy(grad_shape, grad_output->shape, grad_output->ndim * sizeof(int));
    self->grad_input = tensor_create(grad_shape, grad_output->ndim);
    free(grad_shape);

    Tensor* dgamma = tensor_create((int[]){dim}, 1);
    Tensor* dbeta = tensor_create((int[]){dim}, 1);

    for (int d = 0; d < dim; d++) {
        dgamma->data[d] = 0;
        dbeta->data[d] = 0;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < spatial; s++) {
                dgamma->data[d] += grad_output->data[b * dim * spatial + d * spatial + s] *
                                   p->normalized_cache->data[b * dim * spatial + d * spatial + s];
                dbeta->data[d] += grad_output->data[b * dim * spatial + d * spatial + s];
            }
    }

    for (int d = 0; d < dim; d++) {
        p->gamma->data[d] -= 0.01f * dgamma->data[d];
        p->beta->data[d] -= 0.01f * dbeta->data[d];
    }

    for (int b = 0; b < batch; b++)
        for (int d = 0; d < dim; d++)
            for (int s = 0; s < spatial; s++)
                self->grad_input->data[b * dim * spatial + d * spatial + s] =
                    grad_output->data[b * dim * spatial + d * spatial + s] * p->gamma->data[d] / p->std_cache->data[d];

    tensor_free(dgamma);
    tensor_free(dbeta);
}

void batchnorm_free(Layer* self) {
    if (self->params) {
        BatchNormParams* p = (BatchNormParams*)self->params;
        tensor_free(p->gamma);
        tensor_free(p->beta);
        tensor_free(p->running_mean);
        tensor_free(p->running_var);
        tensor_free(p->input_cache);
        tensor_free(p->normalized_cache);
        tensor_free(p->std_cache);
        free(p);
    }
}

Layer* batchnorm_create(int dim, const char* name) {
    if (dim <= 0) return NULL;
    Layer* l = layer_create(LAYER_BATCHNORM, name);
    l->forward = batchnorm_forward;
    l->backward = batchnorm_backward;
    l->free = batchnorm_free;

    BatchNormParams* p = (BatchNormParams*)malloc(sizeof(BatchNormParams));
    p->dim = dim;
    p->momentum = 0.9f;
    p->epsilon = 1e-5f;
    p->gamma = tensor_create((int[]){dim}, 1);
    p->beta = tensor_create((int[]){dim}, 1);
    p->running_mean = tensor_create((int[]){dim}, 1);
    p->running_var = tensor_create((int[]){dim}, 1);
    p->input_cache = NULL;
    p->normalized_cache = NULL;
    p->std_cache = NULL;
    p->training = 1;

    if (!p->gamma || !p->beta || !p->running_mean || !p->running_var) {
        free(p);
        layer_free(l);
        return NULL;
    }

    tensor_fill(p->gamma, 1.0f);
    tensor_fill(p->beta, 0.0f);

    l->params = p;

    return l;
}
