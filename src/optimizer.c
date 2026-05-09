#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "optimizer.h"

Optimizer* optimizer_create(OptimizerType type, float lr) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->type = type;
    opt->learning_rate = lr;
    opt->params = NULL;

    if (type == OPT_ADAM) {
        AdamParams* adam = (AdamParams*)malloc(sizeof(AdamParams));
        adam->learning_rate = lr;
        adam->beta1 = 0.9f;
        adam->beta2 = 0.999f;
        adam->epsilon = 1e-8f;
        adam->t = 0;
        adam->m = NULL;
        adam->v = NULL;
        adam->num_params = 0;
        opt->params = adam;
    }

    return opt;
}

void optimizer_free(Optimizer* opt) {
    if (!opt) return;
    if (opt->type == OPT_ADAM && opt->params) {
        AdamParams* adam = (AdamParams*)opt->params;
        for (int i = 0; i < adam->num_params; i++) {
            tensor_free(adam->m[i]);
            tensor_free(adam->v[i]);
        }
        free(adam->m);
        free(adam->v);
        free(adam);
    }
    free(opt);
}

void optimizer_init_param(Optimizer* opt, Tensor* param) {
    if (opt->type == OPT_ADAM) {
        AdamParams* adam = (AdamParams*)opt->params;
        adam->num_params++;
        adam->m = (Tensor**)realloc(adam->m, adam->num_params * sizeof(Tensor*));
        adam->v = (Tensor**)realloc(adam->v, adam->num_params * sizeof(Tensor*));
        adam->m[adam->num_params - 1] = tensor_create(param->shape, param->ndim);
        adam->v[adam->num_params - 1] = tensor_create(param->shape, param->ndim);
        tensor_fill(adam->m[adam->num_params - 1], 0.0f);
        tensor_fill(adam->v[adam->num_params - 1], 0.0f);
    }
}

void optimizer_step(Optimizer* opt, Tensor* grad, Tensor* param, int param_idx) {
    if (opt->type == OPT_SGD) {
        for (int i = 0; i < param->size; i++) {
            param->data[i] -= opt->learning_rate * grad->data[i];
        }
    } else if (opt->type == OPT_ADAM) {
        AdamParams* adam = (AdamParams*)opt->params;
        adam->t++;

        float lr_t = adam->learning_rate * sqrtf(1 - powf(adam->beta2, adam->t)) / (1 - powf(adam->beta1, adam->t));

        Tensor* m = adam->m[param_idx];
        Tensor* v = adam->v[param_idx];

        for (int i = 0; i < param->size; i++) {
            m->data[i] = adam->beta1 * m->data[i] + (1 - adam->beta1) * grad->data[i];
            v->data[i] = adam->beta2 * v->data[i] + (1 - adam->beta2) * grad->data[i] * grad->data[i];
            param->data[i] -= lr_t * m->data[i] / (sqrtf(v->data[i]) + adam->epsilon);
        }
    }
}
