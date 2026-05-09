#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "tensor.h"

Tensor* tensor_create(int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->ndim = ndim;
    t->shape = (int*)malloc(ndim * sizeof(int));
    if (!t->shape) { free(t); return NULL; }
    memcpy(t->shape, shape, ndim * sizeof(int));
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0 || total > (size_t)INT_MAX / (size_t)shape[i]) {
            free(t->shape);
            free(t);
            return NULL;
        }
        total *= (size_t)shape[i];
    }
    t->size = (int)total;
    t->data = (float*)calloc(t->size, sizeof(float));
    if (!t->data) { free(t->shape); free(t); return NULL; }
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        free(t->data);
        free(t->shape);
        free(t);
    }
}

Tensor* tensor_copy(Tensor* src) {
    Tensor* dst = tensor_create(src->shape, src->ndim);
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}

void tensor_fill(Tensor* t, float value) {
    for (int i = 0; i < t->size; i++) t->data[i] = value;
}

void tensor_print(Tensor* t, const char* name) {
    printf("%s\n", name);
    printf("  Shape: [");
    for (int i = 0; i < t->ndim; i++) {
        if (i > 0) printf(", ");
        printf("%d", t->shape[i]);
    }
    printf("]\n  Data: [");
    int limit = t->size < 12 ? t->size : 12;
    for (int i = 0; i < limit; i++) {
        if (i > 0) printf(", ");
        printf("%.4f", t->data[i]);
    }
    if (t->size > 12) printf(", ...");
    printf("]\n");
}
