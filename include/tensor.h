#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* shape;
    int ndim;
    int size;
} Tensor;

Tensor* tensor_create(int* shape, int ndim);
void tensor_free(Tensor* t);
Tensor* tensor_copy(Tensor* src);
void tensor_fill(Tensor* t, float value);
void tensor_print(Tensor* t, const char* name);

#endif
