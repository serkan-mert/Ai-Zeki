#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef enum {
    LOSS_MSE,
    LOSS_CROSSENTROPY,
    LOSS_BINARY_CROSSENTROPY,
    LOSS_HUBER
} LossType;

float loss_compute(Tensor* pred, Tensor* target, LossType type);
Tensor* loss_gradient(Tensor* pred, Tensor* target, LossType type);

float binary_crossentropy(Tensor* pred, Tensor* target);
float categorical_crossentropy(Tensor* pred, Tensor* target);
Tensor* binary_crossentropy_grad(Tensor* pred, Tensor* target);
Tensor* categorical_crossentropy_grad(Tensor* pred, Tensor* target);

#endif
