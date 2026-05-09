#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "loss.h"

float mse_loss(Tensor* pred, Tensor* target) {
    float sum = 0.0f;
    for (int i = 0; i < pred->size; i++) {
        float diff = pred->data[i] - target->data[i];
        sum += diff * diff;
    }
    return sum / pred->size;
}

Tensor* mse_loss_grad(Tensor* pred, Tensor* target) {
    Tensor* grad = tensor_copy(pred);
    for (int i = 0; i < pred->size; i++) {
        grad->data[i] = 2.0f * (pred->data[i] - target->data[i]) / pred->size;
    }
    return grad;
}

float binary_crossentropy(Tensor* pred, Tensor* target) {
    float sum = 0.0f;
    for (int i = 0; i < pred->size; i++) {
        float p = fmaxf(fminf(pred->data[i], 0.9999f), 0.0001f);
        sum -= target->data[i] * logf(p) + (1 - target->data[i]) * logf(1 - p);
    }
    return sum / pred->size;
}

Tensor* binary_crossentropy_grad(Tensor* pred, Tensor* target) {
    Tensor* grad = tensor_copy(pred);
    for (int i = 0; i < pred->size; i++) {
        float p = fmaxf(fminf(pred->data[i], 0.9999f), 0.0001f);
        grad->data[i] = (p - target->data[i]) / (p * (1 - p) * pred->size);
    }
    return grad;
}

float categorical_crossentropy(Tensor* pred, Tensor* target) {
    float sum = 0.0f;
    int batch = pred->shape[0];
    int classes = pred->shape[1];

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < classes; c++) {
            float p = fmaxf(fminf(pred->data[b * classes + c], 1.0f - 1e-10f), 1e-10f);
            sum -= target->data[b * classes + c] * logf(p);
        }
    }
    return sum / batch;
}

Tensor* categorical_crossentropy_grad(Tensor* pred, Tensor* target) {
    int batch = pred->shape[0];
    int classes = pred->shape[1];
    Tensor* grad = tensor_create(pred->shape, pred->ndim);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < classes; c++) {
            float p = fmaxf(fminf(pred->data[b * classes + c], 1.0f - 1e-10f), 1e-10f);
            grad->data[b * classes + c] = -target->data[b * classes + c] / p / batch;
        }
    }
    return grad;
}

float huber_loss(Tensor* pred, Tensor* target, float delta) {
    float sum = 0.0f;
    for (int i = 0; i < pred->size; i++) {
        float diff = pred->data[i] - target->data[i];
        if (fabsf(diff) <= delta) {
            sum += 0.5f * diff * diff;
        } else {
            sum += delta * (fabsf(diff) - 0.5f * delta);
        }
    }
    return sum / pred->size;
}

Tensor* huber_loss_grad(Tensor* pred, Tensor* target, float delta) {
    Tensor* grad = tensor_copy(pred);
    for (int i = 0; i < pred->size; i++) {
        float diff = pred->data[i] - target->data[i];
        if (fabsf(diff) <= delta) {
            grad->data[i] = diff / pred->size;
        } else {
            grad->data[i] = (diff > 0 ? delta : -delta) / pred->size;
        }
    }
    return grad;
}

float loss_compute(Tensor* pred, Tensor* target, LossType type) {
    switch (type) {
        case LOSS_MSE: return mse_loss(pred, target);
        case LOSS_CROSSENTROPY: return categorical_crossentropy(pred, target);
        case LOSS_BINARY_CROSSENTROPY: return binary_crossentropy(pred, target);
        case LOSS_HUBER: return huber_loss(pred, target, 1.0f);
        default: return mse_loss(pred, target);
    }
}

Tensor* loss_gradient(Tensor* pred, Tensor* target, LossType type) {
    switch (type) {
        case LOSS_MSE: return mse_loss_grad(pred, target);
        case LOSS_CROSSENTROPY: return categorical_crossentropy_grad(pred, target);
        case LOSS_BINARY_CROSSENTROPY: return binary_crossentropy_grad(pred, target);
        case LOSS_HUBER: return huber_loss_grad(pred, target, 1.0f);
        default: return mse_loss_grad(pred, target);
    }
}
