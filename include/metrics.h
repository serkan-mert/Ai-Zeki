#ifndef METRICS_H
#define METRICS_H

#include "tensor.h"

float accuracy(Tensor* pred, Tensor* target);
float precision(Tensor* pred, Tensor* target, int class_idx);
float recall(Tensor* pred, Tensor* target, int class_idx);
float f1_score(Tensor* pred, Tensor* target, int class_idx);
float auc_roc(Tensor* pred_probs, Tensor* target, int class_idx);
void confusion_matrix(Tensor* pred, Tensor* target, int num_classes);

#endif
