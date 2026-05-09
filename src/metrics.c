#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "metrics.h"

float accuracy(Tensor* pred, Tensor* target) {
    int correct = 0;
    int total = pred->shape[0];

    for (int i = 0; i < total; i++) {
        int pred_class = 0, target_class = 0;
        float max_pred = pred->data[i * pred->shape[1]];
        float max_target = target->data[i * target->shape[1]];

        for (int j = 0; j < pred->shape[1]; j++) {
            if (pred->data[i * pred->shape[1] + j] > max_pred) {
                max_pred = pred->data[i * pred->shape[1] + j];
                pred_class = j;
            }
            if (target->data[i * target->shape[1] + j] > max_target) {
                max_target = target->data[i * target->shape[1] + j];
                target_class = j;
            }
        }

        if (pred_class == target_class) correct++;
    }

    return (float)correct / total;
}

float precision(Tensor* pred, Tensor* target, int class_idx) {
    int tp = 0, fp = 0;

    for (int i = 0; i < pred->shape[0]; i++) {
        int pred_class = 0;
        float max_pred = pred->data[i * pred->shape[1]];
        for (int j = 0; j < pred->shape[1]; j++) {
            if (pred->data[i * pred->shape[1] + j] > max_pred) {
                max_pred = pred->data[i * pred->shape[1] + j];
                pred_class = j;
            }
        }

        int is_target = 0;
        if (target->shape[1] > 1) {
            is_target = (target->data[i * target->shape[1] + class_idx] > 0.5f) ? 1 : 0;
        } else {
            is_target = ((int)target->data[i] == class_idx) ? 1 : 0;
        }

        if (pred_class == class_idx && is_target) tp++;
        else if (pred_class == class_idx && !is_target) fp++;
    }

    return (tp + fp == 0) ? 0 : (float)tp / (tp + fp);
}

float recall(Tensor* pred, Tensor* target, int class_idx) {
    int tp = 0, fn = 0;

    for (int i = 0; i < pred->shape[0]; i++) {
        int pred_class = 0;
        float max_pred = pred->data[i * pred->shape[1]];
        for (int j = 0; j < pred->shape[1]; j++) {
            if (pred->data[i * pred->shape[1] + j] > max_pred) {
                max_pred = pred->data[i * pred->shape[1] + j];
                pred_class = j;
            }
        }

        int is_target = 0;
        if (target->shape[1] > 1) {
            is_target = (target->data[i * target->shape[1] + class_idx] > 0.5f) ? 1 : 0;
        } else {
            is_target = ((int)target->data[i] == class_idx) ? 1 : 0;
        }

        if (pred_class == class_idx && is_target) tp++;
        else if (pred_class != class_idx && is_target) fn++;
    }

    return (tp + fn == 0) ? 0 : (float)tp / (tp + fn);
}

float f1_score(Tensor* pred, Tensor* target, int class_idx) {
    float prec = precision(pred, target, class_idx);
    float rec = recall(pred, target, class_idx);
    return (prec + rec == 0) ? 0 : 2 * prec * rec / (prec + rec);
}

void confusion_matrix(Tensor* pred, Tensor* target, int num_classes) {
    int* cm = (int*)calloc(num_classes * num_classes, sizeof(int));

    for (int i = 0; i < pred->shape[0]; i++) {
        int pred_class = 0;
        float max_pred = pred->data[i * pred->shape[1]];
        for (int j = 0; j < pred->shape[1]; j++) {
            if (pred->data[i * pred->shape[1] + j] > max_pred) {
                max_pred = pred->data[i * pred->shape[1] + j];
                pred_class = j;
            }
        }

        int target_class = 0;
        if (target->shape[1] > 1) {
            for (int j = 0; j < target->shape[1]; j++) {
                if (target->data[i * target->shape[1] + j] > 0.5f) {
                    target_class = j;
                    break;
                }
            }
        } else {
            target_class = (int)target->data[i];
        }

        cm[target_class * num_classes + pred_class]++;
    }

    printf("Confusion Matrix:\n");
    printf("    ");
    for (int j = 0; j < num_classes; j++) printf(" P%d  ", j);
    printf("\n");
    for (int i = 0; i < num_classes; i++) {
        printf("T%d: ", i);
        for (int j = 0; j < num_classes; j++) {
            printf("%4d ", cm[i * num_classes + j]);
        }
        printf("\n");
    }

    free(cm);
}
