#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_loader.h"

Tensor* load_csv(const char* filename, int rows, int cols) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;

    Tensor* t = tensor_create((int[]){rows, cols}, 2);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(fp, "%f", &t->data[i * cols + j]);
            if (j < cols - 1) fgetc(fp);
        }
    }

    fclose(fp);
    return t;
}

Tensor* load_labels(const char* filename, int num_classes) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return NULL;

    int count = 0;
    while (fscanf(fp, "%*f") != EOF) count++;
    rewind(fp);

    Tensor* labels = tensor_create((int[]){count, num_classes}, 2);
    tensor_fill(labels, 0.0f);

    for (int i = 0; i < count; i++) {
        int label;
        fscanf(fp, "%d", &label);
        if (label < num_classes) labels->data[i * num_classes + label] = 1.0f;
    }

    fclose(fp);
    return labels;
}
