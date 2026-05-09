#ifndef SAVE_LOAD_H
#define SAVE_LOAD_H

#include "layer.h"
#include "model.h"

typedef enum {
    FORMAT_BINARY,
    FORMAT_TEXT
} SaveFormat;

#define MAX_TENSOR_SIZE 100000000

int model_save(Model* m, const char* filename, SaveFormat fmt);
int model_save_weights(Model* m, const char* filename);
Model* model_load(const char* filename);
int layer_save(Layer* layer, FILE* fp, SaveFormat fmt);
Layer* layer_load(FILE* fp, SaveFormat fmt);

#endif
