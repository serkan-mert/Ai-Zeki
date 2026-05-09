#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "tensor.h"

Tensor* load_csv(const char* filename, int rows, int cols);
Tensor* load_images(const char* dirname, int img_h, int img_w, int channels);
Tensor* load_labels(const char* filename, int num_classes);

#endif
