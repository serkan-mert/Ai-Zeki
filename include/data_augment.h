#ifndef DATA_AUGMENT_H
#define DATA_AUGMENT_H

#include "tensor.h"

Tensor* augment_rotate(Tensor* img, float angle_deg);
Tensor* augment_flip(Tensor* img, int horizontal);
Tensor* augment_brightness(Tensor* img, float factor);
Tensor* augment_crop(Tensor* img, float scale);
Tensor* augment_noise(Tensor* img, float noise_level);

#endif
