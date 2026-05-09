#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"
#include "data_augment.h"

Tensor* augment_rotate(Tensor* img, float angle_deg) {
    float angle = angle_deg * M_PI / 180.0f;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int h = img->shape[1];
    int w = img->shape[2];
    int c = img->shape[0];

    int new_shape[] = {c, h, w};
    Tensor* rotated = tensor_create(new_shape, 3);

    float cx = w / 2.0f;
    float cy = h / 2.0f;

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float nx = cos_a * (x - cx) - sin_a * (y - cy) + cx;
                float ny = sin_a * (x - cx) + cos_a * (y - cy) + cy;

                int x0 = (int)floorf(nx);
                int y0 = (int)floorf(ny);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float dx = nx - x0;
                float dy = ny - y0;

                float val = 0.0f;
                if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
                    val += (1 - dx) * (1 - dy) * img->data[ch * h * w + y0 * w + x0];
                if (x1 >= 0 && x1 < w && y0 >= 0 && y0 < h)
                    val += dx * (1 - dy) * img->data[ch * h * w + y0 * w + x1];
                if (x0 >= 0 && x0 < w && y1 >= 0 && y1 < h)
                    val += (1 - dx) * dy * img->data[ch * h * w + y1 * w + x0];
                if (x1 >= 0 && x1 < w && y1 >= 0 && y1 < h)
                    val += dx * dy * img->data[ch * h * w + y1 * w + x1];

                rotated->data[ch * h * w + y * w + x] = val;
            }
        }
    }

    return rotated;
}

Tensor* augment_flip(Tensor* img, int horizontal) {
    int c = img->shape[0];
    int h = img->shape[1];
    int w = img->shape[2];

    int new_shape[] = {c, h, w};
    Tensor* flipped = tensor_create(new_shape, 3);

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (horizontal) {
                    flipped->data[ch * h * w + y * w + (w - 1 - x)] = img->data[ch * h * w + y * w + x];
                } else {
                    flipped->data[ch * h * w + (h - 1 - y) * w + x] = img->data[ch * h * w + y * w + x];
                }
            }
        }
    }

    return flipped;
}

Tensor* augment_brightness(Tensor* img, float factor) {
    Tensor* result = tensor_copy(img);
    for (int i = 0; i < result->size; i++) {
        result->data[i] *= factor;
        if (result->data[i] > 1.0f) result->data[i] = 1.0f;
        if (result->data[i] < 0.0f) result->data[i] = 0.0f;
    }
    return result;
}

Tensor* augment_crop(Tensor* img, float scale) {
    int c = img->shape[0];
    int h = img->shape[1];
    int w = img->shape[2];

    int new_h = (int)(h * scale);
    int new_w = (int)(w * scale);
    int start_y = (h - new_h) / 2;
    int start_x = (w - new_w) / 2;

    int new_shape[] = {c, h, w};
    Tensor* cropped = tensor_create(new_shape, 3);
    tensor_fill(cropped, 0.0f);

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                int src_y = y + start_y;
                int src_x = x + start_x;
                if (src_y < h && src_x < w) {
                    cropped->data[ch * h * w + y * w + x] = img->data[ch * h * w + src_y * w + src_x];
                }
            }
        }
    }

    return cropped;
}

Tensor* augment_noise(Tensor* img, float noise_level) {
    Tensor* result = tensor_copy(img);
    for (int i = 0; i < result->size; i++) {
        float noise = (float)rand() / RAND_MAX * noise_level - noise_level / 2;
        result->data[i] += noise;
        if (result->data[i] > 1.0f) result->data[i] = 1.0f;
        if (result->data[i] < 0.0f) result->data[i] = 0.0f;
    }
    return result;
}
