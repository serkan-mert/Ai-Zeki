#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pooling.h"
#include "layer.h"

void pooling_forward(Layer* self, Tensor* input) {
    PoolingParams* p = (PoolingParams*)self->params;
    int batch = input->shape[0];
    int channels = input->shape[1];
    int in_h = input->ndim >= 3 ? input->shape[2] : 0;
    int in_w = input->ndim >= 4 ? input->shape[3] : 0;
    p->input_c = channels;
    p->input_h = in_h;
    p->input_w = in_w;
    int k = p->kernel_size;
    int s = p->stride;
    int out_h = (in_h - k) / s + 1;
    int out_w = (in_w - k) / s + 1;

    if (self->output) tensor_free(self->output);
    int out_shape[] = {batch, channels, out_h, out_w};
    self->output = tensor_create(out_shape, 4);

    if (p->type == POOL_MAX && p->indices_cache) {
        tensor_free(p->indices_cache);
    }
    if (p->type == POOL_MAX) {
        p->indices_cache = tensor_create(out_shape, 4);
    }

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int h_start = oh * s;
                    int w_start = ow * s;
                    float best = input->data[b * channels * in_h * in_w + c * in_h * in_w + h_start * in_w + w_start];
                    int best_idx = 0;
                    if (p->type == POOL_MAX) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = h_start + kh;
                                int iw = w_start + kw;
                                float val = input->data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                                if (val > best) {
                                    best = val;
                                    best_idx = kh * k + kw;
                                }
                            }
                        }
                        self->output->data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = best;
                        p->indices_cache->data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = (float)best_idx;
                    } else {
                        float sum = 0;
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = h_start + kh;
                                int iw = w_start + kw;
                                sum += input->data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                            }
                        }
                        self->output->data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = sum / (k * k);
                    }
                }
            }
        }
    }
}

void pooling_backward(Layer* self, Tensor* grad_output) {
    PoolingParams* p = (PoolingParams*)self->params;
    int batch = grad_output->shape[0];
    int channels = p->input_c;
    int in_h = p->input_h;
    int in_w = p->input_w;
    int k = p->kernel_size;
    int s = p->stride;
    int out_h = grad_output->shape[2];
    int out_w = grad_output->shape[3];

    if (self->grad_input) tensor_free(self->grad_input);
    int grad_shape[] = {batch, channels, in_h, in_w};
    self->grad_input = tensor_create(grad_shape, 4);
    tensor_fill(self->grad_input, 0.0f);

    if (p->type == POOL_MAX) {
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        int idx = (int)p->indices_cache->data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow];
                        int kh = idx / k;
                        int kw = idx % k;
                        int ih = oh * s + kh;
                        int iw = ow * s + kw;
                        self->grad_input->data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] +=
                            grad_output->data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow];
                    }
                }
            }
        }
    } else {
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float grad = grad_output->data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] / (k * k);
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh * s + kh;
                                int iw = ow * s + kw;
                                self->grad_input->data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] += grad;
                            }
                        }
                    }
                }
            }
        }
    }
}

void pooling_free(Layer* self) {
    if (self->params) {
        PoolingParams* p = (PoolingParams*)self->params;
        tensor_free(p->indices_cache);
        free(p);
    }
}

Layer* maxpool2d_create(int kernel_size, int stride, const char* name) {
    Layer* l = layer_create(LAYER_ACTIVATION, name);
    l->forward = pooling_forward;
    l->backward = pooling_backward;
    l->free = pooling_free;

    PoolingParams* p = (PoolingParams*)malloc(sizeof(PoolingParams));
    p->type = POOL_MAX;
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->input_h = 0;
    p->input_w = 0;
    p->input_c = 0;
    p->indices_cache = NULL;
    l->params = p;

    return l;
}
