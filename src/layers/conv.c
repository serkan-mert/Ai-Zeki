#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "conv.h"
#include "layer.h"

void conv_forward(Layer* self, Tensor* input) {
    ConvParams* p = (ConvParams*)self->params;
    int batch = input->shape[0];
    int in_c = p->input_channels;
    int out_c = p->output_channels;
    int k = p->kernel_size;
    int s = p->stride;
    int pad = p->padding;

    int in_h = p->input_h;
    int in_w = p->input_w;
    int out_h = (in_h + 2 * pad - k) / s + 1;
    int out_w = (in_w + 2 * pad - k) / s + 1;

    if (self->output) tensor_free(self->output);
    int out_shape[] = {batch, out_c, out_h, out_w};
    self->output = tensor_create(out_shape, 4);

    p->input_cache = tensor_copy(input);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = p->biases->data[oc];
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh * s + kh - pad;
                                int iw = ow * s + kw - pad;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    sum += input->data[b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw] *
                                           p->kernels->data[oc * in_c * k * k + ic * k * k + kh * k + kw];
                                }
                            }
                        }
                    }
                    self->output->data[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }
}

void conv_backward(Layer* self, Tensor* grad_output) {
    ConvParams* p = (ConvParams*)self->params;
    int batch = grad_output->shape[0];
    int in_c = p->input_channels;
    int out_c = p->output_channels;
    int k = p->kernel_size;
    int s = p->stride;
    int pad = p->padding;
    int in_h = p->input_h;
    int in_w = p->input_w;
    int out_h = grad_output->shape[2];
    int out_w = grad_output->shape[3];

    if (self->grad_input) tensor_free(self->grad_input);
    int grad_shape[] = {batch, in_c, in_h, in_w};
    self->grad_input = tensor_create(grad_shape, 4);
    tensor_fill(self->grad_input, 0.0f);

    if (!p->kernel_grad) p->kernel_grad = tensor_create(p->kernels->shape, p->kernels->ndim);
    if (!p->bias_grad) p->bias_grad = tensor_create(p->biases->shape, p->biases->ndim);
    tensor_fill(p->kernel_grad, 0.0f);
    tensor_fill(p->bias_grad, 0.0f);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float grad = grad_output->data[b * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow];
                    p->bias_grad->data[oc] += grad;
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh * s + kh - pad;
                                int iw = ow * s + kw - pad;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    p->kernel_grad->data[oc * in_c * k * k + ic * k * k + kh * k + kw] +=
                                        grad * p->input_cache->data[b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw];
                                    self->grad_input->data[b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw] +=
                                        grad * p->kernels->data[oc * in_c * k * k + ic * k * k + kh * k + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv_free(Layer* self) {
    if (self->params) {
        ConvParams* p = (ConvParams*)self->params;
        tensor_free(p->kernels);
        tensor_free(p->biases);
        tensor_free(p->kernel_grad);
        tensor_free(p->bias_grad);
        tensor_free(p->input_cache);
        free(p);
    }
}

Layer* conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, const char* name) {
    Layer* l = layer_create(LAYER_CONV2D, name);
    l->forward = conv_forward;
    l->backward = conv_backward;
    l->free = conv_free;

    ConvParams* p = (ConvParams*)calloc(1, sizeof(ConvParams));
    p->input_channels = in_channels;
    p->output_channels = out_channels;
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    p->input_h = 0;
    p->input_w = 0;
    p->input_cache = NULL;
    p->kernel_grad = NULL;
    p->bias_grad = NULL;

    int k_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    p->kernels = tensor_create(k_shape, 4);
    int b_shape[] = {out_channels};
    p->biases = tensor_create(b_shape, 1);

    float scale = sqrtf(2.0f / (in_channels * kernel_size * kernel_size));
    for (int i = 0; i < p->kernels->size; i++) {
        p->kernels->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
    tensor_fill(p->biases, 0.0f);

    l->params = p;
    l->weights = p->kernels;
    l->biases = p->biases;

    return l;
}
