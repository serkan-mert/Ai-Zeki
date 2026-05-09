#ifndef LSTM_H
#define LSTM_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int input_dim;
    int hidden_dim;
    float learning_rate;
    Tensor* Wf;
    Tensor* Wi;
    Tensor* Wc;
    Tensor* Wo;
    Tensor* bf;
    Tensor* bi;
    Tensor* bc;
    Tensor* bo;
    Tensor* Wf_grad;
    Tensor* Wi_grad;
    Tensor* Wc_grad;
    Tensor* Wo_grad;
    Tensor* bf_grad;
    Tensor* bi_grad;
    Tensor* bc_grad;
    Tensor* bo_grad;
    Tensor* h_prev;
    Tensor* c_prev;
    Tensor* h_next;
    Tensor* c_next;
    Tensor* input_cache;
    Tensor* f_cache;
    Tensor* i_cache;
    Tensor* c_hat_cache;
    Tensor* o_cache;
    int batch_size;
    int seq_len;
} LSTMCellParams;

void lstm_cell_forward(Layer* self, Tensor* input);
void lstm_cell_backward(Layer* self, Tensor* grad_output);
void lstm_free(Layer* self);

Layer* lstm_cell_create(int input_dim, int hidden_dim, const char* name);

#endif
