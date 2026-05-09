#ifndef GRU_H
#define GRU_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    int input_dim;
    int hidden_dim;
    Tensor* Wz;
    Tensor* Wr;
    Tensor* Wh;
    Tensor* Uz;
    Tensor* Ur;
    Tensor* Uh;
    Tensor* bz;
    Tensor* br;
    Tensor* bh;
    Tensor* Wz_grad;
    Tensor* Wr_grad;
    Tensor* Wh_grad;
    Tensor* Uz_grad;
    Tensor* Ur_grad;
    Tensor* Uh_grad;
    Tensor* bz_grad;
    Tensor* br_grad;
    Tensor* bh_grad;
    Tensor* h_prev;
    Tensor* input_cache;
    Tensor* z_cache;
    Tensor* r_cache;
    Tensor* h_hat_cache;
} GRUCellParams;

void gru_cell_forward(Layer* self, Tensor* input);
void gru_cell_backward(Layer* self, Tensor* grad_output);
void gru_free(Layer* self);

Layer* gru_cell_create(int input_dim, int hidden_dim, const char* name);

#endif
