#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "dense.h"
#include "dropout.h"
#include "batchnorm.h"
#include "conv.h"
#include "lstm.h"
#include "gru.h"
#include "attention.h"
#include "transformer.h"
#include "loss.h"
#include "metrics.h"

Model* model_create(Layer* input_layer) {
    Model* m = (Model*)malloc(sizeof(Model));
    m->input_layer = input_layer;
    m->output_layer = input_layer;
    m->num_layers = 1;
    m->optimizer = NULL;
    m->loss_type = LOSS_MSE;
    m->loss_fn = NULL;
    m->loss_grad = NULL;
    return m;
}

void model_free(Model* m) {
    if (!m) return;
    Layer* curr = m->input_layer;
    while (curr) {
        Layer* next = curr->next;
        layer_free(curr);
        curr = next;
    }
    if (m->optimizer) optimizer_free(m->optimizer);
    free(m);
}

void model_add_layer(Model* m, Layer* layer) {
    layer_connect(m->output_layer, layer);
    m->output_layer = layer;
    m->num_layers++;
}

void model_remove_layer(Model* m, int index) {
    if (!m || !m->input_layer || index < 0 || index >= m->num_layers) return;
    if (index == 0) return;

    Layer* curr = m->input_layer;
    for (int i = 0; i < index; i++) {
        if (!curr) return;
        curr = curr->next;
    }
    if (!curr) return;

    Layer* prev = curr->prev;
    Layer* next = curr->next;

    if (prev) prev->next = next;
    if (next) next->prev = prev;

    if (curr == m->output_layer) {
        m->output_layer = prev;
    }

    layer_free(curr);
    m->num_layers--;

    if (m->optimizer) {
        optimizer_free(m->optimizer);
        m->optimizer = NULL;
    }
}

void model_set_optimizer(Model* m, Optimizer* opt) {
    if (m->optimizer) optimizer_free(m->optimizer);
    m->optimizer = opt;

    int param_idx = 0;
    Layer* curr = m->input_layer;
    while (curr) {
        if (curr->type == LAYER_DENSE && curr->params) {
            optimizer_init_param(opt, curr->weights); param_idx++;
            optimizer_init_param(opt, curr->biases); param_idx++;
        } else if (curr->type == LAYER_CONV2D && curr->params) {
            ConvParams* p = (ConvParams*)curr->params;
            optimizer_init_param(opt, p->kernels); param_idx++;
            optimizer_init_param(opt, p->biases); param_idx++;
        } else if (curr->type == LAYER_LSTM && curr->params) {
            LSTMCellParams* p = (LSTMCellParams*)curr->params;
            optimizer_init_param(opt, p->Wf); param_idx++;
            optimizer_init_param(opt, p->Wi); param_idx++;
            optimizer_init_param(opt, p->Wc); param_idx++;
            optimizer_init_param(opt, p->Wo); param_idx++;
            optimizer_init_param(opt, p->bf); param_idx++;
            optimizer_init_param(opt, p->bi); param_idx++;
            optimizer_init_param(opt, p->bc); param_idx++;
            optimizer_init_param(opt, p->bo); param_idx++;
        } else if (curr->type == LAYER_GRU && curr->params) {
            GRUCellParams* p = (GRUCellParams*)curr->params;
            optimizer_init_param(opt, p->Wz); param_idx++;
            optimizer_init_param(opt, p->Wr); param_idx++;
            optimizer_init_param(opt, p->Wh); param_idx++;
            optimizer_init_param(opt, p->Uz); param_idx++;
            optimizer_init_param(opt, p->Ur); param_idx++;
            optimizer_init_param(opt, p->Uh); param_idx++;
            optimizer_init_param(opt, p->bz); param_idx++;
            optimizer_init_param(opt, p->br); param_idx++;
            optimizer_init_param(opt, p->bh); param_idx++;
        } else if (curr->type == LAYER_ATTENTION && curr->params) {
            AttentionParams* p = (AttentionParams*)curr->params;
            optimizer_init_param(opt, p->Wq); param_idx++;
            optimizer_init_param(opt, p->Wk); param_idx++;
            optimizer_init_param(opt, p->Wv); param_idx++;
            optimizer_init_param(opt, p->Wo); param_idx++;
        } else if (curr->type == LAYER_TRANSFORMER && curr->params) {
            TransformerBlockParams* p = (TransformerBlockParams*)curr->params;
            if (p->attn_layer && p->attn_layer->type == LAYER_ATTENTION && p->attn_layer->params) {
                AttentionParams* ap = (AttentionParams*)p->attn_layer->params;
                optimizer_init_param(opt, ap->Wq); param_idx++;
                optimizer_init_param(opt, ap->Wk); param_idx++;
                optimizer_init_param(opt, ap->Wv); param_idx++;
                optimizer_init_param(opt, ap->Wo); param_idx++;
            }
            if (p->ff_dense1 && p->ff_dense1->params) {
                optimizer_init_param(opt, p->ff_dense1->weights); param_idx++;
                optimizer_init_param(opt, p->ff_dense1->biases); param_idx++;
            }
            if (p->ff_dense2 && p->ff_dense2->params) {
                optimizer_init_param(opt, p->ff_dense2->weights); param_idx++;
                optimizer_init_param(opt, p->ff_dense2->biases); param_idx++;
            }
        }
        /* LAYER_FLATTEN, LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_INPUT, LAYER_BATCHNORM: no params */
        curr = curr->next;
    }
}

void model_set_training(Model* m, int training) {
    Layer* curr = m->input_layer;
    while (curr) {
        if (curr->type == LAYER_DROPOUT && curr->params) {
            ((DropoutParams*)curr->params)->training = training;
        } else if (curr->type == LAYER_BATCHNORM && curr->params) {
            ((BatchNormParams*)curr->params)->training = training;
        }
        curr = curr->next;
    }
}

void model_set_loss(Model* m, LossType type) {
    m->loss_type = type;
}

void model_train_step(Model* m, Tensor* input, Tensor* target) {
    model_set_training(m, 1);
    layer_forward_pass(m->input_layer, input);
    Tensor* pred = m->output_layer->output;

    Tensor* grad = loss_gradient(pred, target, m->loss_type);
    layer_backward_pass(m->output_layer, grad);
    tensor_free(grad);

    if (m->optimizer) {
        int param_idx = 0;
        Layer* curr = m->input_layer;
        while (curr) {
            if (curr->type == LAYER_DENSE && curr->params) {
                DenseParams* p = (DenseParams*)curr->params;
                if (p->weight_grad) {
                    optimizer_step(m->optimizer, p->weight_grad, curr->weights, param_idx++);
                }
                if (p->bias_grad) {
                    optimizer_step(m->optimizer, p->bias_grad, curr->biases, param_idx++);
                }
            } else if (curr->type == LAYER_CONV2D && curr->params) {
                ConvParams* p = (ConvParams*)curr->params;
                if (p->kernel_grad) {
                    optimizer_step(m->optimizer, p->kernel_grad, p->kernels, param_idx++);
                }
                if (p->bias_grad) {
                    optimizer_step(m->optimizer, p->bias_grad, p->biases, param_idx++);
                }
            } else if (curr->type == LAYER_LSTM && curr->params) {
                LSTMCellParams* p = (LSTMCellParams*)curr->params;
                if (p->Wf_grad) optimizer_step(m->optimizer, p->Wf_grad, p->Wf, param_idx++);
                if (p->Wi_grad) optimizer_step(m->optimizer, p->Wi_grad, p->Wi, param_idx++);
                if (p->Wc_grad) optimizer_step(m->optimizer, p->Wc_grad, p->Wc, param_idx++);
                if (p->Wo_grad) optimizer_step(m->optimizer, p->Wo_grad, p->Wo, param_idx++);
                if (p->bf_grad) optimizer_step(m->optimizer, p->bf_grad, p->bf, param_idx++);
                if (p->bi_grad) optimizer_step(m->optimizer, p->bi_grad, p->bi, param_idx++);
                if (p->bc_grad) optimizer_step(m->optimizer, p->bc_grad, p->bc, param_idx++);
                if (p->bo_grad) optimizer_step(m->optimizer, p->bo_grad, p->bo, param_idx++);
            } else if (curr->type == LAYER_GRU && curr->params) {
                GRUCellParams* p = (GRUCellParams*)curr->params;
                if (p->Wz_grad) optimizer_step(m->optimizer, p->Wz_grad, p->Wz, param_idx++);
                if (p->Wr_grad) optimizer_step(m->optimizer, p->Wr_grad, p->Wr, param_idx++);
                if (p->Wh_grad) optimizer_step(m->optimizer, p->Wh_grad, p->Wh, param_idx++);
                if (p->Uz_grad) optimizer_step(m->optimizer, p->Uz_grad, p->Uz, param_idx++);
                if (p->Ur_grad) optimizer_step(m->optimizer, p->Ur_grad, p->Ur, param_idx++);
                if (p->Uh_grad) optimizer_step(m->optimizer, p->Uh_grad, p->Uh, param_idx++);
                if (p->bz_grad) optimizer_step(m->optimizer, p->bz_grad, p->bz, param_idx++);
                if (p->br_grad) optimizer_step(m->optimizer, p->br_grad, p->br, param_idx++);
                if (p->bh_grad) optimizer_step(m->optimizer, p->bh_grad, p->bh, param_idx++);
            } else if (curr->type == LAYER_ATTENTION && curr->params) {
                /* Attention backward is a pass-through; no gradient-based optimization for now */
            } else if (curr->type == LAYER_TRANSFORMER && curr->params) {
                /* Transformer backward is a pass-through; no gradient-based optimization for now */
            }
            /* LAYER_FLATTEN, LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_INPUT, LAYER_BATCHNORM: no params */
            curr = curr->next;
        }
    }
}

void model_predict(Model* m, Tensor* input) {
    model_set_training(m, 0);
    layer_forward_pass(m->input_layer, input);
}

float model_evaluate(Model* m, Tensor* input, Tensor* target) {
    model_predict(m, input);
    return loss_compute(m->output_layer->output, target, m->loss_type);
}

float model_accuracy(Model* m, Tensor* input, Tensor* target) {
    model_predict(m, input);
    return accuracy(m->output_layer->output, target);
}

void model_validate(Model* m, Tensor* val_input, Tensor* val_target) {
    float val_loss = model_evaluate(m, val_input, val_target);
    float val_acc = model_accuracy(m, val_input, val_target);
    printf("Validation - Loss: %.6f, Accuracy: %.2f%%\n", val_loss, val_acc * 100);
}

void model_test(Model* m, Tensor* test_input, Tensor* test_target) {
    float test_loss = model_evaluate(m, test_input, test_target);
    float test_acc = model_accuracy(m, test_input, test_target);
    printf("Test - Loss: %.6f, Accuracy: %.2f%%\n", test_loss, test_acc * 100);
}
