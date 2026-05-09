#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "optimizer.h"
#include "loss.h"

typedef struct {
    Layer* input_layer;
    Layer* output_layer;
    int num_layers;
    Optimizer* optimizer;
    LossType loss_type;
    float (*loss_fn)(Tensor* pred, Tensor* target);
    Tensor* (*loss_grad)(Tensor* pred, Tensor* target);
} Model;

Model* model_create(Layer* input_layer);
void model_free(Model* m);
void model_add_layer(Model* m, Layer* layer);
void model_remove_layer(Model* m, int index);
void model_set_optimizer(Model* m, Optimizer* opt);
void model_set_loss(Model* m, LossType type);
void model_set_training(Model* m, int training);
void model_train_step(Model* m, Tensor* input, Tensor* target);
void model_predict(Model* m, Tensor* input);
float model_evaluate(Model* m, Tensor* input, Tensor* target);
float model_accuracy(Model* m, Tensor* input, Tensor* target);
void model_validate(Model* m, Tensor* val_input, Tensor* val_target);
void model_test(Model* m, Tensor* test_input, Tensor* test_target);
float mse_loss(Tensor* pred, Tensor* target);
Tensor* mse_loss_grad(Tensor* pred, Tensor* target);

#endif
