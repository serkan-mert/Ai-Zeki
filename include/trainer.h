#ifndef TRAINER_H
#define TRAINER_H

#include "model.h"
#include "optimizer.h"

typedef struct {
    int batch_size;
    int epochs;
    int steps_per_epoch;
    float learning_rate;
    float best_loss;
    int patience;
    int patience_counter;
    int early_stopped;
    float lr_decay;
    int lr_step;
    int current_step;
    char checkpoint_path[256];
    int checkpoint_saved;
} TrainerParams;

typedef struct {
    Model* model;
    Optimizer* optimizer;
    TrainerParams* params;
} Trainer;

Trainer* trainer_create(Model* model, Optimizer* opt, int batch_size, int epochs);
void trainer_free(Trainer* trainer);
void trainer_set_early_stopping(Trainer* trainer, int patience);
void trainer_set_lr_scheduler(Trainer* trainer, float decay, int step);
void trainer_set_checkpoint(Trainer* trainer, const char* path);
void trainer_fit(Trainer* trainer, Tensor* x_train, Tensor* y_train, Tensor* x_val, Tensor* y_val);
void trainer_evaluate(Trainer* trainer, Tensor* x_test, Tensor* y_test);

#endif
