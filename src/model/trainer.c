#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "trainer.h"
#include "save_load.h"

Trainer* trainer_create(Model* model, Optimizer* opt, int batch_size, int epochs) {
    Trainer* trainer = (Trainer*)malloc(sizeof(Trainer));
    trainer->model = model;
    trainer->optimizer = opt;
    
    TrainerParams* params = (TrainerParams*)malloc(sizeof(TrainerParams));
    params->batch_size = batch_size;
    params->epochs = epochs;
    params->steps_per_epoch = 0;
    params->best_loss = 1e10f;
    params->patience = 0;
    params->patience_counter = 0;
    params->early_stopped = 0;
    params->lr_decay = 1.0f;
    params->lr_step = 0;
    params->current_step = 0;
    params->checkpoint_path[0] = '\0';
    params->checkpoint_saved = 0;
    
    trainer->params = params;
    return trainer;
}

void trainer_free(Trainer* trainer) {
    if (trainer) {
        if (trainer->params) free(trainer->params);
        free(trainer);
    }
}

void trainer_set_early_stopping(Trainer* trainer, int patience) {
    trainer->params->patience = patience;
    trainer->params->patience_counter = 0;
}

void trainer_set_lr_scheduler(Trainer* trainer, float decay, int step) {
    trainer->params->lr_decay = decay;
    trainer->params->lr_step = step;
    trainer->params->current_step = 0;
}

void trainer_set_checkpoint(Trainer* trainer, const char* path) {
    strncpy(trainer->params->checkpoint_path, path, 255);
    trainer->params->checkpoint_path[255] = '\0';
    trainer->params->checkpoint_saved = 0;
}

void trainer_fit(Trainer* trainer, Tensor* x_train, Tensor* y_train, Tensor* x_val, Tensor* y_val) {
    TrainerParams* p = trainer->params;
    Model* model = trainer->model;
    
    printf("\n--- Training ---\n");
    fflush(stdout);
    
    int num_samples = x_train->shape[0];
    int elems_per_sample = x_train->size / num_samples;
    int y_elems_per_sample = y_train->size / y_train->shape[0];
    p->steps_per_epoch = (num_samples + p->batch_size -1) / p->batch_size;
    
    int feat_dim_display = (x_train->ndim >= 2) ? x_train->shape[1] : 0;
    printf("  %d samples, %d features, batch %d, %d epochs\n",
           num_samples, feat_dim_display, p->batch_size, p->epochs);
    fflush(stdout);
    
    for (int epoch = 0; epoch < p->epochs && !p->early_stopped; epoch++) {
        printf("  Epoch %d/%d [", epoch + 1, p->epochs);
        fflush(stdout);
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (int step = 0; step < p->steps_per_epoch; step++) {
            int start = step * p->batch_size;
            int end = (start + p->batch_size < num_samples) ? start + p->batch_size : num_samples;
            int current_batch = end - start; 
            
            int* batch_shape = (int*)malloc(x_train->ndim * sizeof(int));
            batch_shape[0] = current_batch;
            for (int i = 1; i < x_train->ndim; i++) batch_shape[i] = x_train->shape[i];
            Tensor* batch_x = tensor_create(batch_shape, x_train->ndim);
            free(batch_shape);
            
            int* target_shape = (int*)malloc(y_train->ndim * sizeof(int));
            target_shape[0] = current_batch;
            for (int i = 1; i < y_train->ndim; i++) target_shape[i] = y_train->shape[i];
            Tensor* batch_y = tensor_create(target_shape, y_train->ndim);
            free(target_shape);
            
            memcpy(batch_x->data, &x_train->data[start * elems_per_sample], 
                   current_batch * elems_per_sample * sizeof(float));
            memcpy(batch_y->data, &y_train->data[start * y_elems_per_sample], 
                   current_batch * y_elems_per_sample * sizeof(float));
            
            model_train_step(model, batch_x, batch_y);
            
            float batch_loss = loss_compute(model->output_layer->output, batch_y, model->loss_type);
            epoch_loss += batch_loss;
            num_batches++;
            
            tensor_free(batch_x);
            tensor_free(batch_y);
            
            p->current_step++;
            if (p->lr_step > 0 && p->current_step % p->lr_step == 0) {
                trainer->optimizer->learning_rate *= p->lr_decay;
            }
            
            if ((step + 1) % (p->steps_per_epoch / 4 + 1) == 0) {
                printf(".");
                fflush(stdout);
            }
        }
        
        printf("] loss: %.4f", epoch_loss / num_batches);
        
        if (x_val && y_val) {
            float val_loss = model_evaluate(model, x_val, y_val);
            float val_acc = model_accuracy(model, x_val, y_val);
            
            if (val_loss < p->best_loss) {
                p->best_loss = val_loss;
                p->patience_counter = 0;
                printf(" | val_loss: %.4f, acc: %.1f%% | best", val_loss, val_acc * 100);
            } else {
                p->patience_counter++;
                printf(" | val_loss: %.4f, acc: %.1f%% | patience: %d/%d",
                       val_loss, val_acc * 100, p->patience_counter, p->patience);
                if (p->patience > 0 && p->patience_counter >= p->patience) {
                    printf(" | EARLY STOP");
                    p->early_stopped = 1;
                }
            }
        }
        printf("\n");
        
        if (p->early_stopped) break;
    }
    
    if (p->early_stopped) {
        printf("  (early stopped)\n");
    }
}

void trainer_evaluate(Trainer* trainer, Tensor* x_test, Tensor* y_test) {
    Model* model = trainer->model;
    float test_loss = model_evaluate(model, x_test, y_test);
    float test_acc = model_accuracy(model, x_test, y_test);
    printf("\n--- Test Results ---\n");
    printf("  Loss: %.6f, Accuracy: %.2f%%\n", test_loss, test_acc * 100);
}
