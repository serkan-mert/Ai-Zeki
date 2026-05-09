#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensor.h"
#include "layer.h"
#include "dense.h"
#include "activation.h"
#include "dropout.h"
#include "batchnorm.h"
#include "conv.h"
#include "pooling.h"
#include "lstm.h"
#include "gru.h"
#include "flatten.h"
#include "attention.h"
#include "transformer.h"
#include "optimizer.h"
#include "save_load.h"
#include "data_loader.h"
#include "data_augment.h"
#include "metrics.h"
#include "loss.h"
#include "model.h"
#include "trainer.h"
#include "mcp_server.h"

void test_adam_optimizer() {
    printf("--- Adam Optimizer ---\n");
    srand((unsigned int)time(NULL));

    Layer* input = layer_create(LAYER_INPUT, "Input");
    int in_shape[] = {1, 5};
    input->output = tensor_create(in_shape, 2);

    Layer* dense1 = dense_create(5, 10, "Dense-1");
    Layer* relu1 = activation_create(ACT_RELU, "ReLU-1");
    Layer* dense2 = dense_create(10, 3, "Output");
    Layer* softmax = activation_create(ACT_SOFTMAX, "Softmax");

    Model* model = model_create(input);
    model_add_layer(model, dense1);
    model_add_layer(model, relu1);
    model_add_layer(model, dense2);
    model_add_layer(model, softmax);

    Optimizer* adam = optimizer_create(OPT_ADAM, 0.001f);
    model_set_optimizer(model, adam);
    model_set_loss(model, LOSS_CROSSENTROPY);

    Tensor* x = tensor_create(in_shape, 2);
    for (int i = 0; i < 5; i++) x->data[i] = (float)rand() / RAND_MAX;

    int target_shape[] = {1, 3};
    Tensor* y = tensor_create(target_shape, 2);
    y->data[1] = 1.0f;

    printf("  Training 50 epochs: ");
    for (int epoch = 0; epoch < 50; epoch++) {
        if (epoch % 10 == 0) { printf("%d ", epoch + 1); fflush(stdout); }
        model_train_step(model, x, y);
    }
    printf("\n");

    printf("  Final prediction:\n");
    tensor_print(model->output_layer->output, "    Output");

    if (model_save(model, "zeki_model.bin", FORMAT_BINARY)) {
        printf("  Model saved: zeki_model.bin\n");
    } else {
        fprintf(stderr, "  Model save FAILED!\n");
    }

    model_free(model);
    tensor_free(x);
    tensor_free(y);
}

void test_lstm() {
    printf("--- LSTM ---\n");

    Layer* lstm = lstm_cell_create(3, 5, "LSTM-1");

    int input_shape[] = {1, 3};
    Tensor* x = tensor_create(input_shape, 2);
    x->data[0] = 0.5f; x->data[1] = 0.3f; x->data[2] = 0.8f;

    lstm->forward(lstm, x);
    tensor_print(lstm->output, "  LSTM Output");

    layer_free(lstm);
    tensor_free(x);
}

void test_gru() {
    printf("--- GRU ---\n");

    Layer* gru = gru_cell_create(3, 5, "GRU-1");

    int input_shape[] = {1, 3};
    Tensor* x = tensor_create(input_shape, 2);
    x->data[0] = 0.5f; x->data[1] = 0.3f; x->data[2] = 0.8f;

    gru->forward(gru, x);
    tensor_print(gru->output, "  GRU Output");

    layer_free(gru);
    tensor_free(x);
}

void test_attention() {
    printf("--- Attention ---\n");

    Layer* attn = attention_create(8, "Attention-1");

    int input_shape[] = {1, 4, 8};
    Tensor* x = tensor_create(input_shape, 3);
    for (int i = 0; i < x->size; i++) x->data[i] = (float)rand() / RAND_MAX;

    attn->forward(attn, x);
    tensor_print(attn->output, "  Attention Output");

    layer_free(attn);
    tensor_free(x);
}

void test_transformer() {
    printf("--- Transformer ---\n");

    Layer* transformer = transformer_block_create(8, 2, 16, "Transformer-1");

    int input_shape[] = {1, 4, 8};
    Tensor* x = tensor_create(input_shape, 3);
    for (int i = 0; i < x->size; i++) x->data[i] = (float)rand() / RAND_MAX;

    transformer->forward(transformer, x);
    tensor_print(transformer->output, "  Transformer Output");

    layer_free(transformer);
    tensor_free(x);
}

void test_data_augmentation() {
    printf("--- Data Augmentation ---\n");

    int shape[] = {1, 28, 28};
    Tensor* img = tensor_create(shape, 3);
    for (int i = 0; i < img->size; i++) img->data[i] = (float)rand() / RAND_MAX;

    tensor_free(augment_rotate(img, 15.0f));
    printf("  rotation, ");
    tensor_free(augment_flip(img, 1));
    printf("flip, ");
    tensor_free(augment_brightness(img, 1.2f));
    printf("brightness, ");
    tensor_free(augment_crop(img, 0.8f));
    printf("crop, ");
    tensor_free(augment_noise(img, 0.1f));
    printf("noise\n");

    tensor_free(img);
}

void test_metrics() {
    printf("--- Metrics ---\n");

    int pred_shape[] = {3, 2};
    Tensor* pred = tensor_create(pred_shape, 2);
    pred->data[0] = 0.9f; pred->data[1] = 0.1f;
    pred->data[2] = 0.2f; pred->data[3] = 0.8f;
    pred->data[4] = 0.3f; pred->data[5] = 0.7f;

    int target_shape[] = {3, 2};
    Tensor* target = tensor_create(target_shape, 2);
    target->data[0] = 1.0f; target->data[1] = 0.0f;
    target->data[2] = 0.0f; target->data[3] = 1.0f;
    target->data[4] = 0.0f; target->data[5] = 1.0f;

    printf("  Accuracy: %.2f%%\n", accuracy(pred, target) * 100);
    printf("  Precision (cls 0): %.4f\n", precision(pred, target, 0));
    printf("  Recall    (cls 0): %.4f\n", recall(pred, target, 0));
    printf("  F1-Score  (cls 0): %.4f\n", f1_score(pred, target, 0));

    confusion_matrix(pred, target, 2);

    tensor_free(pred);
    tensor_free(target);
}

void test_loss_functions() {
    printf("--- Loss Functions ---\n");

    int shape[] = {2, 3};
    Tensor* pred = tensor_create(shape, 2);
    pred->data[0] = 0.9f; pred->data[1] = 0.1f; pred->data[2] = 0.0f;
    pred->data[3] = 0.2f; pred->data[4] = 0.7f; pred->data[5] = 0.1f;

    Tensor* target = tensor_create(shape, 2);
    target->data[0] = 1.0f; target->data[1] = 0.0f; target->data[2] = 0.0f;
    target->data[3] = 0.0f; target->data[4] = 1.0f; target->data[5] = 0.0f;

    printf("  MSE:             %.6f\n", loss_compute(pred, target, LOSS_MSE));
    printf("  Cross-Entropy:   %.6f\n", loss_compute(pred, target, LOSS_CROSSENTROPY));
    printf("  Huber:           %.6f\n", loss_compute(pred, target, LOSS_HUBER));

    tensor_free(pred);
    tensor_free(target);
}

void test_full_cnn_model() {
    printf("--- CNN with BatchNorm ---\n");

    int input_shape[] = {1, 1, 32, 32};
    Layer* input = layer_create(LAYER_INPUT, "Input");
    input->output = tensor_create(input_shape, 4);

    Layer* conv1 = conv2d_create(1, 16, 3, 1, 1, "Conv1");
    ((ConvParams*)conv1->params)->input_h = 32;
    ((ConvParams*)conv1->params)->input_w = 32;
    Layer* bn1 = batchnorm_create(16, "BN1");
    Layer* relu1 = activation_create(ACT_RELU, "ReLU1");
    Layer* pool1 = maxpool2d_create(2, 2, "Pool1");

    Layer* conv2 = conv2d_create(16, 32, 3, 1, 1, "Conv2");
    ((ConvParams*)conv2->params)->input_h = 16;
    ((ConvParams*)conv2->params)->input_w = 16;
    Layer* bn2 = batchnorm_create(32, "BN2");
    Layer* relu2 = activation_create(ACT_RELU, "ReLU2");
    Layer* pool2 = maxpool2d_create(2, 2, "Pool2");

    Layer* flatten = flatten_create("Flatten");
    Layer* relu3 = activation_create(ACT_RELU, "ReLU3");
    Layer* dropout = dropout_create(0.3f, "Dropout");
    Layer* fc = dense_create(2048, 128, "FC");
    Layer* output = dense_create(128, 10, "Output");
    Layer* softmax = activation_create(ACT_SOFTMAX, "Softmax");

    Model* model = model_create(input);
    model_add_layer(model, conv1);
    model_add_layer(model, bn1);
    model_add_layer(model, relu1);
    model_add_layer(model, pool1);
    model_add_layer(model, conv2);
    model_add_layer(model, bn2);
    model_add_layer(model, relu2);
    model_add_layer(model, pool2);
    model_add_layer(model, flatten);
    model_add_layer(model, fc);
    model_add_layer(model, relu3);
    model_add_layer(model, dropout);
    model_add_layer(model, output);
    model_add_layer(model, softmax);

    Optimizer* adam = optimizer_create(OPT_ADAM, 0.001f);
    model_set_optimizer(model, adam);
    model_set_loss(model, LOSS_CROSSENTROPY);

    printf("  %d layers\n", model->num_layers);

    Tensor* x = tensor_create(input_shape, 4);
    for (int i = 0; i < x->size; i++) x->data[i] = (float)rand() / RAND_MAX;

    model_predict(model, x);
    tensor_print(model->output_layer->output, "  CNN Output");

    if (model_save(model, "cnn_model.bin", FORMAT_BINARY)) {
        printf("  Saved: cnn_model.bin\n");
    } else {
        fprintf(stderr, "  Save FAILED!\n");
    }

    model_free(model);
    tensor_free(x);
}

static void generate_synthetic_data(Tensor** out_x, Tensor** out_y,
                                      int n_samples, int n_features, int n_classes) {
    int x_shape[] = {n_samples, n_features};
    int y_shape[] = {n_samples, n_classes};
    *out_x = tensor_create(x_shape, 2);
    *out_y = tensor_create(y_shape, 2);
    tensor_fill(*out_y, 0.0f);

    Tensor* centers = tensor_create((int[]){n_classes, n_features}, 2);
    for (int i = 0; i < centers->size; i++)
        centers->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 6.0f;

    for (int s = 0; s < n_samples; s++) {
        int cls = rand() % n_classes;
        (*out_y)->data[s * n_classes + cls] = 1.0f;
        for (int f = 0; f < n_features; f++) {
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.6f;
            (*out_x)->data[s * n_features + f] = centers->data[cls * n_features + f] + noise;
        }
    }
    tensor_free(centers);
}

void test_validation_and_test() {
    printf("--- Training & Evaluation ---\n");

    int n_features = 8;
    int n_classes = 3;
    int n_train = 1000, n_val = 200, n_test = 200;
    int n_total = n_train + n_val + n_test;

    Tensor *all_x, *all_y;
    generate_synthetic_data(&all_x, &all_y, n_total, n_features, n_classes);

    Tensor* train_x = tensor_create((int[]){n_train, n_features}, 2);
    Tensor* train_y = tensor_create((int[]){n_train, n_classes}, 2);
    Tensor* val_x = tensor_create((int[]){n_val, n_features}, 2);
    Tensor* val_y = tensor_create((int[]){n_val, n_classes}, 2);
    Tensor* test_x = tensor_create((int[]){n_test, n_features}, 2);
    Tensor* test_y = tensor_create((int[]){n_test, n_classes}, 2);

    memcpy(train_x->data, all_x->data, n_train * n_features * sizeof(float));
    memcpy(train_y->data, all_y->data, n_train * n_classes * sizeof(float));
    memcpy(val_x->data, all_x->data + n_train * n_features, n_val * n_features * sizeof(float));
    memcpy(val_y->data, all_y->data + n_train * n_classes, n_val * n_classes * sizeof(float));
    memcpy(test_x->data, all_x->data + (n_train + n_val) * n_features, n_test * n_features * sizeof(float));
    memcpy(test_y->data, all_y->data + (n_train + n_val) * n_classes, n_test * n_classes * sizeof(float));

    tensor_free(all_x);
    tensor_free(all_y);

    printf("  Dataset: %d train / %d val / %d test, %d features, %d classes\n",
           n_train, n_val, n_test, n_features, n_classes);

    Layer* input = layer_create(LAYER_INPUT, "Input");
    int in_shape[] = {1, n_features};
    input->output = tensor_create(in_shape, 2);

    Layer* dense1 = dense_create(n_features, 16, "Dense-1");
    Layer* relu1 = activation_create(ACT_RELU, "ReLU-1");
    Layer* dropout1 = dropout_create(0.2f, "Dropout-1");
    Layer* dense2 = dense_create(16, 16, "Dense-2");
    Layer* relu2 = activation_create(ACT_RELU, "ReLU-2");
    Layer* output_layer = dense_create(16, n_classes, "Output");
    Layer* softmax = activation_create(ACT_SOFTMAX, "Softmax");

    Model* model = model_create(input);
    model_add_layer(model, dense1);
    model_add_layer(model, relu1);
    model_add_layer(model, dropout1);
    model_add_layer(model, dense2);
    model_add_layer(model, relu2);
    model_add_layer(model, output_layer);
    model_add_layer(model, softmax);

    Optimizer* adam = optimizer_create(OPT_ADAM, 0.001f);
    model_set_optimizer(model, adam);
    model_set_loss(model, LOSS_CROSSENTROPY);

    Trainer* trainer = trainer_create(model, adam, 32, 15);
    trainer_set_early_stopping(trainer, 5);
    trainer_set_lr_scheduler(trainer, 0.5f, 300);
    trainer_set_checkpoint(trainer, "best_model.bin");

    trainer_fit(trainer, train_x, train_y, val_x, val_y);
    trainer_evaluate(trainer, test_x, test_y);

    trainer_free(trainer);
    model_free(model);
    tensor_free(train_x); tensor_free(train_y);
    tensor_free(val_x); tensor_free(val_y);
    tensor_free(test_x); tensor_free(test_y);
}

int main(int argc, char* argv[]) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (argc > 1 && strcmp(argv[1], "--mcp") == 0) {
        Model* model = NULL;
        fprintf(stderr, "Zeki MCP Server (stdio) starting...\n");
        return mcp_server_run(&model);
    }

    if (argc > 1 && strcmp(argv[1], "--sse") == 0) {
        int port = 8080;
        if (argc > 2) port = atoi(argv[2]);
        Model* model = NULL;
        fprintf(stderr, "Zeki MCP Server (SSE) starting on port %d...\n", port);
        return mcp_server_run_sse(&model, port);
    }

    printf("====================================\n");
    printf("  ZEKI AI Model - Full Test Suite\n");
    printf("====================================\n\n");

    srand((unsigned int)time(NULL));

    test_adam_optimizer();
    test_lstm();
    test_gru();
    test_attention();
    test_transformer();
    test_data_augmentation();
    test_metrics();
    test_loss_functions();
    test_full_cnn_model();
    test_validation_and_test();

    printf("\n====================================\n");
    printf("  All tests completed successfully!\n");
    printf("====================================\n");
    return 0;
}
