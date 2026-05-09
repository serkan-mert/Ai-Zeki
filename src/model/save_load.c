#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "save_load.h"
#include "dense.h"
#include "activation.h"
#include "dropout.h"
#include "batchnorm.h"
#include "conv.h"
#include "lstm.h"
#include "flatten.h"
#include "gru.h"
#include "attention.h"
#include "transformer.h"

int tensor_save(Tensor* t, FILE* fp, SaveFormat fmt) {
    if (fmt == FORMAT_BINARY) {
        if (t->size < 1 || t->size > MAX_TENSOR_SIZE) {
            fprintf(stderr, "Error: tensor_save rejected size=%d (limit=%d)\n", t->size, MAX_TENSOR_SIZE);
            return 0;
        }
        if (fwrite(&t->ndim, sizeof(int), 1, fp) != 1) return 0;
        if (fwrite(t->shape, sizeof(int), t->ndim, fp) != (size_t)t->ndim) return 0;
        if (fwrite(&t->size, sizeof(int), 1, fp) != 1) return 0;
        if (fwrite(t->data, sizeof(float), t->size, fp) != (size_t)t->size) return 0;
    } else {
        if (fprintf(fp, "%d ", t->ndim) < 0) return 0;
        for (int i = 0; i < t->ndim; i++)
            if (fprintf(fp, "%d ", t->shape[i]) < 0) return 0;
        if (fprintf(fp, "%d\n", t->size) < 0) return 0;
        for (int i = 0; i < t->size; i++)
            if (fprintf(fp, "%.6f ", t->data[i]) < 0) return 0;
        if (fprintf(fp, "\n") < 0) return 0;
    }
    return 1;
}

Tensor* tensor_load(FILE* fp, SaveFormat fmt) {
    int ndim, size;
    if (fmt == FORMAT_BINARY) {
        if (fread(&ndim, sizeof(int), 1, fp) != 1) return NULL;
        if (ndim < 1 || ndim > 64) return NULL;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if (fread(shape, sizeof(int), ndim, fp) != (size_t)ndim) { free(shape); return NULL; }
        if (fread(&size, sizeof(int), 1, fp) != 1) { free(shape); return NULL; }
        if (size < 1 || size > MAX_TENSOR_SIZE) { free(shape); return NULL; }
        Tensor* t = tensor_create(shape, ndim);
        if (fread(t->data, sizeof(float), size, fp) != (size_t)size) { tensor_free(t); free(shape); return NULL; }
        free(shape);
        return t;
    } else {
        if (fscanf(fp, "%d", &ndim) != 1) return NULL;
        if (ndim < 1 || ndim > 64) return NULL;
        int* shape = (int*)malloc(ndim * sizeof(int));
        for (int i = 0; i < ndim; i++) if (fscanf(fp, "%d", &shape[i]) != 1) { free(shape); return NULL; }
        if (fscanf(fp, "%d", &size) != 1) { free(shape); return NULL; }
        if (size < 1 || size > MAX_TENSOR_SIZE) { free(shape); return NULL; }
        Tensor* t = tensor_create(shape, ndim);
        for (int i = 0; i < size; i++) if (fscanf(fp, "%f", &t->data[i]) != 1) { tensor_free(t); free(shape); return NULL; }
        free(shape);
        return t;
    }
}

int layer_save(Layer* layer, FILE* fp, SaveFormat fmt) {
    if (fmt == FORMAT_BINARY) {
        fwrite(&layer->type, sizeof(LayerType), 1, fp);
        int name_len = (int)strlen(layer->name) + 1;
        fwrite(&name_len, sizeof(int), 1, fp);
        fwrite(layer->name, sizeof(char), name_len, fp);
    } else {
        fprintf(fp, "%d %s\n", layer->type, layer->name);
    }

    if (layer->type == LAYER_DENSE && layer->params) {
        DenseParams* p = (DenseParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->input_dim, sizeof(int), 1, fp);
            fwrite(&p->output_dim, sizeof(int), 1, fp);
            fwrite(&p->learning_rate, sizeof(float), 1, fp);
        } else {
            fprintf(fp, "%d %d %.6f\n", p->input_dim, p->output_dim, p->learning_rate);
        }
        if (layer->weights) tensor_save(layer->weights, fp, fmt);
        if (layer->biases) tensor_save(layer->biases, fp, fmt);
    } else if (layer->type == LAYER_CONV2D && layer->params) {
        ConvParams* p = (ConvParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->input_channels, sizeof(int), 1, fp);
            fwrite(&p->output_channels, sizeof(int), 1, fp);
            fwrite(&p->kernel_size, sizeof(int), 1, fp);
            fwrite(&p->stride, sizeof(int), 1, fp);
            fwrite(&p->padding, sizeof(int), 1, fp);
            fwrite(&p->input_h, sizeof(int), 1, fp);
            fwrite(&p->input_w, sizeof(int), 1, fp);
        } else {
            fprintf(fp, "%d %d %d %d %d %d %d\n",
                p->input_channels, p->output_channels, p->kernel_size,
                p->stride, p->padding, p->input_h, p->input_w);
        }
        tensor_save(p->kernels, fp, fmt);
        tensor_save(p->biases, fp, fmt);
    } else if (layer->type == LAYER_LSTM && layer->params) {
        LSTMCellParams* p = (LSTMCellParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->input_dim, sizeof(int), 1, fp);
            fwrite(&p->hidden_dim, sizeof(int), 1, fp);
        } else {
            fprintf(fp, "%d %d\n", p->input_dim, p->hidden_dim);
        }
        tensor_save(p->Wf, fp, fmt); tensor_save(p->Wi, fp, fmt);
        tensor_save(p->Wc, fp, fmt); tensor_save(p->Wo, fp, fmt);
        tensor_save(p->bf, fp, fmt); tensor_save(p->bi, fp, fmt);
        tensor_save(p->bc, fp, fmt); tensor_save(p->bo, fp, fmt);
    } else if (layer->type == LAYER_FLATTEN) {
        /* no metadata or tensors to save */
    } else if (layer->type == LAYER_GRU && layer->params) {
        GRUCellParams* p = (GRUCellParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->input_dim, sizeof(int), 1, fp);
            fwrite(&p->hidden_dim, sizeof(int), 1, fp);
        } else {
            fprintf(fp, "%d %d\n", p->input_dim, p->hidden_dim);
        }
        tensor_save(p->Wz, fp, fmt); tensor_save(p->Wr, fp, fmt);
        tensor_save(p->Wh, fp, fmt);
        tensor_save(p->Uz, fp, fmt); tensor_save(p->Ur, fp, fmt);
        tensor_save(p->Uh, fp, fmt);
        tensor_save(p->bz, fp, fmt); tensor_save(p->br, fp, fmt);
        tensor_save(p->bh, fp, fmt);
    } else if (layer->type == LAYER_BATCHNORM && layer->params) {
        BatchNormParams* p = (BatchNormParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->dim, sizeof(int), 1, fp);
        } else {
            fprintf(fp, "%d\n", p->dim);
        }
        tensor_save(p->gamma, fp, fmt);
        tensor_save(p->beta, fp, fmt);
        tensor_save(p->running_mean, fp, fmt);
        tensor_save(p->running_var, fp, fmt);
    } else if (layer->type == LAYER_ATTENTION && layer->params) {
        AttentionParams* p = (AttentionParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->embed_dim, sizeof(int), 1, fp);
        } else {
            fprintf(fp, "%d\n", p->embed_dim);
        }
        tensor_save(p->Wq, fp, fmt);
        tensor_save(p->Wk, fp, fmt);
        tensor_save(p->Wv, fp, fmt);
        tensor_save(p->Wo, fp, fmt);
    } else if (layer->type == LAYER_TRANSFORMER && layer->params) {
        TransformerBlockParams* p = (TransformerBlockParams*)layer->params;
        if (fmt == FORMAT_BINARY) {
            fwrite(&p->embed_dim, sizeof(int), 1, fp);
            fwrite(&p->num_heads, sizeof(int), 1, fp);
            fwrite(&p->ff_dim, sizeof(int), 1, fp);
        } else {
            fprintf(fp, "%d %d %d\n", p->embed_dim, p->num_heads, p->ff_dim);
        }
        layer_save(p->attn_layer, fp, fmt);
        layer_save(p->ff_dense1, fp, fmt);
        layer_save(p->ff_activation, fp, fmt);
        layer_save(p->ff_dense2, fp, fmt);
    }

    return 1;
}

Layer* layer_load(FILE* fp, SaveFormat fmt) {
    LayerType type;
    char name[64];

    if (fmt == FORMAT_BINARY) {
        if (fread(&type, sizeof(LayerType), 1, fp) != 1) return NULL;
        int name_len;
        if (fread(&name_len, sizeof(int), 1, fp) != 1) return NULL;
        if (name_len < 1 || name_len > 63) return NULL;
        if (fread(name, sizeof(char), name_len, fp) != (size_t)name_len) return NULL;
        name[63] = '\0';
    } else {
        int t;
        if (fscanf(fp, "%d %63s", &t, name) != 2) return NULL;
        type = (LayerType)t;
    }

    Layer* layer = NULL;

    if (type == LAYER_DENSE) {
        int in_dim, out_dim;
        float lr;
        if (fmt == FORMAT_BINARY) {
            if (fread(&in_dim, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&out_dim, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&lr, sizeof(float), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d %d %f", &in_dim, &out_dim, &lr) != 3) return NULL;
        }
        layer = dense_create(in_dim, out_dim, name);
        if (layer && layer->weights) {
            tensor_free(layer->weights);
            layer->weights = tensor_load(fp, fmt);
        }
        if (layer && layer->biases) {
            tensor_free(layer->biases);
            layer->biases = tensor_load(fp, fmt);
        }
        if (layer && (!layer->weights || !layer->biases)) {
            fprintf(stderr, "Warning: Failed to load weights/biases for Dense layer '%s'\n", name);
            layer_free(layer);
            return NULL;
        }
    } else if (type == LAYER_ACTIVATION) {
        layer = activation_create(ACT_RELU, name);
    } else if (type == LAYER_CONV2D) {
        int in_c, out_c, k, s, pad, in_h, in_w;
        if (fmt == FORMAT_BINARY) {
            if (fread(&in_c, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&out_c, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&k, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&s, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&pad, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&in_h, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&in_w, sizeof(int), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d %d %d %d %d %d %d", &in_c, &out_c, &k, &s, &pad, &in_h, &in_w) != 7) return NULL;
        }
        layer = conv2d_create(in_c, out_c, k, s, pad, name);
        if (layer && layer->params) {
            ConvParams* p = (ConvParams*)layer->params;
            p->input_h = in_h;
            p->input_w = in_w;
            tensor_free(p->kernels);
            p->kernels = tensor_load(fp, fmt);
            tensor_free(p->biases);
            p->biases = tensor_load(fp, fmt);
            layer->weights = p->kernels;
            layer->biases = p->biases;
            if (!p->kernels || !p->biases) {
                fprintf(stderr, "Warning: Failed to load kernels/biases for Conv2D layer '%s'\n", name);
                layer_free(layer);
                return NULL;
            }
        }
    } else if (type == LAYER_LSTM) {
        int in_dim, h_dim;
        if (fmt == FORMAT_BINARY) {
            if (fread(&in_dim, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&h_dim, sizeof(int), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d %d", &in_dim, &h_dim) != 2) return NULL;
        }
        layer = lstm_cell_create(in_dim, h_dim, name);
        if (layer && layer->params) {
            LSTMCellParams* p = (LSTMCellParams*)layer->params;
            tensor_free(p->Wf); tensor_free(p->Wi); tensor_free(p->Wc); tensor_free(p->Wo);
            tensor_free(p->bf); tensor_free(p->bi); tensor_free(p->bc); tensor_free(p->bo);
            p->Wf = tensor_load(fp, fmt); p->Wi = tensor_load(fp, fmt);
            p->Wc = tensor_load(fp, fmt); p->Wo = tensor_load(fp, fmt);
            p->bf = tensor_load(fp, fmt); p->bi = tensor_load(fp, fmt);
            p->bc = tensor_load(fp, fmt); p->bo = tensor_load(fp, fmt);
            layer->weights = p->Wf;
            layer->biases = p->bf;
            if (!p->Wf || !p->Wi || !p->Wc || !p->Wo || !p->bf || !p->bi || !p->bc || !p->bo) {
                fprintf(stderr, "Warning: Failed to load tensors for LSTM layer '%s'\n", name);
                layer_free(layer);
                return NULL;
            }
        }
    } else if (type == LAYER_FLATTEN) {
        layer = flatten_create(name);
    } else if (type == LAYER_GRU) {
        int in_dim, h_dim;
        if (fmt == FORMAT_BINARY) {
            if (fread(&in_dim, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&h_dim, sizeof(int), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d %d", &in_dim, &h_dim) != 2) return NULL;
        }
        layer = gru_cell_create(in_dim, h_dim, name);
        if (layer && layer->params) {
            GRUCellParams* p = (GRUCellParams*)layer->params;
            tensor_free(p->Wz); tensor_free(p->Wr); tensor_free(p->Wh);
            tensor_free(p->Uz); tensor_free(p->Ur); tensor_free(p->Uh);
            tensor_free(p->bz); tensor_free(p->br); tensor_free(p->bh);
            p->Wz = tensor_load(fp, fmt); p->Wr = tensor_load(fp, fmt);
            p->Wh = tensor_load(fp, fmt);
            p->Uz = tensor_load(fp, fmt); p->Ur = tensor_load(fp, fmt);
            p->Uh = tensor_load(fp, fmt);
            p->bz = tensor_load(fp, fmt); p->br = tensor_load(fp, fmt);
            p->bh = tensor_load(fp, fmt);
            layer->weights = p->Wz;
            layer->biases = p->bz;
            if (!p->Wz || !p->Wr || !p->Wh || !p->Uz || !p->Ur || !p->Uh || !p->bz || !p->br || !p->bh) {
                fprintf(stderr, "Warning: Failed to load tensors for GRU layer '%s'\n", name);
                layer_free(layer);
                return NULL;
            }
        }
    } else if (type == LAYER_BATCHNORM) {
        int dim;
        if (fmt == FORMAT_BINARY) {
            if (fread(&dim, sizeof(int), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d", &dim) != 1) return NULL;
        }
        layer = batchnorm_create(dim, name);
        if (layer && layer->params) {
            BatchNormParams* p = (BatchNormParams*)layer->params;
            tensor_free(p->gamma); tensor_free(p->beta);
            tensor_free(p->running_mean); tensor_free(p->running_var);
            p->gamma = tensor_load(fp, fmt);
            p->beta = tensor_load(fp, fmt);
            p->running_mean = tensor_load(fp, fmt);
            p->running_var = tensor_load(fp, fmt);
            if (!p->gamma || !p->beta || !p->running_mean || !p->running_var) {
                fprintf(stderr, "Warning: Failed to load tensors for BatchNorm layer '%s'\n", name);
                layer_free(layer);
                return NULL;
            }
        }
    } else if (type == LAYER_ATTENTION) {
        int embed_dim;
        if (fmt == FORMAT_BINARY) {
            if (fread(&embed_dim, sizeof(int), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d", &embed_dim) != 1) return NULL;
        }
        layer = attention_create(embed_dim, name);
        if (layer && layer->params) {
            AttentionParams* p = (AttentionParams*)layer->params;
            tensor_free(p->Wq); tensor_free(p->Wk); tensor_free(p->Wv); tensor_free(p->Wo);
            p->Wq = tensor_load(fp, fmt);
            p->Wk = tensor_load(fp, fmt);
            p->Wv = tensor_load(fp, fmt);
            p->Wo = tensor_load(fp, fmt);
            layer->weights = p->Wq;
            if (!p->Wq || !p->Wk || !p->Wv || !p->Wo) {
                fprintf(stderr, "Warning: Failed to load tensors for Attention layer '%s'\n", name);
                layer_free(layer);
                return NULL;
            }
        }
    } else if (type == LAYER_TRANSFORMER) {
        int embed_dim, num_heads, ff_dim;
        if (fmt == FORMAT_BINARY) {
            if (fread(&embed_dim, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&num_heads, sizeof(int), 1, fp) != 1) return NULL;
            if (fread(&ff_dim, sizeof(int), 1, fp) != 1) return NULL;
        } else {
            if (fscanf(fp, "%d %d %d", &embed_dim, &num_heads, &ff_dim) != 3) return NULL;
        }
        layer = transformer_block_create(embed_dim, num_heads, ff_dim, name);
        if (layer && layer->params) {
            TransformerBlockParams* p = (TransformerBlockParams*)layer->params;
            layer_free(p->attn_layer);
            p->attn_layer = layer_load(fp, fmt);
            layer_free(p->ff_dense1);
            p->ff_dense1 = layer_load(fp, fmt);
            layer_free(p->ff_activation);
            p->ff_activation = layer_load(fp, fmt);
            layer_free(p->ff_dense2);
            p->ff_dense2 = layer_load(fp, fmt);
            if (!p->attn_layer || !p->ff_dense1 || !p->ff_activation || !p->ff_dense2) {
                fprintf(stderr, "Warning: Failed to load sub-layers for Transformer layer '%s'\n", name);
                layer_free(layer);
                return NULL;
            }
        }
    }

    return layer;
}

int model_save(Model* m, const char* filename, SaveFormat fmt) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s' for writing\n", filename);
        return 0;
    }

    if (fwrite(&m->num_layers, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to write num_layers\n");
        fclose(fp);
        return 0;
    }
    if (m->input_layer && m->input_layer->output) {
        Tensor* input_shape_t = tensor_create(m->input_layer->output->shape, m->input_layer->output->ndim);
        tensor_save(input_shape_t, fp, fmt);
        tensor_free(input_shape_t);
    }

    Layer* curr = m->input_layer ? m->input_layer->next : NULL;
    while (curr) {
        layer_save(curr, fp, fmt);
        curr = curr->next;
    }

    if (fflush(fp) != 0) {
        fprintf(stderr, "Error: Failed to flush file\n");
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

int model_save_weights(Model* m, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return 0;

    Layer* curr = m->input_layer;
    while (curr) {
        if (curr->weights)
            if (!tensor_save(curr->weights, fp, FORMAT_BINARY)) { fclose(fp); return 0; }
        if (curr->biases)
            if (!tensor_save(curr->biases, fp, FORMAT_BINARY)) { fclose(fp); return 0; }
        curr = curr->next;
    }

    fclose(fp);
    return 1;
}

Model* model_load(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;

    int num_layers;
    if (fread(&num_layers, sizeof(int), 1, fp) != 1) { fclose(fp); return NULL; }
    if (num_layers < 1 || num_layers > 1000) { fclose(fp); return NULL; }

    Tensor* input_shape = tensor_load(fp, FORMAT_BINARY);
    if (!input_shape) { fclose(fp); return NULL; }

    Layer* input = layer_create(LAYER_INPUT, "Input");
    input->output = tensor_create(input_shape->shape, input_shape->ndim);
    tensor_free(input_shape);

    Model* m = model_create(input);
    int layers_loaded = 0;

    for (int i = 1; i < num_layers; i++) {
        Layer* layer = layer_load(fp, FORMAT_BINARY);
        if (layer) {
            model_add_layer(m, layer);
            layers_loaded++;
        }
    }

    fclose(fp);

    if (layers_loaded == 0) {
        fprintf(stderr, "Warning: No layers could be loaded (file may be corrupt or incompatible)\n");
        model_free(m);
        return NULL;
    }

    if (layers_loaded != num_layers - 1) {
        fprintf(stderr, "Warning: Loaded %d/%d layers (incompatible format?)\n", layers_loaded, num_layers - 1);
    }

    return m;
}
