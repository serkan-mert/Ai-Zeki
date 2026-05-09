#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "mcp_server.h"
#include "json_utils.h"
#include "http_server.h"
#include "tensor.h"
#include "save_load.h"
#include "dense.h"
#include "activation.h"
#include "dropout.h"
#include "conv.h"
#include "lstm.h"
#include "gru.h"
#include "flatten.h"
#include "batchnorm.h"
#include "attention.h"
#include "transformer.h"
#include "loss.h"
#include "metrics.h"
#include "trainer.h"

static Model** g_model;
static int g_initialized = 0;
static void (*g_sse_send_fn)(const char* event, const char* data) = NULL;

static char* read_one_line(void) {
    int cap = 4096, len = 0;
    char* buf = (char*)malloc(cap);
    int c;
    while ((c = getchar()) != EOF) {
        if (c == '\n') break;
        if (len + 1 >= cap) {
            cap *= 2;
            buf = (char*)realloc(buf, cap);
        }
        buf[len++] = (char)c;
    }
    if (len == 0 && c == EOF) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

static void write_msg(const char* json_str) {
    fprintf(stdout, "%s\n", json_str);
    fflush(stdout);
}

static JsonValue* make_error(int code, const char* msg, JsonValue* data) {
    JsonValue* err = json_object();
    json_object_set(err, "code", json_number(code));
    json_object_set(err, "message", json_string(msg));
    if (data) json_object_set(err, "data", data);
    return err;
}

static JsonValue* make_response(int id, JsonValue* result, JsonValue* error) {
    JsonValue* resp = json_object();
    json_object_set(resp, "jsonrpc", json_string("2.0"));
    json_object_set(resp, "id", json_number(id));
    if (result) json_object_set(resp, "result", result);
    if (error) json_object_set(resp, "error", error);
    return resp;
}

static void send_response(int id, JsonValue* result, JsonValue* error) {
    JsonValue* resp = make_response(id, result, error);
    char* s = json_serialize(resp);
    if (g_sse_send_fn) g_sse_send_fn("message", s);
    else write_msg(s);
    free(s);
    json_free(resp);
}

static void send_result(int id, JsonValue* result) {
    send_response(id, result, NULL);
}

static void send_error(int id, int code, const char* msg) {
    send_response(id, NULL, make_error(code, msg, NULL));
}

static JsonValue* make_content_text(const char* text) {
    JsonValue* c = json_object();
    json_object_set(c, "type", json_string("text"));
    json_object_set(c, "text", json_string(text));
    return c;
}

static JsonValue* tensor_to_json_array(Tensor* t) {
    JsonValue* arr = json_array();
    for (int i = 0; i < t->size; i++) json_array_append(arr, json_number(t->data[i]));
    return arr;
}

static int json_array_to_float(JsonValue* arr, float** out, int* out_len) {
    if (!arr || arr->type != JSON_ARRAY) return 0;
    *out_len = arr->array.count;
    *out = (float*)malloc(*out_len * sizeof(float));
    for (int i = 0; i < *out_len; i++) {
        JsonValue* item = arr->array.items[i];
        if (item->type == JSON_NUMBER) (*out)[i] = (float)item->num_val;
        else if (item->type == JSON_BOOL) (*out)[i] = (float)item->bool_val;
        else (*out)[i] = 0.0f;
    }
    return 1;
}

static int json_array_to_int(JsonValue* arr, int** out, int* out_len) {
    if (!arr || arr->type != JSON_ARRAY) return 0;
    *out_len = arr->array.count;
    *out = (int*)malloc(*out_len * sizeof(int));
    for (int i = 0; i < *out_len; i++) {
        JsonValue* item = arr->array.items[i];
        if (item->type == JSON_NUMBER) (*out)[i] = (int)item->num_val;
        else (*out)[i] = 0;
    }
    return 1;
}

static float* extract_float_array(JsonValue* params, const char* key, int* len) {
    JsonValue* arr = json_object_get(params, key);
    int n = 0;
    float* data = NULL;
    if (!json_array_to_float(arr, &data, &n)) return NULL;
    *len = n;
    return data;
}

static int* extract_int_array(JsonValue* params, const char* key, int* len) {
    JsonValue* arr = json_object_get(params, key);
    int n = 0;
    int* data = NULL;
    if (!json_array_to_int(arr, &data, &n)) return NULL;
    *len = n;
    return data;
}

static void handle_initialize(int id, JsonValue* params) {
    (void)id; (void)params;
    JsonValue* cap = json_object();
    json_object_set(cap, "tools", json_object());
    json_object_set(cap, "resources", json_object());
    json_object_set(cap, "prompts", json_object());

    JsonValue* server_info = json_object();
    json_object_set(server_info, "name", json_string("zeki-mcp"));
    json_object_set(server_info, "version", json_string("1.0.0"));

    JsonValue* result = json_object();
    json_object_set(result, "protocolVersion", json_string("2024-11-05"));
    json_object_set(result, "capabilities", cap);
    json_object_set(result, "serverInfo", server_info);

    send_result(id, result);
}

static void handle_tools_list(int id) {
    JsonValue* tools = json_array();

    JsonValue* t1 = json_object();
    json_object_set(t1, "name", json_string("predict"));
    json_object_set(t1, "description", json_string("Run inference on the Zeki model with input data"));
    JsonValue* s1 = json_object();
    json_object_set(s1, "type", json_string("object"));
    JsonValue* props1 = json_object();
    JsonValue* p_input = json_object();
    json_object_set(p_input, "type", json_string("array"));
    json_object_set(p_input, "description", json_string("Input data as flat array of floats"));
    json_object_set(p_input, "items", json_object());
    json_object_set(json_object_get(p_input, "items"), "type", json_string("number"));
    json_object_set(props1, "input_data", p_input);
    JsonValue* p_shape = json_object();
    json_object_set(p_shape, "type", json_string("array"));
    json_object_set(p_shape, "description", json_string("Input shape as array of ints (e.g. [1, 784])"));
    json_object_set(p_shape, "items", json_object());
    json_object_set(json_object_get(p_shape, "items"), "type", json_string("number"));
    json_object_set(props1, "input_shape", p_shape);
    json_object_set(s1, "properties", props1);
    JsonValue* req1 = json_array();
    json_array_append(req1, json_string("input_data"));
    json_array_append(req1, json_string("input_shape"));
    json_object_set(s1, "required", req1);
    json_object_set(t1, "inputSchema", s1);
    json_array_append(tools, t1);

    JsonValue* t2 = json_object();
    json_object_set(t2, "name", json_string("train_step"));
    json_object_set(t2, "description", json_string("Run one training step on the model"));
    JsonValue* s2 = json_object();
    json_object_set(s2, "type", json_string("object"));
    JsonValue* props2 = json_object();
    JsonValue* p_in = json_object();
    json_object_set(p_in, "type", json_string("array"));
    json_object_set(p_in, "description", json_string("Training input data as flat array"));
    json_object_set(p_in, "items", json_object());
    json_object_set(json_object_get(p_in, "items"), "type", json_string("number"));
    json_object_set(props2, "input_data", p_in);
    JsonValue* p_tar = json_object();
    json_object_set(p_tar, "type", json_string("array"));
    json_object_set(p_tar, "description", json_string("Target data as flat array"));
    json_object_set(p_tar, "items", json_object());
    json_object_set(json_object_get(p_tar, "items"), "type", json_string("number"));
    json_object_set(props2, "target_data", p_tar);
    JsonValue* p_ishape = json_object();
    json_object_set(p_ishape, "type", json_string("array"));
    json_object_set(p_ishape, "description", json_string("Input shape"));
    json_object_set(p_ishape, "items", json_object());
    json_object_set(json_object_get(p_ishape, "items"), "type", json_string("number"));
    json_object_set(props2, "input_shape", p_ishape);
    JsonValue* p_tshape = json_object();
    json_object_set(p_tshape, "type", json_string("array"));
    json_object_set(p_tshape, "description", json_string("Target shape"));
    json_object_set(p_tshape, "items", json_object());
    json_object_set(json_object_get(p_tshape, "items"), "type", json_string("number"));
    json_object_set(props2, "target_shape", p_tshape);
    json_object_set(s2, "properties", props2);
    JsonValue* req2 = json_array();
    json_array_append(req2, json_string("input_data"));
    json_array_append(req2, json_string("target_data"));
    json_array_append(req2, json_string("input_shape"));
    json_array_append(req2, json_string("target_shape"));
    json_object_set(s2, "required", req2);
    json_object_set(t2, "inputSchema", s2);
    json_array_append(tools, t2);

    JsonValue* t3 = json_object();
    json_object_set(t3, "name", json_string("load_model"));
    json_object_set(t3, "description", json_string("Load a Zeki model from a .bin file"));
    JsonValue* s3 = json_object();
    json_object_set(s3, "type", json_string("object"));
    JsonValue* props3 = json_object();
    JsonValue* p_fn = json_object();
    json_object_set(p_fn, "type", json_string("string"));
    json_object_set(p_fn, "description", json_string("Path to the .bin model file"));
    json_object_set(props3, "filename", p_fn);
    json_object_set(s3, "properties", props3);
    JsonValue* req3 = json_array();
    json_array_append(req3, json_string("filename"));
    json_object_set(s3, "required", req3);
    json_object_set(t3, "inputSchema", s3);
    json_array_append(tools, t3);

    JsonValue* t4 = json_object();
    json_object_set(t4, "name", json_string("save_model"));
    json_object_set(t4, "description", json_string("Save the current model to a .bin file"));
    JsonValue* s4 = json_object();
    json_object_set(s4, "type", json_string("object"));
    JsonValue* props4 = json_object();
    JsonValue* p_fn2 = json_object();
    json_object_set(p_fn2, "type", json_string("string"));
    json_object_set(p_fn2, "description", json_string("Path to save the model file"));
    json_object_set(props4, "filename", p_fn2);
    json_object_set(s4, "properties", props4);
    JsonValue* req4 = json_array();
    json_array_append(req4, json_string("filename"));
    json_object_set(s4, "required", req4);
    json_object_set(t4, "inputSchema", s4);
    json_array_append(tools, t4);

    JsonValue* t5 = json_object();
    json_object_set(t5, "name", json_string("get_model_info"));
    json_object_set(t5, "description", json_string("Get information about the currently loaded model"));
    JsonValue* s5 = json_object();
    json_object_set(s5, "type", json_string("object"));
    json_object_set(s5, "properties", json_object());
    json_object_set(t5, "inputSchema", s5);
    json_array_append(tools, t5);

    JsonValue* t6 = json_object();
    json_object_set(t6, "name", json_string("list_layers"));
    json_object_set(t6, "description", json_string("List all layers in the model"));
    JsonValue* s6 = json_object();
    json_object_set(s6, "type", json_string("object"));
    json_object_set(s6, "properties", json_object());
    json_object_set(t6, "inputSchema", s6);
    json_array_append(tools, t6);

    JsonValue* t7 = json_object();
    json_object_set(t7, "name", json_string("evaluate"));
    json_object_set(t7, "description", json_string("Evaluate the model on test data"));
    JsonValue* s7 = json_object();
    json_object_set(s7, "type", json_string("object"));
    JsonValue* props7 = json_object();
    JsonValue* p_in2 = json_object();
    json_object_set(p_in2, "type", json_string("array"));
    json_object_set(p_in2, "description", json_string("Test input data"));
    json_object_set(p_in2, "items", json_object());
    json_object_set(json_object_get(p_in2, "items"), "type", json_string("number"));
    json_object_set(props7, "input_data", p_in2);
    JsonValue* p_tar2 = json_object();
    json_object_set(p_tar2, "type", json_string("array"));
    json_object_set(p_tar2, "description", json_string("Test target data"));
    json_object_set(p_tar2, "items", json_object());
    json_object_set(json_object_get(p_tar2, "items"), "type", json_string("number"));
    json_object_set(props7, "target_data", p_tar2);
    JsonValue* p_ishape2 = json_object();
    json_object_set(p_ishape2, "type", json_string("array"));
    json_object_set(p_ishape2, "description", json_string("Input shape"));
    json_object_set(p_ishape2, "items", json_object());
    json_object_set(json_object_get(p_ishape2, "items"), "type", json_string("number"));
    json_object_set(props7, "input_shape", p_ishape2);
    JsonValue* p_tshape2 = json_object();
    json_object_set(p_tshape2, "type", json_string("array"));
    json_object_set(p_tshape2, "description", json_string("Target shape"));
    json_object_set(p_tshape2, "items", json_object());
    json_object_set(json_object_get(p_tshape2, "items"), "type", json_string("number"));
    json_object_set(props7, "target_shape", p_tshape2);
    json_object_set(s7, "properties", props7);
    JsonValue* req7 = json_array();
    json_array_append(req7, json_string("input_data"));
    json_array_append(req7, json_string("target_data"));
    json_array_append(req7, json_string("input_shape"));
    json_array_append(req7, json_string("target_shape"));
    json_object_set(s7, "required", req7);
    json_object_set(t7, "inputSchema", s7);
    json_array_append(tools, t7);

    JsonValue* t8 = json_object();
    json_object_set(t8, "name", json_string("create_model"));
    json_object_set(t8, "description", json_string("Create a new empty model with specified input shape"));
    JsonValue* s8 = json_object();
    json_object_set(s8, "type", json_string("object"));
    JsonValue* props8 = json_object();
    JsonValue* p_is = json_object();
    json_object_set(p_is, "type", json_string("array"));
    json_object_set(p_is, "description", json_string("Input shape (e.g. [1, 784] for batch_size, features)"));
    json_object_set(p_is, "items", json_object());
    json_object_set(json_object_get(p_is, "items"), "type", json_string("number"));
    json_object_set(props8, "input_shape", p_is);
    json_object_set(s8, "properties", props8);
    JsonValue* req8 = json_array();
    json_array_append(req8, json_string("input_shape"));
    json_object_set(s8, "required", req8);
    json_object_set(t8, "inputSchema", s8);
    json_array_append(tools, t8);

    JsonValue* t9 = json_object();
    json_object_set(t9, "name", json_string("add_layer"));
    json_object_set(t9, "description", json_string("Add a layer to the model"));
    JsonValue* s9 = json_object();
    json_object_set(s9, "type", json_string("object"));
    JsonValue* props9 = json_object();
    JsonValue* p_lt = json_object();
    json_object_set(p_lt, "type", json_string("string"));
    json_object_set(p_lt, "description", json_string("Layer type: dense, conv2d, relu, sigmoid, tanh, softmax, dropout, flatten, lstm, gru, batchnorm, attention, transformer"));
    json_object_set(props9, "layer_type", p_lt);
    JsonValue* p_ln = json_object();
    json_object_set(p_ln, "type", json_string("string"));
    json_object_set(p_ln, "description", json_string("Layer name (optional)"));
    json_object_set(props9, "name", p_ln);
    JsonValue* p_lp = json_object();
    json_object_set(p_lp, "type", json_string("object"));
    json_object_set(p_lp, "description", json_string("Layer parameters: for dense: {\"input_dim\": int, \"output_dim\": int}; conv2d: {\"in_channels\": int, \"out_channels\": int, \"kernel_size\": int, ...}; dropout: {\"rate\": float}; lstm: {\"input_dim\": int, \"hidden_dim\": int}; gru: {\"input_dim\": int, \"hidden_dim\": int}; batchnorm: {\"dim\": int}; attention: {\"embed_dim\": int}; transformer: {\"embed_dim\": int, \"num_heads\": int, \"ff_dim\": int}"));
    json_object_set(props9, "params", p_lp);
    json_object_set(s9, "properties", props9);
    JsonValue* req9 = json_array();
    json_array_append(req9, json_string("layer_type"));
    json_object_set(s9, "required", req9);
    json_object_set(t9, "inputSchema", s9);
    json_array_append(tools, t9);

    JsonValue* t10 = json_object();
    json_object_set(t10, "name", json_string("set_training_mode"));
    json_object_set(t10, "description", json_string("Set model to training or evaluation mode"));
    JsonValue* s10 = json_object();
    json_object_set(s10, "type", json_string("object"));
    JsonValue* props10 = json_object();
    JsonValue* p_tm = json_object();
    json_object_set(p_tm, "type", json_string("boolean"));
    json_object_set(p_tm, "description", json_string("true for training, false for evaluation"));
    json_object_set(props10, "training", p_tm);
    json_object_set(s10, "properties", props10);
    JsonValue* req10 = json_array();
    json_array_append(req10, json_string("training"));
    json_object_set(s10, "required", req10);
    json_object_set(t10, "inputSchema", s10);
    json_array_append(tools, t10);

    JsonValue* t11 = json_object();
    json_object_set(t11, "name", json_string("train"));
    json_object_set(t11, "description", json_string("Train the model for multiple epochs"));
    JsonValue* s11 = json_object();
    json_object_set(s11, "type", json_string("object"));
    JsonValue* props11 = json_object();
    JsonValue* p_ti = json_object();
    json_object_set(p_ti, "type", json_string("array"));
    json_object_set(p_ti, "description", json_string("Training input data"));
    json_object_set(p_ti, "items", json_object());
    json_object_set(json_object_get(p_ti, "items"), "type", json_string("number"));
    json_object_set(props11, "train_input", p_ti);
    JsonValue* p_tt = json_object();
    json_object_set(p_tt, "type", json_string("array"));
    json_object_set(p_tt, "description", json_string("Training target data"));
    json_object_set(p_tt, "items", json_object());
    json_object_set(json_object_get(p_tt, "items"), "type", json_string("number"));
    json_object_set(props11, "train_target", p_tt);
    JsonValue* p_ep = json_object();
    json_object_set(p_ep, "type", json_string("number"));
    json_object_set(p_ep, "description", json_string("Number of epochs"));
    json_object_set(props11, "epochs", p_ep);
    JsonValue* p_bs = json_object();
    json_object_set(p_bs, "type", json_string("number"));
    json_object_set(p_bs, "description", json_string("Batch size"));
    json_object_set(props11, "batch_size", p_bs);
    JsonValue* p_vi = json_object();
    json_object_set(p_vi, "type", json_string("array"));
    json_object_set(p_vi, "description", json_string("Validation input data (optional)"));
    json_object_set(p_vi, "items", json_object());
    json_object_set(json_object_get(p_vi, "items"), "type", json_string("number"));
    json_object_set(props11, "val_input", p_vi);
    JsonValue* p_vt = json_object();
    json_object_set(p_vt, "type", json_string("array"));
    json_object_set(p_vt, "description", json_string("Validation target data (optional)"));
    json_object_set(p_vt, "items", json_object());
    json_object_set(json_object_get(p_vt, "items"), "type", json_string("number"));
    json_object_set(props11, "val_target", p_vt);
    JsonValue* p_ish3 = json_object();
    json_object_set(p_ish3, "type", json_string("array"));
    json_object_set(p_ish3, "description", json_string("Input shape"));
    json_object_set(p_ish3, "items", json_object());
    json_object_set(json_object_get(p_ish3, "items"), "type", json_string("number"));
    json_object_set(props11, "input_shape", p_ish3);
    JsonValue* p_tsh3 = json_object();
    json_object_set(p_tsh3, "type", json_string("array"));
    json_object_set(p_tsh3, "description", json_string("Target shape"));
    json_object_set(p_tsh3, "items", json_object());
    json_object_set(json_object_get(p_tsh3, "items"), "type", json_string("number"));
    json_object_set(props11, "target_shape", p_tsh3);
    json_object_set(s11, "properties", props11);
    JsonValue* req11 = json_array();
    json_array_append(req11, json_string("train_input"));
    json_array_append(req11, json_string("train_target"));
    json_array_append(req11, json_string("epochs"));
    json_array_append(req11, json_string("batch_size"));
    json_object_set(s11, "required", req11);
    json_object_set(t11, "inputSchema", s11);
    json_array_append(tools, t11);

    JsonValue* t12 = json_object();
    json_object_set(t12, "name", json_string("get_layer_output"));
    json_object_set(t12, "description", json_string("Get the output tensor of a specific layer by index"));
    JsonValue* s12 = json_object();
    json_object_set(s12, "type", json_string("object"));
    JsonValue* props12 = json_object();
    JsonValue* p_li = json_object();
    json_object_set(p_li, "type", json_string("number"));
    json_object_set(p_li, "description", json_string("Layer index (0-based)"));
    json_object_set(props12, "layer_index", p_li);
    json_object_set(s12, "properties", props12);
    JsonValue* req12 = json_array();
    json_array_append(req12, json_string("layer_index"));
    json_object_set(s12, "required", req12);
    json_object_set(t12, "inputSchema", s12);
    json_array_append(tools, t12);

    JsonValue* t13 = json_object();
    json_object_set(t13, "name", json_string("destroy_model"));
    json_object_set(t13, "description", json_string("Destroy/clear the currently loaded model"));
    JsonValue* s13 = json_object();
    json_object_set(s13, "type", json_string("object"));
    JsonValue* props13 = json_object();
    json_object_set(s13, "properties", props13);
    json_object_set(t13, "inputSchema", s13);
    json_array_append(tools, t13);

    JsonValue* t14 = json_object();
    json_object_set(t14, "name", json_string("remove_layer"));
    json_object_set(t14, "description", json_string("Remove a layer from the model by index (cannot remove input layer at index 0)"));
    JsonValue* s14 = json_object();
    json_object_set(s14, "type", json_string("object"));
    JsonValue* props14 = json_object();
    JsonValue* p_li_rem = json_object();
    json_object_set(p_li_rem, "type", json_string("number"));
    json_object_set(p_li_rem, "description", json_string("Index of the layer to remove (0-based)"));
    json_object_set(props14, "layer_index", p_li_rem);
    json_object_set(s14, "properties", props14);
    JsonValue* req14 = json_array();
    json_array_append(req14, json_string("layer_index"));
    json_object_set(s14, "required", req14);
    json_object_set(t14, "inputSchema", s14);
    json_array_append(tools, t14);

    JsonValue* result = json_object();
    json_object_set(result, "tools", tools);
    send_result(id, result);
}

static void handle_resources_list(int id) {
    JsonValue* resources = json_array();

    JsonValue* r1 = json_object();
    json_object_set(r1, "uri", json_string("model://info"));
    json_object_set(r1, "name", json_string("Model Information"));
    json_object_set(r1, "description", json_string("Information about the currently loaded Zeki model"));
    json_object_set(r1, "mimeType", json_string("application/json"));
    json_array_append(resources, r1);

    JsonValue* r2 = json_object();
    json_object_set(r2, "uri", json_string("model://layers"));
    json_object_set(r2, "name", json_string("Layer List"));
    json_object_set(r2, "description", json_string("List of all layers in the model"));
    json_object_set(r2, "mimeType", json_string("application/json"));
    json_array_append(resources, r2);

    JsonValue* r3 = json_object();
    json_object_set(r3, "uri", json_string("model://layers/0"));
    json_object_set(r3, "name", json_string("First Layer Info"));
    json_object_set(r3, "description", json_string("Information about the first layer"));
    json_object_set(r3, "mimeType", json_string("application/json"));
    json_array_append(resources, r3);

    JsonValue* r4 = json_object();
    json_object_set(r4, "uri", json_string("model://weights"));
    json_object_set(r4, "name", json_string("Model Weights"));
    json_object_set(r4, "description", json_string("Access layer weights by index: model://weights/0"));
    json_object_set(r4, "mimeType", json_string("application/json"));
    json_array_append(resources, r4);

    JsonValue* result = json_object();
    json_object_set(result, "resources", resources);
    send_result(id, result);
}

static Layer* get_layer_by_index(int idx) {
    if (!g_model || !*g_model) return NULL;
    Layer* curr = (*g_model)->input_layer;
    int i = 0;
    while (curr) {
        if (i == idx) return curr;
        curr = curr->next;
        i++;
    }
    return NULL;
}

static void handle_resources_read(int id, JsonValue* params) {
    JsonValue* uri_val = json_object_get(params, "uri");
    if (!uri_val || uri_val->type != JSON_STRING) {
        send_error(id, -32602, "Missing or invalid 'uri' parameter");
        return;
    }
    const char* uri = uri_val->str_val;

    if (strcmp(uri, "model://info") == 0) {
        if (!g_model || !*g_model) {
            send_error(id, -32000, "No model loaded");
            return;
        }
        Model* m = *g_model;
        JsonValue* info = json_object();
        json_object_set(info, "num_layers", json_number(m->num_layers));
        json_object_set(info, "loss_type", json_string(
            m->loss_type == LOSS_MSE ? "MSE" :
            m->loss_type == LOSS_CROSSENTROPY ? "CrossEntropy" :
            m->loss_type == LOSS_BINARY_CROSSENTROPY ? "BinaryCrossEntropy" : "Huber"));
        if (m->optimizer) {
            json_object_set(info, "optimizer", json_string(
                m->optimizer->type == OPT_ADAM ? "Adam" :
                m->optimizer->type == OPT_SGD ? "SGD" : "RMSprop"));
            json_object_set(info, "learning_rate", json_number(m->optimizer->learning_rate));
        }
        if (m->input_layer && m->input_layer->output) {
            JsonValue* shape_arr = json_array();
            for (int i = 0; i < m->input_layer->output->ndim; i++)
                json_array_append(shape_arr, json_number(m->input_layer->output->shape[i]));
            json_object_set(info, "input_shape", shape_arr);
        }
        char* infojson = json_serialize(info);
        JsonValue* contents = json_array();
        JsonValue* c = json_object();
        json_object_set(c, "uri", json_string(uri));
        json_object_set(c, "mimeType", json_string("application/json"));
        json_object_set(c, "text", json_string(infojson));
        json_array_append(contents, c);
        JsonValue* result = json_object();
        json_object_set(result, "contents", contents);
        send_result(id, result);
        free(infojson);
        json_free(info);
        return;
    }

    if (strcmp(uri, "model://layers") == 0) {
        if (!g_model || !*g_model) {
            send_error(id, -32000, "No model loaded");
            return;
        }
        JsonValue* layers_arr = json_array();
        Layer* curr = (*g_model)->input_layer;
        while (curr) {
            JsonValue* l = json_object();
            json_object_set(l, "name", json_string(curr->name));
            json_object_set(l, "type", json_string(
                curr->type == LAYER_DENSE ? "Dense" :
                curr->type == LAYER_ACTIVATION ? "Activation" :
                curr->type == LAYER_DROPOUT ? "Dropout" :
                curr->type == LAYER_INPUT ? "Input" :
                curr->type == LAYER_CONV2D ? "Conv2D" :
                curr->type == LAYER_LSTM ? "LSTM" :
                curr->type == LAYER_GRU ? "GRU" :
                curr->type == LAYER_FLATTEN ? "Flatten" :
                curr->type == LAYER_BATCHNORM ? "BatchNorm" :
                curr->type == LAYER_ATTENTION ? "Attention" :
                curr->type == LAYER_TRANSFORMER ? "Transformer" : "Unknown"));
        if (curr->output) {
            JsonValue* sh = json_array();
            for (int i = 0; i < curr->output->ndim; i++)
                json_array_append(sh, json_number(curr->output->shape[i]));
            json_object_set(l, "output_shape", sh);
        }
        if (curr->params) {
            JsonValue* p = json_object();
            if (curr->type == LAYER_DENSE) {
                DenseParams* dp = (DenseParams*)curr->params;
                json_object_set(p, "input_dim", json_number(dp->input_dim));
                json_object_set(p, "output_dim", json_number(dp->output_dim));
                json_object_set(l, "input_dim", json_number(dp->input_dim));
                json_object_set(l, "output_dim", json_number(dp->output_dim));
            } else if (curr->type == LAYER_CONV2D) {
                ConvParams* cp = (ConvParams*)curr->params;
                json_object_set(p, "in_channels", json_number(cp->input_channels));
                json_object_set(p, "out_channels", json_number(cp->output_channels));
                json_object_set(p, "kernel_size", json_number(cp->kernel_size));
                json_object_set(p, "stride", json_number(cp->stride));
                json_object_set(p, "padding", json_number(cp->padding));
                json_object_set(l, "input_dim", json_number(cp->input_channels));
                json_object_set(l, "output_dim", json_number(cp->output_channels));
            } else if (curr->type == LAYER_DROPOUT) {
                DropoutParams* dp = (DropoutParams*)curr->params;
                json_object_set(p, "rate", json_number(dp->rate));
            } else if (curr->type == LAYER_LSTM) {
                LSTMCellParams* lp = (LSTMCellParams*)curr->params;
                json_object_set(p, "input_dim", json_number(lp->input_dim));
                json_object_set(p, "hidden_dim", json_number(lp->hidden_dim));
                json_object_set(l, "input_dim", json_number(lp->input_dim));
                json_object_set(l, "output_dim", json_number(lp->hidden_dim));
            } else if (curr->type == LAYER_GRU) {
                GRUCellParams* gp = (GRUCellParams*)curr->params;
                json_object_set(p, "input_dim", json_number(gp->input_dim));
                json_object_set(p, "hidden_dim", json_number(gp->hidden_dim));
                json_object_set(l, "input_dim", json_number(gp->input_dim));
                json_object_set(l, "output_dim", json_number(gp->hidden_dim));
            } else if (curr->type == LAYER_BATCHNORM) {
                BatchNormParams* bp = (BatchNormParams*)curr->params;
                json_object_set(p, "dim", json_number(bp->dim));
                json_object_set(l, "input_dim", json_number(bp->dim));
                json_object_set(l, "output_dim", json_number(bp->dim));
            } else if (curr->type == LAYER_ATTENTION) {
                AttentionParams* ap = (AttentionParams*)curr->params;
                json_object_set(p, "embed_dim", json_number(ap->embed_dim));
                json_object_set(l, "input_dim", json_number(ap->embed_dim));
                json_object_set(l, "output_dim", json_number(ap->embed_dim));
            } else if (curr->type == LAYER_TRANSFORMER) {
                TransformerBlockParams* tp = (TransformerBlockParams*)curr->params;
                json_object_set(p, "embed_dim", json_number(tp->embed_dim));
                json_object_set(p, "num_heads", json_number(tp->num_heads));
                json_object_set(p, "ff_dim", json_number(tp->ff_dim));
                json_object_set(l, "input_dim", json_number(tp->embed_dim));
                json_object_set(l, "output_dim", json_number(tp->embed_dim));
            } else if (curr->type == LAYER_ACTIVATION) {
                ActivationParams* ap = (ActivationParams*)curr->params;
                const char* act_type = ap->type == ACT_RELU ? "relu" :
                                       ap->type == ACT_SIGMOID ? "sigmoid" :
                                       ap->type == ACT_TANH ? "tanh" : "softmax";
                json_object_set(p, "activation", json_string(act_type));
            }
            if (p->object.count > 0) { json_object_set(l, "params", p); }
            else { json_free(p); }
        }
        json_array_append(layers_arr, l);
            curr = curr->next;
        }
        char* ljson = json_serialize(layers_arr);
        JsonValue* contents = json_array();
        JsonValue* c = json_object();
        json_object_set(c, "uri", json_string(uri));
        json_object_set(c, "mimeType", json_string("application/json"));
        json_object_set(c, "text", json_string(ljson));
        json_array_append(contents, c);
        JsonValue* result = json_object();
        json_object_set(result, "contents", contents);
        send_result(id, result);
        free(ljson);
        json_free(layers_arr);
        return;
    }

    if (strncmp(uri, "model://layers/", 15) == 0) {
        if (!g_model || !*g_model) {
            send_error(id, -32000, "No model loaded");
            return;
        }
        int idx = atoi(uri + 15);
        Layer* layer = get_layer_by_index(idx);
        if (!layer) {
            send_error(id, -32000, "Layer not found");
            return;
        }
        JsonValue* l = json_object();
        json_object_set(l, "name", json_string(layer->name));
        json_object_set(l, "type", json_string(
            layer->type == LAYER_DENSE ? "Dense" :
            layer->type == LAYER_ACTIVATION ? "Activation" :
            layer->type == LAYER_DROPOUT ? "Dropout" :
            layer->type == LAYER_INPUT ? "Input" :
            layer->type == LAYER_CONV2D ? "Conv2D" :
            layer->type == LAYER_LSTM ? "LSTM" :
            layer->type == LAYER_GRU ? "GRU" :
            layer->type == LAYER_FLATTEN ? "Flatten" :
            layer->type == LAYER_BATCHNORM ? "BatchNorm" :
            layer->type == LAYER_ATTENTION ? "Attention" :
            layer->type == LAYER_TRANSFORMER ? "Transformer" : "Unknown"));
        if (layer->output) {
            JsonValue* sh = json_array();
            for (int i = 0; i < layer->output->ndim; i++)
                json_array_append(sh, json_number(layer->output->shape[i]));
            json_object_set(l, "output_shape", sh);
        }
        if (layer->params) {
            JsonValue* p = json_object();
            if (layer->type == LAYER_DENSE) {
                DenseParams* dp = (DenseParams*)layer->params;
                json_object_set(p, "input_dim", json_number(dp->input_dim));
                json_object_set(p, "output_dim", json_number(dp->output_dim));
                json_object_set(l, "input_dim", json_number(dp->input_dim));
                json_object_set(l, "output_dim", json_number(dp->output_dim));
            } else if (layer->type == LAYER_CONV2D) {
                ConvParams* cp = (ConvParams*)layer->params;
                json_object_set(p, "in_channels", json_number(cp->input_channels));
                json_object_set(p, "out_channels", json_number(cp->output_channels));
                json_object_set(p, "kernel_size", json_number(cp->kernel_size));
                json_object_set(p, "stride", json_number(cp->stride));
                json_object_set(p, "padding", json_number(cp->padding));
                json_object_set(l, "input_dim", json_number(cp->input_channels));
                json_object_set(l, "output_dim", json_number(cp->output_channels));
            } else if (layer->type == LAYER_DROPOUT) {
                DropoutParams* dp = (DropoutParams*)layer->params;
                json_object_set(p, "rate", json_number(dp->rate));
            } else if (layer->type == LAYER_LSTM) {
                LSTMCellParams* lp = (LSTMCellParams*)layer->params;
                json_object_set(p, "input_dim", json_number(lp->input_dim));
                json_object_set(p, "hidden_dim", json_number(lp->hidden_dim));
                json_object_set(l, "input_dim", json_number(lp->input_dim));
                json_object_set(l, "output_dim", json_number(lp->hidden_dim));
            } else if (layer->type == LAYER_GRU) {
                GRUCellParams* gp = (GRUCellParams*)layer->params;
                json_object_set(p, "input_dim", json_number(gp->input_dim));
                json_object_set(p, "hidden_dim", json_number(gp->hidden_dim));
                json_object_set(l, "input_dim", json_number(gp->input_dim));
                json_object_set(l, "output_dim", json_number(gp->hidden_dim));
            } else if (layer->type == LAYER_BATCHNORM) {
                BatchNormParams* bp = (BatchNormParams*)layer->params;
                json_object_set(p, "dim", json_number(bp->dim));
                json_object_set(l, "input_dim", json_number(bp->dim));
                json_object_set(l, "output_dim", json_number(bp->dim));
            } else if (layer->type == LAYER_ATTENTION) {
                AttentionParams* ap = (AttentionParams*)layer->params;
                json_object_set(p, "embed_dim", json_number(ap->embed_dim));
                json_object_set(l, "output_dim", json_number(ap->embed_dim));
            } else if (layer->type == LAYER_TRANSFORMER) {
                TransformerBlockParams* tp = (TransformerBlockParams*)layer->params;
                json_object_set(p, "embed_dim", json_number(tp->embed_dim));
                json_object_set(p, "num_heads", json_number(tp->num_heads));
                json_object_set(p, "ff_dim", json_number(tp->ff_dim));
                json_object_set(l, "output_dim", json_number(tp->embed_dim));
            } else if (layer->type == LAYER_ACTIVATION) {
                ActivationParams* ap = (ActivationParams*)layer->params;
                const char* act_type = ap->type == ACT_RELU ? "relu" :
                                       ap->type == ACT_SIGMOID ? "sigmoid" :
                                       ap->type == ACT_TANH ? "tanh" : "softmax";
                json_object_set(p, "activation", json_string(act_type));
            }
            if (p->object.count > 0) { json_object_set(l, "params", p); }
            else { json_free(p); }
        }
        if (layer->weights) {
            json_object_set(l, "weights_shape", json_string(
                layer->weights->ndim == 1 ? "1D" :
                layer->weights->ndim == 2 ? "2D" :
                layer->weights->ndim == 3 ? "3D" : "4D"));
            json_object_set(l, "weights_size", json_number(layer->weights->size));
        }
        if (layer->biases) {
            json_object_set(l, "biases_size", json_number(layer->biases->size));
        }
        char* ljson = json_serialize(l);
        JsonValue* contents = json_array();
        JsonValue* c = json_object();
        json_object_set(c, "uri", json_string(uri));
        json_object_set(c, "mimeType", json_string("application/json"));
        json_object_set(c, "text", json_string(ljson));
        json_array_append(contents, c);
        JsonValue* result = json_object();
        json_object_set(result, "contents", contents);
        send_result(id, result);
        free(ljson);
        json_free(l);
        return;
    }

    if (strncmp(uri, "model://weights/", 16) == 0) {
        if (!g_model || !*g_model) {
            send_error(id, -32000, "No model loaded");
            return;
        }
        int idx = atoi(uri + 16);
        Layer* layer = get_layer_by_index(idx);
        if (!layer || !layer->weights) {
            send_error(id, -32000, "Layer weights not found");
            return;
        }
        JsonValue* w = json_object();
        JsonValue* data_arr = tensor_to_json_array(layer->weights);
        json_object_set(w, "data", data_arr);
        JsonValue* shape_arr = json_array();
        for (int i = 0; i < layer->weights->ndim; i++)
            json_array_append(shape_arr, json_number(layer->weights->shape[i]));
        json_object_set(w, "shape", shape_arr);
        char* wjson = json_serialize(w);
        JsonValue* contents = json_array();
        JsonValue* c = json_object();
        json_object_set(c, "uri", json_string(uri));
        json_object_set(c, "mimeType", json_string("application/json"));
        json_object_set(c, "text", json_string(wjson));
        json_array_append(contents, c);
        JsonValue* result = json_object();
        json_object_set(result, "contents", contents);
        send_result(id, result);
        free(wjson);
        json_free(w);
        return;
    }

    send_error(id, -32602, "Resource not found");
}

static void handle_prompts_list(int id) {
    JsonValue* prompts = json_array();

    JsonValue* p1 = json_object();
    json_object_set(p1, "name", json_string("predict"));
    json_object_set(p1, "description", json_string("Generate a prompt for making a prediction with the Zeki model"));
    JsonValue* args1 = json_array();
    JsonValue* a1 = json_object();
    json_object_set(a1, "name", json_string("input_description"));
    json_object_set(a1, "description", json_string("Description of the input data for prediction"));
    json_object_set(a1, "required", json_bool(1));
    json_array_append(args1, a1);
    json_object_set(p1, "arguments", args1);
    json_array_append(prompts, p1);

    JsonValue* p2 = json_object();
    json_object_set(p2, "name", json_string("train"));
    json_object_set(p2, "description", json_string("Generate a prompt for training the Zeki model"));
    JsonValue* args2 = json_array();
    JsonValue* a2 = json_object();
    json_object_set(a2, "name", json_string("task_description"));
    json_object_set(a2, "description", json_string("Description of the training task"));
    json_object_set(a2, "required", json_bool(1));
    json_array_append(args2, a2);
    JsonValue* a2b = json_object();
    json_object_set(a2b, "name", json_string("epochs"));
    json_object_set(a2b, "description", json_string("Number of training epochs"));
    json_object_set(a2b, "required", json_bool(0));
    json_array_append(args2, a2b);
    json_object_set(p2, "arguments", args2);
    json_array_append(prompts, p2);

    JsonValue* result = json_object();
    json_object_set(result, "prompts", prompts);
    send_result(id, result);
}

static void handle_prompts_get(int id, JsonValue* params) {
    JsonValue* name_val = json_object_get(params, "name");
    if (!name_val || name_val->type != JSON_STRING) {
        send_error(id, -32602, "Missing or invalid 'name' parameter");
        return;
    }
    const char* name = name_val->str_val;

    JsonValue* arguments = json_object_get(params, "arguments");

    if (strcmp(name, "predict") == 0) {
        const char* input_desc = "unknown input";
        if (arguments) {
            JsonValue* id_val = json_object_get(arguments, "input_description");
            if (id_val && id_val->type == JSON_STRING) input_desc = id_val->str_val;
        }
        char prompt[2048];
        snprintf(prompt, sizeof(prompt),
            "I want to use the Zeki AI model to make a prediction. "
            "Input description: %s. "
            "Please use the predict tool with the appropriate input data.",
            input_desc);

        JsonValue* messages = json_array();
        JsonValue* msg = json_object();
        json_object_set(msg, "role", json_string("user"));
        JsonValue* content = json_object();
        json_object_set(content, "type", json_string("text"));
        json_object_set(content, "text", json_string(prompt));
        json_object_set(msg, "content", content);
        json_array_append(messages, msg);

        JsonValue* result = json_object();
        json_object_set(result, "messages", messages);
        json_object_set(result, "description", json_string("Prediction prompt template"));
        send_result(id, result);
        return;
    }

    if (strcmp(name, "train") == 0) {
        const char* task_desc = "unknown task";
        const char* epochs = "10";
        if (arguments) {
            JsonValue* td = json_object_get(arguments, "task_description");
            if (td && td->type == JSON_STRING) task_desc = td->str_val;
            JsonValue* ep = json_object_get(arguments, "epochs");
            if (ep && ep->type == JSON_STRING) epochs = ep->str_val;
        }
        char prompt[2048];
        snprintf(prompt, sizeof(prompt),
            "I want to train the Zeki AI model. "
            "Task: %s. "
            "Epochs: %s. "
            "Please use the train tool with the appropriate training data.",
            task_desc, epochs);

        JsonValue* messages = json_array();
        JsonValue* msg = json_object();
        json_object_set(msg, "role", json_string("user"));
        JsonValue* content = json_object();
        json_object_set(content, "type", json_string("text"));
        json_object_set(content, "text", json_string(prompt));
        json_object_set(msg, "content", content);
        json_array_append(messages, msg);

        JsonValue* result = json_object();
        json_object_set(result, "messages", messages);
        json_object_set(result, "description", json_string("Training prompt template"));
        send_result(id, result);
        return;
    }

    send_error(id, -32602, "Prompt not found");
}

static void handle_tool_predict(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded. Use load_model or create_model first.");
        return;
    }
    int input_len = 0, shape_len = 0;
    float* input_data = extract_float_array(args, "input_data", &input_len);
    int* input_shape = extract_int_array(args, "input_shape", &shape_len);
    if (!input_data || !input_shape || shape_len < 1) {
        free(input_data); free(input_shape);
        send_error(id, -32602, "Invalid or missing input_data or input_shape");
        return;
    }
    Tensor* input = tensor_create(input_shape, shape_len);
    int copy_len = input->size < input_len ? input->size : input_len;
    memcpy(input->data, input_data, copy_len * sizeof(float));
    free(input_data); free(input_shape);

    model_predict(*g_model, input);

    Tensor* out = (*g_model)->output_layer->output;
    JsonValue* result_data = tensor_to_json_array(out);
    JsonValue* result_shape = json_array();
    for (int i = 0; i < out->ndim; i++)
        json_array_append(result_shape, json_number(out->shape[i]));

    JsonValue* result_obj = json_object();
    json_object_set(result_obj, "output_data", result_data);
    json_object_set(result_obj, "output_shape", result_shape);

    char* text = json_serialize(result_obj);
    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);

    free(text);
    json_free(result_obj);
    tensor_free(input);
}

static void handle_tool_train_step(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    int in_len = 0, tar_len = 0, ish_len = 0, tsh_len = 0;
    float* in_data = extract_float_array(args, "input_data", &in_len);
    float* tar_data = extract_float_array(args, "target_data", &tar_len);
    int* in_shape = extract_int_array(args, "input_shape", &ish_len);
    int* tar_shape = extract_int_array(args, "target_shape", &tsh_len);
    if (!in_data || !tar_data || !in_shape || !tar_shape) {
        free(in_data); free(tar_data); free(in_shape); free(tar_shape);
        send_error(id, -32602, "Missing required parameters");
        return;
    }

    Tensor* x = tensor_create(in_shape, ish_len);
    Tensor* y = tensor_create(tar_shape, tsh_len);
    int cp_x = x->size < in_len ? x->size : in_len;
    int cp_y = y->size < tar_len ? y->size : tar_len;
    memcpy(x->data, in_data, cp_x * sizeof(float));
    memcpy(y->data, tar_data, cp_y * sizeof(float));
    free(in_data); free(tar_data); free(in_shape); free(tar_shape);

    model_train_step(*g_model, x, y);

    float loss = loss_compute((*g_model)->output_layer->output, y, (*g_model)->loss_type);
    char result_text[128];
    snprintf(result_text, sizeof(result_text), "{\"loss\": %.6f}", (double)loss);

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(result_text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);

    tensor_free(x);
    tensor_free(y);
}

static void handle_tool_load_model(int id, JsonValue* args) {
    JsonValue* fn = json_object_get(args, "filename");
    if (!fn || fn->type != JSON_STRING) {
        send_error(id, -32602, "Missing 'filename' parameter");
        return;
    }
    Model* new_model = model_load(fn->str_val);
    if (!new_model) {
        send_error(id, -32000, "Failed to load model from file");
        return;
    }
    if (*g_model) model_free(*g_model);
    *g_model = new_model;

    JsonValue* content_arr = json_array();
    char text[256];
    snprintf(text, sizeof(text), "{\"status\": \"loaded\", \"layers\": %d, \"file\": \"%s\"}",
             new_model->num_layers, fn->str_val);
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
}

static void handle_tool_save_model(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    JsonValue* fn = json_object_get(args, "filename");
    if (!fn || fn->type != JSON_STRING) {
        send_error(id, -32602, "Missing 'filename' parameter");
        return;
    }
    int ok = model_save(*g_model, fn->str_val, FORMAT_BINARY);
    char text[256];
    snprintf(text, sizeof(text), "{\"status\": \"%s\", \"file\": \"%s\"}",
             ok ? "saved" : "failed", fn->str_val);
    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
}

static void handle_tool_get_model_info(int id, JsonValue* args) {
    (void)args;
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    Model* m = *g_model;
    JsonValue* info = json_object();
    json_object_set(info, "num_layers", json_number(m->num_layers));
    json_object_set(info, "model_name", json_string("Zeki AI Model"));
    if (m->input_layer && m->input_layer->output) {
        JsonValue* sh = json_array();
        for (int i = 0; i < m->input_layer->output->ndim; i++)
            json_array_append(sh, json_number(m->input_layer->output->shape[i]));
        json_object_set(info, "input_shape", sh);
    }
    if (m->optimizer) {
        json_object_set(info, "optimizer", json_string(
            m->optimizer->type == OPT_ADAM ? "Adam" :
            m->optimizer->type == OPT_SGD ? "SGD" : "RMSprop"));
        json_object_set(info, "learning_rate", json_number(m->optimizer->learning_rate));
    }
    json_object_set(info, "loss_type", json_string(
        m->loss_type == LOSS_MSE ? "MSE" :
        m->loss_type == LOSS_CROSSENTROPY ? "CrossEntropy" :
        m->loss_type == LOSS_BINARY_CROSSENTROPY ? "BinaryCrossEntropy" : "Huber"));

    Layer* curr = m->input_layer;
    int layer_idx = 0;
    JsonValue* layers_arr = json_array();
    while (curr) {
        JsonValue* l = json_object();
        json_object_set(l, "index", json_number(layer_idx));
        json_object_set(l, "name", json_string(curr->name));
        json_object_set(l, "type", json_string(
            curr->type == LAYER_DENSE ? "Dense" :
            curr->type == LAYER_ACTIVATION ? "Activation" :
            curr->type == LAYER_DROPOUT ? "Dropout" :
            curr->type == LAYER_INPUT ? "Input" :
            curr->type == LAYER_CONV2D ? "Conv2D" :
            curr->type == LAYER_LSTM ? "LSTM" :
            curr->type == LAYER_GRU ? "GRU" :
            curr->type == LAYER_FLATTEN ? "Flatten" :
            curr->type == LAYER_BATCHNORM ? "BatchNorm" :
            curr->type == LAYER_ATTENTION ? "Attention" :
            curr->type == LAYER_TRANSFORMER ? "Transformer" : "Unknown"));
        if (curr->output) {
            JsonValue* sh = json_array();
            for (int i = 0; i < curr->output->ndim; i++)
                json_array_append(sh, json_number(curr->output->shape[i]));
            json_object_set(l, "output_shape", sh);
        }
        json_array_append(layers_arr, l);
        curr = curr->next;
        layer_idx++;
    }
    json_object_set(info, "layers", layers_arr);

    char* text = json_serialize(info);
    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
    free(text);
    json_free(info);
}

static void handle_tool_list_layers(int id, JsonValue* args) {
    (void)args;
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    JsonValue* layers_arr = json_array();
    Layer* curr = (*g_model)->input_layer;
    int idx = 0;
    while (curr) {
        JsonValue* l = json_object();
        json_object_set(l, "index", json_number(idx));
        json_object_set(l, "name", json_string(curr->name));
        json_object_set(l, "type", json_string(
            curr->type == LAYER_DENSE ? "Dense" :
            curr->type == LAYER_ACTIVATION ? "Activation" :
            curr->type == LAYER_DROPOUT ? "Dropout" :
            curr->type == LAYER_INPUT ? "Input" :
            curr->type == LAYER_CONV2D ? "Conv2D" :
            curr->type == LAYER_LSTM ? "LSTM" :
            curr->type == LAYER_GRU ? "GRU" :
            curr->type == LAYER_FLATTEN ? "Flatten" :
            curr->type == LAYER_BATCHNORM ? "BatchNorm" :
            curr->type == LAYER_ATTENTION ? "Attention" :
            curr->type == LAYER_TRANSFORMER ? "Transformer" : "Unknown"));
        if (curr->output) {
            JsonValue* sh = json_array();
            for (int i = 0; i < curr->output->ndim; i++)
                json_array_append(sh, json_number(curr->output->shape[i]));
            json_object_set(l, "output_shape", sh);
        }
        if (curr->params) {
            JsonValue* p = json_object();
            if (curr->type == LAYER_DENSE) {
                DenseParams* dp = (DenseParams*)curr->params;
                json_object_set(p, "input_dim", json_number(dp->input_dim));
                json_object_set(p, "output_dim", json_number(dp->output_dim));
                json_object_set(l, "input_dim", json_number(dp->input_dim));
                json_object_set(l, "output_dim", json_number(dp->output_dim));
            } else if (curr->type == LAYER_CONV2D) {
                ConvParams* cp = (ConvParams*)curr->params;
                json_object_set(p, "in_channels", json_number(cp->input_channels));
                json_object_set(p, "out_channels", json_number(cp->output_channels));
                json_object_set(p, "kernel_size", json_number(cp->kernel_size));
                json_object_set(p, "stride", json_number(cp->stride));
                json_object_set(p, "padding", json_number(cp->padding));
                json_object_set(l, "input_dim", json_number(cp->input_channels));
                json_object_set(l, "output_dim", json_number(cp->output_channels));
            } else if (curr->type == LAYER_DROPOUT) {
                DropoutParams* dp = (DropoutParams*)curr->params;
                json_object_set(p, "rate", json_number(dp->rate));
            } else if (curr->type == LAYER_LSTM) {
                LSTMCellParams* lp = (LSTMCellParams*)curr->params;
                json_object_set(p, "input_dim", json_number(lp->input_dim));
                json_object_set(p, "hidden_dim", json_number(lp->hidden_dim));
                json_object_set(l, "input_dim", json_number(lp->input_dim));
                json_object_set(l, "output_dim", json_number(lp->hidden_dim));
            } else if (curr->type == LAYER_GRU) {
                GRUCellParams* gp = (GRUCellParams*)curr->params;
                json_object_set(p, "input_dim", json_number(gp->input_dim));
                json_object_set(p, "hidden_dim", json_number(gp->hidden_dim));
                json_object_set(l, "input_dim", json_number(gp->input_dim));
                json_object_set(l, "output_dim", json_number(gp->hidden_dim));
            } else if (curr->type == LAYER_BATCHNORM) {
                BatchNormParams* bp = (BatchNormParams*)curr->params;
                json_object_set(p, "dim", json_number(bp->dim));
                json_object_set(l, "input_dim", json_number(bp->dim));
                json_object_set(l, "output_dim", json_number(bp->dim));
            } else if (curr->type == LAYER_ATTENTION) {
                AttentionParams* ap = (AttentionParams*)curr->params;
                json_object_set(p, "embed_dim", json_number(ap->embed_dim));
                json_object_set(l, "input_dim", json_number(ap->embed_dim));
                json_object_set(l, "output_dim", json_number(ap->embed_dim));
            } else if (curr->type == LAYER_TRANSFORMER) {
                TransformerBlockParams* tp = (TransformerBlockParams*)curr->params;
                json_object_set(p, "embed_dim", json_number(tp->embed_dim));
                json_object_set(p, "num_heads", json_number(tp->num_heads));
                json_object_set(p, "ff_dim", json_number(tp->ff_dim));
                json_object_set(l, "input_dim", json_number(tp->embed_dim));
                json_object_set(l, "output_dim", json_number(tp->embed_dim));
            } else if (curr->type == LAYER_ACTIVATION) {
                ActivationParams* ap = (ActivationParams*)curr->params;
                const char* act_type = ap->type == ACT_RELU ? "relu" :
                                       ap->type == ACT_SIGMOID ? "sigmoid" :
                                       ap->type == ACT_TANH ? "tanh" : "softmax";
                json_object_set(p, "activation", json_string(act_type));
            }
            if (p->object.count > 0) { json_object_set(l, "params", p); }
            else { json_free(p); }
        }
        json_array_append(layers_arr, l);
        curr = curr->next;
        idx++;
    }
    char* text = json_serialize(layers_arr);
    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
    free(text);
    json_free(layers_arr);
}

static void handle_tool_evaluate(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    int in_len = 0, tar_len = 0, ish_len = 0, tsh_len = 0;
    float* in_data = extract_float_array(args, "input_data", &in_len);
    float* tar_data = extract_float_array(args, "target_data", &tar_len);
    int* in_shape = extract_int_array(args, "input_shape", &ish_len);
    int* tar_shape = extract_int_array(args, "target_shape", &tsh_len);
    if (!in_data || !tar_data || !in_shape || !tar_shape) {
        free(in_data); free(tar_data); free(in_shape); free(tar_shape);
        send_error(id, -32602, "Missing required parameters");
        return;
    }

    Tensor* x = tensor_create(in_shape, ish_len);
    Tensor* y = tensor_create(tar_shape, tsh_len);
    int cp_x = x->size < in_len ? x->size : in_len;
    int cp_y = y->size < tar_len ? y->size : tar_len;
    memcpy(x->data, in_data, cp_x * sizeof(float));
    memcpy(y->data, tar_data, cp_y * sizeof(float));
    free(in_data); free(tar_data); free(in_shape); free(tar_shape);

    float loss = model_evaluate(*g_model, x, y);
    float acc = model_accuracy(*g_model, x, y);

    char text[256];
    snprintf(text, sizeof(text), "{\"loss\": %.6f, \"accuracy\": %.4f}", (double)loss, (double)acc);

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);

    tensor_free(x);
    tensor_free(y);
}

static void handle_tool_create_model(int id, JsonValue* args) {
    int shape_len = 0;
    int* input_shape = extract_int_array(args, "input_shape", &shape_len);
    if (!input_shape || shape_len < 1) {
        free(input_shape);
        send_error(id, -32602, "Invalid or missing 'input_shape'");
        return;
    }

    Layer* input = layer_create(LAYER_INPUT, "Input");
    input->output = tensor_create(input_shape, shape_len);
    free(input_shape);

    Model* new_model = model_create(input);

    if (*g_model) model_free(*g_model);
    *g_model = new_model;

    char text[128];
    snprintf(text, sizeof(text), "{\"status\": \"created\", \"layers\": 1}");

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
}

static void handle_tool_destroy_model(int id, JsonValue* args) {
    (void)args;
    if (g_model && *g_model) {
        model_free(*g_model);
        *g_model = NULL;
    }
    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text("{\"status\": \"destroyed\"}"));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
}

static void handle_tool_remove_layer(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    JsonValue* li_val = json_object_get(args, "layer_index");
    if (!li_val || li_val->type != JSON_NUMBER) {
        send_error(id, -32602, "Missing or invalid 'layer_index'");
        return;
    }
    int idx = (int)li_val->num_val;

    if (idx == 0) {
        send_error(id, -32000, "Cannot remove the input layer (index 0)");
        return;
    }

    if (idx < 0 || idx >= (*g_model)->num_layers) {
        send_error(id, -32000, "Layer index out of range");
        return;
    }

    model_remove_layer(*g_model, idx);

    char text[128];
    snprintf(text, sizeof(text), "{\"status\": \"removed\", \"layer_index\": %d, \"total_layers\": %d}",
             idx, (*g_model)->num_layers);

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
}

static void handle_tool_add_layer(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded. Use create_model first.");
        return;
    }
    JsonValue* lt_val = json_object_get(args, "layer_type");
    if (!lt_val || lt_val->type != JSON_STRING) {
        send_error(id, -32602, "Missing or invalid 'layer_type'");
        return;
    }
    const char* layer_type = lt_val->str_val;

    JsonValue* name_val = json_object_get(args, "name");
    const char* layer_name = (name_val && name_val->type == JSON_STRING) ? name_val->str_val : layer_type;

    JsonValue* params = json_object_get(args, "params");

    Layer* layer = NULL;

    if (strcmp(layer_type, "dense") == 0) {
        if (!params) { send_error(id, -32602, "dense layer requires 'params' with input_dim and output_dim"); return; }
        JsonValue* idim = json_object_get(params, "input_dim");
        JsonValue* odim = json_object_get(params, "output_dim");
        if (!idim || !odim) { send_error(id, -32602, "dense layer requires input_dim and output_dim"); return; }
        int idim_val = (int)idim->num_val;
        int odim_val = (int)odim->num_val;
        if (idim_val <= 0) { send_error(id, -32000, "input_dim must be positive"); return; }
        if (odim_val <= 0) { send_error(id, -32000, "output_dim must be positive"); return; }
        if ((long long)idim_val * odim_val > MAX_TENSOR_SIZE) {
            char msg[192];
            snprintf(msg, sizeof(msg), "Layer too large: %d * %d = %lld elements (max %d). Reduce output_dim.", idim_val, odim_val, (long long)idim_val * odim_val, MAX_TENSOR_SIZE);
            send_error(id, -32000, msg);
            return;
        }
        layer = dense_create(idim_val, odim_val, layer_name);
    } else if (strcmp(layer_type, "relu") == 0) {
        layer = activation_create(ACT_RELU, layer_name);
    } else if (strcmp(layer_type, "sigmoid") == 0) {
        layer = activation_create(ACT_SIGMOID, layer_name);
    } else if (strcmp(layer_type, "tanh") == 0) {
        layer = activation_create(ACT_TANH, layer_name);
    } else if (strcmp(layer_type, "softmax") == 0) {
        layer = activation_create(ACT_SOFTMAX, layer_name);
    } else if (strcmp(layer_type, "dropout") == 0) {
        if (!params) { send_error(id, -32602, "dropout layer requires 'params' with rate"); return; }
        JsonValue* rate = json_object_get(params, "rate");
        if (!rate) { send_error(id, -32602, "dropout requires 'rate' parameter"); return; }
        layer = dropout_create((float)rate->num_val, layer_name);
    } else if (strcmp(layer_type, "flatten") == 0) {
        layer = flatten_create(layer_name);
    } else if (strcmp(layer_type, "conv2d") == 0) {
        if (!params) { send_error(id, -32602, "conv2d requires params"); return; }
        JsonValue* ic = json_object_get(params, "in_channels");
        JsonValue* oc = json_object_get(params, "out_channels");
        JsonValue* ks = json_object_get(params, "kernel_size");
        JsonValue* st = json_object_get(params, "stride");
        JsonValue* pd = json_object_get(params, "padding");
        if (!ic || !oc || !ks) { send_error(id, -32602, "conv2d requires in_channels, out_channels, kernel_size"); return; }
        layer = conv2d_create((int)ic->num_val, (int)oc->num_val, (int)ks->num_val,
                              st ? (int)st->num_val : 1, pd ? (int)pd->num_val : 0, layer_name);
    } else if (strcmp(layer_type, "lstm") == 0) {
        if (!params) { send_error(id, -32602, "lstm requires params"); return; }
        JsonValue* idim = json_object_get(params, "input_dim");
        JsonValue* hdim = json_object_get(params, "hidden_dim");
        if (!idim || !hdim) { send_error(id, -32602, "lstm requires input_dim and hidden_dim"); return; }
        layer = lstm_cell_create((int)idim->num_val, (int)hdim->num_val, layer_name);
    } else if (strcmp(layer_type, "gru") == 0) {
        if (!params) { send_error(id, -32602, "gru requires params"); return; }
        JsonValue* idim = json_object_get(params, "input_dim");
        JsonValue* hdim = json_object_get(params, "hidden_dim");
        if (!idim || !hdim) { send_error(id, -32602, "gru requires input_dim and hidden_dim"); return; }
        layer = gru_cell_create((int)idim->num_val, (int)hdim->num_val, layer_name);
    } else if (strcmp(layer_type, "batchnorm") == 0) {
        if (!params) { send_error(id, -32602, "batchnorm requires params with dim"); return; }
        JsonValue* dim = json_object_get(params, "dim");
        if (!dim) { send_error(id, -32602, "batchnorm requires 'dim' parameter"); return; }
        layer = batchnorm_create((int)dim->num_val, layer_name);
    } else if (strcmp(layer_type, "attention") == 0) {
        if (!params) { send_error(id, -32602, "attention requires params with embed_dim"); return; }
        JsonValue* ed = json_object_get(params, "embed_dim");
        if (!ed) { send_error(id, -32602, "attention requires 'embed_dim' parameter"); return; }
        layer = attention_create((int)ed->num_val, layer_name);
    } else if (strcmp(layer_type, "transformer") == 0) {
        if (!params) { send_error(id, -32602, "transformer requires params"); return; }
        JsonValue* ed = json_object_get(params, "embed_dim");
        JsonValue* nh = json_object_get(params, "num_heads");
        JsonValue* ff = json_object_get(params, "ff_dim");
        if (!ed || !nh || !ff) { send_error(id, -32602, "transformer requires embed_dim, num_heads, ff_dim"); return; }
        layer = transformer_block_create((int)ed->num_val, (int)nh->num_val, (int)ff->num_val, layer_name);
    } else {
        send_error(id, -32602, "Unknown layer type. Supported: dense, conv2d, relu, sigmoid, tanh, softmax, dropout, flatten, lstm, gru, batchnorm, attention, transformer");
        return;
    }

    if (!layer) {
        send_error(id, -32000, "Failed to create layer");
        return;
    }

    model_add_layer(*g_model, layer);

    char text[256];
    snprintf(text, sizeof(text), "{\"status\": \"added\", \"layer\": \"%s\", \"type\": \"%s\", \"total_layers\": %d}",
             layer_name, layer_type, (*g_model)->num_layers);

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
}

static void handle_tool_set_training_mode(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    JsonValue* training_val = json_object_get(args, "training");
    if (!training_val || (training_val->type != JSON_BOOL && training_val->type != JSON_NUMBER)) {
        send_error(id, -32602, "Missing or invalid 'training' parameter");
        return;
    }
    int training = training_val->type == JSON_BOOL ? training_val->bool_val : (int)training_val->num_val;
    model_set_training(*g_model, training);

    const char* mode = training ? "training" : "evaluation";
    char text[128];
    snprintf(text, sizeof(text), "{\"status\": \"ok\", \"mode\": \"%s\"}", mode);

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);
    json_free(result);
}

/* Architecture validation: trace element count through layers, return error if mismatch */
static const char* validate_arch(Model* m, int* shape, int ndim) {
    int elems = 1;
    for (int i = 1; i < ndim; i++) elems *= shape[i];

    Layer* l = m->input_layer;
    while (l) {
        if (l->type == LAYER_CONV2D || l->type == LAYER_LSTM ||
            l->type == LAYER_GRU) {
            return NULL; /* skip validation for complex layers */
        }
        if (l->type == LAYER_DENSE && l->params) {
            DenseParams* dp = (DenseParams*)l->params;
            if (elems != dp->input_dim) {
                static char err[256];
                snprintf(err, sizeof(err), "Dense(input_dim=%d) beklenen=%d, gelen=%d. Dense girişini %d yapın.",
                         dp->input_dim, dp->input_dim, elems, elems);
                return err;
            }
            elems = dp->output_dim;
        } else if ((l->type == LAYER_TRANSFORMER || l->type == LAYER_ATTENTION) && l->params) {
            int edim = (l->type == LAYER_TRANSFORMER)
                ? ((TransformerBlockParams*)l->params)->embed_dim
                : ((AttentionParams*)l->params)->embed_dim;
            if (ndim != 3) {
                if (elems % edim != 0) {
                    static char err[256];
                    snprintf(err, sizeof(err), "Transformer(embed_dim=%d): giriş özellik sayısı (%d) embed_dim'e tam bölünmüyor", edim, elems);
                    return err;
                }
            } else if (shape[ndim-1] != edim) {
                static char err[256];
                snprintf(err, sizeof(err), "Transformer(embed_dim=%d) ama giriş son boyutu %d", edim, shape[ndim-1]);
                return err;
            }
        }
        /* Flatten, Activation, Dropout, BatchNorm, Input: elems değişmez */
        l = l->next;
    }
    return NULL;
}

static void handle_tool_train(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }

    int ti_len = 0, tt_len = 0, ish_len = 0, tsh_len = 0;
    int vi_len = 0, vt_len = 0;
    float* train_input = extract_float_array(args, "train_input", &ti_len);
    float* train_target = extract_float_array(args, "train_target", &tt_len);
    int* in_shape = extract_int_array(args, "input_shape", &ish_len);
    int* tar_shape = extract_int_array(args, "target_shape", &tsh_len);
    float* val_input = extract_float_array(args, "val_input", &vi_len);
    float* val_target = extract_float_array(args, "val_target", &vt_len);

    if (!train_input || !train_target || !in_shape || !tar_shape) {
        free(train_input); free(train_target); free(in_shape); free(tar_shape);
        free(val_input); free(val_target);
        send_error(id, -32602, "Missing required parameters: train_input, train_target, input_shape, target_shape");
        return;
    }

    JsonValue* epochs_val = json_object_get(args, "epochs");
    JsonValue* batch_val = json_object_get(args, "batch_size");
    if (!epochs_val || !batch_val) {
        free(train_input); free(train_target); free(in_shape); free(tar_shape);
        free(val_input); free(val_target);
        send_error(id, -32602, "Missing required parameters: epochs, batch_size");
        return;
    }
    int epochs = (int)epochs_val->num_val;
    int batch_size = (int)batch_val->num_val;

    Model* model = *g_model;
    char warning[256] = "";
    int model_ndim = 0, model_feature_dim = 0;
    if (model->input_layer && model->input_layer->output) {
        model_ndim = model->input_layer->output->ndim;
        model_feature_dim = (model_ndim >= 2) ? model->input_layer->output->shape[1] : 0;
    }

    /* Check if model has a Transformer or Attention layer (expects 3D input) */
    int has_3d_layer = 0;
    int target_embed_dim = 0;
    {
        Layer* l = model->input_layer ? model->input_layer->next : NULL;
        while (l) {
            if (l->type == LAYER_TRANSFORMER && l->params) {
                has_3d_layer = 1;
                target_embed_dim = ((TransformerBlockParams*)l->params)->embed_dim;
                break;
            }
            if (l->type == LAYER_ATTENTION && l->params) {
                has_3d_layer = 1;
                target_embed_dim = ((AttentionParams*)l->params)->embed_dim;
                break;
            }
            l = l->next;
        }
    }

    /* Case: model has Transformer/Attention, data is 2D -> reshape to 3D */
    if (has_3d_layer && ish_len == 2) {
        int rows = in_shape[0];
        int cols = in_shape[1];

        /* Pad cols up to next multiple of embed_dim if needed */
        int padded_cols = cols;
        if (cols % target_embed_dim != 0) {
            padded_cols = ((cols / target_embed_dim) + 1) * target_embed_dim;
            fprintf(stderr, "  Giriş %d özellik -> %d'e (embed_dim=%d katı) 0'lanarak genişletildi.\n", cols, padded_cols, target_embed_dim);
            if (!warning[0]) snprintf(warning, sizeof(warning), "%d özellik %d'e 0'lanarak genişletildi", cols, padded_cols);

            float* padded = (float*)calloc(rows * padded_cols, sizeof(float));
            for (int r = 0; r < rows; r++) {
                memcpy(&padded[r * padded_cols], &train_input[r * cols], cols * sizeof(float));
            }
            free(train_input);
            train_input = padded;
            ti_len = rows * padded_cols;

            if (val_input && vi_len > 0) {
                int val_rows = vi_len / cols;
                float* vpadded = (float*)calloc(val_rows * padded_cols, sizeof(float));
                for (int r = 0; r < val_rows; r++) {
                    memcpy(&vpadded[r * padded_cols], &val_input[r * cols], cols * sizeof(float));
                }
                free(val_input);
                val_input = vpadded;
                vi_len = val_rows * padded_cols;
            }
            cols = padded_cols;
        }

        int seq_len = cols / target_embed_dim;
        fprintf(stderr, "  2D -> 3D dönüşümü: [%d, %d] -> [%d, %d, %d]\n", rows, cols, rows, seq_len, target_embed_dim);

        int* new_shape = (int*)malloc(3 * sizeof(int));
        new_shape[0] = rows;
        new_shape[1] = seq_len;
        new_shape[2] = target_embed_dim;
        free(in_shape);
        in_shape = new_shape;
        ish_len = 3;

        model_ndim = 0;
        model_feature_dim = 0;

    } else if (model_ndim > 0 && ish_len != model_ndim) {
        fprintf(stderr, "  Hata: giriş boyutu %d, model %d bekliyor. Otomatik düzeltme mümkün değil.\n", ish_len, model_ndim);
        free(train_input); free(train_target); free(in_shape); free(tar_shape);
        free(val_input); free(val_target);
        send_error(id, -32000, "Giriş boyutu model ile uyuşmuyor");
        return;
    }

    if (model_feature_dim > 0 && ish_len >= 2 && in_shape[1] != model_feature_dim) {
        int new_cols = model_feature_dim;
        int rows = in_shape[0];
        int old_cols = in_shape[1];
        int new_size = rows * new_cols;

        if (old_cols < new_cols) {
            snprintf(warning, sizeof(warning), "Giriş %d özellik içeriyor, model %d bekliyor. %d sıfır eklendi.", old_cols, new_cols, new_cols - old_cols);
            fprintf(stderr, "  Uyari: %s\n", warning);
        } else {
            snprintf(warning, sizeof(warning), "Giriş %d özellik içeriyor, model %d bekliyor. %d'e kırpıldı.", old_cols, new_cols, new_cols);
            fprintf(stderr, "  Uyari: %s\n", warning);
        }

        float* resized = (float*)calloc(new_size, sizeof(float));
        int copy_per_row = old_cols < new_cols ? old_cols : new_cols;
        for (int row = 0; row < rows; row++) {
            memcpy(&resized[row * new_cols], &train_input[row * old_cols], copy_per_row * sizeof(float));
        }
        free(train_input);
        train_input = resized;
        ti_len = new_size;

        if (val_input && vi_len > 0) {
            int val_rows = vi_len / old_cols;
            int val_new_size = val_rows * new_cols;
            float* val_resized = (float*)calloc(val_new_size, sizeof(float));
            for (int row = 0; row < val_rows; row++) {
                memcpy(&val_resized[row * new_cols], &val_input[row * old_cols], copy_per_row * sizeof(float));
            }
            free(val_input);
            val_input = val_resized;
            vi_len = val_new_size;
        }

        int* new_shape = (int*)malloc(2 * sizeof(int));
        new_shape[0] = rows;
        new_shape[1] = new_cols;
        free(in_shape);
        in_shape = new_shape;
    }

    { /* Validate model architecture against actual data shape */
        const char* arch_err = validate_arch(model, in_shape, ish_len);
        if (arch_err) {
            fprintf(stderr, "  Mimari hatasi: %s\n", arch_err);
            free(train_input); free(train_target); free(in_shape); free(tar_shape);
            free(val_input); free(val_target);
            send_error(id, -32000, arch_err);
            return;
        }
        /* Compute expected output elements per sample */
        int expected_out = 1;
        for (int i = 1; i < ish_len; i++) expected_out *= in_shape[i];
        {
            Layer* ll = model->input_layer;
            while (ll) {
                if (ll->type == LAYER_DENSE && ll->params)
                    expected_out = ((DenseParams*)ll->params)->output_dim;
                ll = ll->next;
            }
        }
        /* Compute target elements per sample */
        int target_out = 1;
        for (int i = 1; i < tsh_len; i++) target_out *= tar_shape[i];
        if (expected_out != target_out) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Model çıkışı örnek başına %d değer üretiyor, hedef ise %d. Dense çıkışını %d yapın.",
                     expected_out, target_out, target_out);
            fprintf(stderr, "  Hata: %s\n", msg);
            free(train_input); free(train_target); free(in_shape); free(tar_shape);
            free(val_input); free(val_target);
            send_error(id, -32000, msg);
            return;
        }
    }

    Tensor* x_train = tensor_create(in_shape, ish_len);
    Tensor* y_train = tensor_create(tar_shape, tsh_len);
    if (!x_train || !y_train) {
        tensor_free(x_train); tensor_free(y_train);
        free(train_input); free(train_target); free(in_shape); free(tar_shape);
        free(val_input); free(val_target);
        send_error(id, -32000, "Failed to create training tensors (dimensions too large)");
        return;
    }
    int cp_x = x_train->size < ti_len ? x_train->size : ti_len;
    int cp_y = y_train->size < tt_len ? y_train->size : tt_len;
    memcpy(x_train->data, train_input, cp_x * sizeof(float));
    memcpy(y_train->data, train_target, cp_y * sizeof(float));
    free(train_input); free(train_target);

    Tensor* x_val = NULL;
    Tensor* y_val = NULL;
    if (val_input && val_target && vi_len > 0 && vt_len > 0) {
        x_val = tensor_create(in_shape, ish_len);
        y_val = tensor_create(tar_shape, tsh_len);
        if (!x_val || !y_val) {
            tensor_free(x_val); tensor_free(y_val);
            tensor_free(x_train); tensor_free(y_train);
            free(in_shape); free(tar_shape); free(val_input); free(val_target);
            send_error(id, -32000, "Failed to create validation tensors");
            return;
        }
        int cp_vx = x_val->size < vi_len ? x_val->size : vi_len;
        int cp_vy = y_val->size < vt_len ? y_val->size : vt_len;
        memcpy(x_val->data, val_input, cp_vx * sizeof(float));
        memcpy(y_val->data, val_target, cp_vy * sizeof(float));
    }
    free(val_input); free(val_target); free(in_shape); free(tar_shape);

    if (!model->optimizer) {
        Optimizer* adam = optimizer_create(OPT_ADAM, 0.001f);
        model_set_optimizer(model, adam);
    }

    Trainer* trainer = trainer_create(model, model->optimizer, batch_size, epochs);
    trainer_fit(trainer, x_train, y_train, x_val, y_val);

    float final_loss = model_evaluate(model, x_train, y_train);
    float final_acc = 0;
    if (y_train->ndim >= 2) {
        model_predict(model, x_train);
        final_acc = accuracy(model->output_layer->output, y_train);
    }

    char text[768];
    if (warning[0]) {
        snprintf(text, sizeof(text), "{\"status\": \"completed\", \"epochs\": %d, \"batch_size\": %d, \"final_loss\": %.6f, \"final_accuracy\": %.4f, \"warning\": \"%s\"}",
                 epochs, batch_size, (double)final_loss, (double)final_acc, warning);
    } else {
        snprintf(text, sizeof(text), "{\"status\": \"completed\", \"epochs\": %d, \"batch_size\": %d, \"final_loss\": %.6f, \"final_accuracy\": %.4f}",
                 epochs, batch_size, (double)final_loss, (double)final_acc);
    }

    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);

    trainer_free(trainer);
    tensor_free(x_train);
    tensor_free(y_train);
    if (x_val) tensor_free(x_val);
    if (y_val) tensor_free(y_val);
}

static void handle_tool_get_layer_output(int id, JsonValue* args) {
    if (!g_model || !*g_model) {
        send_error(id, -32000, "No model loaded");
        return;
    }
    JsonValue* li_val = json_object_get(args, "layer_index");
    if (!li_val || li_val->type != JSON_NUMBER) {
        send_error(id, -32602, "Missing or invalid 'layer_index'");
        return;
    }
    int idx = (int)li_val->num_val;
    Layer* layer = get_layer_by_index(idx);
    if (!layer) {
        send_error(id, -32000, "Layer index out of range");
        return;
    }
    if (!layer->output) {
        send_error(id, -32000, "Layer has no output (run a prediction first)");
        return;
    }

    JsonValue* out_data = tensor_to_json_array(layer->output);
    JsonValue* out_shape = json_array();
    for (int i = 0; i < layer->output->ndim; i++)
        json_array_append(out_shape, json_number(layer->output->shape[i]));

    JsonValue* out_obj = json_object();
    json_object_set(out_obj, "output_data", out_data);
    json_object_set(out_obj, "output_shape", out_shape);

    char* text = json_serialize(out_obj);
    JsonValue* content_arr = json_array();
    json_array_append(content_arr, make_content_text(text));
    JsonValue* result = json_object();
    json_object_set(result, "content", content_arr);
    send_result(id, result);

    free(text);
    json_free(out_obj);
}

static void handle_tools_call(int id, JsonValue* params) {
    JsonValue* name_val = json_object_get(params, "name");
    if (!name_val || name_val->type != JSON_STRING) {
        send_error(id, -32602, "Missing or invalid 'name' parameter");
        return;
    }
    const char* name = name_val->str_val;

    JsonValue* args = json_object_get(params, "arguments");
    if (!args || args->type != JSON_OBJECT) {
        JsonValue* empty = json_object();
        if (strcmp(name, "predict") == 0) handle_tool_predict(id, empty);
        else if (strcmp(name, "train_step") == 0) handle_tool_train_step(id, empty);
        else if (strcmp(name, "load_model") == 0) handle_tool_load_model(id, empty);
        else if (strcmp(name, "save_model") == 0) handle_tool_save_model(id, empty);
        else if (strcmp(name, "get_model_info") == 0) handle_tool_get_model_info(id, empty);
        else if (strcmp(name, "list_layers") == 0) handle_tool_list_layers(id, empty);
        else if (strcmp(name, "evaluate") == 0) handle_tool_evaluate(id, empty);
        else if (strcmp(name, "create_model") == 0) handle_tool_create_model(id, empty);
        else if (strcmp(name, "add_layer") == 0) handle_tool_add_layer(id, empty);
        else if (strcmp(name, "set_training_mode") == 0) handle_tool_set_training_mode(id, empty);
        else if (strcmp(name, "train") == 0) handle_tool_train(id, empty);
        else if (strcmp(name, "get_layer_output") == 0) handle_tool_get_layer_output(id, empty);
        else if (strcmp(name, "remove_layer") == 0) handle_tool_remove_layer(id, empty);
        else if (strcmp(name, "destroy_model") == 0) handle_tool_destroy_model(id, empty);
        else send_error(id, -32601, "Tool not found");
        json_free(empty);
        return;
    }

    if (strcmp(name, "predict") == 0) handle_tool_predict(id, args);
    else if (strcmp(name, "train_step") == 0) handle_tool_train_step(id, args);
    else if (strcmp(name, "load_model") == 0) handle_tool_load_model(id, args);
    else if (strcmp(name, "save_model") == 0) handle_tool_save_model(id, args);
    else if (strcmp(name, "get_model_info") == 0) handle_tool_get_model_info(id, args);
    else if (strcmp(name, "list_layers") == 0) handle_tool_list_layers(id, args);
    else if (strcmp(name, "evaluate") == 0) handle_tool_evaluate(id, args);
    else if (strcmp(name, "create_model") == 0) handle_tool_create_model(id, args);
    else if (strcmp(name, "add_layer") == 0) handle_tool_add_layer(id, args);
    else if (strcmp(name, "set_training_mode") == 0) handle_tool_set_training_mode(id, args);
    else if (strcmp(name, "train") == 0) handle_tool_train(id, args);
    else if (strcmp(name, "get_layer_output") == 0) handle_tool_get_layer_output(id, args);
    else if (strcmp(name, "remove_layer") == 0) handle_tool_remove_layer(id, args);
    else if (strcmp(name, "destroy_model") == 0) handle_tool_destroy_model(id, args);
    else send_error(id, -32601, "Tool not found");
}

static void handle_request(JsonValue* msg) {
    JsonValue* id_val = json_object_get(msg, "id");
    JsonValue* method_val = json_object_get(msg, "method");
    JsonValue* params = json_object_get(msg, "params");

    int id = id_val && id_val->type == JSON_NUMBER ? (int)id_val->num_val : 0;
    const char* method = method_val && method_val->type == JSON_STRING ? method_val->str_val : "";

    if (!params || params->type != JSON_OBJECT) {
        JsonValue* empty = json_object();
        if (strcmp(method, "initialize") == 0) handle_initialize(id, empty);
        else if (strcmp(method, "tools/list") == 0) handle_tools_list(id);
        else if (strcmp(method, "tools/call") == 0) handle_tools_call(id, empty);
        else if (strcmp(method, "resources/list") == 0) handle_resources_list(id);
        else if (strcmp(method, "resources/read") == 0) handle_resources_read(id, empty);
        else if (strcmp(method, "prompts/list") == 0) handle_prompts_list(id);
        else if (strcmp(method, "prompts/get") == 0) handle_prompts_get(id, empty);
        else send_error(id, -32601, "Method not found");
        json_free(empty);
        return;
    }

    if (strcmp(method, "initialize") == 0) handle_initialize(id, params);
    else if (strcmp(method, "tools/list") == 0) handle_tools_list(id);
    else if (strcmp(method, "tools/call") == 0) handle_tools_call(id, params);
    else if (strcmp(method, "resources/list") == 0) handle_resources_list(id);
    else if (strcmp(method, "resources/read") == 0) handle_resources_read(id, params);
    else if (strcmp(method, "prompts/list") == 0) handle_prompts_list(id);
    else if (strcmp(method, "prompts/get") == 0) handle_prompts_get(id, params);
    else send_error(id, -32601, "Method not found");
}

static void handle_notification(JsonValue* msg) {
    JsonValue* method_val = json_object_get(msg, "method");
    if (!method_val) return;
    const char* method = method_val->str_val;

    if (strcmp(method, "notifications/initialized") == 0) {
        g_initialized = 1;
    } else if (strcmp(method, "shutdown") == 0) {
        exit(0);
    }
}

int mcp_server_run(Model** model) {
    g_model = model;
    g_sse_send_fn = NULL;

    setbuf(stderr, NULL);
    setbuf(stdout, NULL);

    while (1) {
        char* line = read_one_line();
        if (!line) break;

        if (strlen(line) == 0) { free(line); continue; }

        JsonValue* msg = json_parse(line);
        free(line);

        if (!msg) {
            JsonValue* err = make_error(-32700, "Parse error", NULL);
            JsonValue* resp = json_object();
            json_object_set(resp, "jsonrpc", json_string("2.0"));
            json_object_set(resp, "id", json_null());
            json_object_set(resp, "error", err);
            char* s = json_serialize(resp);
            if (g_sse_send_fn) g_sse_send_fn("message", s);
            else write_msg(s);
            free(s);
            json_free(resp);
            continue;
        }

        JsonValue* method = json_object_get(msg, "method");
        JsonValue* id = json_object_get(msg, "id");

        if (method && id && id->type == JSON_NUMBER) {
            handle_request(msg);
        } else if (method && (!id || id->type == JSON_NULL)) {
            handle_notification(msg);
        } else if (!method && id && id->type == JSON_NUMBER) {
            send_error((int)id->num_val, -32600, "Invalid request: missing method");
        }

        json_free(msg);
    }

    return 0;
}

static void sse_request_handler(const char* json_request,
                                 void (*send_sse_event)(const char* event_name, const char* data)) {
    if (!json_request) return;

    JsonValue* msg = json_parse(json_request);
    if (!msg) return;

    g_sse_send_fn = send_sse_event;

    JsonValue* method = json_object_get(msg, "method");
    JsonValue* id = json_object_get(msg, "id");

    if (method && id && id->type == JSON_NUMBER) {
        handle_request(msg);
    } else if (method && (!id || id->type == JSON_NULL)) {
        handle_notification(msg);
    }

    g_sse_send_fn = NULL;
    json_free(msg);
}

int mcp_server_run_sse(Model** model, int port) {
    g_model = model;
    g_initialized = 0;

    setbuf(stderr, NULL);
    setbuf(stdout, NULL);

    if (http_server_start(port, sse_request_handler) != 0) {
        fprintf(stderr, "Failed to start SSE server\n");
        return -1;
    }

    fprintf(stderr, "Press Ctrl+C to stop the server\n");
    fflush(stderr);

    while (1) Sleep(1000);

    http_server_stop();
    return 0;
}
