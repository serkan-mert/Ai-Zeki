#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "json_utils.h"

static JsonValue* val_alloc(JsonType type) {
    JsonValue* v = (JsonValue*)malloc(sizeof(JsonValue));
    v->type = type;
    if (type == JSON_ARRAY) {
        v->array.items = NULL;
        v->array.count = 0;
        v->array.capacity = 0;
    } else if (type == JSON_OBJECT) {
        v->object.keys = NULL;
        v->object.values = NULL;
        v->object.count = 0;
        v->object.capacity = 0;
    } else if (type == JSON_STRING) {
        v->str_val = NULL;
    }
    return v;
}

JsonValue* json_null(void) { return val_alloc(JSON_NULL); }
JsonValue* json_bool(int v) { JsonValue* r = val_alloc(JSON_BOOL); r->bool_val = v; return r; }
JsonValue* json_number(double v) { JsonValue* r = val_alloc(JSON_NUMBER); r->num_val = v; return r; }

JsonValue* json_string(const char* v) {
    JsonValue* r = val_alloc(JSON_STRING);
    r->str_val = v ? strdup(v) : strdup("");
    return r;
}

JsonValue* json_array(void) { return val_alloc(JSON_ARRAY); }

JsonValue* json_object(void) { return val_alloc(JSON_OBJECT); }

void json_free(JsonValue* v) {
    if (!v) return;
    if (v->type == JSON_STRING) {
        free(v->str_val);
    } else if (v->type == JSON_ARRAY) {
        for (int i = 0; i < v->array.count; i++) json_free(v->array.items[i]);
        free(v->array.items);
    } else if (v->type == JSON_OBJECT) {
        for (int i = 0; i < v->object.count; i++) {
            free(v->object.keys[i]);
            json_free(v->object.values[i]);
        }
        free(v->object.keys);
        free(v->object.values);
    }
    free(v);
}

void json_array_append(JsonValue* arr, JsonValue* item) {
    if (arr->array.count >= arr->array.capacity) {
        arr->array.capacity = arr->array.capacity ? arr->array.capacity * 2 : 8;
        arr->array.items = (JsonValue**)realloc(arr->array.items, arr->array.capacity * sizeof(JsonValue*));
    }
    arr->array.items[arr->array.count++] = item;
}

int json_array_len(JsonValue* arr) { return arr->array.count; }

JsonValue* json_array_get(JsonValue* arr, int idx) {
    if (idx < 0 || idx >= arr->array.count) return NULL;
    return arr->array.items[idx];
}

void json_object_set(JsonValue* obj, const char* key, JsonValue* val) {
    for (int i = 0; i < obj->object.count; i++) {
        if (strcmp(obj->object.keys[i], key) == 0) {
            json_free(obj->object.values[i]);
            obj->object.values[i] = val;
            return;
        }
    }
    if (obj->object.count >= obj->object.capacity) {
        obj->object.capacity = obj->object.capacity ? obj->object.capacity * 2 : 8;
        obj->object.keys = (char**)realloc(obj->object.keys, obj->object.capacity * sizeof(char*));
        obj->object.values = (JsonValue**)realloc(obj->object.values, obj->object.capacity * sizeof(JsonValue*));
    }
    obj->object.keys[obj->object.count] = strdup(key);
    obj->object.values[obj->object.count] = val;
    obj->object.count++;
}

JsonValue* json_object_get(JsonValue* obj, const char* key) {
    for (int i = 0; i < obj->object.count; i++) {
        if (strcmp(obj->object.keys[i], key) == 0) return obj->object.values[i];
    }
    return NULL;
}

static void serialize_value(JsonValue* v, char** buf, int* len, int* cap);
static void buf_append(char** buf, int* len, int* cap, const char* s, int slen);
static void buf_append_char(char** buf, int* len, int* cap, char c);

static void buf_append(char** buf, int* len, int* cap, const char* s, int slen) {
    if (*len + slen + 1 > *cap) {
        *cap = (*cap ? *cap * 2 : 4096);
        while (*len + slen + 1 > *cap) *cap *= 2;
        *buf = (char*)realloc(*buf, *cap);
    }
    memcpy(*buf + *len, s, slen);
    *len += slen;
}

static void buf_append_char(char** buf, int* len, int* cap, char c) {
    if (*len + 2 > *cap) {
        *cap = *cap ? *cap * 2 : 4096;
        *buf = (char*)realloc(*buf, *cap);
    }
    (*buf)[(*len)++] = c;
}

static void serialize_string(char** buf, int* len, int* cap, const char* s) {
    buf_append_char(buf, len, cap, '"');
    while (*s) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
            case '"':  buf_append(buf, len, cap, "\\\"", 2); break;
            case '\\': buf_append(buf, len, cap, "\\\\", 2); break;
            case '\b': buf_append(buf, len, cap, "\\b", 2); break;
            case '\f': buf_append(buf, len, cap, "\\f", 2); break;
            case '\n': buf_append(buf, len, cap, "\\n", 2); break;
            case '\r': buf_append(buf, len, cap, "\\r", 2); break;
            case '\t': buf_append(buf, len, cap, "\\t", 2); break;
            default:
                if (c < 0x20) {
                    char hex[8];
                    sprintf(hex, "\\u%04x", c);
                    buf_append(buf, len, cap, hex, 6);
                } else {
                    buf_append_char(buf, len, cap, c);
                }
                break;
        }
        s++;
    }
    buf_append_char(buf, len, cap, '"');
}

static void serialize_value(JsonValue* v, char** buf, int* len, int* cap) {
    if (!v) { buf_append(buf, len, cap, "null", 4); return; }
    switch (v->type) {
        case JSON_NULL:
            buf_append(buf, len, cap, "null", 4);
            break;
        case JSON_BOOL:
            buf_append(buf, len, cap, v->bool_val ? "true" : "false", v->bool_val ? 4 : 5);
            break;
        case JSON_NUMBER: {
            char num[64];
            if (v->num_val == (double)(int)v->num_val && fabs(v->num_val) < 1e15)
                sprintf(num, "%d", (int)v->num_val);
            else
                sprintf(num, "%.17g", v->num_val);
            buf_append(buf, len, cap, num, (int)strlen(num));
            break;
        }
        case JSON_STRING:
            serialize_string(buf, len, cap, v->str_val ? v->str_val : "");
            break;
        case JSON_ARRAY: {
            buf_append_char(buf, len, cap, '[');
            for (int i = 0; i < v->array.count; i++) {
                if (i > 0) buf_append_char(buf, len, cap, ',');
                serialize_value(v->array.items[i], buf, len, cap);
            }
            buf_append_char(buf, len, cap, ']');
            break;
        }
        case JSON_OBJECT: {
            buf_append_char(buf, len, cap, '{');
            for (int i = 0; i < v->object.count; i++) {
                if (i > 0) buf_append_char(buf, len, cap, ',');
                serialize_string(buf, len, cap, v->object.keys[i]);
                buf_append_char(buf, len, cap, ':');
                serialize_value(v->object.values[i], buf, len, cap);
            }
            buf_append_char(buf, len, cap, '}');
            break;
        }
    }
}

char* json_serialize(JsonValue* v) {
    char* buf = NULL;
    int len = 0, cap = 0;
    serialize_value(v, &buf, &len, &cap);
    if (buf) buf[len] = '\0';
    return buf;
}

typedef struct {
    const char* s;
    int pos;
    int len;
} ParseCtx;

static void skip_ws(ParseCtx* ctx) {
    while (ctx->pos < ctx->len && isspace((unsigned char)ctx->s[ctx->pos])) ctx->pos++;
}

static JsonValue* parse_value(ParseCtx* ctx);

static JsonValue* parse_string(ParseCtx* ctx) {
    if (ctx->pos >= ctx->len || ctx->s[ctx->pos] != '"') return NULL;
    ctx->pos++;
    char* result = (char*)malloc(1);
    int rlen = 0, rcap = 1;
    result[0] = '\0';
    while (ctx->pos < ctx->len) {
        char c = ctx->s[ctx->pos];
        if (c == '"') {
            ctx->pos++;
            result[rlen] = '\0';
            JsonValue* v = val_alloc(JSON_STRING);
            v->str_val = result;
            return v;
        }
        if (c == '\\') {
            ctx->pos++;
            if (ctx->pos >= ctx->len) { free(result); return NULL; }
            char esc = ctx->s[ctx->pos];
            switch (esc) {
                case '"': buf_append_char(&result, &rlen, &rcap, '"'); break;
                case '\\': buf_append_char(&result, &rlen, &rcap, '\\'); break;
                case '/': buf_append_char(&result, &rlen, &rcap, '/'); break;
                case 'b': buf_append_char(&result, &rlen, &rcap, '\b'); break;
                case 'f': buf_append_char(&result, &rlen, &rcap, '\f'); break;
                case 'n': buf_append_char(&result, &rlen, &rcap, '\n'); break;
                case 'r': buf_append_char(&result, &rlen, &rcap, '\r'); break;
                case 't': buf_append_char(&result, &rlen, &rcap, '\t'); break;
                case 'u': {
                    if (ctx->pos + 4 >= ctx->len) { free(result); return NULL; }
                    char hex[5] = {ctx->s[ctx->pos+1], ctx->s[ctx->pos+2], ctx->s[ctx->pos+3], ctx->s[ctx->pos+4], 0};
                    int code = (int)strtol(hex, NULL, 16);
                    ctx->pos += 4;
                    if (code < 0x80) {
                        buf_append_char(&result, &rlen, &rcap, (char)code);
                    } else if (code < 0x800) {
                        buf_append_char(&result, &rlen, &rcap, (char)(0xC0 | (code >> 6)));
                        buf_append_char(&result, &rlen, &rcap, (char)(0x80 | (code & 0x3F)));
                    } else {
                        buf_append_char(&result, &rlen, &rcap, (char)(0xE0 | (code >> 12)));
                        buf_append_char(&result, &rlen, &rcap, (char)(0x80 | ((code >> 6) & 0x3F)));
                        buf_append_char(&result, &rlen, &rcap, (char)(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default: buf_append_char(&result, &rlen, &rcap, esc); break;
            }
        } else {
            buf_append_char(&result, &rlen, &rcap, c);
        }
        ctx->pos++;
    }
    free(result);
    return NULL;
}

static JsonValue* parse_number(ParseCtx* ctx) {
    int start = ctx->pos;
    if (ctx->pos < ctx->len && ctx->s[ctx->pos] == '-') ctx->pos++;
    while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->s[ctx->pos])) ctx->pos++;
    if (ctx->pos < ctx->len && ctx->s[ctx->pos] == '.') {
        ctx->pos++;
        while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->s[ctx->pos])) ctx->pos++;
    }
    if (ctx->pos < ctx->len && (ctx->s[ctx->pos] == 'e' || ctx->s[ctx->pos] == 'E')) {
        ctx->pos++;
        if (ctx->pos < ctx->len && (ctx->s[ctx->pos] == '+' || ctx->s[ctx->pos] == '-')) ctx->pos++;
        while (ctx->pos < ctx->len && isdigit((unsigned char)ctx->s[ctx->pos])) ctx->pos++;
    }
    if (ctx->pos == start) return NULL;
    int len = ctx->pos - start;
    char* numstr = (char*)malloc(len + 1);
    memcpy(numstr, ctx->s + start, len);
    numstr[len] = '\0';
    double val = strtod(numstr, NULL);
    free(numstr);
    return json_number(val);
}

static JsonValue* parse_array(ParseCtx* ctx) {
    if (ctx->pos >= ctx->len || ctx->s[ctx->pos] != '[') return NULL;
    ctx->pos++;
    JsonValue* arr = json_array();
    skip_ws(ctx);
    if (ctx->pos < ctx->len && ctx->s[ctx->pos] == ']') {
        ctx->pos++;
        return arr;
    }
    while (1) {
        skip_ws(ctx);
        JsonValue* val = parse_value(ctx);
        if (!val) { json_free(arr); return NULL; }
        json_array_append(arr, val);
        skip_ws(ctx);
        if (ctx->pos >= ctx->len) { json_free(arr); return NULL; }
        if (ctx->s[ctx->pos] == ']') { ctx->pos++; return arr; }
        if (ctx->s[ctx->pos] != ',') { json_free(arr); return NULL; }
        ctx->pos++;
    }
}

static JsonValue* parse_object(ParseCtx* ctx) {
    if (ctx->pos >= ctx->len || ctx->s[ctx->pos] != '{') return NULL;
    ctx->pos++;
    JsonValue* obj = json_object();
    skip_ws(ctx);
    if (ctx->pos < ctx->len && ctx->s[ctx->pos] == '}') {
        ctx->pos++;
        return obj;
    }
    while (1) {
        skip_ws(ctx);
        JsonValue* key = parse_string(ctx);
        if (!key || key->type != JSON_STRING) { json_free(key); json_free(obj); return NULL; }
        skip_ws(ctx);
        if (ctx->pos >= ctx->len || ctx->s[ctx->pos] != ':') { json_free(key); json_free(obj); return NULL; }
        ctx->pos++;
        skip_ws(ctx);
        JsonValue* val = parse_value(ctx);
        if (!val) { json_free(key); json_free(obj); return NULL; }
        json_object_set(obj, key->str_val, val);
        json_free(key);
        skip_ws(ctx);
        if (ctx->pos >= ctx->len) { json_free(obj); return NULL; }
        if (ctx->s[ctx->pos] == '}') { ctx->pos++; return obj; }
        if (ctx->s[ctx->pos] != ',') { json_free(obj); return NULL; }
        ctx->pos++;
    }
}

static JsonValue* parse_value(ParseCtx* ctx) {
    skip_ws(ctx);
    if (ctx->pos >= ctx->len) return NULL;
    char c = ctx->s[ctx->pos];
    if (c == '"') return parse_string(ctx);
    if (c == '{') return parse_object(ctx);
    if (c == '[') return parse_array(ctx);
    if (c == '-' || isdigit((unsigned char)c)) return parse_number(ctx);
    if (ctx->len - ctx->pos >= 4 && memcmp(ctx->s + ctx->pos, "true", 4) == 0) {
        ctx->pos += 4;
        return json_bool(1);
    }
    if (ctx->len - ctx->pos >= 5 && memcmp(ctx->s + ctx->pos, "false", 5) == 0) {
        ctx->pos += 5;
        return json_bool(0);
    }
    if (ctx->len - ctx->pos >= 4 && memcmp(ctx->s + ctx->pos, "null", 4) == 0) {
        ctx->pos += 4;
        return json_null();
    }
    return NULL;
}

JsonValue* json_parse(const char* str) {
    if (!str) return NULL;
    ParseCtx ctx;
    ctx.s = str;
    ctx.pos = 0;
    ctx.len = (int)strlen(str);
    JsonValue* v = parse_value(&ctx);
    return v;
}
