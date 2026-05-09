#ifndef JSON_UTILS_H
#define JSON_UTILS_H

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue {
    JsonType type;
    union {
        int bool_val;
        double num_val;
        char* str_val;
        struct {
            struct JsonValue** items;
            int count;
            int capacity;
        } array;
        struct {
            char** keys;
            struct JsonValue** values;
            int count;
            int capacity;
        } object;
    };
} JsonValue;

JsonValue* json_null(void);
JsonValue* json_bool(int v);
JsonValue* json_number(double v);
JsonValue* json_string(const char* v);
JsonValue* json_array(void);
JsonValue* json_object(void);
void json_free(JsonValue* v);

void json_array_append(JsonValue* arr, JsonValue* item);
int json_array_len(JsonValue* arr);
JsonValue* json_array_get(JsonValue* arr, int idx);

void json_object_set(JsonValue* obj, const char* key, JsonValue* val);
JsonValue* json_object_get(JsonValue* obj, const char* key);

char* json_serialize(JsonValue* v);
JsonValue* json_parse(const char* str);

#endif
