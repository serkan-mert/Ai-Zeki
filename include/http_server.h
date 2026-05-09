#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

typedef void (*mcp_request_handler)(const char* json_request, void (*send_sse_event)(const char* event_name, const char* data));

int http_server_start(int port, mcp_request_handler handler);
void http_server_stop(void);

#endif
