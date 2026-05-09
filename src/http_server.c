#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <process.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include "http_server.h"

static SOCKET g_listen_sock = INVALID_SOCKET;
static volatile int g_running = 0;
static mcp_request_handler g_handler = NULL;

static CRITICAL_SECTION g_sse_lock;
static SOCKET g_sse_sock = INVALID_SOCKET;
static char g_session_id[64] = "";
static int g_have_sse = 0;

static CRITICAL_SECTION g_response_lock;

static void send_sse_event(const char* event_name, const char* data) {
    EnterCriticalSection(&g_sse_lock);
    if (g_sse_sock != INVALID_SOCKET) {
        char buf[65536];
        int n = snprintf(buf, sizeof(buf), "event: %s\ndata: %s\n\n", event_name, data);
        send(g_sse_sock, buf, n, 0);
    }
    LeaveCriticalSection(&g_sse_lock);
}

static int read_http_request(SOCKET sock, char* req_buf, int buf_size, char** body, int* body_len) {
    int total = 0;
    while (total < buf_size - 1) {
        int n = recv(sock, req_buf + total, buf_size - 1 - total, 0);
        if (n <= 0) return -1;
        total += n;
        req_buf[total] = '\0';
        char* hdr_end = strstr(req_buf, "\r\n\r\n");
        if (hdr_end) break;
    }
    char* hdr_end = strstr(req_buf, "\r\n\r\n");
    if (!hdr_end) return -1;
    int header_len = (int)(hdr_end - req_buf) + 4;

    char* cl = strstr(req_buf, "Content-Length: ");
    if (!cl) cl = strstr(req_buf, "content-length: ");
    if (cl) {
        int content_len = atoi(cl + 16);
        int had = total - header_len;
        while (had < content_len && total < buf_size - 1) {
            int n = recv(sock, req_buf + total, buf_size - 1 - total, 0);
            if (n <= 0) return -1;
            total += n;
            had = total - header_len;
        }
        *body = req_buf + header_len;
        *body_len = content_len;
    } else {
        *body = NULL;
        *body_len = 0;
    }
    return total;
}

static void send_http_response(SOCKET sock, int status, const char* status_text,
                                const char* content_type, const char* body, int body_len) {
    char buf[65536];
    int n = snprintf(buf, sizeof(buf),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "\r\n"
        "%s",
        status, status_text, content_type, body_len, body ? body : "");
    send(sock, buf, n, 0);
}

static void send_http_ok(SOCKET sock, const char* content_type, const char* body) {
    send_http_response(sock, 200, "OK", content_type, body, body ? (int)strlen(body) : 0);
}

static void send_http_accepted(SOCKET sock) {
    const char* msg = "Accepted";
    send_http_response(sock, 202, "Accepted", "text/plain", msg, (int)strlen(msg));
}

static void handle_sse(SOCKET sock) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n");
    send(sock, buf, n, 0);

    EnterCriticalSection(&g_sse_lock);
    if (g_sse_sock != INVALID_SOCKET) closesocket(g_sse_sock);
    g_sse_sock = sock;
    g_have_sse = 1;
    snprintf(g_session_id, sizeof(g_session_id), "sse-%lu", (unsigned long)GetCurrentThreadId());
    LeaveCriticalSection(&g_sse_lock);

    char ep_buf[256];
    n = snprintf(ep_buf, sizeof(ep_buf), "/messages?sessionId=%s", g_session_id);
    send_sse_event("endpoint", ep_buf);

    fd_set read_fds;
    struct timeval tv = {30, 0};
    int keepalive_count = 0;
    while (g_running) {
        FD_ZERO(&read_fds);
        FD_SET(sock, &read_fds);
        int ret = select(0, &read_fds, NULL, NULL, &tv);
        if (ret < 0) break;
        if (ret == 0) {
            send_sse_event("keepalive", "ping");
            keepalive_count++;
            if (keepalive_count > 20) break;
            tv.tv_sec = 30;
            tv.tv_usec = 0;
            continue;
        }
        char discard[256];
        int r = recv(sock, discard, sizeof(discard), 0);
        if (r <= 0) break;
    }

    EnterCriticalSection(&g_sse_lock);
    if (g_sse_sock == sock) {
        g_sse_sock = INVALID_SOCKET;
        g_have_sse = 0;
    }
    LeaveCriticalSection(&g_sse_lock);
    closesocket(sock);
}

static void handle_post(SOCKET sock, const char* body, int body_len) {
    if (!body || body_len <= 0 || !g_handler) {
        send_http_accepted(sock);
        closesocket(sock);
        return;
    }
    char* json = (char*)malloc(body_len + 1);
    memcpy(json, body, body_len);
    json[body_len] = '\0';

    g_handler(json, send_sse_event);
    free(json);
    send_http_accepted(sock);
    closesocket(sock);
}

static void handle_options(SOCKET sock) {
    send_http_response(sock, 204, "No Content", "text/plain", "", 0);
    closesocket(sock);
}

static unsigned __stdcall connection_thread(void* arg) {
    SOCKET client = (SOCKET)(uintptr_t)arg;
    char req_buf[65536];
    char* body = NULL;
    int body_len = 0;

    int total = read_http_request(client, req_buf, sizeof(req_buf), &body, &body_len);
    if (total <= 0) { closesocket(client); return 0; }

    char method[16] = {0}, path[1024] = {0};
    if (sscanf(req_buf, "%15s %1023s", method, path) < 2) {
        closesocket(client);
        return 0;
    }

    if (strcmp(method, "OPTIONS") == 0) {
        handle_options(client);
    } else if (strcmp(method, "GET") == 0 && strncmp(path, "/sse", 4) == 0) {
        handle_sse(client);
    } else if (strcmp(method, "POST") == 0 && strncmp(path, "/messages", 9) == 0) {
        handle_post(client, body, body_len);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        send_http_ok(client, "application/json", "{\"status\":\"ok\",\"server\":\"zeki-mcp\"}");
        closesocket(client);
    } else {
        send_http_response(client, 404, "Not Found", "text/plain", "Not Found", 9);
        closesocket(client);
    }
    return 0;
}

static unsigned __stdcall accept_thread(void* arg) {
    (void)arg;
    while (g_running) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(g_listen_sock, &read_fds);
        struct timeval tv = {1, 0};
        int ret = select(0, &read_fds, NULL, NULL, &tv);
        if (ret < 0) break;
        if (ret == 0) continue;

        SOCKET client = accept(g_listen_sock, NULL, NULL);
        if (client == INVALID_SOCKET) continue;

        unsigned tid;
        uintptr_t h = _beginthreadex(NULL, 0, connection_thread, (void*)(uintptr_t)client, 0, &tid);
        if (h != 0) CloseHandle((HANDLE)h);
    }
    return 0;
}

int http_server_start(int port, mcp_request_handler handler) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return -1;
    }

    g_listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_listen_sock == INVALID_SOCKET) {
        fprintf(stderr, "socket() failed: %d\n", WSAGetLastError());
        WSACleanup();
        return -1;
    }

    int opt = 1;
    setsockopt(g_listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)port);

    if (bind(g_listen_sock, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "bind() failed on port %d: %d\n", port, WSAGetLastError());
        closesocket(g_listen_sock);
        WSACleanup();
        return -1;
    }

    if (listen(g_listen_sock, SOMAXCONN) == SOCKET_ERROR) {
        fprintf(stderr, "listen() failed: %d\n", WSAGetLastError());
        closesocket(g_listen_sock);
        WSACleanup();
        return -1;
    }

    InitializeCriticalSection(&g_sse_lock);
    InitializeCriticalSection(&g_response_lock);
    g_handler = handler;
    g_running = 1;

    unsigned tid;
    uintptr_t h = _beginthreadex(NULL, 0, accept_thread, NULL, 0, &tid);
    if (h == 0) {
        fprintf(stderr, "Failed to start accept thread\n");
        g_running = 0;
        closesocket(g_listen_sock);
        DeleteCriticalSection(&g_sse_lock);
        DeleteCriticalSection(&g_response_lock);
        WSACleanup();
        return -1;
    }
    CloseHandle((HANDLE)h);

    fprintf(stderr, "MCP SSE server listening on port %d\n", port);
    fprintf(stderr, "  SSE endpoint:  http://localhost:%d/sse\n", port);
    fprintf(stderr, "  Message POST:  http://localhost:%d/messages\n", port);
    fflush(stderr);

    return 0;
}

void http_server_stop(void) {
    g_running = 0;
    if (g_listen_sock != INVALID_SOCKET) {
        closesocket(g_listen_sock);
        g_listen_sock = INVALID_SOCKET;
    }
    EnterCriticalSection(&g_sse_lock);
    if (g_sse_sock != INVALID_SOCKET) {
        closesocket(g_sse_sock);
        g_sse_sock = INVALID_SOCKET;
    }
    LeaveCriticalSection(&g_sse_lock);
    DeleteCriticalSection(&g_sse_lock);
    DeleteCriticalSection(&g_response_lock);
    WSACleanup();
    fprintf(stderr, "MCP SSE server stopped\n");
}
