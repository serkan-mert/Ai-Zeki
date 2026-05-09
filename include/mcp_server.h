#ifndef MCP_SERVER_H
#define MCP_SERVER_H

#include "model.h"

int mcp_server_run(Model** model);
int mcp_server_run_sse(Model** model, int port);

#endif
