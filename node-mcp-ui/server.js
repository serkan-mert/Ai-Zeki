const http = require('http');
const fs = require('fs');
const path = require('path');

const ZEKI_HOST = 'localhost';
const ZEKI_PORT = 8080;
const UI_PORT = 3000;

let sseSessionEndpoint = '/messages';
let sseBuffer = '';
let sseConnected = false;
let pendingRequests = new Map();
let requestIdCounter = 1;

function sseConnect() {
  const req = http.get(`http://${ZEKI_HOST}:${ZEKI_PORT}/sse`, (res) => {
    console.log('[SSE] Connected');
    sseConnected = true;

    res.on('data', (chunk) => {
      sseBuffer += chunk.toString();
      const parts = sseBuffer.split('\n\n');
      sseBuffer = parts.pop();

      for (const part of parts) {
        const lines = part.split('\n');
        let eventType = 'message';
        let data = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) data += line.slice(6).trim();
        }
        handleSseEvent(eventType, data);
      }
    });

    res.on('end', () => {
      console.log('[SSE] Disconnected, reconnecting...');
      sseConnected = false;
      setTimeout(sseConnect, 1000);
    });

    res.on('error', (err) => {
      console.error('[SSE] Error:', err.message);
      sseConnected = false;
      setTimeout(sseConnect, 2000);
    });
  });

  req.on('error', (err) => {
    console.error('[SSE] Connection failed:', err.message);
    sseConnected = false;
    setTimeout(sseConnect, 2000);
  });

  req.end();
}

function handleSseEvent(eventType, data) {
  if (eventType === 'endpoint') {
    sseSessionEndpoint = data;
    console.log('[SSE] Endpoint:', data);
    return;
  }

  if (eventType === 'keepalive') return;

  if (eventType === 'message' && data) {
    try {
      const msg = JSON.parse(data);
      const id = msg.id;
      if (id != null && pendingRequests.has(id)) {
        const { resolve } = pendingRequests.get(id);
        pendingRequests.delete(id);
        resolve(msg);
      }
    } catch (e) {
      console.error('[SSE] Parse error:', e.message);
    }
  }
}

function sendMcpRequest(method, params) {
  return new Promise((resolve, reject) => {
    const id = requestIdCounter++;
    const body = JSON.stringify({ jsonrpc: '2.0', id, method, params });

    pendingRequests.set(id, { resolve, reject, timeout: setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error('Request timeout'));
    }, 30000) });

    const postData = body;
    const options = {
      hostname: ZEKI_HOST,
      port: ZEKI_PORT,
      path: sseSessionEndpoint,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(postData) }
    };

    const req = http.request(options, (res) => {
      res.on('data', () => {});
      res.on('end', () => {});
    });

    req.on('error', (err) => {
      const p = pendingRequests.get(id);
      if (p) { clearTimeout(p.timeout); pendingRequests.delete(id); }
      reject(err);
    });

    req.write(postData);
    req.end();
  });
}

async function initialize() {
  await sendMcpRequest('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'zeki-ui', version: '1.0.0' } });
  await sendMcpRequest('notifications/initialized', {});
  console.log('[MCP] Initialized');
}

const mimeTypes = {
  '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml'
};

const uiServer = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://${req.headers.host}`);
  const pathname = url.pathname;

  if (pathname === '/api/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ connected: sseConnected, endpoint: sseSessionEndpoint }));
    return;
  }

  if (pathname.startsWith('/api/mcp/')) {
    const toolName = pathname.slice(9);
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const params = body ? JSON.parse(body) : {};

        let result;
        if (toolName === 'initialize') {
          result = await sendMcpRequest('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'zeki-ui', version: '1.0.0' } });
          await sendMcpRequest('notifications/initialized', {});
        } else {
          result = await sendMcpRequest('tools/call', { name: toolName, arguments: params });
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (pathname.startsWith('/api/resource/')) {
    const uri = decodeURIComponent(pathname.slice(13));
    sendMcpRequest('resources/read', { uri }).then(result => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    }).catch(err => {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    });
    return;
  }

  if (pathname.startsWith('/api/prompt/')) {
    const name = decodeURIComponent(pathname.slice(12));
    sendMcpRequest('prompts/get', { name }).then(result => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    }).catch(err => {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    });
    return;
  }

  let filePath = path.join(__dirname, 'public', pathname === '/' ? 'index.html' : pathname);
  const ext = path.extname(filePath);
  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('404 Not Found');
      return;
    }
    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'text/plain' });
    res.end(content);
  });
});

sseConnect();
setTimeout(() => {
  initialize().catch(err => console.error('[MCP] Init error:', err.message));
}, 500);

uiServer.listen(UI_PORT, () => {
  console.log(`Zeki MCP UI: http://localhost:${UI_PORT}`);
});
