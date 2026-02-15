use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::SqlitePool;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::{Mutex, RwLock};
use tokio::time::timeout;

use crate::db;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_MAX_RETRIES: u32 = 3;

// --- JSON-RPC 2.0 Wire Types ---

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
    #[allow(dead_code)]
    id: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

// --- MCP Tool Info ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

// --- Internal Connection State ---

struct McpConnection {
    transport: String,
    endpoint: String,
    tcp_reader: Option<BufReader<OwnedReadHalf>>,
    tcp_writer: Option<OwnedWriteHalf>,
    cached_tools: Option<Vec<McpToolInfo>>,
    initialized: bool,
}

// --- Public Manager ---

pub struct McpManager {
    connections: RwLock<HashMap<String, Arc<Mutex<McpConnection>>>>,
    http_client: reqwest::Client,
    next_id: AtomicU64,
}

impl McpManager {
    pub fn new() -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create MCP HTTP client");

        Self {
            connections: RwLock::new(HashMap::new()),
            http_client,
            next_id: AtomicU64::new(1),
        }
    }

    /// Ensure a connection to the named MCP server is established and initialized.
    /// Looks up the server in the database, connects (TCP) or validates (HTTP),
    /// runs the MCP handshake, and caches discovered tools.
    pub async fn ensure_connected(&self, pool: &SqlitePool, server_name: &str) -> Result<()> {
        // Fast path: already connected and initialized
        {
            let conns = self.connections.read().await;
            if let Some(conn_arc) = conns.get(server_name) {
                let conn = conn_arc.lock().await;
                if conn.initialized {
                    return Ok(());
                }
            }
        }

        // Look up server in DB
        let server = db::mcp_server_get(pool, server_name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("MCP server '{}' not found", server_name))?;

        if !server.enabled {
            anyhow::bail!("MCP server '{}' is disabled", server_name);
        }

        match server.transport.as_str() {
            "tcp" | "http" | "sse" | "streamable_http" => {}
            other => anyhow::bail!(
                "Transport '{}' not supported for MCP connections (supported: tcp, http, sse, streamable_http)",
                other
            ),
        }

        let mut conn = McpConnection {
            transport: server.transport.clone(),
            endpoint: server.endpoint.clone(),
            tcp_reader: None,
            tcp_writer: None,
            cached_tools: None,
            initialized: false,
        };

        // Establish TCP socket if needed
        if server.transport == "tcp" {
            let (host, port) = parse_tcp_endpoint(&server.endpoint)?;
            let addr = format!("{}:{}", host, port);
            tracing::info!(server = server_name, addr = %addr, "Connecting to MCP server via TCP");

            let stream = timeout(CONNECT_TIMEOUT, TcpStream::connect(&addr))
                .await
                .context("TCP connect timeout (10s)")?
                .context(format!("TCP connect to {} failed", addr))?;

            let (read_half, write_half) = stream.into_split();
            conn.tcp_reader = Some(BufReader::new(read_half));
            conn.tcp_writer = Some(write_half);
        }

        // Run MCP handshake: initialize → notifications/initialized → tools/list
        let tools = run_handshake(&mut conn, &self.http_client, &self.next_id).await?;
        let tool_count = tools.len();
        conn.cached_tools = Some(tools);
        conn.initialized = true;

        tracing::info!(
            server = server_name,
            transport = %server.transport,
            tools = tool_count,
            "MCP server connected and initialized"
        );

        // Store the connection
        let conn_arc = Arc::new(Mutex::new(conn));
        let mut conns = self.connections.write().await;
        conns.insert(server_name.to_string(), conn_arc);

        Ok(())
    }

    /// Re-fetch the tool list from a connected server (forces refresh).
    pub async fn discover_tools(&self, server_name: &str) -> Result<Vec<McpToolInfo>> {
        let conn_arc = {
            let conns = self.connections.read().await;
            conns
                .get(server_name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Not connected to '{}'. Call ensure_connected first.",
                        server_name
                    )
                })?
                .clone()
        };

        let mut conn = conn_arc.lock().await;
        let list_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: "tools/list".to_string(),
            params: None,
            id: Some(list_id),
        };

        let response = send_rpc(&mut conn, &self.http_client, &request).await?;
        if let Some(err) = response.error {
            anyhow::bail!("tools/list error: {} (code {})", err.message, err.code);
        }

        let tools_value = response.result.unwrap_or(json!({}));
        let tools: Vec<McpToolInfo> =
            serde_json::from_value(tools_value.get("tools").cloned().unwrap_or(json!([])))?;

        conn.cached_tools = Some(tools.clone());
        Ok(tools)
    }

    /// Get the cached tool list for a connected server.
    pub async fn cached_tools(&self, server_name: &str) -> Option<Vec<McpToolInfo>> {
        let conns = self.connections.read().await;
        let conn_arc = conns.get(server_name)?;
        let conn = conn_arc.lock().await;
        conn.cached_tools.clone()
    }

    /// Call a tool on an MCP server. Auto-connects if needed.
    /// On TCP transport failure, attempts one reconnection before failing.
    pub async fn call_tool(
        &self,
        pool: &SqlitePool,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<Value> {
        self.ensure_connected(pool, server_name).await?;

        let conn_arc = {
            let conns = self.connections.read().await;
            conns
                .get(server_name)
                .ok_or_else(|| anyhow::anyhow!("Not connected to '{}'", server_name))?
                .clone()
        };

        let mut conn = conn_arc.lock().await;

        let call_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": tool_name,
                "arguments": arguments,
            })),
            id: Some(call_id),
        };

        let response = match send_rpc(&mut conn, &self.http_client, &request).await {
            Ok(resp) => resp,
            Err(e) if conn.transport == "tcp" => {
                // TCP failure — attempt one reconnect
                tracing::warn!(
                    error = %e,
                    server = server_name,
                    "TCP call failed, attempting reconnect"
                );
                reconnect_tcp(&mut conn).await?;
                let tools = run_handshake(&mut conn, &self.http_client, &self.next_id).await?;
                conn.cached_tools = Some(tools);
                conn.initialized = true;
                // Retry the original call
                send_rpc(&mut conn, &self.http_client, &request).await?
            }
            Err(e) => return Err(e),
        };

        if let Some(err) = response.error {
            return Ok(json!({
                "error": err.message,
                "code": err.code,
            }));
        }

        Ok(response.result.unwrap_or(json!(null)))
    }

    /// Disconnect from an MCP server, dropping the connection.
    #[allow(dead_code)]
    pub async fn disconnect(&self, server_name: &str) {
        let mut conns = self.connections.write().await;
        conns.remove(server_name);
    }
}

// --- Helper Functions ---

fn parse_tcp_endpoint(endpoint: &str) -> Result<(String, u16)> {
    let addr = endpoint.strip_prefix("tcp://").unwrap_or(endpoint);
    let (host, port_str) = addr
        .rsplit_once(':')
        .context("TCP endpoint must be in format host:port or tcp://host:port")?;
    let port: u16 = port_str
        .parse()
        .context(format!("Invalid port number: '{}'", port_str))?;
    Ok((host.to_string(), port))
}

async fn reconnect_tcp(conn: &mut McpConnection) -> Result<()> {
    let (host, port) = parse_tcp_endpoint(&conn.endpoint)?;
    let addr = format!("{}:{}", host, port);
    tracing::info!(addr = %addr, "Reconnecting TCP to MCP server");

    let stream = timeout(CONNECT_TIMEOUT, TcpStream::connect(&addr))
        .await
        .context("TCP reconnect timeout (10s)")?
        .context(format!("TCP reconnect to {} failed", addr))?;

    let (read_half, write_half) = stream.into_split();
    conn.tcp_reader = Some(BufReader::new(read_half));
    conn.tcp_writer = Some(write_half);
    conn.initialized = false;
    Ok(())
}

async fn run_handshake(
    conn: &mut McpConnection,
    http_client: &reqwest::Client,
    next_id: &AtomicU64,
) -> Result<Vec<McpToolInfo>> {
    // Step 1: initialize
    let init_id = next_id.fetch_add(1, Ordering::Relaxed);
    let init_req = JsonRpcRequest {
        jsonrpc: "2.0",
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "astartebot",
                "version": "0.1.0"
            }
        })),
        id: Some(init_id),
    };

    let init_resp = send_rpc(conn, http_client, &init_req)
        .await
        .context("MCP initialize request failed")?;

    if let Some(err) = init_resp.error {
        anyhow::bail!("MCP initialize error: {} (code {})", err.message, err.code);
    }

    tracing::debug!(result = ?init_resp.result, "MCP initialize response");

    // Step 2: notifications/initialized (notification — no id, no response expected)
    let notif = JsonRpcRequest {
        jsonrpc: "2.0",
        method: "notifications/initialized".to_string(),
        params: None,
        id: None,
    };
    send_notification(conn, http_client, &notif).await?;

    // Step 3: tools/list
    let list_id = next_id.fetch_add(1, Ordering::Relaxed);
    let list_req = JsonRpcRequest {
        jsonrpc: "2.0",
        method: "tools/list".to_string(),
        params: None,
        id: Some(list_id),
    };

    let list_resp = send_rpc(conn, http_client, &list_req)
        .await
        .context("MCP tools/list request failed")?;

    if let Some(err) = list_resp.error {
        anyhow::bail!("MCP tools/list error: {} (code {})", err.message, err.code);
    }

    let tools_value = list_resp.result.unwrap_or(json!({}));
    let tools: Vec<McpToolInfo> =
        serde_json::from_value(tools_value.get("tools").cloned().unwrap_or(json!([])))?;

    tracing::info!(count = tools.len(), "MCP tools discovered");
    Ok(tools)
}

// --- Transport Dispatch ---

async fn send_rpc(
    conn: &mut McpConnection,
    http_client: &reqwest::Client,
    request: &JsonRpcRequest,
) -> Result<JsonRpcResponse> {
    match conn.transport.as_str() {
        "tcp" => timeout(REQUEST_TIMEOUT, tcp_send_recv(conn, request))
            .await
            .context("MCP TCP request timeout (30s)")?,
        "http" | "sse" | "streamable_http" => {
            http_rpc_with_retry(http_client, &conn.endpoint, request).await
        }
        other => anyhow::bail!("Unsupported MCP transport: {}", other),
    }
}

async fn send_notification(
    conn: &mut McpConnection,
    http_client: &reqwest::Client,
    request: &JsonRpcRequest,
) -> Result<()> {
    match conn.transport.as_str() {
        "tcp" => {
            let writer = conn.tcp_writer.as_mut().context("TCP not connected")?;
            let mut data = serde_json::to_string(request)?;
            data.push('\n');
            writer.write_all(data.as_bytes()).await?;
            writer.flush().await?;
            Ok(())
        }
        "http" | "sse" | "streamable_http" => {
            // Fire-and-forget POST for HTTP notifications
            let _ = http_client
                .post(&conn.endpoint)
                .json(request)
                .timeout(Duration::from_secs(5))
                .send()
                .await;
            Ok(())
        }
        _ => Ok(()),
    }
}

// --- TCP Transport ---

async fn tcp_send_recv(
    conn: &mut McpConnection,
    request: &JsonRpcRequest,
) -> Result<JsonRpcResponse> {
    // Send
    {
        let writer = conn.tcp_writer.as_mut().context("TCP not connected")?;
        let mut data = serde_json::to_string(request)?;
        data.push('\n');
        writer.write_all(data.as_bytes()).await?;
        writer.flush().await?;
    }

    // Receive
    let reader = conn.tcp_reader.as_mut().context("TCP not connected")?;
    tcp_read_response(reader).await
}

/// Read a JSON-RPC response from TCP, skipping any server-initiated notifications.
async fn tcp_read_response(reader: &mut BufReader<OwnedReadHalf>) -> Result<JsonRpcResponse> {
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).await?;
        if bytes_read == 0 {
            anyhow::bail!("TCP connection closed by MCP server");
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parsed: Value = serde_json::from_str(trimmed)
            .context(format!("Invalid JSON from MCP server: {}", trimmed))?;

        // Skip server-initiated notifications (have "method" but no "id")
        if parsed.get("method").is_some() && (parsed.get("id").is_none() || parsed["id"].is_null())
        {
            tracing::debug!(
                method = %parsed["method"],
                "Skipping MCP server notification"
            );
            continue;
        }

        let response: JsonRpcResponse = serde_json::from_value(parsed)?;
        return Ok(response);
    }
}

// --- HTTP Transport ---

async fn http_rpc_with_retry(
    client: &reqwest::Client,
    endpoint: &str,
    request: &JsonRpcRequest,
) -> Result<JsonRpcResponse> {
    let mut last_error = None;

    for attempt in 0..HTTP_MAX_RETRIES {
        if attempt > 0 {
            let delay = Duration::from_millis(500 * 2u64.pow(attempt));
            tracing::warn!(
                attempt,
                delay_ms = delay.as_millis(),
                "Retrying MCP HTTP request"
            );
            tokio::time::sleep(delay).await;
        }

        match timeout(REQUEST_TIMEOUT, client.post(endpoint).json(request).send()).await {
            Ok(Ok(resp)) => {
                let status = resp.status();
                if status.is_success() {
                    let body = resp.text().await?;
                    return serde_json::from_str(&body).context(format!(
                        "Invalid JSON-RPC response from MCP server: {}",
                        body
                    ));
                } else if status.is_server_error() {
                    let body = resp.text().await.unwrap_or_default();
                    tracing::warn!(status = status.as_u16(), body = %body, "MCP HTTP server error");
                    last_error = Some(anyhow::anyhow!("HTTP {}: {}", status, body));
                    continue;
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!("MCP HTTP {}: {}", status, body));
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, attempt, "MCP HTTP request failed");
                last_error = Some(e.into());
                continue;
            }
            Err(_) => {
                tracing::warn!(attempt, "MCP HTTP request timeout (30s)");
                last_error = Some(anyhow::anyhow!("MCP HTTP request timeout (30s)"));
                continue;
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All MCP HTTP retries exhausted")))
}
