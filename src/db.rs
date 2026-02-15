use anyhow::Result;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Column, Row, SqlitePool};
use std::str::FromStr;

use crate::types::{ConversationRow, MemoryRow, McpServerRow, NoteRow};

pub async fn create_pool(db_path: &str) -> Result<SqlitePool> {
    let options = SqliteConnectOptions::from_str(db_path)?
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    run_migrations(&pool).await?;
    Ok(pool)
}

async fn run_migrations(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    let current_version: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM schema_version")
        .fetch_one(pool)
        .await?;

    let migrations: Vec<(i64, &str)> = vec![
        (1, "CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"),
        (2, "CREATE TABLE IF NOT EXISTS name_map (
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            display_name TEXT NOT NULL,
            PRIMARY KEY (entity_type, entity_id)
        )"),
        (3, "CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(segment, key)
        )"),
        (4, "CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"),
        (5, "CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_call_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"),
        (6, "CREATE TABLE IF NOT EXISTS tool_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"),
        (7, "CREATE INDEX IF NOT EXISTS idx_conversation_chat ON conversation_history(chat_id, created_at)"),
        (8, "CREATE INDEX IF NOT EXISTS idx_notes_segment ON notes(segment)"),
        (9, "CREATE INDEX IF NOT EXISTS idx_memory_segment ON memory(segment)"),
        (10, "ALTER TABLE conversation_history ADD COLUMN user_name TEXT NOT NULL DEFAULT ''"),
        (11, "CREATE INDEX IF NOT EXISTS idx_conversation_user ON conversation_history(chat_id, user_id)"),
        (12, "CREATE INDEX IF NOT EXISTS idx_conversation_content ON conversation_history(chat_id, content)"),
        (13, "ALTER TABLE conversation_history ADD COLUMN reply_to_id INTEGER"),
        (14, "ALTER TABLE conversation_history ADD COLUMN message_id INTEGER"),
        (15, "ALTER TABLE name_map ADD COLUMN username TEXT NOT NULL DEFAULT ''"),
        (16, "CREATE TABLE IF NOT EXISTS trigger_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE COLLATE NOCASE
        )"),
        (17, "CREATE TABLE IF NOT EXISTS mcp_servers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            transport TEXT NOT NULL DEFAULT 'stdio',
            endpoint TEXT NOT NULL DEFAULT '',
            command TEXT NOT NULL DEFAULT '',
            args TEXT NOT NULL DEFAULT '[]',
            environment TEXT NOT NULL DEFAULT '{}',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_by INTEGER,
            updated_by INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"),
    ];

    for (version, sql) in migrations {
        if version > current_version {
            sqlx::query(sql).execute(pool).await?;
            sqlx::query("INSERT INTO schema_version (version) VALUES (?)")
                .bind(version)
                .execute(pool)
                .await?;
            tracing::info!(version, "Applied migration");
        }
    }

    Ok(())
}

// --- Config ---

pub async fn config_get(pool: &SqlitePool, key: &str) -> Result<Option<String>> {
    let row: Option<(String,)> = sqlx::query_as("SELECT value FROM config WHERE key = ?")
        .bind(key)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|r| r.0))
}

pub async fn config_set(pool: &SqlitePool, key: &str, value: &str) -> Result<()> {
    sqlx::query(
        "INSERT INTO config (key, value) VALUES (?, ?)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    )
    .bind(key)
    .bind(value)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn config_list(pool: &SqlitePool) -> Result<Vec<(String, String)>> {
    let rows: Vec<(String, String)> = sqlx::query_as("SELECT key, value FROM config ORDER BY key")
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

// --- Name Map ---

pub async fn name_map_set(
    pool: &SqlitePool,
    entity_type: &str,
    entity_id: i64,
    display_name: &str,
    username: &str,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO name_map (entity_type, entity_id, display_name, username) VALUES (?, ?, ?, ?)
         ON CONFLICT(entity_type, entity_id) DO UPDATE SET display_name = excluded.display_name, username = excluded.username",
    )
    .bind(entity_type)
    .bind(entity_id)
    .bind(display_name)
    .bind(username)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn name_map_get(pool: &SqlitePool, entity_type: &str, entity_id: i64) -> Result<Option<String>> {
    let row: Option<(String,)> =
        sqlx::query_as("SELECT display_name FROM name_map WHERE entity_type = ? AND entity_id = ?")
            .bind(entity_type)
            .bind(entity_id)
            .fetch_optional(pool)
            .await?;
    Ok(row.map(|r| r.0))
}

/// Get both display_name and username for an entity
pub async fn name_map_get_full(pool: &SqlitePool, entity_type: &str, entity_id: i64) -> Result<Option<(String, String)>> {
    let row: Option<(String, String)> =
        sqlx::query_as("SELECT display_name, username FROM name_map WHERE entity_type = ? AND entity_id = ?")
            .bind(entity_type)
            .bind(entity_id)
            .fetch_optional(pool)
            .await?;
    Ok(row)
}

// --- Conversation History ---

pub async fn conversation_save(
    pool: &SqlitePool,
    chat_id: i64,
    user_id: i64,
    user_name: &str,
    role: &str,
    content: &str,
    tool_call_id: Option<&str>,
    message_id: Option<i64>,
    reply_to_id: Option<i64>,
) -> Result<i64> {
    let result = sqlx::query(
        "INSERT INTO conversation_history (chat_id, user_id, user_name, role, content, tool_call_id, message_id, reply_to_id)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(chat_id)
    .bind(user_id)
    .bind(user_name)
    .bind(role)
    .bind(content)
    .bind(tool_call_id)
    .bind(message_id)
    .bind(reply_to_id)
    .execute(pool)
    .await?;

    let row_id = result.last_insert_rowid();

    tracing::debug!(
        chat_id,
        user_id,
        user_name,
        role,
        content_len = content.len(),
        row_id,
        "Conversation saved to DB"
    );

    Ok(row_id)
}

pub async fn conversation_load(pool: &SqlitePool, chat_id: i64, limit: i64) -> Result<Vec<ConversationRow>> {
    let rows = sqlx::query(
        "SELECT id, chat_id, user_id, user_name, role, content, tool_call_id, message_id, reply_to_id, created_at
         FROM conversation_history
         WHERE chat_id = ?
         ORDER BY id DESC
         LIMIT ?",
    )
    .bind(chat_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let mut result: Vec<ConversationRow> = rows
        .iter()
        .map(|row| ConversationRow {
            id: row.get("id"),
            chat_id: row.get("chat_id"),
            user_id: row.get("user_id"),
            user_name: row.get("user_name"),
            role: row.get("role"),
            content: row.get("content"),
            tool_call_id: row.get("tool_call_id"),
            message_id: row.get("message_id"),
            reply_to_id: row.get("reply_to_id"),
            created_at: row.get("created_at"),
        })
        .collect();

    result.reverse(); // oldest first
    Ok(result)
}

pub async fn conversation_clear(pool: &SqlitePool, chat_id: i64) -> Result<u64> {
    let result = sqlx::query("DELETE FROM conversation_history WHERE chat_id = ?")
        .bind(chat_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

/// Search conversation history with various filters
pub async fn conversation_search(
    pool: &SqlitePool,
    chat_id: i64,
    keyword: Option<&str>,
    sender_name: Option<&str>,
    sender_id: Option<i64>,
    date_from: Option<&str>,
    date_to: Option<&str>,
    limit: i64,
    offset: i64,
) -> Result<Vec<ConversationRow>> {
    let mut sql = String::from(
        "SELECT id, chat_id, user_id, user_name, role, content, tool_call_id, message_id, reply_to_id, created_at
         FROM conversation_history WHERE chat_id = ?",
    );
    let mut bind_values: Vec<String> = vec![];

    if let Some(kw) = keyword {
        sql.push_str(" AND content LIKE ?");
        bind_values.push(format!("%{}%", kw));
    }
    if let Some(name) = sender_name {
        sql.push_str(" AND user_name LIKE ?");
        bind_values.push(format!("%{}%", name));
    }
    if let Some(sid) = sender_id {
        sql.push_str(" AND user_id = ?");
        bind_values.push(sid.to_string());
    }
    if let Some(df) = date_from {
        sql.push_str(" AND created_at >= ?");
        bind_values.push(df.to_string());
    }
    if let Some(dt) = date_to {
        sql.push_str(" AND created_at <= ?");
        bind_values.push(dt.to_string());
    }

    sql.push_str(" ORDER BY id DESC LIMIT ? OFFSET ?");

    let mut query = sqlx::query(&sql).bind(chat_id);
    for val in &bind_values {
        query = query.bind(val);
    }
    query = query.bind(limit).bind(offset);

    let rows = query.fetch_all(pool).await?;

    let mut result: Vec<ConversationRow> = rows
        .iter()
        .map(|row| ConversationRow {
            id: row.get("id"),
            chat_id: row.get("chat_id"),
            user_id: row.get("user_id"),
            user_name: row.get("user_name"),
            role: row.get("role"),
            content: row.get("content"),
            tool_call_id: row.get("tool_call_id"),
            message_id: row.get("message_id"),
            reply_to_id: row.get("reply_to_id"),
            created_at: row.get("created_at"),
        })
        .collect();

    result.reverse();
    Ok(result)
}

/// Search conversation history across ALL chats (no chat_id filter)
pub async fn conversation_search_global(
    pool: &SqlitePool,
    keyword: Option<&str>,
    sender_name: Option<&str>,
    sender_id: Option<i64>,
    date_from: Option<&str>,
    date_to: Option<&str>,
    limit: i64,
    offset: i64,
) -> Result<Vec<ConversationRow>> {
    let mut sql = String::from(
        "SELECT id, chat_id, user_id, user_name, role, content, tool_call_id, message_id, reply_to_id, created_at
         FROM conversation_history WHERE 1=1",
    );
    let mut bind_values: Vec<String> = vec![];

    if let Some(kw) = keyword {
        sql.push_str(" AND content LIKE ?");
        bind_values.push(format!("%{}%", kw));
    }
    if let Some(name) = sender_name {
        sql.push_str(" AND user_name LIKE ?");
        bind_values.push(format!("%{}%", name));
    }
    if let Some(sid) = sender_id {
        sql.push_str(" AND user_id = ?");
        bind_values.push(sid.to_string());
    }
    if let Some(df) = date_from {
        sql.push_str(" AND created_at >= ?");
        bind_values.push(df.to_string());
    }
    if let Some(dt) = date_to {
        sql.push_str(" AND created_at <= ?");
        bind_values.push(dt.to_string());
    }

    sql.push_str(" ORDER BY id DESC LIMIT ? OFFSET ?");

    let mut query = sqlx::query(&sql);
    for val in &bind_values {
        query = query.bind(val);
    }
    query = query.bind(limit).bind(offset);

    let rows = query.fetch_all(pool).await?;

    let mut result: Vec<ConversationRow> = rows
        .iter()
        .map(|row| ConversationRow {
            id: row.get("id"),
            chat_id: row.get("chat_id"),
            user_id: row.get("user_id"),
            user_name: row.get("user_name"),
            role: row.get("role"),
            content: row.get("content"),
            tool_call_id: row.get("tool_call_id"),
            message_id: row.get("message_id"),
            reply_to_id: row.get("reply_to_id"),
            created_at: row.get("created_at"),
        })
        .collect();

    result.reverse();
    Ok(result)
}

/// Get the "important" pinned memory for a chat segment
pub async fn get_important_memory(pool: &SqlitePool, segment: &str) -> Result<Option<String>> {
    let key = "__important__";
    memory_get(pool, segment, key).await
}

/// Set the "important" pinned memory for a segment (max 500 chars)
pub async fn set_important_memory(pool: &SqlitePool, segment: &str, value: &str) -> Result<()> {
    if value.len() > 500 {
        anyhow::bail!("Important memory is limited to 500 characters (got {})", value.len());
    }
    let key = "__important__";
    memory_set(pool, segment, key, value).await
}

/// Clear the "important" pinned memory for a segment
pub async fn clear_important_memory(pool: &SqlitePool, segment: &str) -> Result<bool> {
    let result = sqlx::query("DELETE FROM memory WHERE segment = ? AND key = '__important__'")
        .bind(segment)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// --- Notes ---

pub async fn note_create(
    pool: &SqlitePool,
    segment: &str,
    title: &str,
    content: &str,
    tags: &str,
) -> Result<i64> {
    let result = sqlx::query(
        "INSERT INTO notes (segment, title, content, tags) VALUES (?, ?, ?, ?)",
    )
    .bind(segment)
    .bind(title)
    .bind(content)
    .bind(tags)
    .execute(pool)
    .await?;
    let id = result.last_insert_rowid();

    tracing::info!(id, segment, title, tags, "Note saved to DB");

    Ok(id)
}

pub async fn note_read(pool: &SqlitePool, note_id: i64) -> Result<Option<NoteRow>> {
    let row = sqlx::query(
        "SELECT id, segment, title, content, tags, created_at, updated_at FROM notes WHERE id = ?",
    )
    .bind(note_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| NoteRow {
        id: r.get("id"),
        segment: r.get("segment"),
        title: r.get("title"),
        content: r.get("content"),
        tags: r.get("tags"),
        created_at: r.get("created_at"),
        updated_at: r.get("updated_at"),
    }))
}

pub async fn note_delete(pool: &SqlitePool, note_id: i64) -> Result<bool> {
    let result = sqlx::query("DELETE FROM notes WHERE id = ?")
        .bind(note_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn note_search(pool: &SqlitePool, query: &str, segment: Option<&str>) -> Result<Vec<NoteRow>> {
    let pattern = format!("%{}%", query);

    let rows = if let Some(seg) = segment {
        sqlx::query(
            "SELECT id, segment, title, content, tags, created_at, updated_at FROM notes
             WHERE segment = ? AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)
             ORDER BY updated_at DESC LIMIT 20",
        )
        .bind(seg)
        .bind(&pattern)
        .bind(&pattern)
        .bind(&pattern)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query(
            "SELECT id, segment, title, content, tags, created_at, updated_at FROM notes
             WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
             ORDER BY updated_at DESC LIMIT 20",
        )
        .bind(&pattern)
        .bind(&pattern)
        .bind(&pattern)
        .fetch_all(pool)
        .await?
    };

    Ok(rows
        .iter()
        .map(|r| NoteRow {
            id: r.get("id"),
            segment: r.get("segment"),
            title: r.get("title"),
            content: r.get("content"),
            tags: r.get("tags"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        })
        .collect())
}

// --- Memory ---

pub async fn memory_set(pool: &SqlitePool, segment: &str, key: &str, value: &str) -> Result<()> {
    sqlx::query(
        "INSERT INTO memory (segment, key, value) VALUES (?, ?, ?)
         ON CONFLICT(segment, key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')",
    )
    .bind(segment)
    .bind(key)
    .bind(value)
    .execute(pool)
    .await?;

    tracing::info!(segment, key, value, "Memory saved to DB");

    Ok(())
}

pub async fn memory_get(pool: &SqlitePool, segment: &str, key: &str) -> Result<Option<String>> {
    let row: Option<(String,)> =
        sqlx::query_as("SELECT value FROM memory WHERE segment = ? AND key = ?")
            .bind(segment)
            .bind(key)
            .fetch_optional(pool)
            .await?;
    Ok(row.map(|r| r.0))
}

pub async fn memory_list(pool: &SqlitePool, segment: &str) -> Result<Vec<MemoryRow>> {
    let rows = sqlx::query(
        "SELECT id, segment, key, value, created_at, updated_at FROM memory
         WHERE segment = ? ORDER BY key",
    )
    .bind(segment)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .iter()
        .map(|r| MemoryRow {
            id: r.get("id"),
            segment: r.get("segment"),
            key: r.get("key"),
            value: r.get("value"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        })
        .collect())
}

// --- Tool Call Log ---

pub async fn tool_call_log(
    pool: &SqlitePool,
    chat_id: i64,
    user_id: i64,
    tool_name: &str,
    arguments: &str,
    result: &str,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO tool_call_log (chat_id, user_id, tool_name, arguments, result)
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(chat_id)
    .bind(user_id)
    .bind(tool_name)
    .bind(arguments)
    .bind(result)
    .execute(pool)
    .await?;
    Ok(())
}

// --- Trigger Keywords ---

pub async fn trigger_keywords_list(pool: &SqlitePool) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as("SELECT keyword FROM trigger_keywords ORDER BY keyword")
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(|r| r.0).collect())
}

pub async fn trigger_keyword_add(pool: &SqlitePool, keyword: &str) -> Result<bool> {
    let result = sqlx::query("INSERT OR IGNORE INTO trigger_keywords (keyword) VALUES (?)")
        .bind(keyword)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn trigger_keyword_remove(pool: &SqlitePool, keyword: &str) -> Result<bool> {
    let result = sqlx::query("DELETE FROM trigger_keywords WHERE keyword = ? COLLATE NOCASE")
        .bind(keyword)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// --- MCP Servers ---

pub async fn mcp_server_list(pool: &SqlitePool, include_disabled: bool) -> Result<Vec<McpServerRow>> {
    let sql = if include_disabled {
        "SELECT id, name, description, transport, endpoint, command, args, environment, enabled, created_by, updated_by, created_at, updated_at
         FROM mcp_servers
         ORDER BY name ASC"
    } else {
        "SELECT id, name, description, transport, endpoint, command, args, environment, enabled, created_by, updated_by, created_at, updated_at
         FROM mcp_servers
         WHERE enabled = 1
         ORDER BY name ASC"
    };

    let rows = sqlx::query(sql).fetch_all(pool).await?;

    Ok(rows
        .iter()
        .map(|row| McpServerRow {
            id: row.get("id"),
            name: row.get("name"),
            description: row.get("description"),
            transport: row.get("transport"),
            endpoint: row.get("endpoint"),
            command: row.get("command"),
            args: row.get("args"),
            environment: row.get("environment"),
            enabled: row.get::<i64, _>("enabled") != 0,
            created_by: row.try_get("created_by").ok(),
            updated_by: row.try_get("updated_by").ok(),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect())
}

pub async fn mcp_server_get(pool: &SqlitePool, name: &str) -> Result<Option<McpServerRow>> {
    let row = sqlx::query(
        "SELECT id, name, description, transport, endpoint, command, args, environment, enabled, created_by, updated_by, created_at, updated_at
         FROM mcp_servers WHERE name = ?",
    )
    .bind(name)
    .fetch_optional(pool)
    .await?;

    let row = match row {
        Some(row) => row,
        None => return Ok(None),
    };

    Ok(Some(McpServerRow {
        id: row.get("id"),
        name: row.get("name"),
        description: row.get("description"),
        transport: row.get("transport"),
        endpoint: row.get("endpoint"),
        command: row.get("command"),
        args: row.get("args"),
        environment: row.get("environment"),
        enabled: row.get::<i64, _>("enabled") != 0,
        created_by: row.try_get("created_by").ok(),
        updated_by: row.try_get("updated_by").ok(),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }))
}

pub async fn mcp_server_create(
    pool: &SqlitePool,
    name: &str,
    description: &str,
    transport: &str,
    endpoint: &str,
    command: &str,
    args: &str,
    environment: &str,
    enabled: bool,
    actor_id: Option<i64>,
) -> Result<i64> {
    if name.trim().is_empty() {
        anyhow::bail!("MCP server name cannot be empty");
    }

    if mcp_server_get(pool, name).await?.is_some() {
        anyhow::bail!("MCP server '{}' already exists", name);
    }

    let result = sqlx::query(
        "INSERT INTO mcp_servers (name, description, transport, endpoint, command, args, environment, enabled, created_by, updated_by)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(name)
    .bind(description)
    .bind(transport)
    .bind(endpoint)
    .bind(command)
    .bind(args)
    .bind(environment)
    .bind(if enabled { 1 } else { 0 })
    .bind(actor_id)
    .bind(actor_id)
    .execute(pool)
    .await?;

    Ok(result.last_insert_rowid())
}

pub async fn mcp_server_update(
    pool: &SqlitePool,
    current_name: &str,
    name: &str,
    description: &str,
    transport: &str,
    endpoint: &str,
    command: &str,
    args: &str,
    environment: &str,
    enabled: bool,
    actor_id: Option<i64>,
) -> Result<bool> {
    if name.trim().is_empty() {
        anyhow::bail!("MCP server name cannot be empty");
    }

    let result = sqlx::query(
        "UPDATE mcp_servers
         SET name = ?, description = ?, transport = ?, endpoint = ?, command = ?, args = ?, environment = ?, enabled = ?, updated_at = datetime('now'), updated_by = ?
         WHERE name = ?",
    )
    .bind(name)
    .bind(description)
    .bind(transport)
    .bind(endpoint)
    .bind(command)
    .bind(args)
    .bind(environment)
    .bind(if enabled { 1 } else { 0 })
    .bind(actor_id)
    .bind(current_name)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

pub async fn mcp_server_delete(pool: &SqlitePool, name: &str) -> Result<bool> {
    let result = sqlx::query("DELETE FROM mcp_servers WHERE name = ?")
        .bind(name)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

// --- Raw SQL (CLI only) ---

pub async fn raw_query(pool: &SqlitePool, sql: &str) -> Result<Vec<Vec<(String, String)>>> {
    let rows = sqlx::query(sql).fetch_all(pool).await?;
    let mut result = Vec::new();
    for row in &rows {
        let mut cols = Vec::new();
        for col in row.columns() {
            let name = col.name().to_string();
            let val: String = row
                .try_get::<String, _>(col.ordinal())
                .or_else(|_| row.try_get::<i64, _>(col.ordinal()).map(|v| v.to_string()))
                .or_else(|_| row.try_get::<f64, _>(col.ordinal()).map(|v| v.to_string()))
                .unwrap_or_else(|_| "NULL".to_string());
            cols.push((name, val));
        }
        result.push(cols);
    }
    Ok(result)
}

pub async fn raw_modify(pool: &SqlitePool, sql: &str) -> Result<u64> {
    // Block modification of tg_bot_token via raw SQL
    let lower = sql.to_lowercase();
    if lower.contains("tg_bot_token") {
        anyhow::bail!("Cannot modify tg_bot_token via raw SQL. Use 'astartebot config set tg_bot_token <value>' instead.");
    }
    let result = sqlx::query(sql).execute(pool).await?;
    Ok(result.rows_affected())
}
