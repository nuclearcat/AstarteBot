use anyhow::Result;
use sqlx::SqlitePool;

use crate::db;

pub async fn get(pool: &SqlitePool, key: &str) -> Result<Option<String>> {
    db::config_get(pool, key).await
}

pub async fn set(pool: &SqlitePool, key: &str, value: &str) -> Result<()> {
    db::config_set(pool, key, value).await
}

pub async fn list(pool: &SqlitePool) -> Result<Vec<(String, String)>> {
    db::config_list(pool).await
}

pub async fn get_or_default(pool: &SqlitePool, key: &str, default: &str) -> Result<String> {
    match db::config_get(pool, key).await? {
        Some(v) => Ok(v),
        None => Ok(default.to_string()),
    }
}

pub async fn get_required(pool: &SqlitePool, key: &str) -> Result<String> {
    db::config_get(pool, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Required config key '{}' not set. Use: astartebot config set {} <value>", key, key))
}

/// Get the Telegram bot token from config or env var
pub async fn get_telegram_token(pool: &SqlitePool) -> Result<String> {
    // Try env var first
    if let Ok(token) = std::env::var("TELEGRAM_BOT_TOKEN") {
        if !token.is_empty() {
            return Ok(token);
        }
    }
    // Fall back to config
    get_required(pool, "tg_bot_token").await
}
