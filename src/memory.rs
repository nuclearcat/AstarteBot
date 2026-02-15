use anyhow::Result;
use sqlx::SqlitePool;

use crate::db;
use crate::types::MemoryRow;

/// Validate segment format: "chat:{id}", "person:{id}", "global", "bot"
pub fn validate_segment(segment: &str) -> Result<()> {
    if segment == "global" || segment == "bot" {
        return Ok(());
    }
    if let Some(id_str) = segment.strip_prefix("chat:") {
        id_str
            .parse::<i64>()
            .map_err(|_| anyhow::anyhow!("Invalid chat segment ID: {}", id_str))?;
        return Ok(());
    }
    if let Some(id_str) = segment.strip_prefix("person:") {
        id_str
            .parse::<i64>()
            .map_err(|_| anyhow::anyhow!("Invalid person segment ID: {}", id_str))?;
        return Ok(());
    }
    anyhow::bail!(
        "Invalid segment '{}'. Must be 'global', 'bot', 'chat:{{id}}', or 'person:{{id}}'",
        segment
    )
}

pub async fn set(pool: &SqlitePool, segment: &str, key: &str, value: &str) -> Result<()> {
    validate_segment(segment)?;
    db::memory_set(pool, segment, key, value).await
}

pub async fn get(pool: &SqlitePool, segment: &str, key: &str) -> Result<Option<String>> {
    validate_segment(segment)?;
    db::memory_get(pool, segment, key).await
}

pub async fn list(pool: &SqlitePool, segment: &str) -> Result<Vec<MemoryRow>> {
    validate_segment(segment)?;
    db::memory_list(pool, segment).await
}
