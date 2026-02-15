use anyhow::Result;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Create a backup of the current working directory as a .tgz archive.
///
/// The archive is placed one level above the working directory with the name
/// `astarte_backup_YYYY-MM-DD-HH-MM-<reason>.tgz`.
pub async fn create_backup(reason: &str) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let timestamp = chrono::Local::now().format("%Y-%m-%d-%H-%M").to_string();
    let filename = format!("astarte_backup_{}-{}.tgz", timestamp, reason);

    let dest = cwd
        .parent()
        .ok_or_else(|| anyhow::anyhow!("working directory has no parent"))?
        .join(&filename);

    tracing::info!(dest = %dest.display(), reason, "Creating backup");

    let output = tokio::process::Command::new("tar")
        .arg("czf")
        .arg(&dest)
        .arg("-C")
        .arg(&cwd)
        .arg(".")
        .output()
        .await?;

    if output.status.success() {
        tracing::info!(dest = %dest.display(), "Backup created successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!(%stderr, "Backup failed");
        Err(anyhow::anyhow!("tar failed: {}", stderr))
    }
}

/// Spawn a background task that creates an hourly backup.
///
/// Returns the join handle so the caller can abort it on shutdown if needed.
pub fn start_hourly_backup() -> JoinHandle<()> {
    tokio::spawn(async {
        loop {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            if let Err(e) = create_backup("hourly").await {
                tracing::error!(error = %e, "Hourly backup failed");
            }
        }
    })
}
