mod backup;
mod bot;
mod config;
mod db;
mod llm;
mod logging;
mod mcp;
mod memory;
mod rag;
mod tools;
mod types;

use anyhow::Result;
use clap::{Parser, Subcommand};

const DB_PATH: &str = "sqlite:astartebot.db";
const LOG_DIR: &str = "logs";

fn mask_sensitive_value(value: &str) -> String {
    let prefix: String = value.chars().take(4).collect();
    let suffix: String = value
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{}...{}", prefix, suffix)
}

#[derive(Parser)]
#[command(
    name = "astartebot",
    version,
    about = "AstarteBot - Telegram AI Assistant"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Telegram bot
    Run,
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Direct database access (read/write)
    Db {
        #[command(subcommand)]
        action: DbAction,
    },
    /// Manage trigger keywords (words that activate the LLM in group chats)
    Trigger {
        #[command(subcommand)]
        action: TriggerAction,
    },
    /// RAG semantic search engine management
    Rag {
        #[command(subcommand)]
        action: RagAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Set a config value
    Set { key: String, value: String },
    /// Get a config value
    Get { key: String },
    /// List all config values
    List,
}

#[derive(Subcommand)]
enum DbAction {
    /// Run a read-only SQL query
    Query { sql: String },
    /// Run a write SQL statement (blocks tg_bot_token modification)
    Modify { sql: String },
}

#[derive(Subcommand)]
enum TriggerAction {
    /// Add a trigger keyword
    Add { keyword: String },
    /// Remove a trigger keyword
    Remove { keyword: String },
    /// List all trigger keywords
    List,
}

#[derive(Subcommand)]
enum RagAction {
    /// Rebuild the RAG index from all existing SQLite data
    Reindex,
    /// Show RAG engine statistics
    Stats,
    /// Test embedding quality with diagnostic pairs
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Run => {
            logging::init(LOG_DIR)?;
            let pool = db::create_pool(DB_PATH).await?;
            bot::run(pool).await?;
        }
        Commands::Config { action } => {
            let pool = db::create_pool(DB_PATH).await?;
            match action {
                ConfigAction::Set { key, value } => {
                    config::set(&pool, key, value).await?;
                    let display_val = if key.contains("token") || key.contains("secret") {
                        mask_sensitive_value(value)
                    } else {
                        value.clone()
                    };
                    println!("Set {} = {}", key, display_val);
                }
                ConfigAction::Get { key } => match config::get(&pool, key).await? {
                    Some(value) => {
                        let display_val = if key.contains("token") || key.contains("secret") {
                            mask_sensitive_value(&value)
                        } else {
                            value
                        };
                        println!("{} = {}", key, display_val);
                    }
                    None => println!("Key '{}' not found", key),
                },
                ConfigAction::List => {
                    let items = config::list(&pool).await?;
                    if items.is_empty() {
                        println!("No config values set.");
                    } else {
                        for (key, value) in items {
                            let display_val = if key.contains("token") || key.contains("secret") {
                                mask_sensitive_value(&value)
                            } else {
                                value
                            };
                            println!("{} = {}", key, display_val);
                        }
                    }
                }
            }
        }
        Commands::Db { action } => {
            let pool = db::create_pool(DB_PATH).await?;
            match action {
                DbAction::Query { sql } => {
                    let rows = db::raw_query(&pool, sql).await?;
                    if rows.is_empty() {
                        println!("(no results)");
                    } else {
                        for row in &rows {
                            let line: Vec<String> =
                                row.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
                            println!("{}", line.join(" | "));
                        }
                        println!("({} rows)", rows.len());
                    }
                }
                DbAction::Modify { sql } => {
                    let affected = db::raw_modify(&pool, sql).await?;
                    println!("{} row(s) affected", affected);
                }
            }
        }
        Commands::Rag { action } => {
            let pool = db::create_pool(DB_PATH).await?;
            let rag_engine = rag::RagEngine::init(&std::path::PathBuf::from("rag_data")).await?;
            match action {
                RagAction::Reindex => {
                    let count = rag_engine.reindex_all(&pool).await?;
                    println!("Reindexed {} records into RAG engine", count);
                }
                RagAction::Stats => {
                    let (vector_count, metadata_count, next_id) = rag_engine.stats();
                    println!("RAG Engine Statistics:");
                    println!("  Vectors in index:  {}", vector_count);
                    println!("  Metadata entries:  {}", metadata_count);
                    println!("  Next vector ID:    {}", next_id);
                }
                RagAction::Test => {
                    println!("=== RAG Embedding Diagnostic ===\n");
                    let pairs: Vec<(&str, &str)> = vec![
                        (
                            "birthday celebration with cake and party",
                            "Lera's birthday at Atlantis with a wonderful cake",
                        ),
                        (
                            "cooking Italian pasta with tomato sauce",
                            "making spaghetti bolognese for dinner",
                        ),
                        ("cooking Italian pasta", "quantum physics lecture notes"),
                        ("Hello world", "Hello world"),
                        ("cats ginger and black", "рыжий кот и чёрная кошка"),
                    ];
                    for (a, b) in &pairs {
                        let va = rag_engine.embed_text(a)?;
                        let vb = rag_engine.embed_text(b)?;
                        let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
                        let norm_a: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm_b: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        println!("A: \"{}\"", a);
                        println!("B: \"{}\"", b);
                        println!(
                            "  dot={:.6}  |a|={:.6}  |b|={:.6}  cosine={:.6}\n",
                            dot,
                            norm_a,
                            norm_b,
                            dot / (norm_a * norm_b)
                        );
                    }
                    let v = rag_engine.embed_text("Hello world")?;
                    println!("Embedding 'Hello world' first 10 dims: {:?}", &v[..10]);
                    println!(
                        "Norm: {:.6}  Dim: {}",
                        v.iter().map(|x| x * x).sum::<f32>().sqrt(),
                        v.len()
                    );
                }
            }
        }
        Commands::Trigger { action } => {
            let pool = db::create_pool(DB_PATH).await?;
            match action {
                TriggerAction::Add { keyword } => {
                    if db::trigger_keyword_add(&pool, keyword).await? {
                        println!("Added trigger keyword: {}", keyword);
                    } else {
                        println!("Keyword '{}' already exists", keyword);
                    }
                }
                TriggerAction::Remove { keyword } => {
                    if db::trigger_keyword_remove(&pool, keyword).await? {
                        println!("Removed trigger keyword: {}", keyword);
                    } else {
                        println!("Keyword '{}' not found", keyword);
                    }
                }
                TriggerAction::List => {
                    let keywords = db::trigger_keywords_list(&pool).await?;
                    if keywords.is_empty() {
                        println!(
                            "No trigger keywords set. The bot will only respond to @mentions and replies in groups."
                        );
                    } else {
                        println!("Trigger keywords ({}):", keywords.len());
                        for kw in &keywords {
                            println!("  {}", kw);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
