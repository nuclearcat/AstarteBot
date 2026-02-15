use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokio::task::spawn_blocking;

const EMBEDDING_DIM: usize = 384;
const MAX_TOKENS: usize = 128; // paraphrase-multilingual-MiniLM-L12-v2 limit
const SEP_TOKEN_ID: u32 = 102;

/// Simple brute-force vector index (replaces FAISS FlatIndex with InnerProduct).
/// For L2-normalized vectors, inner product == cosine similarity.
struct FlatIndex {
    vectors: Vec<f32>, // flat storage: vectors.len() == n * EMBEDDING_DIM
    count: usize,
}

impl FlatIndex {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            count: 0,
        }
    }

    fn add(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), EMBEDDING_DIM);
        self.vectors.extend_from_slice(vec);
        self.count += 1;
    }

    fn ntotal(&self) -> usize {
        self.count
    }

    /// Search for top-k nearest neighbors by inner product (descending).
    /// Returns (positions, scores) sorted by score descending.
    fn search(&self, query: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        if self.count == 0 || k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Compute inner products with all vectors
        let mut scores: Vec<(usize, f32)> = (0..self.count)
            .map(|i| {
                let start = i * EMBEDDING_DIM;
                let end = start + EMBEDDING_DIM;
                let dot: f32 = query
                    .iter()
                    .zip(&self.vectors[start..end])
                    .map(|(a, b)| a * b)
                    .sum();
                (i, dot)
            })
            .collect();

        // Partial sort: only need top-k
        let k = k.min(self.count);
        scores.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let positions: Vec<usize> = scores.iter().map(|(pos, _)| *pos).collect();
        let dists: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
        (positions, dists)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagMetadata {
    pub source_type: String,
    pub source_id: i64,
    pub chat_id: i64,
    pub segment: String,
    pub content_preview: String,
    pub user_name: String,
    pub created_at: String,
}

pub struct RagResult {
    pub score: f32,
    pub metadata: RagMetadata,
}

pub struct RagEngine {
    model: candle_onnx::onnx::ModelProto,
    tokenizer: tokenizers::Tokenizer,
    index: Mutex<FlatIndex>,
    position_to_id: Mutex<Vec<i64>>,
    meta_db: rocksdb::DB,
    next_vector_id: Mutex<i64>,
}

// Safety: candle_onnx::onnx::ModelProto is read-only after init, tokenizers::Tokenizer::encode takes &self,
// rocksdb::DB is internally thread-safe. All mutable state is behind Mutex.
unsafe impl Send for RagEngine {}
unsafe impl Sync for RagEngine {}

impl RagEngine {
    pub async fn init(data_dir: &Path) -> Result<Self> {
        let data_dir = data_dir.to_path_buf();
        std::fs::create_dir_all(&data_dir)
            .context("Failed to create RAG data directory")?;

        // Download model files via hf-hub (cached after first download)
        let (model_path, tokenizer_path) = {
            spawn_blocking(move || -> Result<(PathBuf, PathBuf)> {
                let api = hf_hub::api::sync::Api::new()
                    .context("Failed to create HuggingFace API client")?;
                let repo = api.model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string());
                let model_path = repo
                    .get("onnx/model.onnx")
                    .context("Failed to download model.onnx")?;
                let tokenizer_path = repo
                    .get("tokenizer.json")
                    .context("Failed to download tokenizer.json")?;
                Ok((model_path, tokenizer_path))
            })
            .await??
        };

        tracing::info!(?model_path, ?tokenizer_path, "Model files ready");

        // Load ONNX model (CPU-bound)
        let mp = model_path.clone();
        let model = spawn_blocking(move || {
            candle_onnx::read_file(mp).context("Failed to load ONNX model")
        })
        .await??;

        // Load tokenizer
        let tp = tokenizer_path.clone();
        let tokenizer = spawn_blocking(move || {
            tokenizers::Tokenizer::from_file(tp)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
        })
        .await??;

        // Open RocksDB
        let rocks_path = data_dir.join("rocksdb");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let meta_db = rocksdb::DB::open(&opts, &rocks_path)
            .context("Failed to open RocksDB")?;

        // Load next_vector_id from RocksDB
        let next_id = match meta_db.get(b"__next_vector_id__")? {
            Some(bytes) if bytes.len() == 8 => {
                i64::from_le_bytes(bytes[..8].try_into().unwrap())
            }
            _ => 0,
        };

        let engine = Self {
            model,
            tokenizer,
            index: Mutex::new(FlatIndex::new()),
            position_to_id: Mutex::new(Vec::new()),
            meta_db,
            next_vector_id: Mutex::new(next_id),
        };

        // Rebuild index from stored vectors
        let count = engine.rebuild_index_from_rocksdb()?;
        tracing::info!(count, next_id, "RAG engine initialized");

        Ok(engine)
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {}", e))?;

        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let mut token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

        // Truncate to MAX_TOKENS
        if input_ids.len() > MAX_TOKENS {
            input_ids.truncate(MAX_TOKENS);
            attention_mask.truncate(MAX_TOKENS);
            token_type_ids.truncate(MAX_TOKENS);
            // Fix last token to [SEP]
            if let Some(last) = input_ids.last_mut() {
                *last = SEP_TOKEN_ID;
            }
        }

        let seq_len = input_ids.len();
        let device = &candle_core::Device::Cpu;

        // Create tensors with shape [1, seq_len]
        let ids_tensor = candle_core::Tensor::from_vec(
            input_ids.iter().map(|&x| x as i64).collect::<Vec<i64>>(),
            (1, seq_len),
            device,
        )?;
        let mask_tensor = candle_core::Tensor::from_vec(
            attention_mask.iter().map(|&x| x as i64).collect::<Vec<i64>>(),
            (1, seq_len),
            device,
        )?;
        let type_tensor = candle_core::Tensor::from_vec(
            token_type_ids.iter().map(|&x| x as i64).collect::<Vec<i64>>(),
            (1, seq_len),
            device,
        )?;

        // Build input map for ONNX model
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("input_ids".to_string(), ids_tensor);
        inputs.insert("attention_mask".to_string(), mask_tensor);
        inputs.insert("token_type_ids".to_string(), type_tensor);

        // Run inference
        let outputs = candle_onnx::simple_eval(&self.model, inputs)
            .map_err(|e| anyhow::anyhow!("ONNX eval failed: {}", e))?;

        // Get last_hidden_state [1, seq_len, 384]
        let hidden = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No output from ONNX model"))?;

        // Mean pooling with attention mask
        let hidden = hidden.squeeze(0)?; // [seq_len, 384]
        let mask_f32 = candle_core::Tensor::from_vec(
            attention_mask.iter().map(|&x| x as f32).collect::<Vec<f32>>(),
            (seq_len, 1),
            device,
        )?;

        // Multiply hidden states by mask, then sum and divide
        let masked = hidden.broadcast_mul(&mask_f32)?; // [seq_len, 384]
        let summed = masked.sum(0)?; // [384]
        let mask_sum = mask_f32.sum(0)?.broadcast_as(summed.shape())?; // [384]
        let pooled = summed.broadcast_div(&mask_sum)?; // [384]

        // L2 normalize
        let norm = pooled
            .sqr()?
            .sum_all()?
            .sqrt()?
            .to_scalar::<f32>()?;

        let normalized = if norm > 0.0 {
            (pooled / norm as f64)?
        } else {
            pooled
        };

        let vec: Vec<f32> = normalized.to_vec1()?;
        Ok(vec)
    }

    pub fn index_record_sync(
        &self,
        source_type: &str,
        source_id: i64,
        chat_id: i64,
        segment: &str,
        content: &str,
        user_name: &str,
        created_at: &str,
    ) -> Result<()> {
        // Skip short content
        if content.trim().len() < 10 {
            return Ok(());
        }

        let dedup_key = format!("dedup:{}:{}", source_type, source_id);

        // Check for duplicates
        if let Some(existing_bytes) = self.meta_db.get(dedup_key.as_bytes())? {
            if source_type == "conversation" {
                // Conversations are immutable — skip
                return Ok(());
            }
            // For notes/memory, remove old entry
            if existing_bytes.len() == 8 {
                let old_id = i64::from_le_bytes(existing_bytes[..8].try_into().unwrap());
                let vec_key = format!("vec:{}", old_id);
                let raw_key = format!("raw:{}", old_id);
                self.meta_db.delete(vec_key.as_bytes())?;
                self.meta_db.delete(raw_key.as_bytes())?;
                self.meta_db.delete(dedup_key.as_bytes())?;
                // Orphan in index: search will skip results with missing metadata
            }
        }

        // Embed the content
        let embedding = self.embed_text(content)?;

        // Allocate vector_id
        let vector_id = {
            let mut next = self.next_vector_id.lock().unwrap();
            let id = *next;
            *next += 1;
            // Persist counter
            self.meta_db
                .put(b"__next_vector_id__", (*next).to_le_bytes())?;
            id
        };

        // Build content preview (first 200 chars)
        let content_preview: String = content.chars().take(200).collect();

        let metadata = RagMetadata {
            source_type: source_type.to_string(),
            source_id,
            chat_id,
            segment: segment.to_string(),
            content_preview,
            user_name: user_name.to_string(),
            created_at: created_at.to_string(),
        };

        // Store metadata in RocksDB
        let vec_key = format!("vec:{}", vector_id);
        let meta_json = serde_json::to_vec(&metadata)?;
        self.meta_db.put(vec_key.as_bytes(), &meta_json)?;

        // Store raw vector bytes
        let raw_key = format!("raw:{}", vector_id);
        let raw_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.meta_db.put(raw_key.as_bytes(), &raw_bytes)?;

        // Store dedup key
        self.meta_db
            .put(dedup_key.as_bytes(), vector_id.to_le_bytes())?;

        // Add to in-memory index
        {
            let mut index = self.index.lock().unwrap();
            let mut pos_map = self.position_to_id.lock().unwrap();
            index.add(&embedding);
            pos_map.push(vector_id);
        }

        Ok(())
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
        source_type_filter: Option<&str>,
    ) -> Result<Vec<RagResult>> {
        let index = self.index.lock().unwrap();

        if index.ntotal() == 0 {
            return Ok(Vec::new());
        }

        let query_vec = self.embed_text(query)?;

        // Over-fetch to compensate for filtered/orphaned results
        let k = (limit * 3).min(index.ntotal()).max(1);

        let (positions, scores) = index.search(&query_vec, k);

        let pos_map = self.position_to_id.lock().unwrap();
        let mut results = Vec::new();

        for (i, &position) in positions.iter().enumerate() {
            if position >= pos_map.len() {
                continue;
            }

            let vector_id = pos_map[position];
            let vec_key = format!("vec:{}", vector_id);

            // Look up metadata in RocksDB
            let meta_bytes = match self.meta_db.get(vec_key.as_bytes())? {
                Some(b) => b,
                None => continue, // Orphaned vector, skip
            };

            let metadata: RagMetadata = match serde_json::from_slice(&meta_bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Apply source_type filter
            if let Some(filter) = source_type_filter {
                if metadata.source_type != filter {
                    continue;
                }
            }

            results.push(RagResult {
                score: scores[i],
                metadata,
            });

            if results.len() >= limit {
                break;
            }
        }

        Ok(results)
    }

    pub async fn reindex_all(&self, pool: &SqlitePool) -> Result<usize> {
        tracing::info!("Starting full RAG reindex...");

        // Clear everything
        {
            let mut index = self.index.lock().unwrap();
            *index = FlatIndex::new();
        }
        {
            let mut pos_map = self.position_to_id.lock().unwrap();
            pos_map.clear();
        }
        {
            let mut next = self.next_vector_id.lock().unwrap();
            *next = 0;
            self.meta_db.put(b"__next_vector_id__", 0i64.to_le_bytes())?;
        }

        // Clear all RocksDB keys
        let mut batch = rocksdb::WriteBatch::default();
        let iter = self.meta_db.iterator(rocksdb::IteratorMode::Start);
        for item in iter {
            let (key, _) = item?;
            batch.delete(&key);
        }
        self.meta_db.write(batch)?;

        let mut count: usize = 0;

        // Index conversation_history
        let conv_rows: Vec<(i64, i64, i64, String, String, String, String)> = sqlx::query_as(
            "SELECT id, chat_id, user_id, user_name, role, content, created_at
             FROM conversation_history
             WHERE role IN ('user', 'assistant') AND LENGTH(content) >= 10
             ORDER BY id",
        )
        .fetch_all(pool)
        .await?;

        tracing::info!(total = conv_rows.len(), "Indexing conversations...");
        for (id, chat_id, _user_id, user_name, _role, content, created_at) in &conv_rows {
            let segment = format!("chat:{}", chat_id);
            if let Err(e) = self.index_record_sync(
                "conversation",
                *id,
                *chat_id,
                &segment,
                content,
                user_name,
                created_at,
            ) {
                tracing::warn!(id, error = %e, "Failed to index conversation");
            }
            count += 1;
            if count % 500 == 0 {
                tracing::info!(count, "Reindex progress...");
            }
        }

        // Index notes
        let note_rows: Vec<(i64, String, String, String, String, String)> = sqlx::query_as(
            "SELECT id, segment, title, content, tags, created_at FROM notes ORDER BY id",
        )
        .fetch_all(pool)
        .await?;

        tracing::info!(total = note_rows.len(), "Indexing notes...");
        for (id, segment, title, content, tags, created_at) in &note_rows {
            let embed_text = format!("{}\n{}\n{}", title, content, tags);
            if let Err(e) =
                self.index_record_sync("note", *id, 0, segment, &embed_text, "", created_at)
            {
                tracing::warn!(id, error = %e, "Failed to index note");
            }
            count += 1;
            if count % 500 == 0 {
                tracing::info!(count, "Reindex progress...");
            }
        }

        // Index memory (exclude __important__)
        let mem_rows: Vec<(i64, String, String, String, String)> = sqlx::query_as(
            "SELECT id, segment, key, value, created_at FROM memory
             WHERE key != '__important__' AND LENGTH(value) >= 10
             ORDER BY id",
        )
        .fetch_all(pool)
        .await?;

        tracing::info!(total = mem_rows.len(), "Indexing memory...");
        for (id, segment, key, value, created_at) in &mem_rows {
            let embed_text = format!("{}: {}", key, value);
            if let Err(e) =
                self.index_record_sync("memory", *id, 0, segment, &embed_text, "", created_at)
            {
                tracing::warn!(id, error = %e, "Failed to index memory");
            }
            count += 1;
            if count % 500 == 0 {
                tracing::info!(count, "Reindex progress...");
            }
        }

        tracing::info!(count, "RAG reindex complete");
        Ok(count)
    }

    fn rebuild_index_from_rocksdb(&self) -> Result<usize> {
        // Collect all raw:* keys sorted by vector_id
        let mut entries: Vec<(i64, Vec<f32>)> = Vec::new();

        let iter = self.meta_db.prefix_iterator(b"raw:");
        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with("raw:") {
                break; // prefix_iterator may go beyond prefix
            }
            if let Some(id_str) = key_str.strip_prefix("raw:") {
                if let Ok(vector_id) = id_str.parse::<i64>() {
                    if value.len() == EMBEDDING_DIM * 4 {
                        let vec: Vec<f32> = value
                            .chunks_exact(4)
                            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                            .collect();
                        entries.push((vector_id, vec));
                    }
                }
            }
        }

        entries.sort_by_key(|(id, _)| *id);

        let mut index = self.index.lock().unwrap();
        let mut pos_map = self.position_to_id.lock().unwrap();

        for (vector_id, vec) in &entries {
            index.add(vec);
            pos_map.push(*vector_id);
        }

        Ok(entries.len())
    }

    pub fn stats(&self) -> (usize, usize, i64) {
        let index = self.index.lock().unwrap();
        let next_id = *self.next_vector_id.lock().unwrap();

        // Count metadata entries
        let mut meta_count: usize = 0;
        let iter = self.meta_db.prefix_iterator(b"vec:");
        for item in iter {
            let (key, _) = match item {
                Ok(kv) => kv,
                Err(_) => break,
            };
            if !key.starts_with(b"vec:") {
                break;
            }
            meta_count += 1;
        }

        (index.ntotal(), meta_count, next_id)
    }
}
