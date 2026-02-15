use anyhow::Result;
use base64::Engine;
use sqlx::SqlitePool;
use std::sync::Arc;
use teloxide::prelude::*;
use teloxide::types::{MediaKind, MessageKind, ParseMode, PhotoSize, Voice};

use crate::config;
use crate::db;
use crate::llm::LlmClient;
use crate::mcp::McpManager;
use crate::rag::RagEngine;
use crate::types::*;

const MAX_TELEGRAM_MSG_LEN: usize = 4096;
const DEFAULT_HISTORY_LIMIT: i64 = 50;

struct BotState {
    pool: SqlitePool,
    llm: LlmClient,
    rag: RagEngine,
    mcp: McpManager,
    bot_name: String,
    bot_username: String,
    bot_user_id: i64,
    system_prompt: String,
    trigger_keywords: Vec<String>,
}

pub async fn run(pool: SqlitePool) -> Result<()> {
    let tg_token = config::get_telegram_token(&pool).await?;
    let llm_token = config::get_required(&pool, "llm_token").await?;
    let llm_model =
        config::get_or_default(&pool, "llm_model", "anthropic/claude-sonnet-4-5-20250929").await?;
    let bot_name = config::get_or_default(&pool, "bot_name", "Astarte").await?;
    let system_prompt = config::get_or_default(
        &pool,
        "system_prompt",
        &format!(
            "You are {}, a helpful and knowledgeable AI assistant in a Telegram chat. \
             You have access to tools for managing notes and memory. \
             Use them when users ask you to remember things, take notes, or recall information. \
             You can also search conversation history to recall past discussions. \
             Be conversational, helpful, and concise.\n\n\
            You can manage MCP server registrations via the `crud_mcp_server` tool with actions: create/read/list/update/delete. \
            For create/update, transport can be omitted:\n\
            - command only => transport inferred as stdio (requires non-empty command)\n\
            - `tcp://host:port` / `host:port` => transport inferred as tcp\n\
            - `http://...` / `https://...` => transport inferred as http\n\
            - explicit `transport='stdio'` => requires non-empty command\n\
            - explicit `transport='http'`/`sse`/`streamable_http` => requires non-empty endpoint URL.\n\
            You can manage files via `crud_file` with actions: create/read/list/update/delete. File operations are restricted to the `files` directory by default (or `config set files_root <path>` to override). \
            Use `read` with `offset` (0-based) and optional `line_count` (default 200) for paging. \
            Use `update` with `mode` (`replace`, `insert`, `delete`, `append`), and optional `offset`/`line_count` for patch-style updates. \
            `offset` defaults to 0 when meaningful; `line_count` defaults to 1 for partial replace/delete and 200 for read. File paths must be relative and cannot include path traversal segments.\n\
            If required fields are missing, do not call the tool; ask the user for the missing fields instead.",
            bot_name
        ),
    )
    .await?;

    let bot = Bot::new(&tg_token);

    // Get bot username and ID
    let me = bot.get_me().await?;
    let bot_username = me.username().to_string();
    let bot_user_id = me.id.0 as i64;

    // Load trigger keywords
    let trigger_keywords = db::trigger_keywords_list(&pool).await?;

    // Initialize RAG engine
    let rag = RagEngine::init(&std::path::PathBuf::from("rag_data")).await?;

    tracing::info!(
        bot_name = %bot_name,
        bot_username = %bot_username,
        llm_model = %llm_model,
        trigger_keywords = ?trigger_keywords,
        "Starting Telegram bot"
    );

    let state = Arc::new(BotState {
        pool,
        llm: LlmClient::new(llm_token, llm_model),
        rag,
        mcp: McpManager::new(),
        bot_name,
        bot_username,
        bot_user_id,
        system_prompt,
        trigger_keywords,
    });

    // Start hourly background backups
    let _backup_handle = crate::backup::start_hourly_backup();

    let handler = Update::filter_message().endpoint(handle_message);

    Dispatcher::builder(bot, handler)
        .default_handler(|_upd| async {})
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

async fn handle_message(
    bot: Bot,
    msg: Message,
    state: Arc<BotState>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let user = msg.from.as_ref();
    let user_id = user.map(|u| u.id.0 as i64).unwrap_or(0);
    let chat_id = msg.chat.id.0;
    let user_display_name = user.map(|u| u.full_name()).unwrap_or_default();
    let tg_message_id = Some(msg.id.0 as i64);
    let reply_to_msg_id = msg.reply_to_message().map(|r| r.id.0 as i64);

    // Log received message
    let is_voice =
        matches!(&msg.kind, MessageKind::Common(c) if matches!(&c.media_kind, MediaKind::Voice(_)));
    let text_preview = if is_voice {
        "[voice message]"
    } else {
        msg.text().or(msg.caption()).unwrap_or("[non-text]")
    };
    tracing::info!(
        chat_id,
        user_id,
        user_name = %user_display_name,
        message_id = ?tg_message_id,
        reply_to = ?reply_to_msg_id,
        text = %text_preview,
        "Received message"
    );

    // Update name map for every message we see
    if let Some(u) = user {
        let username = u.username.as_deref().unwrap_or("");
        let _ = db::name_map_set(&state.pool, "user", user_id, &u.full_name(), username).await;
    }
    if let Some(title) = msg.chat.title() {
        let _ = db::name_map_set(&state.pool, "chat", chat_id, title, "").await;
    }

    // Log ALL text messages to conversation history (for searchable context)
    if let Some(text) = msg.text() {
        if let Ok(row_id) = db::conversation_save(
            &state.pool,
            chat_id,
            user_id,
            &user_display_name,
            "user",
            text,
            None,
            tg_message_id,
            reply_to_msg_id,
        )
        .await
        {
            let segment = format!("chat:{}", chat_id);
            let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let _ = state.rag.index_record_sync(
                "conversation",
                row_id,
                chat_id,
                &segment,
                text,
                &user_display_name,
                &now,
            );
        }
    }

    // Check for commands first
    if let Some(text) = msg.text() {
        let text_trimmed = text.trim();
        if text_trimmed == "/start" || text_trimmed == format!("/start@{}", state.bot_username) {
            bot.send_message(
                msg.chat.id,
                format!(
                    "Hello! I'm {}. Send me a message and I'll do my best to help!",
                    state.bot_name
                ),
            )
            .await?;
            return Ok(());
        }
        if text_trimmed == "/help" || text_trimmed == format!("/help@{}", state.bot_username) {
            let help_text = format!(
                "I'm {}, your AI assistant.\n\n\
                 Commands:\n\
                 /start - Start the bot\n\
                 /help - Show this help\n\
                 /reset - Clear conversation history\n\n\
                 I can remember things using notes and memory. Just ask!\n\
                 In groups, mention me or reply to my messages.",
                state.bot_name
            );
            bot.send_message(msg.chat.id, help_text).await?;
            return Ok(());
        }
        if text_trimmed == "/reset" || text_trimmed == format!("/reset@{}", state.bot_username) {
            let deleted = db::conversation_clear(&state.pool, msg.chat.id.0)
                .await
                .unwrap_or(0);
            bot.send_message(
                msg.chat.id,
                format!(
                    "Conversation history cleared ({} messages removed).",
                    deleted
                ),
            )
            .await?;
            return Ok(());
        }
    }

    // Check if this message should trigger an LLM response
    if !should_respond(&msg, &state) {
        // Message is already logged to history above, just don't call LLM
        return Ok(());
    }

    // Extract content (text, photo, or voice message)
    let user_content = match extract_content(&bot, &msg, &state.pool).await {
        Some(content) => content,
        None => return Ok(()),
    };

    // If this is not a plain text message, save the extracted content to history
    if msg.text().is_none() {
        let content_text = match &user_content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Text { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
        };
        if let Ok(row_id) = db::conversation_save(
            &state.pool,
            chat_id,
            user_id,
            &user_display_name,
            "user",
            &content_text,
            None,
            tg_message_id,
            reply_to_msg_id,
        )
        .await
        {
            let segment = format!("chat:{}", chat_id);
            let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let _ = state.rag.index_record_sync(
                "conversation",
                row_id,
                chat_id,
                &segment,
                &content_text,
                &user_display_name,
                &now,
            );
        }
    }

    // Build messages for LLM
    let messages =
        build_llm_messages(&state, chat_id, user_id, tg_message_id, user_content).await?;

    // Call LLM
    match state
        .llm
        .chat(
            &state.pool,
            &bot,
            &state.rag,
            &state.mcp,
            messages,
            chat_id,
            user_id,
        )
        .await
    {
        Ok(response) => {
            if response.is_empty() {
                return Ok(());
            }

            // Save assistant response to history
            if let Ok(row_id) = db::conversation_save(
                &state.pool,
                chat_id,
                0,
                &state.bot_name,
                "assistant",
                &response,
                None,
                None,
                tg_message_id,
            )
            .await
            {
                let segment = format!("chat:{}", chat_id);
                let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
                let _ = state.rag.index_record_sync(
                    "conversation",
                    row_id,
                    chat_id,
                    &segment,
                    &response,
                    &state.bot_name,
                    &now,
                );
            }

            // Send response, splitting if needed
            send_split_message(&bot, msg.chat.id, &response, msg.id).await?;
        }
        Err(e) => {
            tracing::error!(error = %e, chat_id, user_id, "LLM error");
            bot.send_message(
                msg.chat.id,
                "Sorry, I encountered an error processing your message. Please try again.",
            )
            .await?;
        }
    }

    Ok(())
}

fn should_respond(msg: &Message, state: &BotState) -> bool {
    // Always respond in private/DM chats
    if msg.chat.is_private() {
        return true;
    }

    // Reply to bot's own message (check by bot user ID, not just any bot)
    if let Some(reply) = msg.reply_to_message() {
        if let Some(from) = reply.from.as_ref() {
            if from.id.0 as i64 == state.bot_user_id {
                return true;
            }
        }
    }

    // Voice messages in groups only trigger if replying to the bot (handled above)
    // If we get here for a voice message in a group (not a reply to bot), skip
    if matches!(&msg.kind, MessageKind::Common(c) if matches!(&c.media_kind, MediaKind::Voice(_))) {
        return false;
    }

    // Check text OR caption (photos have captions, not text)
    let check_text = msg.text().or(msg.caption());
    if let Some(text) = check_text {
        // Check @mention of bot username
        let mention = format!("@{}", state.bot_username);
        if text.contains(&mention) {
            return true;
        }

        // Check trigger keywords (case-insensitive)
        let text_lower = text.to_lowercase();
        for keyword in &state.trigger_keywords {
            if text_lower.contains(&keyword.to_lowercase()) {
                return true;
            }
        }
    }

    // Check mention entities (in text or caption)
    // Note: Telegram entity offset/length are in UTF-16 code units, not bytes
    let entities = msg.entities().or(msg.caption_entities());
    if let Some(entities) = entities {
        let entity_text = msg.text().or(msg.caption());
        for entity in entities {
            if entity.kind == teloxide::types::MessageEntityKind::Mention {
                if let Some(text) = entity_text {
                    if let Some(mention) = utf16_substr(text, entity.offset, entity.length) {
                        if mention
                            .trim_start_matches('@')
                            .eq_ignore_ascii_case(&state.bot_username)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

async fn extract_content(bot: &Bot, msg: &Message, pool: &SqlitePool) -> Option<MessageContent> {
    if let MessageKind::Common(common) = &msg.kind {
        // Handle photo messages
        if let MediaKind::Photo(photo) = &common.media_kind {
            let best_photo = photo.photo.iter().max_by_key(|p| p.width * p.height)?;

            if let Ok(image_content) = download_photo_as_base64(bot, best_photo).await {
                let caption = photo.caption.clone().unwrap_or_default();
                let mut parts = vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: format!("data:image/jpeg;base64,{}", image_content),
                        detail: Some("auto".to_string()),
                    },
                }];
                if !caption.is_empty() {
                    parts.push(ContentPart::Text { text: caption });
                } else {
                    parts.push(ContentPart::Text {
                        text: "What do you see in this image?".to_string(),
                    });
                }
                return Some(MessageContent::Parts(parts));
            }
        }

        // Handle voice messages
        if let MediaKind::Voice(voice_media) = &common.media_kind {
            let voice_mode = config::get_or_default(pool, "voice_mode", "auto")
                .await
                .unwrap_or_else(|_| "auto".to_string());
            // auto: use whisper if openai_api_key is set, otherwise openrouter
            let use_whisper = match voice_mode.as_str() {
                "whisper" => true,
                "openrouter" => false,
                _ => config::get(pool, "openai_api_key")
                    .await
                    .ok()
                    .flatten()
                    .is_some_and(|k| !k.is_empty()),
            };
            if use_whisper {
                // Whisper mode: transcribe first, then send text to LLM
                match transcribe_voice(bot, pool, &voice_media.voice).await {
                    Ok(transcript) => {
                        let text = format!("[voice message, transcribed] {}", transcript);
                        tracing::info!(
                            transcript_len = transcript.len(),
                            "Voice transcribed via Whisper"
                        );
                        return Some(MessageContent::Text(text));
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Whisper transcription failed");
                        return Some(MessageContent::Text(
                            "[voice message — transcription failed]".to_string(),
                        ));
                    }
                }
            } else {
                // OpenRouter mode: send audio directly to LLM (model must support audio)
                match download_voice_as_base64(bot, &voice_media.voice).await {
                    Ok(audio_b64) => {
                        tracing::info!(
                            audio_size = audio_b64.len(),
                            "Voice sent directly to LLM via OpenRouter"
                        );
                        let mut parts = vec![ContentPart::InputAudio {
                            input_audio: AudioInput {
                                data: audio_b64,
                                format: "ogg".to_string(),
                            },
                        }];
                        parts.push(ContentPart::Text {
                            text: "[voice message, audio attached] The user spoke this message (not typed). Listen to the audio and respond naturally. Consider replying with voice using send_voice tool.".to_string(),
                        });
                        return Some(MessageContent::Parts(parts));
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to download voice for OpenRouter");
                        return Some(MessageContent::Text(
                            "[voice message — download failed]".to_string(),
                        ));
                    }
                }
            }
        }
    }

    // Handle text messages
    msg.text().map(|t| MessageContent::Text(t.to_string()))
}

async fn download_photo_as_base64(bot: &Bot, photo: &PhotoSize) -> Result<String> {
    let file = bot.get_file(photo.file.id.clone()).await?;

    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        bot.token(),
        file.path
    );
    let resp = reqwest::get(&url).await?;
    let buf = resp.bytes().await?.to_vec();

    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

async fn download_voice_as_base64(bot: &Bot, voice: &Voice) -> Result<String> {
    let file = bot.get_file(voice.file.id.clone()).await?;
    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        bot.token(),
        file.path
    );
    let resp = reqwest::get(&url).await?;
    let buf = resp.bytes().await?.to_vec();
    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

async fn transcribe_voice(bot: &Bot, pool: &SqlitePool, voice: &Voice) -> Result<String> {
    // Get OpenAI API key
    let api_key = config::get(pool, "openai_api_key")
        .await?
        .ok_or_else(|| anyhow::anyhow!("openai_api_key not configured"))?;

    // Download voice file from Telegram
    let file = bot.get_file(voice.file.id.clone()).await?;
    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        bot.token(),
        file.path
    );
    let resp = reqwest::get(&url).await?;
    let audio_bytes = resp.bytes().await?.to_vec();

    tracing::info!(
        file_size = audio_bytes.len(),
        duration_secs = voice.duration.seconds(),
        "Downloading voice for transcription"
    );

    // Call OpenAI Whisper API
    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(audio_bytes)
        .file_name("voice.ogg")
        .mime_str("audio/ogg")?;

    let form = reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .text("response_format", "text")
        .part("file", part);

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", api_key))
        .multipart(form)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let error_body = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "Whisper API error HTTP {}: {}",
            status,
            error_body
        ));
    }

    let transcript = response.text().await?.trim().to_string();
    Ok(transcript)
}

async fn build_llm_messages(
    state: &BotState,
    chat_id: i64,
    user_id: i64,
    current_message_id: Option<i64>,
    current_content: MessageContent,
) -> Result<Vec<ChatMessage>> {
    let mut messages = Vec::new();

    // Resolve names (full: display_name + @username)
    let (user_display, user_tg_username) = db::name_map_get_full(&state.pool, "user", user_id)
        .await?
        .unwrap_or_else(|| (format!("User#{}", user_id), String::new()));
    let chat_name = db::name_map_get(&state.pool, "chat", chat_id)
        .await?
        .unwrap_or_else(|| {
            if chat_id > 0 {
                "Direct Message".to_string()
            } else {
                format!("Chat#{}", chat_id)
            }
        });

    let now = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    // Build user identity string
    let user_identity = if user_tg_username.is_empty() {
        format!("{} (id: {})", user_display, user_id)
    } else {
        format!("{} (@{}, id: {})", user_display, user_tg_username, user_id)
    };

    // Build system prompt
    let mut system_text = format!(
        "{}\n\n\
         Context:\n\
         - Current time: {}\n\
         - Bot name: {}\n\
         - Chat: {} (id: {})\n\
         - Current user: {}\n\
         - Available memory segments for this context: global, bot, chat:{}, person:{}\n\
         - Voice: Messages prefixed with [voice message, transcribed] were spoken by the user (not typed). \
         You can reply with voice using the send_voice tool. When a user speaks to you, consider replying with voice too for a natural conversation.",
        state.system_prompt,
        now,
        state.bot_name,
        chat_name,
        chat_id,
        user_identity,
        chat_id,
        user_id
    );
    system_text.push_str(
        "\n\nImportant MCP rules for `crud_mcp_server`:\n\
         - Use exactly one action: create, read, list, update, delete.\n\
         - transport is optional for create/update. If omitted:\n\
         -- command-only -> transport is inferred as stdio (requires command).\n\
         -- tcp-style endpoint (`tcp://host:port` or `host:port`) -> inferred transport=tcp.\n\
         -- http/https endpoint (`http://...`, `https://...`) -> inferred transport=http.\n\
         - explicit transport stdio requires non-empty command.\n\
         - explicit transport http/sse/streamable_http requires non-empty endpoint URL.\n\
         - do not default transport to stdio unless command-only is clearly provided.\n\
         - If a required field is missing, do not call the tool and ask the user for it.",
    );
    system_text.push_str(
        "\n\nYou can interact with registered MCP servers (tcp/http transports):\n\
         1. `mcp_list_tools` — discover available methods on MCP servers. Call this first.\n\
         2. `mcp_call` — invoke a method on an MCP server with server_name, method, and arguments.\n\
         Workflow: list tools first to see what's available, then call specific methods with the correct arguments.",
    );

    // Inject important/pinned memory for this chat
    let chat_segment = format!("chat:{}", chat_id);
    if let Ok(Some(important)) = db::get_important_memory(&state.pool, &chat_segment).await {
        system_text.push_str(&format!("\n\nPinned memory for this chat:\n{}", important));
    }
    if let Ok(Some(important)) = db::get_important_memory(&state.pool, "global").await {
        system_text.push_str(&format!("\n\nGlobal pinned memory:\n{}", important));
    }
    if let Ok(Some(important)) = db::get_important_memory(&state.pool, "bot").await {
        system_text.push_str(&format!("\n\nBot pinned memory:\n{}", important));
    }
    // Inject person-specific pinned memory for the current user
    let person_segment = format!("person:{}", user_id);
    if let Ok(Some(important)) = db::get_important_memory(&state.pool, &person_segment).await {
        system_text.push_str(&format!(
            "\n\nPinned memory about current user ({}):\n{}",
            user_display, important
        ));
    }

    messages.push(ChatMessage {
        role: "system".to_string(),
        content: Some(MessageContent::Text(system_text)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    // Load recent conversation history (for LLM context window)
    let history = db::conversation_load(&state.pool, chat_id, DEFAULT_HISTORY_LIMIT).await?;
    for row in &history {
        // The current incoming message is already persisted before this call.
        // Skip it in history to avoid duplicating the same user turn in prompt context.
        if row.role == "user" && row.user_id == user_id && row.message_id == current_message_id {
            continue;
        }

        // Build rich content with sender name, timestamp, and reply context
        let mut content = String::new();
        if row.role == "user" && !row.user_name.is_empty() {
            content.push_str(&format!("[{} at {}]", row.user_name, row.created_at));
            if let Some(reply_id) = row.reply_to_id {
                // Find the message being replied to in history
                if let Some(replied_to) = history.iter().find(|r| r.message_id == Some(reply_id)) {
                    let reply_preview: String = replied_to.content.chars().take(80).collect();
                    let replied_name = if replied_to.user_name.is_empty() {
                        &replied_to.role
                    } else {
                        &replied_to.user_name
                    };
                    content.push_str(&format!(
                        " (replying to {}: \"{}...\")",
                        replied_name, reply_preview
                    ));
                }
            }
            content.push_str(&format!(": {}", row.content));
        } else {
            content.push_str(&row.content);
        }
        messages.push(ChatMessage {
            role: row.role.clone(),
            content: Some(MessageContent::Text(content)),
            tool_calls: None,
            tool_call_id: row.tool_call_id.clone(),
            name: None,
        });
    }

    // Add current message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: Some(current_content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    Ok(messages)
}

async fn send_split_message(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: teloxide::types::MessageId,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use teloxide::types::ReplyParameters;

    let chunks = if text.len() <= MAX_TELEGRAM_MSG_LEN {
        vec![text.to_string()]
    } else {
        split_message(text, MAX_TELEGRAM_MSG_LEN)
    };

    let reply_params = ReplyParameters::new(reply_to).allow_sending_without_reply();

    for (i, chunk) in chunks.iter().enumerate() {
        // Only reply to the original message for the first chunk
        let mut request = bot.send_message(chat_id, chunk);
        if i == 0 {
            request = request.reply_parameters(reply_params.clone());
        }
        // Try MarkdownV2 first, fall back to plain text
        let md_result = request.parse_mode(ParseMode::MarkdownV2).await;

        if md_result.is_err() {
            let mut request = bot.send_message(chat_id, chunk);
            if i == 0 {
                request = request.reply_parameters(reply_params.clone());
            }
            request.await?;
        }
    }

    Ok(())
}

fn split_message(text: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        // Find the largest char boundary <= max_len
        let mut boundary = max_len;
        while boundary > 0 && !remaining.is_char_boundary(boundary) {
            boundary -= 1;
        }

        // Try to split at a newline or space within the boundary
        let split_at = remaining[..boundary]
            .rfind('\n')
            .unwrap_or_else(|| remaining[..boundary].rfind(' ').unwrap_or(boundary));

        let (chunk, rest) = remaining.split_at(split_at);
        chunks.push(chunk.to_string());
        remaining = rest.trim_start_matches('\n');
    }

    chunks
}

/// Extract a substring using UTF-16 offset and length (Telegram's encoding).
fn utf16_substr(text: &str, utf16_offset: usize, utf16_len: usize) -> Option<String> {
    let utf16_units: Vec<u16> = text.encode_utf16().collect();
    let start = utf16_offset;
    let end = utf16_offset + utf16_len;
    if end > utf16_units.len() {
        return None;
    }
    String::from_utf16(&utf16_units[start..end]).ok()
}
