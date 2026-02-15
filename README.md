# AstarteBot

A Telegram AI assistant bot built in Rust, powered by OpenRouter.ai LLM backend with SQLite for persistent storage, tool-calling capabilities, and conversation memory.

## Features

- **LLM-powered conversations** via OpenRouter (supports any model: Claude, GPT-4, Llama, etc.)
- **Persistent notes & memory** — the bot can save, search, and recall information across sessions
- **Conversation history** — all messages logged to SQLite, searchable by keyword/sender/date
- **Important/pinned memory** — critical info injected into every prompt automatically
- **Image support** — forward photos to vision-capable models
- **Group awareness** — only responds when @mentioned or replied to in groups
- **Tool calling** — LLM can use structured tools for notes, memory, and history search
- **CLI management** — configure the bot and query the database from the command line
- **Message splitting** — auto-splits responses exceeding Telegram's 4096 char limit
- **Retry logic** — exponential backoff on OpenRouter failures
- **Structured logging** — console + rolling JSON log files

## Prerequisites

- **Rust 1.93+** (edition 2024)
- **Telegram Bot Token** — get one from [@BotFather](https://t.me/BotFather)
- **OpenRouter API Key** — get one from [openrouter.ai](https://openrouter.ai/)
- **OpenAI API Key** (optional, for voice messages) — get one from [platform.openai.com](https://platform.openai.com/api-keys)

## Build

```bash
cargo build --release
```

The binary will be at `target/release/astartebot`.

## Initial Setup

### 1. Configure BotFather settings

In Telegram, open [@BotFather](https://t.me/BotFather) and configure your bot:

```
/setprivacy → Select your bot → Disable
```

**This is critical for group chats.** With privacy mode enabled (default), the bot only receives messages that @mention it or reply to it. With privacy mode **disabled**, the bot receives ALL group messages, enabling:
- Full conversation history logging
- Searchable chat history via `search_history` tool
- Better context awareness for the LLM

> Note: You need to remove and re-add the bot to existing groups after changing this setting.

### 2. Configure the bot

```bash
# Set your Telegram bot token (from @BotFather)
./astartebot config set tg_bot_token "YOUR_TELEGRAM_BOT_TOKEN"

# Set your OpenRouter API key
./astartebot config set llm_token "sk-or-v1-..."

# Set the LLM model (optional, defaults to claude-sonnet-4-5)
./astartebot config set llm_model "anthropic/claude-sonnet-4-5-20250929"

# Set the bot's display name (optional, defaults to "Astarte")
./astartebot config set bot_name "Astarte"

# Set a custom system prompt (optional)
./astartebot config set system_prompt "You are Astarte, a friendly and helpful AI assistant..."

# Set OpenAI API key for voice messages (optional)
./astartebot config set openai_api_key "sk-..."
```

Alternatively, you can set the Telegram token via environment variable:
```bash
export TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
```

### 3. Verify configuration

```bash
# List all config values (tokens are masked in output)
./astartebot config list

# Get a specific value
./astartebot config get bot_name
```

### 4. Start the bot

```bash
./astartebot run
```

The bot will:
- Connect to Telegram
- Print its username to the console
- Begin listening for messages

## Usage

### In Direct Messages
Simply send a message — the bot responds to everything.

### In Groups
Add the bot to a group, then:
- **@mention** it: `@YourBotUsername what's the weather?`
- **Reply** to one of its messages

The bot logs ALL group messages for context (even when not mentioned), so it has full conversation awareness.

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Greeting message |
| `/help` | Show available commands |
| `/reset` | Clear conversation history for this chat |

### What the Bot Can Do (via LLM Tools)

The bot has access to these tools, which it uses automatically based on conversation:

**Notes** (permanent documents):
- Store notes with titles, content, and tags
- Search notes by keyword or regex
- Read and delete notes
- Organized by scope: per-chat, per-user, global, or bot-private

**Memory** (key-value quick facts):
- Store/retrieve quick facts (e.g., user preferences, settings)
- List all stored memory in a scope

**Pinned Memory** (always-visible):
- Set a single "important" memory per scope that's included in every prompt
- Useful for persistent instructions: "Always respond in Spanish in this chat"

**Conversation History**:
- Search past messages by keyword, sender, date range
- Browse older messages beyond the recent context window

**Voice Messages** (requires `openai_api_key`):
- Send voice messages as Telegram audio with emotional speech
- 5 female voices: nova, shimmer, fable, coral, sage
- Controllable emotion and delivery style (cheerful, dramatic, whispering, etc.)
- Uses OpenAI `gpt-4o-mini-tts` for expressive speech

**Voice Recognition** (incoming voice messages):
- Understands voice messages sent in DMs or as replies in groups
- Three modes configurable via `voice_mode`:
  - `auto` (default) — uses Whisper if `openai_api_key` is set, otherwise sends audio directly to LLM via OpenRouter
  - `whisper` — always transcribes via OpenAI Whisper first, then sends text to LLM. Requires `openai_api_key`. Works with any LLM model.
  - `openrouter` — sends raw audio directly to the LLM. No extra API key, but model must support audio input (e.g. Gemini, GPT-4o). Does NOT work with Grok, Claude, Llama.

## CLI Reference

```bash
# Run the bot
astartebot run

# Configuration
astartebot config set <key> <value>
astartebot config get <key>
astartebot config list

# Database access (read-only query)
astartebot db query "SELECT * FROM notes"
astartebot db query "SELECT COUNT(*) FROM conversation_history"

# Database access (write — blocks tg_bot_token modification)
astartebot db modify "UPDATE config SET value='NewName' WHERE key='bot_name'"
```

## Configuration Keys

| Key | Required | Description |
|-----|----------|-------------|
| `tg_bot_token` | Yes* | Telegram bot token (or use `TELEGRAM_BOT_TOKEN` env var) |
| `llm_token` | Yes | OpenRouter API key |
| `llm_model` | No | Model ID (default: `anthropic/claude-sonnet-4-5-20250929`) |
| `bot_name` | No | Bot display name (default: `Astarte`) |
| `system_prompt` | No | Custom system prompt for the LLM |
| `openai_api_key` | No | OpenAI API key (for voice messages via TTS and Whisper transcription) |
| `voice_mode` | No | Voice recognition: `auto` (default), `whisper`, or `openrouter` |

## Database

AstarteBot uses SQLite (`astartebot.db` in the working directory). The schema is auto-migrated on startup.

**Tables:**
- `config` — key-value configuration
- `name_map` — Telegram user/chat display names
- `memory` — segmented key-value memory store
- `notes` — persistent notes with tags
- `conversation_history` — all messages with sender info, timestamps, reply tracking
- `tool_call_log` — audit log of all LLM tool invocations
- `schema_version` — migration tracking

## Logging

Logs are written to:
- **Console**: compact format, colored
- **File**: `logs/astartebot.log.*` (JSON format, daily rotation)

Control log level with `RUST_LOG` environment variable:
```bash
RUST_LOG=debug ./astartebot run
```

## Supported Models

Any model available on [OpenRouter](https://openrouter.ai/models) works. Popular choices:

```bash
# Anthropic Claude
astartebot config set llm_model "anthropic/claude-sonnet-4-5-20250929"

# OpenAI GPT-4o
astartebot config set llm_model "openai/gpt-4o"

# Meta Llama
astartebot config set llm_model "meta-llama/llama-3.1-70b-instruct"

# Google Gemini
astartebot config set llm_model "google/gemini-2.0-flash-001"
```

For image support, use a vision-capable model (Claude, GPT-4o, Gemini).

## Memory Segments

| Segment | Format | Scope |
|---------|--------|-------|
| Chat | `chat:{chat_id}` | Specific group/DM |
| Person | `person:{user_id}` | About a specific user (cross-chat) |
| Global | `global` | All chats |
| Bot | `bot` | Bot's own private knowledge |
