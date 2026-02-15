use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{json, Value};
use sqlx::SqlitePool;
use teloxide::prelude::*;
use teloxide::types::InputFile;

use crate::db;
use crate::mcp::McpManager;
use crate::memory;
use crate::rag::RagEngine;
use crate::types::ToolDefinition;

fn tool(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::types::FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        },
    }
}

/// Return all tool definitions for the LLM
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // --- Notes ---
        tool(
            "store_note",
            "Create and permanently save a new note. Use this when a user asks you to remember something, save information, or when you want to persist important details for later. Notes survive conversation resets. Each note belongs to a segment that controls its visibility scope:\n\
             - 'chat:{chat_id}': visible only in this specific chat/group\n\
             - 'person:{user_id}': private notes about a specific user (visible across all chats)\n\
             - 'global': shared across all chats and users\n\
             - 'bot': your own private knowledge (personality, learned preferences)\n\
             Returns the created note's ID for future reference.",
            json!({
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "Which scope to store the note in. Examples: 'global', 'bot', 'chat:-1001234567890', 'person:123456789'. Use the chat_id and user_id from your system prompt context."
                    },
                    "title": {
                        "type": "string",
                        "description": "A short, descriptive title for the note (used in search results)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full note content. Can be any length."
                    },
                    "tags": {
                        "type": "string",
                        "description": "Optional comma-separated tags for categorization and easier searching. Example: 'recipe,cooking,italian'"
                    }
                },
                "required": ["segment", "title", "content"]
            }),
        ),
        tool(
            "search_notes",
            "Search through all saved notes by keyword or regex pattern. Use this when a user asks 'do you remember...', 'what did I say about...', or when you need to find previously saved information. Returns a list of matching notes with their IDs (use read_note to get full content). Searches across title, content, and tags fields.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term. For keyword search: any word or phrase (e.g., 'birthday', 'project deadline'). For regex: a valid regex pattern (e.g., '\\bmeeting\\b.*2024')."
                    },
                    "segment": {
                        "type": "string",
                        "description": "Optional. Restrict search to a specific segment (e.g., 'global', 'chat:-1001234567890'). If omitted, searches ALL segments."
                    },
                    "use_regex": {
                        "type": "boolean",
                        "description": "Set to true to interpret 'query' as a regex pattern instead of a plain keyword. Default: false."
                    }
                },
                "required": ["query"]
            }),
        ),
        tool(
            "read_note",
            "Retrieve the full content of a specific note by its ID. Use this after search_notes returns matching results and you need to see the complete note content (search results only show titles and metadata).",
            json!({
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "integer",
                        "description": "The numeric ID of the note to read (obtained from search_notes results)."
                    }
                },
                "required": ["note_id"]
            }),
        ),
        tool(
            "delete_note",
            "Permanently delete a note by its ID. Use this when a user asks to forget something or when information is no longer relevant. This action cannot be undone.",
            json!({
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "integer",
                        "description": "The numeric ID of the note to delete."
                    }
                },
                "required": ["note_id"]
            }),
        ),

        // --- Memory (key-value store) ---
        tool(
            "memory_set",
            "Store or update a key-value pair in a memory segment. Memory is a simple key-value store (like a dictionary) — different from notes which are longer-form documents. Use memory for quick facts, preferences, settings, or flags. If the key already exists, its value is overwritten.\n\
             Segments control scope: 'chat:{chat_id}' (this chat only), 'person:{user_id}' (about this user), 'global' (everywhere), 'bot' (your own data).",
            json!({
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "Scope for this memory. Examples: 'global', 'bot', 'chat:-1001234567890', 'person:123456789'."
                    },
                    "key": {
                        "type": "string",
                        "description": "The key name. Use descriptive, namespaced keys like 'user_language', 'preferred_name', 'timezone'."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to store. Keep it concise — memory is for quick lookups, not long documents."
                    }
                },
                "required": ["segment", "key", "value"]
            }),
        ),
        tool(
            "memory_get",
            "Retrieve a single value from memory by its key. Use this to look up previously stored facts, preferences, or settings. Returns null if the key doesn't exist.",
            json!({
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "The memory segment to search in. Example: 'person:123456789', 'chat:-1001234567890', 'global', 'bot'."
                    },
                    "key": {
                        "type": "string",
                        "description": "The exact key to look up."
                    }
                },
                "required": ["segment", "key"]
            }),
        ),
        tool(
            "memory_list",
            "List ALL key-value pairs stored in a specific memory segment. Use this to see everything stored about a person, chat, or globally. Helpful when you need an overview of what you know.",
            json!({
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "The memory segment to list. Example: 'person:123456789', 'chat:-1001234567890', 'global', 'bot'."
                    }
                },
                "required": ["segment"]
            }),
        ),

        // --- MCP Servers (CRUD) ---
        tool(
            "crud_mcp_server",
            "Manage MCP server registrations.\n\n\
             ACTIONS:\n\
             - list: no extra fields needed.\n\
             - read: requires `name`.\n\
             - delete: requires `name`.\n\
             - update: requires `name` + at least one field to change.\n\
             - create: requires `name` + connection info (see below).\n\n\
             CREATE RULES (IMPORTANT — follow these strictly):\n\
             1. For a LOCAL COMMAND server (stdio): set `command` (e.g. 'npx -y @modelcontextprotocol/server-sqlite'). Transport is auto-detected as stdio.\n\
             2. For an HTTP/SSE server: set `endpoint` to a full URL (e.g. 'http://localhost:3000/mcp'). Transport is auto-detected as http.\n\
             3. For a TCP server: set `transport` to 'tcp' and `endpoint` to 'tcp://host:port' or 'host:port'.\n\
             4. You may optionally set `transport` explicitly, but if you do, you MUST also provide the matching field (`command` for stdio, `endpoint` for tcp/http/sse/streamable_http).\n\
             5. NEVER call create with only `name` and `transport` — you MUST include `command` or `endpoint`.\n\
             6. If the user has not told you the command or endpoint, ASK THEM FIRST — do not guess.",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "read", "list", "update", "delete"],
                        "description": "The CRUD action to perform."
                    },
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Server name. Required for create/read/update/delete."
                    },
                    "new_name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "New name when updating (rename)."
                    },
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Description for the server."
                    },
                    "transport": {
                        "type": "string",
                        "enum": ["stdio", "tcp", "http", "sse", "streamable_http"],
                        "description": "Optional transport mode. If omitted, inferred from command/endpoint. IMPORTANT: if you set transport to tcp/http/sse/streamable_http you MUST also provide endpoint. If you set transport to stdio you MUST provide command."
                    },
                    "command": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Shell command to start the server process. Required for stdio transport. Example: 'npx -y @modelcontextprotocol/server-sqlite'."
                    },
                    "endpoint": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Server URL. Required for http/sse/streamable_http transport. Examples: 'http://localhost:3000/mcp', 'tcp://127.0.0.1:5000', 'host:port'."
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Startup arguments for command mode (stdio)."
                    },
                    "environment": {
                        "type": "object",
                        "additionalProperties": { "type": "string" },
                        "description": "Environment variables for the server process."
                    }
                },
                "required": ["action"]
            }),
        ),

        // --- Conversation History ---
        tool(
            "search_history",
            "Search through the conversation history of the current chat. The last 50 messages are already visible in your context, but this tool lets you search OLDER messages beyond that window. Use this when a user asks about something said days/weeks ago, or when you need to find a specific past discussion. You can filter by keyword, sender name, sender ID, and/or date range. All filters are combined with AND logic.",
            json!({
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Optional. Search for messages containing this word or phrase (case-insensitive substring match). Example: 'project deadline', 'restaurant recommendation'."
                    },
                    "sender_name": {
                        "type": "string",
                        "description": "Optional. Filter by sender's display name (partial match). Example: 'John', 'Alice'. Useful when the user says 'what did John say about...'."
                    },
                    "sender_id": {
                        "type": "integer",
                        "description": "Optional. Filter by exact Telegram user ID. Use this if you know the user's ID from context."
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional. Only include messages from this date/time onward. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'. Example: '2024-01-15'."
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional. Only include messages up to this date/time. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'. Example: '2024-02-01'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return. Default: 20. Max: 100."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results (for pagination). Default: 0. Use with limit to page through results."
                    }
                },
                "required": []
            }),
        ),
        tool(
            "browse_history",
            "Browse older conversation messages by loading a specific page of history. Unlike search_history which filters by criteria, this simply loads N messages starting from a given offset. Use this when you want to 'scroll back' through the conversation chronologically — e.g., when a user says 'what were we talking about yesterday?' or 'go back further'.",
            json!({
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": "How many messages to skip from the most recent. 0 = most recent messages (already in your context). Use 50 to see messages just before your context window, 100 for even older, etc."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many messages to load. Default: 20. Max: 50."
                    }
                },
                "required": ["offset"]
            }),
        ),

        tool(
            "search_all_chats",
            "Search conversation history across ALL chats and groups at once. Unlike search_history (which only searches the current chat), this tool searches every chat the bot has ever participated in. Use this when:\n\
             - A user asks 'did anyone in any group mention...' or 'where was X discussed?'\n\
             - You need to find a conversation but don't know which chat it happened in\n\
             - You want to find all messages from a specific person across all groups\n\n\
             At least one filter is required to prevent unbounded searches. All filters are combined with AND logic. Works with any language (multilingual). Each result includes chat_id so you can tell which chat/group it came from.",
            json!({
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Optional. Search for messages containing this word or phrase (case-insensitive substring match). Works with any language including CJK, Cyrillic, Arabic, etc."
                    },
                    "sender_name": {
                        "type": "string",
                        "description": "Optional. Filter by sender's display name (partial match). Example: 'John', 'Alice'."
                    },
                    "sender_id": {
                        "type": "integer",
                        "description": "Optional. Filter by exact Telegram user ID."
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional. Only include messages from this date/time onward. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional. Only include messages up to this date/time. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return. Default: 20. Max: 100."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results (for pagination). Default: 0."
                    }
                },
                "required": []
            }),
        ),

        // --- RAG Semantic Search ---
        tool(
            "rag_search",
            "Semantic (meaning-based) search across ALL data: conversations, notes, and memory. Unlike keyword search tools (search_history, search_notes), this finds results by MEANING — so searching 'cooking Italian food' will find messages about 'making pasta with tomato sauce' even though they share no keywords. Use this when:\n\
             - You don't know the exact keywords to search for\n\
             - Keyword search (search_history, search_notes) returned nothing useful\n\
             - You want to find conceptually related content across all data types\n\
             Results are ranked by semantic similarity (score 0-1, higher = more relevant).",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural language description of what you're looking for. Be descriptive — 'recipes for Italian pasta dishes' works better than just 'pasta'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Default: 10. Max: 50."
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["conversation", "note", "memory"],
                        "description": "Optional. Filter results to only this data type. If omitted, searches all types."
                    }
                },
                "required": ["query"]
            }),
        ),

        // --- Important/Pinned Memory ---
        tool(
            "set_important_memory",
            "Set a pinned 'important' memory for a segment. This is a special single memory entry that gets automatically included in EVERY system prompt for the relevant scope — the bot always sees it without needing to look it up. Use this for critical information that should always be top-of-mind:\n\
             - For 'chat:{chat_id}': e.g., 'This is a work group. Be professional. The project deadline is March 15.'\n\
             - For 'global': e.g., 'Always respond in English unless asked otherwise.'\n\
             - For 'bot': e.g., 'I prefer to be called Astarte. I should be cheerful and use casual language.'\n\
             There is only ONE important memory per segment (setting a new one overwrites the previous). Max 500 characters.",
            json!({
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "The segment to pin this memory to. Example: 'chat:-1001234567890', 'global', 'bot'. Person segments are also supported: 'person:123456789'."
                    },
                    "content": {
                        "type": "string",
                        "description": "The important information to always remember. Max 500 characters. Be concise — this is injected into every prompt."
                    }
                },
                "required": ["segment", "content"]
            }),
        ),
        tool(
            "clear_important_memory",
            "Remove the pinned 'important' memory from a segment. After clearing, the information will no longer be automatically included in prompts for that scope.",
            json!({
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "description": "The segment to clear the pinned memory from. Example: 'chat:-1001234567890', 'global', 'bot'."
                    }
                },
                "required": ["segment"]
            }),
        ),

        // --- HTTP Request ---
        tool(
            "generic_http_request",
            "Make an HTTP request to any URL. This is your general-purpose web tool — use it to:\n\
             - Fetch web pages (HTML auto-converted to clean text by default, 5-20x smaller)\n\
             - Call REST APIs (weather, GitHub, YouTube, etc.)\n\
             - POST data to endpoints\n\
             - Check if a URL is reachable\n\n\
             IMPORTANT — API Key Security:\n\
             NEVER hardcode API keys in the URL or headers. Instead, first use memory_get to retrieve stored keys \
             (e.g., memory_get segment='person:USER_ID' key='openweather_api_key'), then use the returned value. \
             If the user gives you an API key, store it with memory_set first, then retrieve it.\n\n\
             Retries: Automatically retries up to 3 times on server errors (5xx) and timeouts with exponential backoff.\n\
             Timeout: 10 seconds default.\n\n\
             Examples:\n\
             - Web page: {\"url\": \"https://en.wikipedia.org/wiki/Rust\"}\n\
             - Weather API: {\"url\": \"https://wttr.in/London\", \"headers\": {\"User-Agent\": \"curl/7.0\"}}\n\
             - REST API: {\"url\": \"https://api.github.com/repos/rust-lang/rust\", \"headers\": {\"Accept\": \"application/json\"}}\n\
             - POST: {\"url\": \"https://httpbin.org/post\", \"method\": \"POST\", \"headers\": {\"Content-Type\": \"application/json\"}, \"body\": \"{\\\"hello\\\":\\\"world\\\"}\"}\n\
             - With query params: {\"url\": \"https://api.example.com/search\", \"query_params\": {\"q\": \"rust lang\", \"limit\": \"10\"}}",
            json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The base URL. Must start with http:// or https://. Query params can be in the URL or in query_params (which is safer for special characters)."
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                        "description": "HTTP method. Default: 'GET'."
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs. Example: {\"Authorization\": \"Bearer sk-...\", \"Accept\": \"application/json\"}. For APIs, set Content-Type and Accept headers appropriately.",
                        "additionalProperties": { "type": "string" }
                    },
                    "query_params": {
                        "type": "object",
                        "description": "Optional query parameters as key-value pairs. These are URL-encoded and appended to the URL automatically. Safer than putting them in the URL manually. Example: {\"q\": \"search term\", \"page\": \"1\"}.",
                        "additionalProperties": { "type": "string" }
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST/PUT/PATCH). Send as a string — for JSON, stringify it first. Example: \"{\\\"name\\\": \\\"value\\\"}\"."
                    },
                    "response_type": {
                        "type": "string",
                        "enum": ["auto", "json", "text", "html"],
                        "description": "How to process the response. 'auto' (DEFAULT): detect from Content-Type — JSON is parsed to object, HTML is converted to plain text. 'json': force parse as JSON. 'text': return raw text as-is. 'html': force HTML-to-text conversion."
                    },
                    "strip_html": {
                        "type": "boolean",
                        "description": "For HTML responses: if true (DEFAULT), convert to clean readable text (lynx-style). If false, return raw HTML. Only relevant when response is HTML."
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum response body characters to return. Default: 8000. Max: 30000. Truncates at word boundary with a note."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Request timeout in seconds. Default: 10. Max: 60."
                    }
                },
                "required": ["url"]
            }),
        ),

        // --- Python Execution ---
        tool(
            "run_python",
            "Execute Python 3 code in a sandboxed environment (bubblewrap). Use this for:\n\
             - Calculations, math, data processing\n\
             - Generating text, parsing data, regex operations\n\
             - Fetching data from the internet (requests/urllib work)\n\
             - File I/O within the sandbox (read/write files in /workspace)\n\
             - Any task that benefits from actual code execution\n\n\
             Security: The code runs in an isolated filesystem — it can only read system libraries (read-only) and write to /workspace. \
             Internet access is allowed. Execution is killed after the timeout.\n\n\
             The sandbox has Python 3 standard library available. For pip packages, they must be pre-installed on the host.\n\n\
             Tips:\n\
             - Print results to stdout — that's what gets returned to you\n\
             - Use /workspace/ for any file operations\n\
             - Keep code concise and focused\n\
             - For long-running tasks, increase timeout_secs\n\n\
             Examples:\n\
             - Math: {\"code\": \"import math\\nprint(math.factorial(100))\"}\n\
             - Web fetch: {\"code\": \"import urllib.request\\ndata = urllib.request.urlopen('https://httpbin.org/ip').read()\\nprint(data.decode())\"}\n\
             - File I/O: {\"code\": \"with open('/workspace/result.txt', 'w') as f:\\n    f.write('hello')\\nprint('saved')\"}\n\
             - Data: {\"code\": \"import json\\ndata = [{'name': 'a', 'val': 1}, {'name': 'b', 'val': 2}]\\nprint(json.dumps(data, indent=2))\"}",
            json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python 3 code to execute. Use print() for output. Use /workspace/ for file operations."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds. Default: 30. Max: 120. The process is killed if it exceeds this."
                    },
                    "input_files": {
                        "type": "object",
                        "description": "Optional files to create in /workspace/ before execution. Keys are filenames, values are file contents. Example: {\"data.csv\": \"name,value\\na,1\\nb,2\"}",
                        "additionalProperties": { "type": "string" }
                    }
                },
                "required": ["code"]
            }),
        ),

        // --- Maigret OSINT ---
        tool(
            "maigret_osint",
            "Run an OSINT (Open Source Intelligence) username search across 500+ websites using Maigret. \
             Finds matching profiles on social networks, forums, coding platforms, and other sites.\n\
             Use ONLY when a user explicitly asks for an OSINT search, username lookup, or online profile search.\n\n\
             Depth levels control speed vs coverage:\n\
             - 'light' (default): Top 50 popular sites, ~30 seconds\n\
             - 'medium': Top 200 sites, ~2 minutes\n\
             - 'full': All 500+ sites, ~5 minutes\n\n\
             You can also specify individual sites to check instead of using depth.\n\
             Prerequisite: Maigret must be installed on the host (sudo pip3 install maigret).\n\n\
             Examples:\n\
             - Quick search: {\"query\": \"johndoe\"}\n\
             - Deep scan: {\"query\": \"johndoe\", \"depth\": \"full\", \"timeout_secs\": 300}\n\
             - Specific sites: {\"query\": \"johndoe\", \"sites\": \"github,twitter,reddit\"}",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Username/handle to search. Only alphanumeric, dots, underscores, hyphens. Max 50 chars."
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["light", "medium", "full"],
                        "description": "Search depth. 'light' = top 50 sites (~30s), 'medium' = top 200 (~2min), 'full' = all 500+ (~5min). Default: 'light'."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Max scan time in seconds. Default: 120. Max: 300."
                    },
                    "sites": {
                        "type": "string",
                        "description": "Optional comma-separated specific sites to check (e.g., 'github,twitter,reddit'). Overrides depth."
                    }
                },
                "required": ["query"]
            }),
        ),

        // --- Voice Message (TTS) ---
        tool(
            "send_voice",
            "Generate speech from text and send it as a Telegram voice message in the current chat.\n\
             The voice message is sent DIRECTLY to the chat as a playable audio — users hear your voice. \
             After calling this tool, do NOT repeat the same text in your text reply (avoid duplication).\n\n\
             When to use:\n\
             - User explicitly asks you to speak, say something out loud, or send a voice/audio message\n\
             - You want to add emotional impact (greetings, congratulations, dramatic readings, storytelling)\n\
             - User sends a voice message and you want to reply in kind\n\n\
             The 'instructions' parameter controls emotion and delivery — be creative and match the mood:\n\
             - 'Speak warmly and cheerfully, like greeting a close friend'\n\
             - 'Whisper mysteriously, slow pace'\n\
             - 'Sound excited and energetic, fast pace'\n\
             - 'Calm, soothing bedtime story narrator'\n\
             - 'Dramatic movie trailer voice'\n\
             - 'Sarcastic and dry, deadpan delivery'\n\n\
             Examples:\n\
             - Simple: {\"text\": \"Hello! How are you today?\"}\n\
             - Emotional: {\"text\": \"Happy birthday! Wishing you the best!\", \"instructions\": \"Sing-song cheerful voice, celebratory\", \"voice\": \"shimmer\"}\n\
             - Story: {\"text\": \"Once upon a time...\", \"instructions\": \"Gentle bedtime story narrator, slow and calming\", \"voice\": \"fable\"}",
            json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak and send as a voice message. Keep under 4096 chars."
                    },
                    "voice": {
                        "type": "string",
                        "enum": ["coral", "fable", "nova", "sage", "shimmer"],
                        "description": "Female voice to use. 'nova' (default) — warm and natural, 'shimmer' — soft and bright, 'fable' — expressive British, 'coral' — clear and friendly, 'sage' — calm and composed."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "How to speak: emotion, pace, style. Examples: 'cheerful and upbeat', 'slow dramatic whisper', 'professional newsreader'. Be descriptive for best results."
                    }
                },
                "required": ["text"]
            }),
        ),

        // --- MCP Server Interaction ---
        tool(
            "mcp_list_tools",
            "Discover tools/methods available on registered MCP servers (tcp/http transports only). \
             Call this FIRST before using mcp_call, so you know what methods exist and what arguments they accept.\n\
             Returns tool names, descriptions, and input schemas for each server.\n\
             If server_name is omitted, queries ALL enabled tcp/http MCP servers at once.\n\
             Use refresh=true to force re-fetching the tool list from the server (e.g., after server restart or update).",
            json!({
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Optional. Name of a specific MCP server to query (as registered via crud_mcp_server). If omitted, discovers tools from ALL enabled tcp/http servers."
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": "Optional. If true, forces re-fetching the tool list from the server instead of using cached results. Default: false."
                    }
                },
                "required": []
            }),
        ),
        tool(
            "mcp_call",
            "Invoke a specific tool/method on an MCP server. You MUST call mcp_list_tools first to discover available methods and their expected arguments.\n\
             The server is auto-connected if not already connected.\n\
             Returns the server's response or an error message.\n\n\
             CRITICAL: Most methods require arguments. You MUST pass the required fields from the method's inputSchema \
             in the 'arguments' parameter as a JSON object. For example, if create_category requires 'name', call:\n\
             mcp_call(server_name=\"myserver\", method=\"create_category\", arguments={\"name\": \"Electronics\"})\n\
             Calling without arguments when the method requires them will ALWAYS fail.",
            json!({
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Name of the MCP server to call (as registered via crud_mcp_server)."
                    },
                    "method": {
                        "type": "string",
                        "description": "The tool/method name to invoke (exactly as shown in mcp_list_tools results)."
                    },
                    "arguments": {
                        "type": "object",
                        "description": "IMPORTANT: The method's input parameters as a JSON object. Check the inputSchema from mcp_list_tools for required fields. Most methods need this — e.g. {\"name\": \"value\"}. Only omit for methods with no required parameters."
                    }
                },
                "required": ["server_name", "method"]
            }),
        ),
    ]
}

const MCP_DYNAMIC_CONNECT_RETRY_SECS: u64 = 300;
static MCP_DYNAMIC_CONNECT_RETRY_AFTER: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();

fn mcp_dynamic_retry_map() -> &'static Mutex<HashMap<String, Instant>> {
    MCP_DYNAMIC_CONNECT_RETRY_AFTER.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mcp_dynamic_connect_allowed(server_name: &str, now: Instant) -> bool {
    let map = mcp_dynamic_retry_map().lock().unwrap_or_else(|e| e.into_inner());
    match map.get(server_name) {
        Some(next_allowed_at) => *next_allowed_at <= now,
        None => true,
    }
}

fn mcp_dynamic_mark_connect_failure(server_name: &str, now: Instant) {
    let mut map = mcp_dynamic_retry_map().lock().unwrap_or_else(|e| e.into_inner());
    map.insert(
        server_name.to_string(),
        now + Duration::from_secs(MCP_DYNAMIC_CONNECT_RETRY_SECS),
    );
}

fn mcp_dynamic_clear_connect_failure(server_name: &str) {
    let mut map = mcp_dynamic_retry_map().lock().unwrap_or_else(|e| e.into_inner());
    map.remove(server_name);
}

/// Generate dynamic tool definitions from connected MCP servers.
/// Each MCP tool becomes a first-class LLM tool named `mcp__{server}__{method}`
/// so the LLM fills in parameters naturally without the mcp_call indirection.
pub async fn mcp_dynamic_definitions(
    mcp: &McpManager,
    pool: &SqlitePool,
) -> Vec<ToolDefinition> {
    let mut defs = Vec::new();

    let servers = match db::mcp_server_list(pool, false).await {
        Ok(s) => s,
        Err(_) => return defs,
    };

    for server in &servers {
        if !matches!(server.transport.as_str(), "tcp" | "http" | "sse" | "streamable_http") {
            continue;
        }

        let tools = if let Some(cached) = mcp.cached_tools(&server.name).await {
            cached
        } else {
            let now = Instant::now();
            if !mcp_dynamic_connect_allowed(&server.name, now) {
                continue;
            }

            match mcp.ensure_connected(pool, &server.name).await {
                Ok(()) => {
                    mcp_dynamic_clear_connect_failure(&server.name);
                    match mcp.cached_tools(&server.name).await {
                        Some(t) => t,
                        None => continue,
                    }
                }
                Err(e) => {
                    mcp_dynamic_mark_connect_failure(&server.name, now);
                    tracing::warn!(
                        server = %server.name,
                        error = %e,
                        retry_after_secs = MCP_DYNAMIC_CONNECT_RETRY_SECS,
                        "MCP auto-connect failed, skipping dynamic tools"
                    );
                    continue;
                }
            }
        };

        for mcp_tool in &tools {
            let prefixed_name = format!("mcp__{}__{}", server.name, mcp_tool.name);
            let description = format!(
                "[MCP server: {}] {}",
                server.name,
                if mcp_tool.description.is_empty() { &mcp_tool.name } else { &mcp_tool.description }
            );

            // Resolve $ref / $defs in the schema so the LLM sees flat parameters
            let parameters = if mcp_tool.input_schema.is_null() || mcp_tool.input_schema == json!({}) {
                json!({"type": "object", "properties": {}})
            } else {
                resolve_json_schema_refs(&mcp_tool.input_schema)
            };

            defs.push(ToolDefinition {
                tool_type: "function".to_string(),
                function: crate::types::FunctionDefinition {
                    name: prefixed_name,
                    description,
                    parameters,
                },
            });
        }
    }

    defs
}

/// Resolve `$ref` references in a JSON Schema by inlining definitions.
/// Removes `$defs`, `definitions`, and `$schema` from the top level.
fn resolve_json_schema_refs(schema: &Value) -> Value {
    let defs = schema.get("$defs")
        .or(schema.get("definitions"))
        .cloned();

    let mut resolved = inline_refs(schema, &defs);

    if let Some(obj) = resolved.as_object_mut() {
        obj.remove("$defs");
        obj.remove("definitions");
        obj.remove("$schema");
    }

    resolved
}

fn inline_refs(value: &Value, defs: &Option<Value>) -> Value {
    match value {
        Value::Object(map) => {
            // If this object is a $ref, replace it with the referenced definition
            if let Some(ref_val) = map.get("$ref") {
                if let Some(ref_str) = ref_val.as_str() {
                    let name = ref_str.strip_prefix("#/$defs/")
                        .or(ref_str.strip_prefix("#/definitions/"));
                    if let Some(name) = name {
                        if let Some(defs_val) = defs {
                            if let Some(def) = defs_val.get(name) {
                                return inline_refs(def, defs);
                            }
                        }
                    }
                }
                return value.clone();
            }

            // Recursively resolve all values in the object
            let new_map: serde_json::Map<String, Value> = map.iter()
                .map(|(k, v)| (k.clone(), inline_refs(v, defs)))
                .collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(|v| inline_refs(v, defs)).collect())
        }
        _ => value.clone(),
    }
}

/// Execute a tool call and return the result as JSON string
pub async fn execute(
    pool: &SqlitePool,
    bot: &Bot,
    rag: &RagEngine,
    mcp: &McpManager,
    tool_name: &str,
    arguments: &str,
    chat_id: i64,
    user_id: i64,
) -> Result<String> {
    let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));

    let result = match tool_name {
        "store_note" => execute_store_note(pool, rag, &args).await,
        "search_notes" => execute_search_notes(pool, &args).await,
        "read_note" => execute_read_note(pool, &args).await,
        "delete_note" => execute_delete_note(pool, &args).await,
        "memory_set" => execute_memory_set(pool, rag, &args).await,
        "memory_get" => execute_memory_get(pool, &args).await,
        "memory_list" => execute_memory_list(pool, &args).await,
        "crud_mcp_server" => execute_crud_mcp_server(pool, &args, user_id).await,
        // Backward-compatible aliases (if model still calls old tool names)
        "list_mcp_servers" => {
            execute_crud_mcp_server(pool, &with_action(&args, "list"), user_id).await
        }
        "add_mcp_server" => {
            execute_crud_mcp_server(pool, &with_action(&args, "create"), user_id).await
        }
        "update_mcp_server" => {
            execute_crud_mcp_server(pool, &with_action(&args, "update"), user_id).await
        }
        "delete_mcp_server" => {
            execute_crud_mcp_server(pool, &with_action(&args, "delete"), user_id).await
        }
        "search_history" => execute_search_history(pool, &args, chat_id).await,
        "browse_history" => execute_browse_history(pool, &args, chat_id).await,
        "search_all_chats" => execute_search_all_chats(pool, &args).await,
        "set_important_memory" => execute_set_important_memory(pool, &args).await,
        "clear_important_memory" => execute_clear_important_memory(pool, &args).await,
        "generic_http_request" => execute_http_request(&args).await,
        "run_python" => execute_run_python(&args).await,
        "maigret_osint" => execute_maigret_osint(&args).await,
        "send_voice" => execute_send_voice(pool, bot, &args, chat_id).await,
        "rag_search" => execute_rag_search(rag, &args).await,
        "mcp_list_tools" => execute_mcp_list_tools(pool, mcp, &args).await,
        "mcp_call" => execute_mcp_call(pool, mcp, &args).await,
        // Dynamic MCP tools: mcp__{server}__{method} → direct invocation
        name if name.starts_with("mcp__") => execute_mcp_dynamic(pool, mcp, name, &args).await,
        _ => Ok(json!({"error": format!("Unknown tool: {}", tool_name)}).to_string()),
    };

    let result_str = match &result {
        Ok(r) => r.clone(),
        Err(e) => json!({"error": e.to_string()}).to_string(),
    };

    // Log the tool call
    if let Err(e) = db::tool_call_log(pool, chat_id, user_id, tool_name, arguments, &result_str).await {
        tracing::error!(error = %e, "Failed to log tool call");
    }

    tracing::info!(
        tool_name,
        arguments,
        result = %result_str,
        chat_id,
        user_id,
        "Tool call executed"
    );

    Ok(result_str)
}

// --- Note Tools ---

async fn execute_store_note(pool: &SqlitePool, rag: &RagEngine, args: &Value) -> Result<String> {
    let segment = args["segment"].as_str().unwrap_or("global");
    let title = args["title"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'title'"))?;
    let content = args["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'content'"))?;
    let tags = args["tags"].as_str().unwrap_or("");

    let note_id = db::note_create(pool, segment, title, content, tags).await?;

    // Index in RAG
    let embed_text = format!("{}\n{}\n{}", title, content, tags);
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let _ = rag.index_record_sync("note", note_id, 0, segment, &embed_text, "", &now);

    Ok(json!({
        "success": true,
        "note_id": note_id,
        "message": format!("Note '{}' stored with ID {} in segment '{}'", title, note_id, segment)
    }).to_string())
}

async fn execute_search_notes(pool: &SqlitePool, args: &Value) -> Result<String> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'query'"))?;
    let segment = args["segment"].as_str();
    let use_regex = args["use_regex"].as_bool().unwrap_or(false);

    if use_regex {
        let all_notes = db::note_search(pool, "", segment).await?;
        let re = regex::Regex::new(query)
            .map_err(|e| anyhow::anyhow!("Invalid regex '{}': {}", query, e))?;

        let matching: Vec<_> = all_notes
            .into_iter()
            .filter(|n| re.is_match(&n.title) || re.is_match(&n.content) || re.is_match(&n.tags))
            .map(|n| json!({
                "id": n.id, "segment": n.segment, "title": n.title,
                "tags": n.tags, "created_at": n.created_at,
            }))
            .collect();

        Ok(json!({"results": matching, "count": matching.len()}).to_string())
    } else {
        let notes = db::note_search(pool, query, segment).await?;
        let results: Vec<_> = notes.iter().map(|n| json!({
            "id": n.id, "segment": n.segment, "title": n.title,
            "tags": n.tags, "created_at": n.created_at,
        })).collect();
        let count = results.len();
        Ok(json!({"results": results, "count": count}).to_string())
    }
}

async fn execute_read_note(pool: &SqlitePool, args: &Value) -> Result<String> {
    let note_id = args["note_id"]
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("Missing 'note_id'"))?;

    match db::note_read(pool, note_id).await? {
        Some(note) => Ok(json!({
            "id": note.id, "segment": note.segment, "title": note.title,
            "content": note.content, "tags": note.tags,
            "created_at": note.created_at, "updated_at": note.updated_at,
        }).to_string()),
        None => Ok(json!({"error": format!("Note with ID {} not found", note_id)}).to_string()),
    }
}

async fn execute_delete_note(pool: &SqlitePool, args: &Value) -> Result<String> {
    let note_id = args["note_id"]
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("Missing 'note_id'"))?;

    let deleted = db::note_delete(pool, note_id).await?;
    if deleted {
        Ok(json!({"success": true, "message": format!("Note {} deleted", note_id)}).to_string())
    } else {
        Ok(json!({"success": false, "message": format!("Note {} not found", note_id)}).to_string())
    }
}

// --- Memory Tools ---

async fn execute_memory_set(pool: &SqlitePool, rag: &RagEngine, args: &Value) -> Result<String> {
    let segment = args["segment"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'segment'"))?;
    let key = args["key"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'key'"))?;
    let value = args["value"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'value'"))?;

    memory::set(pool, segment, key, value).await?;

    // Index in RAG — query the memory row ID for dedup
    if let Ok(Some((mem_id,))) = sqlx::query_as::<_, (i64,)>(
        "SELECT id FROM memory WHERE segment = ? AND key = ?",
    )
    .bind(segment)
    .bind(key)
    .fetch_optional(pool)
    .await
    {
        let embed_text = format!("{}: {}", key, value);
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let _ = rag.index_record_sync("memory", mem_id, 0, segment, &embed_text, "", &now);
    }

    Ok(json!({"success": true, "message": format!("Stored {}[{}] = '{}'", segment, key, value)}).to_string())
}

async fn execute_memory_get(pool: &SqlitePool, args: &Value) -> Result<String> {
    let segment = args["segment"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'segment'"))?;
    let key = args["key"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'key'"))?;

    match memory::get(pool, segment, key).await? {
        Some(value) => Ok(json!({"key": key, "value": value, "segment": segment}).to_string()),
        None => Ok(json!({"key": key, "value": null, "segment": segment, "message": "Key not found in this segment"}).to_string()),
    }
}

async fn execute_memory_list(pool: &SqlitePool, args: &Value) -> Result<String> {
    let segment = args["segment"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'segment'"))?;

    let entries = memory::list(pool, segment).await?;
    let items: Vec<_> = entries.iter().map(|m| json!({
        "key": m.key, "value": m.value, "updated_at": m.updated_at,
    })).collect();
    let count = items.len();
    Ok(json!({"segment": segment, "entries": items, "count": count}).to_string())
}

fn with_action(args: &Value, action: &str) -> Value {
    let mut enriched = args.clone();
    enriched["action"] = json!(action);
    enriched
}

fn parse_json_or_default(value: Option<&Value>, fallback: &str) -> Result<String> {
    Ok(match value {
        Some(v) => serde_json::to_string(v)?,
        None => fallback.to_string(),
    })
}

fn parse_json_fallback(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| json!(raw))
}

fn normalize_transport(transport: &str) -> Result<String> {
    let transport = transport.trim().to_lowercase();
    let transport = transport.as_str();
    match transport {
        "stdio" | "tcp" | "http" | "sse" | "streamable_http" | "streamable-http" => {
            Ok(match transport {
                "streamable-http" => "streamable_http".to_string(),
                other => other.to_string(),
            })
        }
        _ => anyhow::bail!(
            "Invalid transport '{}'. Allowed values: stdio, tcp, http, sse, streamable_http",
            transport
        ),
    }
}

fn validate_mcp_runtime(
    transport: &str,
    command: &str,
    endpoint: &str,
) -> Result<()> {
    match transport {
        "stdio" => {
            if command.trim().is_empty() {
                anyhow::bail!("Transport 'stdio' requires a `command` field (e.g. 'npx -y @modelcontextprotocol/server-sqlite'). Ask the user what command to run.");
            }
        }
        "tcp" => {
            if command.trim().is_empty() {
                if endpoint.trim().is_empty() {
                    anyhow::bail!(
                        "Transport 'tcp' requires a `command` or `endpoint` field (e.g. 'tcp://127.0.0.1:3001'). Ask the user for the connection details."
                    );
                }
                validate_tcp_endpoint(endpoint)?;
            }
        }
        "http" | "sse" | "streamable_http" => {
            if endpoint.trim().is_empty() {
                anyhow::bail!(
                    "Transport '{}' requires an `endpoint` URL (e.g. 'http://localhost:3000/mcp'). Ask the user for the server URL.",
                    transport
                );
            }
            if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                anyhow::bail!("'endpoint' must start with http:// or https://");
            }
        }
        _ => {}
    }
    Ok(())
}

fn infer_mcp_transport(
    explicit_transport: Option<&str>,
    command: &str,
    endpoint: &str,
) -> Result<String> {
    if let Some(transport_raw) = explicit_transport {
        let transport = transport_raw.trim();
        if !transport.is_empty() {
            return normalize_transport(transport);
        }
    }

    let command = command.trim();
    let endpoint = endpoint.trim();

    if !endpoint.is_empty() {
        if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            return Ok("http".to_string());
        }
        if validate_tcp_endpoint(endpoint).is_ok() {
            return Ok("tcp".to_string());
        }
        anyhow::bail!(
            "Invalid endpoint '{}': expected `tcp://host:port`, `host:port`, `http://host`, or `https://host`.",
            endpoint
        );
    }

    if !command.is_empty() {
        return Ok("stdio".to_string());
    }

    anyhow::bail!(
        "Cannot create MCP server without knowing how to connect. Provide `command` (for stdio) or `endpoint` (for http/tcp). Ask the user which type of server they want to add and the connection details."
    );
}

fn validate_tcp_endpoint(endpoint: &str) -> Result<()> {
    let trimmed = endpoint.trim();
    if trimmed.starts_with("tcp://") {
        return Ok(());
    }

    // Accept host:port shorthand
    let has_colon = trimmed.contains(':');
    let has_slash = trimmed.contains('/');
    let has_spaces = trimmed.contains(char::is_whitespace);

    if !has_colon || has_slash || has_spaces {
        anyhow::bail!("'endpoint' for transport 'tcp' must be 'tcp://host:port' or 'host:port'");
    }

    Ok(())
}

async fn execute_crud_mcp_server(
    pool: &SqlitePool,
    args: &Value,
    actor_id: i64,
) -> Result<String> {
    let action = args["action"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'action'"))?
        .to_lowercase();

    match action.as_str() {
        "list" => execute_crud_mcp_server_list(pool, args).await,
        "create" => execute_crud_mcp_server_create(pool, args, actor_id).await,
        "read" => execute_crud_mcp_server_read(pool, args).await,
        "update" => execute_crud_mcp_server_update(pool, args, actor_id).await,
        "delete" => execute_crud_mcp_server_delete(pool, args).await,
        _ => Ok(json!({
            "error": "Invalid action. Expected one of: create, read, list, update, delete",
        })
        .to_string()),
    }
}

async fn execute_crud_mcp_server_list(pool: &SqlitePool, _args: &Value) -> Result<String> {
    let servers = db::mcp_server_list(pool, false).await?;
    let items: Vec<_> = servers
        .iter()
        .map(|server| json!({
            "id": server.id,
            "name": server.name,
            "description": server.description,
            "transport": server.transport,
            "endpoint": server.endpoint,
            "command": server.command,
            "args": parse_json_fallback(&server.args),
            "environment": parse_json_fallback(&server.environment),
            "created_by": server.created_by,
            "updated_by": server.updated_by,
            "created_at": server.created_at,
            "updated_at": server.updated_at,
        }))
        .collect();

    let count = items.len();
    Ok(json!({
        "count": count,
        "servers": items,
    })
    .to_string())
}

async fn execute_crud_mcp_server_create(
    pool: &SqlitePool,
    args: &Value,
    actor_id: i64,
) -> Result<String> {
    let name = match args["name"].as_str() {
        Some(n) if !n.trim().is_empty() => n,
        _ => return Ok(json!({
            "error": "Missing 'name'. You must provide a name for the MCP server.",
            "correct_call_example": {"action": "create", "name": "my_server", "endpoint": "http://localhost:9090"}
        }).to_string()),
    };
    let command = args["command"].as_str().unwrap_or("").to_string();
    let endpoint = args["endpoint"].as_str().unwrap_or("").to_string();

    // Check for the common LLM mistake: transport set but no command/endpoint
    let explicit_transport = args["transport"].as_str();
    if let Some(t) = explicit_transport {
        let t = t.trim();
        match t {
            "tcp" if endpoint.trim().is_empty() => {
                return Ok(json!({
                    "error": "You set transport='tcp' but did not provide 'endpoint'. You MUST include an endpoint when using tcp transport.",
                    "fix": "Add the 'endpoint' field with the server address. Ask the user for the host and port if you do not know them.",
                    "correct_call_example": {
                        "action": "create",
                        "name": name,
                        "transport": "tcp",
                        "endpoint": "tcp://HOST:PORT"
                    }
                }).to_string());
            }
            "http" | "sse" | "streamable_http" if endpoint.trim().is_empty() => {
                return Ok(json!({
                    "error": format!("You set transport='{}' but did not provide 'endpoint'. You MUST include an endpoint URL when using this transport.", t),
                    "fix": "Add the 'endpoint' field with the server URL. Ask the user for the URL if you do not know it.",
                    "correct_call_example": {
                        "action": "create",
                        "name": name,
                        "transport": t,
                        "endpoint": "http://HOST:PORT"
                    }
                }).to_string());
            }
            "stdio" if command.trim().is_empty() => {
                return Ok(json!({
                    "error": "You set transport='stdio' but did not provide 'command'. You MUST include a command when using stdio transport.",
                    "fix": "Add the 'command' field with the shell command to start the server. Ask the user for the command if you do not know it.",
                    "correct_call_example": {
                        "action": "create",
                        "name": name,
                        "transport": "stdio",
                        "command": "npx -y @modelcontextprotocol/server-example"
                    }
                }).to_string());
            }
            _ => {}
        }
    }

    if command.trim().is_empty() && endpoint.trim().is_empty() {
        return Ok(json!({
            "error": "You must provide either 'command' (for stdio servers) or 'endpoint' (for http/tcp servers). Ask the user which type of server they want and the connection details.",
            "examples": [
                {"action": "create", "name": name, "command": "npx -y @modelcontextprotocol/server-sqlite /path/to/db"},
                {"action": "create", "name": name, "endpoint": "http://127.0.0.1:9090"},
                {"action": "create", "name": name, "endpoint": "127.0.0.1:9090"}
            ]
        }).to_string());
    }

    let transport = infer_mcp_transport(explicit_transport, &command, &endpoint)?;
    let description = args["description"].as_str().unwrap_or("");
    let args_json = parse_json_or_default(args.get("args"), "[]")?;
    let env_json = parse_json_or_default(args.get("environment"), "{}")?;

    validate_mcp_runtime(&transport, &command, &endpoint)?;

    let id = db::mcp_server_create(
        pool,
        name,
        description,
        &transport,
        &endpoint,
        &command,
        &args_json,
        &env_json,
        true,
        Some(actor_id),
    )
    .await?;

    Ok(json!({
        "success": true,
        "id": id,
        "name": name,
        "message": format!("MCP server '{}' added", name),
        "transport": transport,
    })
    .to_string())
}

async fn execute_crud_mcp_server_read(pool: &SqlitePool, args: &Value) -> Result<String> {
    let name = args["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'name'"))?;

    match db::mcp_server_get(pool, name).await? {
        Some(server) => Ok(json!({
            "success": true,
            "server": {
                "id": server.id,
                "name": server.name,
                "description": server.description,
            "transport": server.transport,
            "endpoint": server.endpoint,
            "command": server.command,
            "args": parse_json_fallback(&server.args),
            "environment": parse_json_fallback(&server.environment),
            "created_by": server.created_by,
            "updated_by": server.updated_by,
            "created_at": server.created_at,
            "updated_at": server.updated_at,
        }
        })
        .to_string()),
        None => Ok(json!({
            "success": false,
            "message": format!("MCP server '{}' not found", name),
        })
        .to_string()),
    }
}

async fn execute_crud_mcp_server_update(
    pool: &SqlitePool,
    args: &Value,
    actor_id: i64,
) -> Result<String> {
    let current_name = args["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'name'"))?;

    let existing = db::mcp_server_get(pool, current_name)
        .await?
        .ok_or_else(|| anyhow::anyhow!("MCP server '{}' not found", current_name))?;

    let new_name = args["new_name"].as_str().unwrap_or(current_name);

    if new_name != current_name {
        if db::mcp_server_get(pool, new_name).await?.is_some() {
            anyhow::bail!("MCP server '{}' already exists", new_name);
        }
    }

    let description = args["description"].as_str().unwrap_or(&existing.description);
    let command = args["command"]
        .as_str()
        .unwrap_or(&existing.command)
        .to_string();
    let endpoint = args["endpoint"]
        .as_str()
        .unwrap_or(&existing.endpoint)
        .to_string();
    let transport = if args.get("transport").is_none()
        && args.get("command").is_none()
        && args.get("endpoint").is_none()
    {
        existing.transport.clone()
    } else {
        infer_mcp_transport(
            args.get("transport").and_then(|v| v.as_str()),
            &command,
            &endpoint,
        )?
    };
    let args_json = match args.get("args") {
        Some(v) => parse_json_or_default(Some(v), &existing.args)?,
        None => existing.args.clone(),
    };
    let env_json = match args.get("environment") {
        Some(v) => parse_json_or_default(Some(v), &existing.environment)?,
        None => existing.environment.clone(),
    };

    if current_name == new_name
        && description == existing.description
        && transport == existing.transport
        && command == existing.command
        && endpoint == existing.endpoint
        && args_json == existing.args
        && env_json == existing.environment
    {
        return Ok(json!({
            "success": false,
            "message": "No changes were provided for update",
        })
        .to_string());
    }

    validate_mcp_runtime(&transport, &command, &endpoint)?;

    let updated = db::mcp_server_update(
        pool,
        current_name,
        new_name,
        description,
        &transport,
        &endpoint,
        &command,
        &args_json,
        &env_json,
        existing.enabled,
        Some(actor_id),
    )
    .await?;

    if updated {
        Ok(json!({
            "success": true,
            "name": new_name,
            "message": format!("MCP server '{}' updated", new_name),
        })
        .to_string())
    } else {
        Ok(json!({
            "success": false,
            "message": format!("Failed to update MCP server '{}'", current_name),
        })
        .to_string())
    }
}

async fn execute_crud_mcp_server_delete(pool: &SqlitePool, args: &Value) -> Result<String> {
    let name = args["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'name'"))?;

    let deleted = db::mcp_server_delete(pool, name).await?;
    if deleted {
        Ok(json!({
            "success": true,
            "message": format!("MCP server '{}' deleted", name),
        })
        .to_string())
    } else {
        Ok(json!({
            "success": false,
            "message": format!("MCP server '{}' not found", name),
        })
        .to_string())
    }
}

// --- History Tools ---

async fn execute_search_history(pool: &SqlitePool, args: &Value, chat_id: i64) -> Result<String> {
    let keyword = args["keyword"].as_str();
    let sender_name = args["sender_name"].as_str();
    let sender_id = args["sender_id"].as_i64();
    let date_from = args["date_from"].as_str();
    let date_to = args["date_to"].as_str();
    let limit = args["limit"].as_i64().unwrap_or(20).min(100);
    let offset = args["offset"].as_i64().unwrap_or(0);

    let rows = db::conversation_search(
        pool, chat_id, keyword, sender_name, sender_id,
        date_from, date_to, limit, offset,
    ).await?;

    let messages: Vec<_> = rows.iter().map(|r| json!({
        "id": r.id,
        "message_id": r.message_id,
        "reply_to_id": r.reply_to_id,
        "user_name": r.user_name,
        "user_id": r.user_id,
        "role": r.role,
        "content": r.content,
        "created_at": r.created_at,
    })).collect();

    let count = messages.len();
    Ok(json!({
        "messages": messages,
        "count": count,
        "has_more": count as i64 == limit,
        "offset": offset,
    }).to_string())
}

async fn execute_browse_history(pool: &SqlitePool, args: &Value, chat_id: i64) -> Result<String> {
    let offset = args["offset"].as_i64().unwrap_or(0);
    let limit = args["limit"].as_i64().unwrap_or(20).min(50);

    let rows = db::conversation_search(
        pool, chat_id, None, None, None, None, None, limit, offset,
    ).await?;

    let messages: Vec<_> = rows.iter().map(|r| json!({
        "id": r.id,
        "message_id": r.message_id,
        "reply_to_id": r.reply_to_id,
        "user_name": r.user_name,
        "user_id": r.user_id,
        "role": r.role,
        "content": r.content,
        "created_at": r.created_at,
    })).collect();

    let count = messages.len();
    Ok(json!({
        "messages": messages,
        "count": count,
        "offset": offset,
        "next_offset": offset + count as i64,
        "has_more": count as i64 == limit,
    }).to_string())
}

async fn execute_search_all_chats(pool: &SqlitePool, args: &Value) -> Result<String> {
    let keyword = args["keyword"].as_str();
    let sender_name = args["sender_name"].as_str();
    let sender_id = args["sender_id"].as_i64();
    let date_from = args["date_from"].as_str();
    let date_to = args["date_to"].as_str();
    let limit = args["limit"].as_i64().unwrap_or(20).min(100);
    let offset = args["offset"].as_i64().unwrap_or(0);

    // Require at least one filter
    if keyword.is_none() && sender_name.is_none() && sender_id.is_none() && date_from.is_none() && date_to.is_none() {
        return Ok(json!({
            "error": "At least one filter (keyword, sender_name, sender_id, date_from, date_to) is required to search across all chats."
        }).to_string());
    }

    let rows = db::conversation_search_global(
        pool, keyword, sender_name, sender_id,
        date_from, date_to, limit, offset,
    ).await?;

    let messages: Vec<_> = rows.iter().map(|r| json!({
        "id": r.id,
        "chat_id": r.chat_id,
        "message_id": r.message_id,
        "reply_to_id": r.reply_to_id,
        "user_name": r.user_name,
        "user_id": r.user_id,
        "role": r.role,
        "content": r.content,
        "created_at": r.created_at,
    })).collect();

    let count = messages.len();
    Ok(json!({
        "messages": messages,
        "count": count,
        "has_more": count as i64 == limit,
        "offset": offset,
    }).to_string())
}

// --- Important Memory Tools ---

async fn execute_set_important_memory(pool: &SqlitePool, args: &Value) -> Result<String> {
    let segment = args["segment"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'segment'"))?;
    let content = args["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'content'"))?;

    memory::validate_segment(segment)?;
    db::set_important_memory(pool, segment, content).await?;
    Ok(json!({
        "success": true,
        "message": format!("Pinned important memory for segment '{}'. This will now be included in every prompt.", segment),
        "segment": segment,
        "content": content,
    }).to_string())
}

async fn execute_clear_important_memory(pool: &SqlitePool, args: &Value) -> Result<String> {
    let segment = args["segment"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'segment'"))?;

    memory::validate_segment(segment)?;
    let cleared = db::clear_important_memory(pool, segment).await?;
    if cleared {
        Ok(json!({"success": true, "message": format!("Cleared pinned memory for segment '{}'", segment)}).to_string())
    } else {
        Ok(json!({"success": false, "message": format!("No pinned memory found for segment '{}'", segment)}).to_string())
    }
}

// --- HTTP Request ---

const DEFAULT_MAX_LENGTH: usize = 8000;
const ABSOLUTE_MAX_LENGTH: usize = 30000;
const DEFAULT_TIMEOUT_SECS: u64 = 10;
const MAX_TIMEOUT_SECS: u64 = 60;
const HTTP_MAX_RETRIES: u32 = 3;

async fn execute_http_request(args: &Value) -> Result<String> {
    let url = args["url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'url'"))?;
    let method = args["method"].as_str().unwrap_or("GET").to_uppercase();
    let strip_html = args["strip_html"].as_bool().unwrap_or(true);
    let response_type = args["response_type"].as_str().unwrap_or("auto");
    let max_length = args["max_length"]
        .as_u64()
        .map(|v| (v as usize).min(ABSOLUTE_MAX_LENGTH))
        .unwrap_or(DEFAULT_MAX_LENGTH);
    let timeout_secs = args["timeout_secs"]
        .as_u64()
        .map(|v| v.min(MAX_TIMEOUT_SECS))
        .unwrap_or(DEFAULT_TIMEOUT_SECS);

    // Validate URL
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Ok(json!({"error": "URL must start with http:// or https://"}).to_string());
    }

    // Build client
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    // Build the request
    let mut request_builder = match method.as_str() {
        "GET" => client.get(url),
        "POST" => client.post(url),
        "PUT" => client.put(url),
        "DELETE" => client.delete(url),
        "PATCH" => client.patch(url),
        "HEAD" => client.head(url),
        _ => return Ok(json!({"error": format!("Unsupported HTTP method: {}", method)}).to_string()),
    };

    // Default headers
    request_builder = request_builder
        .header("User-Agent", "Mozilla/5.0 (compatible; AstarteBot/0.1)")
        .header("Accept-Language", "en-US,en;q=0.9");

    // Custom headers
    if let Some(headers) = args["headers"].as_object() {
        for (key, val) in headers {
            if let Some(v) = val.as_str() {
                request_builder = request_builder.header(key.as_str(), v);
            }
        }
    }

    // Query params
    if let Some(params) = args["query_params"].as_object() {
        let pairs: Vec<(&str, &str)> = params
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.as_str(), s)))
            .collect();
        request_builder = request_builder.query(&pairs);
    }

    // Body
    if let Some(body) = args["body"].as_str() {
        request_builder = request_builder.body(body.to_string());
    }

    // Send with retries (retry on 5xx and timeouts)
    let mut last_error = None;
    let mut response_opt = None;

    for attempt in 0..HTTP_MAX_RETRIES {
        if attempt > 0 {
            let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt));
            tokio::time::sleep(delay).await;
        }

        match request_builder
            .try_clone()
            .unwrap_or_else(|| {
                // If clone fails (body was consumed), rebuild a simple GET
                client.get(url)
            })
            .send()
            .await
        {
            Ok(resp) => {
                if resp.status().is_server_error() && attempt < HTTP_MAX_RETRIES - 1 {
                    last_error = Some(format!("HTTP {} (attempt {}/{})", resp.status().as_u16(), attempt + 1, HTTP_MAX_RETRIES));
                    continue;
                }
                response_opt = Some(resp);
                break;
            }
            Err(e) => {
                let mut error_info = json!({
                    "error": format!("{}", e),
                    "url": url,
                    "method": method,
                    "attempt": attempt + 1,
                    "max_retries": HTTP_MAX_RETRIES,
                });
                if e.is_timeout() {
                    error_info["reason"] = json!("timeout");
                    error_info["detail"] = json!(format!("Request timed out after {}s", timeout_secs));
                } else if e.is_connect() {
                    error_info["reason"] = json!("connection_failed");
                    error_info["detail"] = json!("Could not connect. DNS resolution failed or host is unreachable.");
                } else if e.is_redirect() {
                    error_info["reason"] = json!("too_many_redirects");
                } else {
                    error_info["reason"] = json!("request_error");
                }
                if let Some(source) = StdError::source(&e) {
                    error_info["source"] = json!(format!("{}", source));
                }

                // Only retry on timeout/connect errors
                if (e.is_timeout() || e.is_connect()) && attempt < HTTP_MAX_RETRIES - 1 {
                    last_error = Some(format!("{}", e));
                    continue;
                }
                return Ok(error_info.to_string());
            }
        }
    }

    let response = match response_opt {
        Some(r) => r,
        None => {
            return Ok(json!({
                "error": format!("All {} retries exhausted", HTTP_MAX_RETRIES),
                "last_error": last_error,
                "url": url,
                "method": method,
            }).to_string());
        }
    };

    // Collect response metadata
    let status = response.status().as_u16();
    let status_text = response.status().canonical_reason().unwrap_or("").to_string();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // Collect useful response headers
    let mut resp_headers = json!({});
    for key in ["content-type", "content-length", "location", "retry-after", "x-ratelimit-remaining", "x-ratelimit-reset"] {
        if let Some(val) = response.headers().get(key).and_then(|v| v.to_str().ok()) {
            resp_headers[key] = json!(val);
        }
    }

    // For HEAD requests, return just metadata
    if method == "HEAD" {
        return Ok(json!({
            "url": url,
            "method": "HEAD",
            "status": status,
            "status_text": status_text,
            "headers": resp_headers,
        }).to_string());
    }

    // Read body
    let bytes = match response.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return Ok(json!({
                "error": format!("Failed to read response body: {}", e),
                "url": url,
                "status": status,
            }).to_string());
        }
    };

    if bytes.len() > 2_097_152 {
        return Ok(json!({
            "error": "Response too large (>2MB)",
            "url": url,
            "status": status,
            "size_bytes": bytes.len(),
        }).to_string());
    }

    let raw_body = String::from_utf8_lossy(&bytes).to_string();
    let raw_size = raw_body.len();

    // Non-success: return error with body preview
    if status >= 400 {
        let body_preview: String = raw_body.chars().take(1000).collect();
        return Ok(json!({
            "error": format!("HTTP {} {}", status, status_text),
            "url": url,
            "method": method,
            "status": status,
            "status_text": status_text,
            "content_type": content_type,
            "headers": resp_headers,
            "body_preview": body_preview,
        }).to_string());
    }

    // Determine how to process the response
    let is_json = response_type == "json"
        || (response_type == "auto" && content_type.contains("json"));
    let is_html = response_type == "html"
        || (response_type == "auto" && (content_type.contains("html") || content_type.contains("xhtml")));

    let content = if is_json {
        // Parse as JSON and return structured
        match serde_json::from_str::<Value>(&raw_body) {
            Ok(parsed) => {
                let pretty = serde_json::to_string_pretty(&parsed).unwrap_or(raw_body.clone());
                pretty
            }
            Err(_) => raw_body.clone(),
        }
    } else if is_html && strip_html {
        // Convert HTML to clean plain text
        let text = html2text::from_read(raw_body.as_bytes(), 80)
            .unwrap_or_else(|_| raw_body.clone());

        // Collapse excessive blank lines
        let mut cleaned = String::with_capacity(text.len());
        let mut blank_count = 0;
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                blank_count += 1;
                if blank_count <= 2 {
                    cleaned.push('\n');
                }
            } else {
                blank_count = 0;
                cleaned.push_str(trimmed);
                cleaned.push('\n');
            }
        }
        cleaned
    } else {
        raw_body
    };

    // Truncate to max_length (floor to char boundary to avoid panic on multibyte chars)
    let truncated = content.len() > max_length;
    let final_content = if truncated {
        let mut boundary = max_length;
        while boundary > 0 && !content.is_char_boundary(boundary) {
            boundary -= 1;
        }
        let cut = content[..boundary]
            .rfind(|c: char| c.is_whitespace())
            .unwrap_or(boundary);
        format!("{}...\n\n[TRUNCATED: {}/{} chars]", &content[..cut], cut, content.len())
    } else {
        content
    };

    Ok(json!({
        "url": url,
        "method": method,
        "status": status,
        "status_text": status_text,
        "content_type": content_type,
        "headers": resp_headers,
        "body": final_content,
        "raw_size_bytes": raw_size,
        "returned_size": final_content.len(),
        "truncated": truncated,
        "strip_html": strip_html && is_html,
    }).to_string())
}

// --- Python Execution ---

const PYTHON_DEFAULT_TIMEOUT: u64 = 30;
const PYTHON_MAX_TIMEOUT: u64 = 120;
const PYTHON_MAX_OUTPUT: usize = 15000;
static RUN_PYTHON_SANDBOX_SEQ: AtomicU64 = AtomicU64::new(0);

fn truncate_text_for_output(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    let mut boundary = max_len;
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }

    let cut = text[..boundary]
        .rfind(|c: char| c.is_whitespace())
        .unwrap_or(boundary);

    format!(
        "{}...\n[TRUNCATED: {}/{} chars]",
        &text[..cut],
        cut,
        text.len()
    )
}

async fn execute_run_python(args: &Value) -> Result<String> {
    let code = args["code"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'code'"))?;
    let timeout_secs = args["timeout_secs"]
        .as_u64()
        .map(|v| v.min(PYTHON_MAX_TIMEOUT))
        .unwrap_or(PYTHON_DEFAULT_TIMEOUT);

    // Create a unique temp directory for this invocation to avoid cross-call races.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = RUN_PYTHON_SANDBOX_SEQ.fetch_add(1, Ordering::Relaxed);
    let sandbox_dir = std::env::temp_dir().join(format!(
        "astarte_sandbox_{}_{}_{}",
        std::process::id(),
        ts,
        seq
    ));
    let workspace = sandbox_dir.join("workspace");
    tokio::fs::create_dir_all(&workspace).await?;

    // Write input files if provided
    if let Some(files) = args["input_files"].as_object() {
        for (filename, content) in files {
            if let Some(text) = content.as_str() {
                // Sanitize filename — no path traversal
                let safe_name: String = filename
                    .replace("..", "")
                    .replace('/', "_")
                    .replace('\\', "_");
                let file_path = workspace.join(&safe_name);
                tokio::fs::write(&file_path, text).await?;
            }
        }
    }

    // Write the script
    let script_path = workspace.join("__script__.py");
    tokio::fs::write(&script_path, code).await?;

    // Check which paths exist for bind mounts
    let mut bind_args: Vec<String> = Vec::new();

    // Read-only system mounts needed for Python
    let ro_mounts = [
        "/usr", "/lib", "/lib64", "/lib32",
        "/etc/resolv.conf", "/etc/ssl", "/etc/ca-certificates",
        "/etc/alternatives", "/etc/ld.so.cache",
    ];
    for path in &ro_mounts {
        if tokio::fs::metadata(path).await.is_ok() {
            bind_args.push("--ro-bind".to_string());
            bind_args.push(path.to_string());
            bind_args.push(path.to_string());
        }
    }

    // Build bwrap command
    let mut cmd_args = Vec::new();

    // Filesystem
    cmd_args.extend(bind_args);
    cmd_args.extend([
        "--bind".to_string(), workspace.to_string_lossy().to_string(), "/workspace".to_string(),
        "--proc".to_string(), "/proc".to_string(),
        "--dev".to_string(), "/dev".to_string(),
        "--tmpfs".to_string(), "/tmp".to_string(),
        "--chdir".to_string(), "/workspace".to_string(),
    ]);

    // Isolation: unshare everything except network
    cmd_args.extend([
        "--unshare-user".to_string(),
        "--unshare-pid".to_string(),
        "--unshare-ipc".to_string(),
        "--unshare-cgroup".to_string(),
        "--die-with-parent".to_string(),
    ]);

    // The actual command: timeout + python3
    cmd_args.extend([
        "timeout".to_string(),
        format!("{}", timeout_secs),
        "python3".to_string(),
        "/workspace/__script__.py".to_string(),
    ]);

    tracing::info!(
        timeout_secs,
        code_len = code.len(),
        "Executing Python in sandbox"
    );

    // Run bwrap
    let output = match tokio::process::Command::new("bwrap")
        .args(&cmd_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
    {
        Ok(o) => o,
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&sandbox_dir).await;
            return Ok(json!({
                "error": format!("Failed to launch sandbox: {}", e),
                "hint": "Ensure 'bwrap' (bubblewrap) is installed: apt install bubblewrap",
            }).to_string());
        }
    };

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout_raw = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr_raw = String::from_utf8_lossy(&output.stderr).to_string();

    // Truncate output if too large
    let stdout = truncate_text_for_output(&stdout_raw, PYTHON_MAX_OUTPUT);
    let stderr = truncate_text_for_output(&stderr_raw, PYTHON_MAX_OUTPUT);

    // List files created in workspace (excluding the script itself)
    let mut created_files = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&workspace).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name != "__script__.py" {
                let meta = entry.metadata().await.ok();
                let size = meta.map(|m| m.len()).unwrap_or(0);
                created_files.push(json!({"name": name, "size_bytes": size}));
            }
        }
    }

    // Determine execution status
    let timed_out = exit_code == 124; // timeout command returns 124

    // Clean up sandbox
    let _ = tokio::fs::remove_dir_all(&sandbox_dir).await;

    Ok(json!({
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "timeout_secs": timeout_secs,
        "files_created": created_files,
    }).to_string())
}

// --- Maigret OSINT ---

const MAIGRET_DEFAULT_TIMEOUT: u64 = 120;
const MAIGRET_MAX_TIMEOUT: u64 = 300;

async fn execute_maigret_osint(args: &Value) -> Result<String> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'query'"))?;
    let depth = args["depth"].as_str().unwrap_or("light");
    let timeout_secs = args["timeout_secs"]
        .as_u64()
        .map(|v| v.min(MAIGRET_MAX_TIMEOUT))
        .unwrap_or(MAIGRET_DEFAULT_TIMEOUT);
    let sites = args["sites"].as_str().unwrap_or("");

    // Validate username
    let username_re = regex::Regex::new(r"^[a-zA-Z0-9._-]+$").unwrap();
    if query.is_empty() || query.len() > 50 || !username_re.is_match(query) {
        return Ok(json!({
            "error": "Invalid username. Only letters, digits, dots, underscores, hyphens. 1-50 chars."
        }).to_string());
    }

    // Create sandbox workspace with unique name
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sandbox_dir = std::env::temp_dir().join(format!("astarte_maigret_{}", ts));
    let workspace = sandbox_dir.join("workspace");
    tokio::fs::create_dir_all(&workspace).await?;

    // Write a small Python wrapper that runs maigret and writes clean JSON
    let params = json!({
        "query": query,
        "depth": depth,
        "timeout": timeout_secs,
    });
    tokio::fs::write(
        workspace.join("__params__.json"),
        serde_json::to_string(&params)?,
    )
    .await?;

    // Build depth flags
    let mut maigret_flags: Vec<String> = Vec::new();
    if !sites.is_empty() {
        for site in sites.split(',') {
            let t = site.trim();
            if !t.is_empty() {
                maigret_flags.extend(["--site".into(), t.to_string()]);
            }
        }
    } else {
        match depth {
            "medium" => maigret_flags.extend(["--top-sites".into(), "200".into()]),
            "full" => {}
            _ => maigret_flags.extend(["--top-sites".into(), "50".into()]),
        }
    }

    // Write Python wrapper script
    let script = r#"
import subprocess, json, sys, os

with open('/workspace/__params__.json') as f:
    p = json.load(f)

query = p['query']
depth = p['depth']
timeout = p['timeout']

# Redirect maigret cache writes to writable dirs
os.environ['XDG_DATA_HOME'] = '/workspace/.data'
os.environ['XDG_CONFIG_HOME'] = '/workspace/.config'
os.environ['XDG_CACHE_HOME'] = '/workspace/.cache'
for d in ['/workspace/.data', '/workspace/.config', '/workspace/.cache']:
    os.makedirs(d, exist_ok=True)

# Read extra flags (depth/site flags)
flags = []
flag_file = '/workspace/__flags__.json'
if os.path.exists(flag_file):
    with open(flag_file) as f:
        flags = json.load(f)

# Copy maigret's bundled database to writable workspace so it can read+update it
import shutil, importlib
try:
    maigret_pkg = importlib.import_module('maigret')
    pkg_dir = os.path.dirname(maigret_pkg.__file__)
    src_db = os.path.join(pkg_dir, 'resources', 'data.json')
    dst_db = '/workspace/data.json'
    if os.path.exists(src_db):
        shutil.copy2(src_db, dst_db)
except Exception:
    dst_db = None

# --json takes a format type ('simple' or 'ndjson'), output goes to stdout
cmd = ['python3', '-m', 'maigret', query, '--json', 'simple', '--no-color', '--no-progressbar'] + flags
if dst_db and os.path.exists(dst_db):
    cmd += ['--db', dst_db]

try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd='/workspace')
except subprocess.TimeoutExpired:
    print(json.dumps({"error": "Scan timed out", "query": query, "timeout_secs": timeout}))
    sys.exit(0)
except Exception as e:
    msg = str(e)
    if 'No module' in msg or 'not found' in msg:
        print(json.dumps({"error": "Maigret not installed. Run: sudo pip3 install maigret"}))
    else:
        print(json.dumps({"error": msg}))
    sys.exit(0)

# Parse JSON from stdout
data = None
if proc.stdout:
    raw = proc.stdout.strip()
    # Try direct parse
    try:
        data = json.loads(raw)
    except:
        pass
    # Try finding JSON object/array in mixed output (maigret may print text before JSON)
    if not data:
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = raw.rfind('\n' + start_char)
            if start == -1:
                start = raw.find(start_char)
            else:
                start += 1  # skip newline
            end = raw.rfind(end_char)
            if start != -1 and end != -1 and end >= start:
                try:
                    data = json.loads(raw[start:end+1])
                    break
                except:
                    pass

if not data:
    print(json.dumps({
        "error": "No parseable output from maigret",
        "exit_code": proc.returncode,
        "stderr": (proc.stderr or "")[:3000],
        "stdout": (proc.stdout or "")[:3000],
    }))
    sys.exit(0)

# Extract claimed profiles from maigret output (handles multiple formats)
profiles = []
checked = 0

def get_status(info):
    s = info.get('status', 'Unknown')
    if isinstance(s, dict):
        return s.get('status', str(s))
    return str(s)

def scan_sites(d):
    global checked
    for name, info in d.items():
        if not isinstance(info, dict):
            continue
        if 'url_user' not in info and 'status' not in info:
            continue
        checked += 1
        st = get_status(info)
        if st.lower() in ('claimed', 'taken', 'found', 'detected'):
            profiles.append({
                "site": name,
                "url": info.get('url_user', info.get('url', '')),
                "status": st,
            })

if isinstance(data, list):
    for item in data:
        if isinstance(item, dict):
            if 'sites' in item and isinstance(item['sites'], dict):
                scan_sites(item['sites'])
            elif 'url_user' in item:
                checked += 1
                st = get_status(item)
                if st.lower() in ('claimed', 'taken', 'found', 'detected'):
                    profiles.append({
                        "site": item.get('site_name', item.get('name', '?')),
                        "url": item.get('url_user', ''),
                        "status": st,
                    })
elif isinstance(data, dict):
    target = data.get(query, data)
    if isinstance(target, dict):
        if 'sites' in target and isinstance(target['sites'], dict):
            scan_sites(target['sites'])
        else:
            scan_sites(target)

profiles.sort(key=lambda x: x['site'].lower())

print(json.dumps({
    "query": query,
    "total_found": len(profiles),
    "total_checked": checked,
    "scan_depth": depth,
    "profiles": profiles,
}, indent=2, ensure_ascii=False))
"#;
    tokio::fs::write(workspace.join("__script__.py"), script).await?;
    tokio::fs::write(
        workspace.join("__flags__.json"),
        serde_json::to_string(&maigret_flags)?,
    )
    .await?;

    // Build bwrap command
    let mut bind_args: Vec<String> = Vec::new();

    let ro_mounts = [
        "/usr", "/lib", "/lib64", "/lib32",
        "/etc/resolv.conf", "/etc/ssl", "/etc/ca-certificates",
        "/etc/alternatives", "/etc/ld.so.cache",
    ];
    for path in &ro_mounts {
        if tokio::fs::metadata(path).await.is_ok() {
            bind_args.push("--ro-bind".to_string());
            bind_args.push(path.to_string());
            bind_args.push(path.to_string());
        }
    }

    // Bind ~/.local for pip --user installs
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let local_dir = format!("{}/.local", home);
    if tokio::fs::metadata(&local_dir).await.is_ok() {
        bind_args.push("--ro-bind".to_string());
        bind_args.push(local_dir.clone());
        bind_args.push(local_dir);
    }

    let mut cmd_args = Vec::new();
    cmd_args.extend(bind_args);
    cmd_args.extend([
        "--bind".to_string(), workspace.to_string_lossy().to_string(), "/workspace".to_string(),
        "--proc".to_string(), "/proc".to_string(),
        "--dev".to_string(), "/dev".to_string(),
        "--tmpfs".to_string(), "/tmp".to_string(),
        "--setenv".to_string(), "HOME".to_string(), home,
        "--chdir".to_string(), "/workspace".to_string(),
        "--unshare-user".to_string(),
        "--unshare-pid".to_string(),
        "--unshare-ipc".to_string(),
        "--unshare-cgroup".to_string(),
        "--die-with-parent".to_string(),
    ]);

    let total_timeout = timeout_secs + 30;
    cmd_args.extend([
        "timeout".to_string(),
        format!("{}", total_timeout),
        "python3".to_string(),
        "/workspace/__script__.py".to_string(),
    ]);

    tracing::info!(query, depth, timeout_secs, "Executing Maigret OSINT scan");

    let output = match tokio::process::Command::new("bwrap")
        .args(&cmd_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
    {
        Ok(o) => o,
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&sandbox_dir).await;
            return Ok(json!({
                "error": format!("Failed to launch sandbox: {}", e),
                "hint": "Ensure 'bwrap' (bubblewrap) is installed: sudo apt install bubblewrap",
            }).to_string());
        }
    };

    let exit_code = output.status.code().unwrap_or(-1);
    let stdout_raw = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr_raw = String::from_utf8_lossy(&output.stderr).to_string();

    // Clean up sandbox
    let _ = tokio::fs::remove_dir_all(&sandbox_dir).await;

    // Timeout
    if exit_code == 124 {
        return Ok(json!({
            "error": "Maigret scan timed out",
            "timeout_secs": total_timeout,
            "query": query,
        }).to_string());
    }

    // Parse stdout JSON (Python script outputs clean JSON)
    let stdout_trimmed = stdout_raw.trim();
    if !stdout_trimmed.is_empty() {
        // Try direct parse
        if let Ok(result) = serde_json::from_str::<Value>(stdout_trimmed) {
            return Ok(result.to_string());
        }
        // Try extracting JSON from mixed output (maigret may print progress)
        if let Some(start) = stdout_trimmed.rfind("\n{") {
            if let Ok(result) = serde_json::from_str::<Value>(&stdout_trimmed[start + 1..]) {
                return Ok(result.to_string());
            }
        }
    }

    Ok(json!({
        "error": "Failed to parse maigret output",
        "exit_code": exit_code,
        "stdout": stdout_raw.chars().take(5000).collect::<String>(),
        "stderr": stderr_raw.chars().take(3000).collect::<String>(),
        "query": query,
    }).to_string())
}

// --- Voice Message (TTS) ---

const OPENAI_TTS_URL: &str = "https://api.openai.com/v1/audio/speech";

async fn execute_send_voice(
    pool: &SqlitePool,
    bot: &Bot,
    args: &Value,
    chat_id: i64,
) -> Result<String> {
    let text = args["text"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'text'"))?;
    let voice = args["voice"].as_str().unwrap_or("nova");
    let instructions = args["instructions"].as_str();

    if text.is_empty() {
        return Ok(json!({"error": "Text cannot be empty"}).to_string());
    }
    if text.len() > 4096 {
        return Ok(json!({"error": "Text too long, max 4096 characters"}).to_string());
    }

    // Get OpenAI API key from config
    let api_key = match crate::config::get(pool, "openai_api_key").await? {
        Some(key) if !key.is_empty() => key,
        _ => {
            return Ok(json!({
                "error": "OpenAI API key not configured. Set it with: astartebot config set openai_api_key sk-..."
            }).to_string());
        }
    };

    // Use gpt-4o-mini-tts if instructions are provided (supports steerability), otherwise tts-1
    let model = if instructions.is_some() {
        "gpt-4o-mini-tts"
    } else {
        "tts-1"
    };

    // Build request
    let mut body = json!({
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "opus",
    });
    if let Some(instr) = instructions {
        body["instructions"] = json!(instr);
    }

    tracing::info!(
        voice,
        model,
        text_len = text.len(),
        instructions = instructions.unwrap_or("none"),
        "Generating TTS voice message"
    );

    // Call OpenAI TTS API
    let client = reqwest::Client::new();
    let response = client
        .post(OPENAI_TTS_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await;

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            return Ok(json!({
                "error": format!("Failed to call OpenAI TTS: {}", e),
            }).to_string());
        }
    };

    let status = response.status();
    if !status.is_success() {
        let error_body = response.text().await.unwrap_or_default();
        return Ok(json!({
            "error": format!("OpenAI TTS API error: HTTP {}", status),
            "detail": error_body,
        }).to_string());
    }

    // Get audio bytes
    let audio_bytes = match response.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return Ok(json!({
                "error": format!("Failed to read TTS response: {}", e),
            }).to_string());
        }
    };

    let audio_size = audio_bytes.len();

    // Send as Telegram voice message
    let input_file = InputFile::memory(audio_bytes.to_vec()).file_name("voice.ogg");
    match bot.send_voice(ChatId(chat_id), input_file).await {
        Ok(_) => {
            tracing::info!(audio_size, chat_id, "Voice message sent");
            Ok(json!({
                "success": true,
                "message": "Voice message sent to chat",
                "audio_size_bytes": audio_size,
                "model": model,
                "voice": voice,
            }).to_string())
        }
        Err(e) => {
            Ok(json!({
                "error": format!("Failed to send voice message via Telegram: {}", e),
            }).to_string())
        }
    }
}

// --- RAG Semantic Search ---

async fn execute_rag_search(rag: &RagEngine, args: &Value) -> Result<String> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'query'"))?;
    let limit = args["limit"]
        .as_u64()
        .map(|v| (v as usize).min(50))
        .unwrap_or(10);
    let source_type = args["source_type"].as_str();

    if query.trim().is_empty() {
        return Ok(json!({"error": "Query cannot be empty"}).to_string());
    }

    let results = rag.search(query, limit, source_type)?;

    let items: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "score": format!("{:.4}", r.score),
                "source_type": r.metadata.source_type,
                "source_id": r.metadata.source_id,
                "chat_id": r.metadata.chat_id,
                "segment": r.metadata.segment,
                "content_preview": r.metadata.content_preview,
                "user_name": r.metadata.user_name,
                "created_at": r.metadata.created_at,
            })
        })
        .collect();

    let count = items.len();
    Ok(json!({
        "results": items,
        "count": count,
        "query": query,
    })
    .to_string())
}

// --- MCP Interaction Tools ---

async fn execute_mcp_list_tools(
    pool: &SqlitePool,
    mcp: &McpManager,
    args: &Value,
) -> Result<String> {
    let server_name = args["server_name"].as_str();
    let refresh = args["refresh"].as_bool().unwrap_or(false);

    // Determine which servers to query
    let servers: Vec<String> = if let Some(name) = server_name {
        vec![name.to_string()]
    } else {
        // All enabled servers with tcp/http transports
        let all = db::mcp_server_list(pool, false).await?;
        all.into_iter()
            .filter(|s| matches!(s.transport.as_str(), "tcp" | "http" | "sse" | "streamable_http"))
            .map(|s| s.name)
            .collect()
    };

    if servers.is_empty() {
        return Ok(json!({
            "servers": [],
            "message": "No enabled MCP servers with tcp/http transport found. Register one first with crud_mcp_server."
        }).to_string());
    }

    let mut results = Vec::new();
    for name in &servers {
        match mcp.ensure_connected(pool, name).await {
            Ok(()) => {
                let tools = if refresh {
                    match mcp.discover_tools(name).await {
                        Ok(t) => t,
                        Err(e) => {
                            results.push(json!({
                                "server": name,
                                "error": format!("Failed to refresh tools: {}", e),
                            }));
                            continue;
                        }
                    }
                } else {
                    mcp.cached_tools(name).await.unwrap_or_default()
                };

                let tool_list: Vec<_> = tools.iter().map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })).collect();

                results.push(json!({
                    "server": name,
                    "tools": tool_list,
                    "count": tool_list.len(),
                }));
            }
            Err(e) => {
                results.push(json!({
                    "server": name,
                    "error": format!("Connection failed: {}", e),
                }));
            }
        }
    }

    Ok(json!({"servers": results}).to_string())
}

async fn execute_mcp_call(
    pool: &SqlitePool,
    mcp: &McpManager,
    args: &Value,
) -> Result<String> {
    let server_name = match args["server_name"].as_str() {
        Some(n) if !n.is_empty() => n,
        _ => return Ok(json!({"error": "Missing 'server_name'"}).to_string()),
    };
    let method = match args["method"].as_str() {
        Some(m) if !m.is_empty() => m,
        _ => return Ok(json!({"error": "Missing 'method'"}).to_string()),
    };
    // Accept arguments as object or as a JSON string
    let arguments = match args.get("arguments") {
        Some(v) if v.is_object() => v.clone(),
        Some(v) if v.is_string() => {
            serde_json::from_str(v.as_str().unwrap_or("{}")).unwrap_or(json!({}))
        }
        _ => json!({}),
    };

    // Pre-validate: if arguments is empty but the method requires fields, error
    // immediately with the schema — saves a round-trip and gives clear guidance
    if arguments == json!({}) {
        if let Some(tools) = mcp.cached_tools(server_name).await {
            if let Some(tool_info) = tools.iter().find(|t| t.name == method) {
                if let Some(required) = tool_info.input_schema.get("required") {
                    if let Some(arr) = required.as_array() {
                        if !arr.is_empty() {
                            let required_names: Vec<&str> = arr.iter()
                                .filter_map(|v| v.as_str())
                                .collect();
                            return Ok(json!({
                                "error": format!(
                                    "Method '{}' requires arguments but none were provided. \
                                     Required fields: [{}]. \
                                     You MUST pass them in the 'arguments' parameter. \
                                     inputSchema: {}",
                                    method,
                                    required_names.join(", "),
                                    tool_info.input_schema,
                                )
                            }).to_string());
                        }
                    }
                }
            }
        }
    }

    match mcp.call_tool(pool, server_name, method, arguments).await {
        Ok(result) => {
            if result.get("error").is_some() {
                // MCP server returned a JSON-RPC error — format as a top-level
                // "error" string so the LLM retry mechanism in llm.rs detects it
                // and limits to 2 identical attempts before bailing.
                let server_err = result["error"].as_str().unwrap_or("unknown error");
                let code = result["code"].as_i64().unwrap_or(0);

                // Include the tool's inputSchema so the LLM can self-correct
                let schema_hint = if let Some(tools) = mcp.cached_tools(server_name).await {
                    tools.iter()
                        .find(|t| t.name == method)
                        .map(|t| format!(" inputSchema: {}", t.input_schema))
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                Ok(json!({
                    "error": format!(
                        "MCP method '{}' on '{}' failed (code {}): {}. \
                         Pass the required fields in the 'arguments' parameter.{}",
                        method, server_name, code, server_err, schema_hint
                    )
                }).to_string())
            } else {
                Ok(json!({"success": true, "result": result}).to_string())
            }
        }
        Err(e) => Ok(json!({"error": e.to_string()}).to_string()),
    }
}

/// Execute a dynamically-registered MCP tool (name format: mcp__{server}__{method}).
/// Arguments are passed directly to the MCP server — no wrapping needed.
async fn execute_mcp_dynamic(
    pool: &SqlitePool,
    mcp: &McpManager,
    tool_name: &str,
    args: &Value,
) -> Result<String> {
    let rest = tool_name.strip_prefix("mcp__").unwrap_or(tool_name);
    let (server_name, method) = match rest.split_once("__") {
        Some((s, m)) if !s.is_empty() && !m.is_empty() => (s, m),
        _ => return Ok(json!({"error": format!("Invalid MCP tool name format: {}", tool_name)}).to_string()),
    };

    match mcp.call_tool(pool, server_name, method, args.clone()).await {
        Ok(result) => {
            if result.get("error").is_some() {
                let server_err = result["error"].as_str().unwrap_or("unknown error");
                let code = result["code"].as_i64().unwrap_or(0);
                Ok(json!({
                    "error": format!("MCP method '{}' on '{}' failed (code {}): {}", method, server_name, code, server_err)
                }).to_string())
            } else {
                Ok(json!({"success": true, "result": result}).to_string())
            }
        }
        Err(e) => Ok(json!({"error": e.to_string()}).to_string()),
    }
}
