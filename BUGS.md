# BUGS

| ID | Title | Status | Summary | References |
|---|---|---|---|---|
| BUG-001 | MCP cache can bypass DB state changes | NEED_RESEARCH | Cached initialized MCP connections may continue to be used after server update/delete/disable operations. | `src/mcp.rs:100`, `src/mcp.rs:112`, `src/mcp.rs:169`, `src/tools.rs:1363`, `src/tools.rs:1399`, `src/tools.rs:2613` |
| BUG-002 | `run_python` sandbox directory is shared per process | NEED_RESEARCH | Sandbox path is keyed by PID, so concurrent requests can overwrite each other or remove active workspace data. | `src/tools.rs:1857`, `src/tools.rs:1991` |
| BUG-003 | Current message appears twice in LLM context | CONFIRMED_FIXED | Incoming user message was persisted before history load and then added again as the explicit current message; prompt build now skips the current persisted row before appending the live turn. | `src/bot.rs:140`, `src/bot.rs:210`, `src/bot.rs:558`, `src/bot.rs:589` |
| BUG-004 | `/reset` does not clear semantic memory | NEED_RESEARCH | Reset clears SQL conversation history but does not remove previously indexed RAG vectors for conversation content. | `src/bot.rs:175`, `src/bot.rs:146`, `src/bot.rs:216`, `src/bot.rs:239`, `src/tools.rs:2463` |
| BUG-005 | Dynamic MCP discovery can add recurring latency | CONFIRMED_FIXED | Dynamic MCP discovery now uses cached tool lists first and applies a retry cooldown after failed auto-connect attempts, preventing per-turn repeated timeout paths. | `src/llm.rs:49`, `src/tools.rs:675`, `src/mcp.rs:18`, `src/mcp.rs:143` |
| BUG-006 | Negative pagination values are not clamped | NEED_RESEARCH | History tool inputs cap upper bounds but do not prevent negative `limit`/`offset` values before SQL execution. | `src/tools.rs:1423`, `src/tools.rs:1453`, `src/tools.rs:1486`, `src/db.rs:322`, `src/db.rs:390` |
| BUG-007 | Byte slicing risks UTF-8 boundary panics | CONFIRMED_FIXED | Secret masking and Python stdout/stderr truncation now use char-boundary-safe slicing, removing UTF-8 byte-index panic paths. | `src/main.rs:107`, `src/main.rs:120`, `src/main.rs:139`, `src/tools.rs:1957`, `src/tools.rs:1966` |
