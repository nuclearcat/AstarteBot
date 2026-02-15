use anyhow::Result;
use serde_json::Value;
use sqlx::SqlitePool;
use std::time::{Duration, Instant};
use teloxide::Bot;

use crate::mcp::McpManager;
use crate::rag::RagEngine;
use crate::tools;
use crate::types::*;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const MAX_TOOL_ROUNDS: usize = 30;
const MAX_RETRIES: u32 = 3;

pub struct LlmClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl LlmClient {
    pub fn new(api_key: String, model: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http,
            api_key,
            model,
        }
    }

    /// Send a chat completion request with tool-call loop
    pub async fn chat(
        &self,
        pool: &SqlitePool,
        bot: &Bot,
        rag: &RagEngine,
        mcp: &McpManager,
        messages: Vec<ChatMessage>,
        chat_id: i64,
        user_id: i64,
    ) -> Result<String> {
        let mut tool_defs = tools::definitions();
        // Auto-discover MCP tools and register them as first-class LLM tools
        let mcp_defs = tools::mcp_dynamic_definitions(mcp, pool).await;
        if !mcp_defs.is_empty() {
            tracing::info!(count = mcp_defs.len(), "Registered dynamic MCP tools");
            tool_defs.extend(mcp_defs);
        }

        let mut current_messages = messages;
        let mut last_tool_error_signature: Option<String> = None;
        let mut repeated_tool_error_count = 0u8;

        for round in 0..MAX_TOOL_ROUNDS {
            let request = ChatRequest {
                model: self.model.clone(),
                messages: current_messages.clone(),
                tools: Some(tool_defs.clone()),
                max_tokens: Some(4096),
            };

            let response = self.send_with_retry(&request).await?;

            // Check for API errors
            if let Some(err) = &response.error {
                tracing::error!(error = %err.message, "OpenRouter API error");
                return Err(anyhow::anyhow!("OpenRouter error: {}", err.message));
            }

            let choices = response
                .choices
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No choices in OpenRouter response"))?;

            if choices.is_empty() {
                return Err(anyhow::anyhow!("Empty choices in OpenRouter response"));
            }

            let choice = &choices[0];
            let assistant_msg = &choice.message;

            // Log token usage
            if let Some(usage) = &response.usage {
                tracing::info!(
                    model = %self.model,
                    prompt_tokens = ?usage.prompt_tokens,
                    completion_tokens = ?usage.completion_tokens,
                    total_tokens = ?usage.total_tokens,
                    round,
                    "LLM token usage"
                );
            }

            // Check if there are tool calls
            if let Some(tool_calls) = &assistant_msg.tool_calls {
                if !tool_calls.is_empty() {
                    // Add assistant message with tool calls to context
                    current_messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: assistant_msg.content.clone(),
                        tool_calls: Some(tool_calls.clone()),
                        tool_call_id: None,
                        name: None,
                    });

                    // Execute each tool call
                    for tc in tool_calls {
                        let result = tools::execute(
                            pool,
                            bot,
                            rag,
                            mcp,
                            &tc.function.name,
                            &tc.function.arguments,
                            chat_id,
                            user_id,
                        )
                        .await?;

                        // Add tool result message
                        let args_signature = tc.function.arguments.trim().to_string();
                        let error = parse_tool_error_message(&result);

                        current_messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(MessageContent::Text(result)),
                            tool_calls: None,
                            tool_call_id: Some(tc.id.clone()),
                            name: Some(tc.function.name.clone()),
                        });

                        if let Some(error_msg) = error {
                            let signature = format!("{}|{}", tc.function.name, args_signature);
                            if Some(&signature) == last_tool_error_signature.as_ref() {
                                repeated_tool_error_count =
                                    repeated_tool_error_count.saturating_add(1);
                            } else {
                                repeated_tool_error_count = 1;
                                last_tool_error_signature = Some(signature);
                            }

                            current_messages.push(ChatMessage {
                                role: "system".to_string(),
                                content: Some(MessageContent::Text(format!(
                                    "Tool '{}' returned an error: {}. If you already know the missing values from the conversation, retry the tool call immediately with the corrected arguments. Only ask the user if you truly do not know what values to use.",
                                    tc.function.name, error_msg
                                ))),
                                tool_calls: None,
                                tool_call_id: None,
                                name: None,
                            });

                            if repeated_tool_error_count >= 2 {
                                return Ok(format!(
                                    "I couldn't apply that MCP change yet: {}. Please provide the missing field(s) and try again.",
                                    error_msg
                                ));
                            }
                            continue;
                        }

                        // Reset repetition tracker on success
                        repeated_tool_error_count = 0;
                        last_tool_error_signature = None;
                    }

                    continue; // Next round
                }
            }

            // No tool calls — return the text response
            let text = assistant_msg
                .content
                .as_ref()
                .and_then(|c| c.as_text())
                .unwrap_or("")
                .to_string();

            return Ok(text);
        }

        Err(anyhow::anyhow!(
            "Exceeded maximum tool call rounds ({})",
            MAX_TOOL_ROUNDS
        ))
    }

    async fn send_with_retry(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut last_error = None;

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt));
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis(),
                    "Retrying OpenRouter request"
                );
                tokio::time::sleep(delay).await;
            }

            let start = Instant::now();

            match self
                .http
                .post(OPENROUTER_URL)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(request)
                .send()
                .await
            {
                Ok(resp) => {
                    let latency = start.elapsed();
                    let status = resp.status();

                    if status.is_success() {
                        let body = resp.text().await?;
                        tracing::debug!(
                            latency_ms = latency.as_millis(),
                            "OpenRouter response received"
                        );

                        match serde_json::from_str::<ChatResponse>(&body) {
                            Ok(parsed) => return Ok(parsed),
                            Err(e) => {
                                tracing::error!(error = %e, body = %body, "Failed to parse OpenRouter response");
                                last_error = Some(anyhow::anyhow!("Parse error: {}", e));
                            }
                        }
                    } else if status.as_u16() == 429 || status.is_server_error() {
                        let body = resp.text().await.unwrap_or_default();
                        tracing::warn!(status = status.as_u16(), body = %body, "OpenRouter returned retryable error");
                        last_error = Some(anyhow::anyhow!("HTTP {}: {}", status, body));
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(anyhow::anyhow!("OpenRouter HTTP {}: {}", status, body));
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, attempt, "OpenRouter request failed");
                    last_error = Some(e.into());
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retries exhausted")))
    }
}

fn parse_tool_error_message(result: &str) -> Option<String> {
    let json = serde_json::from_str::<Value>(result).ok()?;
    json.get("error")?.as_str().map(|s| s.to_string())
}
