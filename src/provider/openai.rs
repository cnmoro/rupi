use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use super::{ChatProvider, StreamEvent, StreamResult};
use crate::agent::session::Message;
use crate::error::AgentError;
use crate::rpc::types::ModelInfo;
use crate::tools::{self, ToolCall};

/// Configuration for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub context_window: u64,
    pub reasoning: bool,
    pub timeout_secs: u64,
}

/// OpenAI chat completion request body.
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallData>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    /// Hint to the server to cache the KV state for this message and all preceding ones.
    /// Supported by servers that implement Anthropic-style prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ToolCallData {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

/// OpenAI chat completion streaming chunk.
#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(rename = "finish_reason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChunkToolCallFunction>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    cost: Option<f64>,
}

/// Accumulated tool call during streaming.
#[derive(Debug, Default)]
struct AccumulatedToolCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

/// OpenAI-compatible API provider.
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: Client,
    cached_tools: Vec<serde_json::Value>,
}

/// Max seconds of silence mid-stream before a provider request is failed.
///
/// Distinct from `--timeout`, which caps a whole run: a long generation is
/// fine, a stream that goes quiet is not. Without this a provider that accepts
/// the request and then stops sending parks the agent forever.
static STREAM_IDLE_TIMEOUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(120);

pub fn set_stream_idle_timeout(seconds: u64) {
    STREAM_IDLE_TIMEOUT.store(seconds.max(1), std::sync::atomic::Ordering::Relaxed);
}

fn stream_idle_timeout() -> u64 {
    STREAM_IDLE_TIMEOUT.load(std::sync::atomic::Ordering::Relaxed)
}

impl OpenAIProvider {
    pub fn new(config: OpenAIConfig) -> Self {
        let mut builder = Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            // Cap the gap *between* bytes, not the whole request: a long
            // generation is fine, a stream that goes quiet is not. Without this
            // a provider that accepts the request and then stops sending parks
            // the agent forever — no event, no error, nothing for a caller to
            // react to.
            .read_timeout(std::time::Duration::from_secs(stream_idle_timeout()));
        if config.timeout_secs > 0 {
            builder = builder.timeout(std::time::Duration::from_secs(config.timeout_secs));
        }
        let client = builder.build().unwrap_or_else(|_| Client::new());
        let cached_tools = tools::serialize_tools(&tools::all_tools());
        OpenAIProvider {
            config,
            client,
            cached_tools,
        }
    }

    /// One non-streaming completion.
    ///
    /// `with_tools` decides whether the serialized tool schemas ride along. They
    /// change nothing about the answer — `tool_choice` is `"none"` — but they keep
    /// the request prefix byte-identical to the streaming calls, which is what lets
    /// the provider serve the whole conversation from its KV cache.
    async fn complete_inner(
        &self,
        model: &str,
        messages: &[Message],
        with_tools: bool,
    ) -> Result<String, AgentError> {
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        let api_key = self.config.api_key.clone();
        let client = self.client.clone();
        let model = model.to_string();
        let api_messages = OpenAIProvider::build_messages(messages);

        let mut body = serde_json::json!({
            "model": model,
            "messages": api_messages,
            "stream": false,
            "max_tokens": 4096,
        });
        if with_tools && !self.cached_tools.is_empty() {
            body["tools"] = serde_json::Value::Array(self.cached_tools.clone());
            body["tool_choice"] = serde_json::json!("none");
        }

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(AgentError::Http)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body_text = response.text().await.unwrap_or_default();
            return Err(AgentError::Api {
                message: body_text,
                status_code: status,
            });
        }

        let data: serde_json::Value = response.json().await.map_err(AgentError::Http)?;
        let msg = &data["choices"][0]["message"];
        if let Some(content) = msg["content"].as_str().filter(|s| !s.is_empty()) {
            Ok(content.to_string())
        } else {
            Err(AgentError::Api {
                message: "model returned tool call instead of text".to_string(),
                status_code: 0,
            })
        }
    }

    /// Marker that asks a server to cache the KV state up to and including a message.
    fn cache_breakpoint() -> serde_json::Value {
        serde_json::json!({"type": "ephemeral"})
    }

    fn build_messages(messages: &[Message]) -> Vec<ChatMessage> {
        let last_index = messages.len().saturating_sub(1);
        messages
            .iter()
            .enumerate()
            .map(|(index, m)| {
                // OpenAI expects content: null for assistant messages with only tool calls
                let has_tool_calls = m.tool_calls.as_ref().map_or(false, |c| !c.is_empty());
                let content = if has_tool_calls && m.content.is_empty() {
                    None
                } else {
                    Some(m.content.clone())
                };

                let mut chat_msg = ChatMessage {
                    role: m.role.clone(),
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                    cache_control: None,
                };

                // Two explicit KV cache breakpoints.
                //
                // The system message is the static one: the prompt is frozen at
                // session creation and the tool schemas are serialized once, so this
                // prefix never changes for the life of the session.
                //
                // The last message is the moving one, and it is the one that matters.
                // A server that needs explicit breakpoints caches only up to the
                // furthest one it is given, so marking the system message alone left
                // the entire conversation — which is nearly all of the tokens —
                // re-prefilled on every single turn. Marking the tail extends the
                // cached span to the whole request. Because the history is append
                // only, this turn's breakpoint is next turn's cache hit.
                //
                // Servers with automatic prefix caching (vLLM, SGLang, OpenAI,
                // DeepSeek) ignore the field and lose nothing by it.
                if m.role == "system" || index == last_index {
                    chat_msg.cache_control = Some(OpenAIProvider::cache_breakpoint());
                }

                // Include reasoning_content for assistant messages (required by reasoning models like DeepSeek)
                if m.role == "assistant" {
                    chat_msg.reasoning_content = m.reasoning_content.clone();
                }

                // Handle assistant messages with tool calls
                if let Some(ref calls) = m.tool_calls {
                    let tool_call_data: Vec<ToolCallData> = calls
                        .iter()
                        .map(|tc| ToolCallData {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: ToolCallFunction {
                                name: tc.name.clone(),
                                arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                            },
                        })
                        .collect();
                    chat_msg.tool_calls = Some(tool_call_data);
                }

                // Handle tool result messages
                if let Some(ref id) = m.tool_call_id {
                    chat_msg.tool_call_id = Some(id.clone());
                }

                chat_msg
            })
            .collect()
    }
}

#[async_trait]
impl ChatProvider for OpenAIProvider {
    async fn stream_chat(
        &self,
        model: &str,
        messages: &[Message],
        mut signal: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        let api_key = self.config.api_key.clone();
        let client = self.client.clone();
        let model = model.to_string();

        // Build messages with proper tool call/result formatting
        let api_messages = OpenAIProvider::build_messages(messages);

        // Use cached tool definitions (serialized once at provider creation)
        let serialized_tools = self.cached_tools.clone();

        let (tx, rx) = mpsc::channel(256);

        // Spawn the streaming task. Panics are caught by tokio and stored
        // in the JoinHandle. Since we don't await the handle, a panic would
        // close the tx channel silently. Wrap the body in catch_unwind to
        // send an error event on panic.
        let tx_catch = tx.clone();
        let handle: tokio::task::JoinHandle<()> = tokio::spawn(async move {
            let body = ChatRequest {
                model: model.clone(),
                messages: api_messages,
                stream: true,
                max_tokens: None,
                tools: Some(serialized_tools),
                tool_choice: Some(serde_json::json!("auto")),
            };

            let response_result = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            let response = match response_result {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("HTTP request error: {}", e))).await;
                    return;
                }
            };

            // Capture X-Generation-Id from response headers before consuming body
            let gen_id = response
                .headers()
                .get("X-Generation-Id")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let body_text = response.text().await.unwrap_or_default();
                let _ = tx
                    .send(StreamEvent::Error(format!("API error ({}): {}", status, body_text)))
                    .await;
                return;
            }

            // Emit generation ID as early as possible
            if let Some(ref id) = gen_id {
                let _ = tx.send(StreamEvent::GenerationId(id.clone())).await;
            }

            let mut full_content = String::new();
            let mut reasoning_content = String::new();
            let mut input_tokens = 0;
            let mut output_tokens = 0;
            let mut cost: Option<super::PromptCost> = None;
            let mut tool_calls: Vec<AccumulatedToolCall> = Vec::new();
            let mut finish_reason: Option<String> = None;
            let mut stream = response.bytes_stream();
            // SSE reassembly buffer: accumulates partial lines across chunk boundaries
            let mut sse_buf = String::new();

            loop {
                tokio::select! {
                    biased;
                    _cancelled = signal.changed() => {
                        if *signal.borrow() {
                            let _ = tx.send(StreamEvent::Error("cancelled".into())).await;
                            return;
                        }
                    }
                    chunk_result = stream.next() => {
                        match chunk_result {
                            Some(Ok(bytes)) => {
                                sse_buf.push_str(&String::from_utf8_lossy(&bytes));
                                // Process complete lines from the buffer
                                loop {
                                    let line_end = match sse_buf.find('\n') {
                                        Some(pos) => pos,
                                        None => break, // wait for more data
                                    };
                                    let line = sse_buf[..line_end].trim().to_string();
                                    sse_buf.drain(..=line_end);
                                    if line.is_empty() || line.starts_with(':') {
                                        continue;
                                    }
                                    if line == "data: [DONE]" || line == "data:[DONE]" {
                                        break;
                                    }
                                    // Handle both "data: " and "data:" prefixes
                                    let data = line.strip_prefix("data: ")
                                        .or_else(|| line.strip_prefix("data:"))
                                        .unwrap_or("");
                                    if !data.is_empty() {
                                        match serde_json::from_str::<ChatChunk>(data) {
                                            Ok(chunk) => {
                                                if let Some(usage) = chunk.usage {
                                                    input_tokens = usage.prompt_tokens;
                                                    output_tokens = usage.completion_tokens;
                                                    if let Some(c) = usage.cost {
                                                        cost = Some(super::PromptCost {
                                                            prompt_cost: c,
                                                            completion_cost: 0.0,
                                                            total_cost: c,
                                                        });
                                                    }
                                                }
                                                for choice in chunk.choices {
                                                    if let Some(reason) = choice.finish_reason {
                                                        finish_reason = Some(reason);
                                                    }
                                                    let delta = choice.delta;
                                                    if let Some(content) = delta.content {
                                                        full_content.push_str(&content);
                                                        let _ = tx.send(StreamEvent::Delta(content)).await;
                                                    }
                                                    if let Some(rc) = delta.reasoning_content {
                                                        reasoning_content.push_str(&rc);
                                                        let _ = tx.send(StreamEvent::Reasoning(rc)).await;
                                                    }
                                                    if let Some(chunk_tool_calls) = delta.tool_calls {
                                                        for tc in chunk_tool_calls {
                                                            let index = tc.index;
                                                            while tool_calls.len() <= index {
                                                                tool_calls.push(AccumulatedToolCall::default());
                                                            }
                                                            let acc = &mut tool_calls[index];
                                                            acc.index = index;
                                                            if let Some(id) = tc.id {
                                                                acc.id = id;
                                                            }
                                                            if let Some(func) = tc.function {
                                                                if let Some(name) = func.name {
                                                                    acc.name = name;
                                                                }
                                                                if let Some(args) = func.arguments {
                                                                    acc.arguments.push_str(&args);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            Err(_) => {}
                                        }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                let _ = tx.send(StreamEvent::Error(format!("Stream error: {}", e))).await;
                                return;
                            }
                            None => {
                                break;
                            }
                        }
                    }
                }
            }

            // Check if there were tool calls
            if !tool_calls.is_empty() {
                let calls: Vec<ToolCall> = tool_calls
                    .into_iter()
                    .map(|tc| {
                        let raw = tc.arguments.clone();
                        let parsed = serde_json::from_str(&raw).ok();
                        let truncated = parsed.is_none() && !raw.is_empty();
                        ToolCall {
                            id: tc.id,
                            name: tc.name,
                            arguments: parsed.unwrap_or_default(),
                            raw_arguments: if truncated { Some(raw) } else { None },
                        }
                    })
                    .collect();

                let _ = tx
                    .send(StreamEvent::ToolCalls {
                        calls,
                        content: full_content,
                        reasoning_content,
                        input_tokens,
                        output_tokens,
                        cost,
                        finish_reason,
                    })
                    .await;
                return;
            }

            let _ = tx
                .send(StreamEvent::Done(StreamResult {
                    content: full_content,
                    reasoning_content,
                    input_tokens,
                    output_tokens,
                    cost,
                }))
                .await;
        });

        // Catch panics in the HTTP task (e.g., TLS/crypto failures on older CPUs)
        let tx_err = tx_catch;
        tokio::spawn(async move {
            if let Err(e) = handle.await {
                if e.is_panic() {
                    let panic = e.into_panic();
                    let msg = panic.downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().map(|s| s.clone()))
                        .unwrap_or_else(|| "unknown panic in HTTP task".to_string());
                    let _ = tx_err.send(StreamEvent::Error(format!("Internal error: {}", msg))).await;
                }
            }
        });

        Ok(rx)
    }

    async fn complete(
        &self,
        model: &str,
        messages: &[Message],
    ) -> Result<String, AgentError> {
        self.complete_inner(model, messages, false).await
    }

    async fn complete_aligned(
        &self,
        model: &str,
        messages: &[Message],
    ) -> Result<String, AgentError> {
        match self.complete_inner(model, messages, true).await {
            Ok(text) => Ok(text),
            Err(e) => {
                // Not every OpenAI-compatible endpoint accepts a tools array or
                // `tool_choice: "none"` on a non-streaming call. Prefix alignment is
                // an optimization; the summary itself is not optional. Fall back to
                // the plain call so compaction still lands, and say why.
                eprintln!(
                    "rupi: cache-aligned summarization call failed ({}) — retrying without tools",
                    e
                );
                self.complete_inner(model, messages, false).await
            }
        }
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            provider: "openai-compatible".to_string(),
            id: self.config.model.clone(),
            context_window: self.config.context_window,
            reasoning: self.config.reasoning,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(messages: &[Message]) -> serde_json::Value {
        serde_json::to_value(OpenAIProvider::build_messages(messages)).unwrap()
    }

    #[test]
    fn cache_breakpoints_cover_the_static_head_and_the_moving_tail() {
        let messages = vec![
            Message::new("system", "SYSTEM"),
            Message::new("user", "do the thing"),
            Message::new("assistant", "on it"),
            Message::tool_result("c1", "output"),
        ];
        let wire = body_of(&messages);

        // The static breakpoint: the system prompt and the tool schemas never change
        // within a session.
        assert_eq!(wire[0]["cache_control"]["type"], "ephemeral");
        // The moving breakpoint: without it a server that needs explicit markers
        // caches only the system prompt and re-prefills the whole conversation.
        assert_eq!(wire[3]["cache_control"]["type"], "ephemeral");
        // Nothing in between, so the request stays within the usual four-breakpoint
        // budget however long the conversation gets.
        assert!(wire[1].get("cache_control").is_none());
        assert!(wire[2].get("cache_control").is_none());
    }

    #[test]
    fn the_moving_breakpoint_follows_the_tail() {
        let mut messages = vec![Message::new("system", "SYSTEM"), Message::new("user", "one")];
        assert_eq!(body_of(&messages)[1]["cache_control"]["type"], "ephemeral");

        messages.push(Message::new("assistant", "two"));
        let wire = body_of(&messages);
        assert!(wire[1].get("cache_control").is_none());
        assert_eq!(wire[2]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn a_system_only_request_carries_one_breakpoint() {
        let wire = body_of(&[Message::new("system", "SYSTEM")]);
        assert_eq!(wire.as_array().unwrap().len(), 1);
        assert_eq!(wire[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn an_empty_request_does_not_panic() {
        assert!(body_of(&[]).as_array().unwrap().is_empty());
    }

    use super::*;

    #[tokio::test]
    async fn test_openai_provider_model_info() {
        let config = OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
            timeout_secs: 0,
            reasoning: false,
        };
        let provider = OpenAIProvider::new(config);
        let info = provider.model_info();
        assert_eq!(info.provider, "openai-compatible");
        assert_eq!(info.id, "gpt-4");
        assert_eq!(info.context_window, 8192);
        assert!(!info.reasoning);
    }

    #[tokio::test]
    async fn test_openai_provider_empty_messages() {
        let config = OpenAIConfig {
            base_url: "http://127.0.0.1:1".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
            timeout_secs: 0,
            reasoning: false,
        };
        let provider = OpenAIProvider::new(config);
        let (_tx, rx_signal) = watch::channel(false);
        let result = provider.stream_chat("gpt-4", &[], rx_signal).await;
        assert!(result.is_ok(), "stream_chat should not fail for empty messages");
    }
}
