use crate::agent::session::Message;
use async_trait::async_trait;
use tokio::sync::mpsc;

pub mod defs;
pub mod openai;

/// Result of streaming a prompt to a model.
#[derive(Debug, Clone, Default)]
pub struct StreamResult {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost: Option<PromptCost>,
    pub reasoning_content: String,
}

/// Cost information from the provider (e.g. OpenRouter).
#[derive(Debug, Clone, Default)]
pub struct PromptCost {
    pub prompt_cost: f64,
    pub completion_cost: f64,
    pub total_cost: f64,
}

use crate::tools::ToolCall;

/// Events emitted during streaming.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Delta(String),
    Reasoning(String),
    GenerationId(String),
    Done(StreamResult),
    ToolCalls {
        calls: Vec<ToolCall>,
        content: String,
        input_tokens: u64,
        output_tokens: u64,
        cost: Option<PromptCost>,
        finish_reason: Option<String>,
        reasoning_content: String,
    },
    Error(String),
}

/// A provider that can stream chat completions from an OpenAI-compatible API.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Stream a chat completion. Returns a receiver that yields StreamEvents.
    async fn stream_chat(
        &self,
        model: &str,
        messages: &[Message],
        signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, crate::error::AgentError>;

    /// Non-streaming completion. Returns the full response text.
    async fn complete(
        &self,
        model: &str,
        messages: &[Message],
    ) -> Result<String, crate::error::AgentError>;

    /// Non-streaming completion that keeps the request prefix aligned with
    /// `stream_chat`.
    ///
    /// Compaction replays the conversation's own system prompt and messages to the
    /// summarizer so the call is a genuine prefix of the last routed request and the
    /// provider's KV cache is reused. That only holds if the tool schemas are
    /// present too, because they sit between the system prompt and the messages in
    /// the serialized request. `complete` omits them and breaks the match, so this
    /// method exists to send them.
    ///
    /// The default implementation falls back to `complete`, which is correct but
    /// gives up the cache alignment.
    async fn complete_aligned(
        &self,
        model: &str,
        messages: &[Message],
    ) -> Result<String, crate::error::AgentError> {
        self.complete(model, messages).await
    }

    /// Get the model info.
    fn model_info(&self) -> crate::rpc::types::ModelInfo;
}
