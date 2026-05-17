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

    /// Get the model info.
    fn model_info(&self) -> crate::rpc::types::ModelInfo;
}
