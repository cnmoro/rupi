use crate::agent::session::Message;
use async_trait::async_trait;
use tokio::sync::mpsc;

pub mod openai;

/// Result of streaming a prompt to a model.
#[derive(Debug, Clone)]
pub struct StreamResult {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Events emitted during streaming.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Delta(String),
    Done(StreamResult),
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

    /// Get the model info.
    fn model_info(&self) -> crate::rpc::types::ModelInfo;
}
