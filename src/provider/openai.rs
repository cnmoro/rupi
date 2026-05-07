use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use super::{ChatProvider, StreamEvent, StreamResult};
use crate::agent::session::Message;
use crate::rpc::types::ModelInfo;
use crate::error::AgentError;

/// Configuration for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub context_window: u64,
    pub reasoning: bool,
}

/// OpenAI chat completion request body.
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
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
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

/// OpenAI-compatible API provider.
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: Client,
}

impl OpenAIProvider {
    pub fn new(config: OpenAIConfig) -> Self {
        OpenAIProvider {
            config,
            client: Client::new(),
        }
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
        let messages: Vec<ChatMessage> = messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let body = ChatRequest {
                model: model.clone(),
                messages,
                stream: true,
                max_tokens: None,
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

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let body_text = response.text().await.unwrap_or_default();
                let _ = tx
                    .send(StreamEvent::Error(format!("API error ({}): {}", status, body_text)))
                    .await;
                return;
            }

            let mut full_content = String::new();
            let mut input_tokens = 0;
            let mut output_tokens = 0;
            let mut stream = response.bytes_stream();

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
                                let text = String::from_utf8_lossy(&bytes);
                                for line in text.lines() {
                                    let line = line.trim();
                                    if line.is_empty() {
                                        continue;
                                    }
                                    if line == "data: [DONE]" {
                                        break;
                                    }
                                    if let Some(data) = line.strip_prefix("data: ") {
                                        match serde_json::from_str::<ChatChunk>(data) {
                                            Ok(chunk) => {
                                                if let Some(usage) = chunk.usage {
                                                    input_tokens = usage.prompt_tokens;
                                                    output_tokens = usage.completion_tokens;
                                                }
                                                for choice in chunk.choices {
                                                    if let Some(content) = choice.delta.content {
                                                        full_content.push_str(&content);
                                                        let _ = tx.send(StreamEvent::Delta(content)).await;
                                                    }
                                                    if choice.finish_reason.is_some() {
                                                        // Stream done
                                                    }
                                                }
                                            }
                                            Err(_) => {
                                                // Skip unparseable chunks
                                            }
                                        }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                let _ = tx.send(StreamEvent::Error(format!("Stream error: {}", e))).await;
                                return;
                            }
                            None => {
                                // Stream ended
                                break;
                            }
                        }
                    }
                }
            }

            let _ = tx
                .send(StreamEvent::Done(StreamResult {
                    content: full_content,
                    input_tokens,
                    output_tokens,
                }))
                .await;
        });

        Ok(rx)
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


    #[tokio::test]
    async fn test_openai_provider_model_info() {
        let config = OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
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
            reasoning: false,
        };
        let provider = OpenAIProvider::new(config);
        let (_tx, rx_signal) = watch::channel(false);
        let result = provider
            .stream_chat("gpt-4", &[], rx_signal)
            .await;
        assert!(result.is_ok(), "stream_chat should not fail for empty messages");
    }
}
