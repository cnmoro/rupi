use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, watch};
use tokio::sync::mpsc;

use crate::provider::openai::{OpenAIConfig, OpenAIProvider};
use crate::provider::{ChatProvider, StreamEvent};
use crate::rpc::types::*;
use crate::error::AgentError;

/// A message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Agent session manages conversation state and model interaction.
pub struct AgentSession {
    provider: Arc<dyn ChatProvider>,
    model: String,
    messages: RwLock<Vec<Message>>,
    is_streaming: Mutex<bool>,
    abort_signal: Mutex<Option<watch::Sender<bool>>>,
    thinking_level: RwLock<String>,
    auto_compaction_enabled: RwLock<bool>,
    message_count: RwLock<u64>,
}

impl AgentSession {
    pub fn new(provider: Arc<dyn ChatProvider>, model: String) -> Self {
        AgentSession {
            provider,
            model,
            messages: RwLock::new(Vec::new()),
            is_streaming: Mutex::new(false),
            abort_signal: Mutex::new(None),
            thinking_level: RwLock::new("off".to_string()),
            auto_compaction_enabled: RwLock::new(false),
            message_count: RwLock::new(0),
        }
    }

    pub fn from_config(config: OpenAIConfig) -> Self {
        let model = config.model.clone();
        let provider = Arc::new(OpenAIProvider::new(config));
        Self::new(provider as Arc<dyn ChatProvider>, model)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn messages(&self) -> Vec<Message> {
        self.messages.read().await.clone()
    }

    pub async fn thinking_level(&self) -> String {
        self.thinking_level.read().await.clone()
    }

    pub async fn set_thinking_level(&self, level: String) {
        *self.thinking_level.write().await = level;
    }

    pub async fn cycle_thinking_level(&self) -> Option<String> {
        let mut level = self.thinking_level.write().await;
        *level = match level.as_str() {
            "off" => "low".to_string(),
            "low" => "medium".to_string(),
            "medium" => "high".to_string(),
            "high" => "off".to_string(),
            _ => "off".to_string(),
        };
        Some(level.clone())
    }

    pub async fn auto_compaction_enabled(&self) -> bool {
        *self.auto_compaction_enabled.read().await
    }

    pub async fn set_auto_compaction_enabled(&self, enabled: bool) {
        *self.auto_compaction_enabled.write().await = enabled;
    }

    pub async fn message_count(&self) -> u64 {
        *self.message_count.read().await
    }

    /// Reset the session (clear messages).
    pub async fn reset(&self) {
        self.messages.write().await.clear();
        *self.message_count.write().await = 0;
    }

    /// Stream a prompt to the model. Events are sent to the event_tx channel.
    pub async fn prompt(
        &self,
        message: &str,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        // Check if already streaming
        {
            let mut streaming = self.is_streaming.lock().await;
            if *streaming {
                return Err(AgentError::Config("Already streaming".into()));
            }
            *streaming = true;
        }

        // Create abort signal
        let (abort_tx, abort_rx) = watch::channel(false);
        {
            let mut signal = self.abort_signal.lock().await;
            *signal = Some(abort_tx);
        }

        // Add user message to history
        let user_msg = Message {
            role: "user".to_string(),
            content: message.to_string(),
        };
        self.messages.write().await.push(user_msg.clone());

        let _ = event_tx.send(AgentEvent::agent_start());
        let _ = event_tx.send(AgentEvent::turn_start());
        let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
            role: "user".to_string(),
            content: vec![MessageContent {
                content_type: "text".to_string(),
                text: Some(message.to_string()),
            }],
            model: None,
            usage: None,
            stop_reason: None,
        }));

        // Get current messages for the API call
        let messages_for_api = self.messages.read().await.clone();
        let stream_result: Result<mpsc::Receiver<StreamEvent>, AgentError> = self
            .provider
            .stream_chat(&self.model, &messages_for_api, abort_rx)
            .await;

        match stream_result {
            Ok(mut rx) => {
                let mut full_content = String::new();
                let mut input_tokens = 0;
                let mut output_tokens = 0;

                while let Some(event) = rx.recv().await {
                    match event {
                        StreamEvent::Delta(delta) => {
                            full_content.push_str(&delta);
                            let _ = event_tx.send(AgentEvent::message_update(delta));
                        }
                        StreamEvent::Done(result) => {
                            full_content = result.content;
                            input_tokens = result.input_tokens;
                            output_tokens = result.output_tokens;
                        }
                        StreamEvent::Error(err) => {
                            let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                                role: "assistant".to_string(),
                                content: if full_content.is_empty() {
                                    vec![]
                                } else {
                                    vec![MessageContent {
                                        content_type: "text".to_string(),
                                        text: Some(full_content.clone()),
                                    }]
                                },
                                model: Some(self.model.clone()),
                                usage: Some(Usage {
                                    input: input_tokens,
                                    output: output_tokens,
                                    total_tokens: input_tokens + output_tokens,
                                }),
                                stop_reason: Some("error".to_string()),
                            }));
                            let _ = event_tx.send(AgentEvent::turn_end());
                            let _ = event_tx.send(AgentEvent::agent_end());

                            {
                                let mut streaming = self.is_streaming.lock().await;
                                *streaming = false;
                            }
                            return if err == "cancelled" {
                                Err(AgentError::Cancelled)
                            } else {
                                Err(AgentError::Api {
                                    message: err,
                                    status_code: 0,
                                })
                            };
                        }
                    }
                }

                // Add assistant message to history
                let assistant_msg = Message {
                    role: "assistant".to_string(),
                    content: full_content.clone(),
                };
                self.messages.write().await.push(assistant_msg);

                let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                    role: "assistant".to_string(),
                    content: vec![MessageContent {
                        content_type: "text".to_string(),
                        text: Some(full_content.clone()),
                    }],
                    model: Some(self.model.clone()),
                    usage: Some(Usage {
                        input: input_tokens,
                        output: output_tokens,
                        total_tokens: input_tokens + output_tokens,
                    }),
                    stop_reason: Some("stop".to_string()),
                }));
                let _ = event_tx.send(AgentEvent::turn_end());
                let _ = event_tx.send(AgentEvent::agent_end());

                *self.message_count.write().await += 2;
            }
            Err(e) => {
                let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                    role: "assistant".to_string(),
                    content: vec![],
                    model: None,
                    usage: None,
                    stop_reason: Some("error".to_string()),
                }));
                let _ = event_tx.send(AgentEvent::turn_end());
                let _ = event_tx.send(AgentEvent::agent_end());
                {
                    let mut streaming = self.is_streaming.lock().await;
                    *streaming = false;
                }
                return Err(e);
            }
        }

        {
            let mut streaming = self.is_streaming.lock().await;
            *streaming = false;
        }

        Ok(())
    }

    /// Abort the current streaming operation.
    pub async fn abort(&self) {
        let mut signal = self.abort_signal.lock().await;
        if let Some(tx) = signal.take() {
            let _ = tx.send(true);
        }
        let mut streaming = self.is_streaming.lock().await;
        *streaming = false;
    }

    pub fn provider_model_info(&self) -> ModelInfo {
        self.provider.model_info()
    }

    pub async fn get_state(&self) -> SessionState {
        SessionState {
            model: Some(self.provider_model_info()),
            thinking_level: self.thinking_level.read().await.clone(),
            is_streaming: *self.is_streaming.lock().await,
            is_compacting: false,
            steering_mode: "all".to_string(),
            follow_up_mode: "all".to_string(),
            auto_compaction_enabled: *self.auto_compaction_enabled.read().await,
            message_count: self.messages.read().await.len(),
            pending_message_count: 0,
        }
    }

    pub async fn get_messages_as_rpc(&self) -> Vec<AgentMessage> {
        self.messages
            .read()
            .await
            .iter()
            .map(|m| AgentMessage {
                role: m.role.clone(),
                content: vec![MessageContent {
                    content_type: "text".to_string(),
                    text: Some(m.content.clone()),
                }],
                model: None,
                usage: None,
                stop_reason: None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAIConfig;

    fn create_test_session() -> AgentSession {
        let config = OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
            reasoning: false,
        };
        AgentSession::from_config(config)
    }

    #[tokio::test]
    async fn test_session_creation() {
        let session = create_test_session();
        assert_eq!(session.model(), "gpt-4");
        assert_eq!(session.thinking_level().await, "off");
        assert_eq!(session.message_count().await, 0);
        assert!(!session.auto_compaction_enabled().await);
    }

    #[tokio::test]
    async fn test_cycle_thinking_level() {
        let session = create_test_session();
        assert_eq!(session.thinking_level().await, "off");
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("low"));
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("medium"));
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("high"));
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("off"));
    }

    #[tokio::test]
    async fn test_set_thinking_level() {
        let session = create_test_session();
        session.set_thinking_level("high".into()).await;
        assert_eq!(session.thinking_level().await, "high");
    }

    #[tokio::test]
    async fn test_auto_compaction() {
        let session = create_test_session();
        assert!(!session.auto_compaction_enabled().await);
        session.set_auto_compaction_enabled(true).await;
        assert!(session.auto_compaction_enabled().await);
    }

    #[tokio::test]
    async fn test_get_state() {
        let session = create_test_session();
        let state = session.get_state().await;
        assert!(state.model.is_some());
        assert_eq!(state.thinking_level, "off");
        assert_eq!(state.message_count, 0);
        assert!(!state.is_streaming);
    }

    #[tokio::test]
    async fn test_abort_when_not_streaming() {
        let session = create_test_session();
        session.abort().await;
    }

    #[tokio::test]
    async fn test_provider_model_info() {
        let session = create_test_session();
        let info = session.provider_model_info();
        assert_eq!(info.provider, "openai-compatible");
        assert_eq!(info.id, "gpt-4");
    }

    #[tokio::test]
    async fn test_reset() {
        let session = create_test_session();
        // Add a message directly
        session
            .messages
            .write()
            .await
            .push(Message {
                role: "user".into(),
                content: "hello".into(),
            });
        *session.message_count.write().await = 1;

        assert_eq!(session.messages().await.len(), 1);
        session.reset().await;
        assert_eq!(session.messages().await.len(), 0);
        assert_eq!(session.message_count().await, 0);
    }

    #[tokio::test]
    async fn test_messages_rpc_conversion() {
        let session = create_test_session();
        session
            .messages
            .write()
            .await
            .push(Message {
                role: "user".into(),
                content: "hello".into(),
            });
        let msgs = session.get_messages_as_rpc().await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content[0].text.as_deref(), Some("hello"));
    }
}
