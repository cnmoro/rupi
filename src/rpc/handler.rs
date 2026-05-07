use std::sync::Arc;
use tokio::sync::mpsc;

use super::types::*;
use crate::agent::session::AgentSession;


/// Handles RPC commands and produces responses/events.
pub struct RpcHandler {
    session: Arc<tokio::sync::RwLock<AgentSession>>,
}

impl RpcHandler {
    pub fn new(session: AgentSession) -> Self {
        RpcHandler {
            session: Arc::new(tokio::sync::RwLock::new(session)),
        }
    }

    /// Handle a single RPC command, writing responses and events to output_tx.
    /// Returns immediately. Async commands spawn background tasks.
    pub async fn handle(
        &self,
        command: RpcCommand,
        output_tx: mpsc::UnboundedSender<String>,
    ) {
        let session = self.session.clone();
        let tx = output_tx.clone();

        match command {
            RpcCommand::Ping { id } => {
                let state = session.read().await.get_state().await;
                write_success(tx, id, "ping", Some(state)).await;
            }
            RpcCommand::GetState { id } => {
                let state = session.read().await.get_state().await;
                write_success(tx, id, "get_state", Some(state)).await;
            }
            RpcCommand::SetModel { id, provider, model_id } => {
                if provider != "openai-compatible" {
                    write_error(
                        tx,
                        id,
                        "set_model",
                        format!("Unknown provider: {}. Only 'openai-compatible' is supported.", provider),
                    )
                    .await;
                } else {
                    let data = serde_json::json!({
                        "provider": provider,
                        "id": model_id,
                        "context_window": 128000,
                        "reasoning": false,
                    });
                    write_success(tx, id, "set_model", Some(data)).await;
                }
            }
            RpcCommand::CycleModel { id } => {
                write_success::<()>(tx, id, "cycle_model", None).await;
            }
            RpcCommand::GetAvailableModels { id } => {
                let info = session.read().await.provider_model_info();
                let models = vec![serde_json::json!({
                    "provider": info.provider,
                    "id": info.id,
                    "context_window": info.context_window,
                    "reasoning": info.reasoning,
                })];
                write_success(tx, id, "get_available_models", Some(serde_json::json!({ "models": models }))).await;
            }
            RpcCommand::SetThinkingLevel { id, level } => {
                session.read().await.set_thinking_level(level).await;
                write_success::<()>(tx, id, "set_thinking_level", None).await;
            }
            RpcCommand::CycleThinkingLevel { id } => {
                let level = session.read().await.cycle_thinking_level().await;
                let data = level.map(|l| serde_json::json!({ "level": l }));
                write_success(tx, id, "cycle_thinking_level", data).await;
            }
            RpcCommand::SetAutoCompaction { id, enabled } => {
                session.read().await.set_auto_compaction_enabled(enabled).await;
                write_success::<()>(tx, id, "set_auto_compaction", None).await;
            }
            RpcCommand::GetMessages { id } => {
                let msgs = session.read().await.get_messages_as_rpc().await;
                write_success(tx, id, "get_messages", Some(serde_json::json!({ "messages": msgs }))).await;
            }
            RpcCommand::Compact { id, .. } => {
                let data = serde_json::json!({
                    "summary": "Compaction not implemented in minimal mode",
                    "tokensBefore": 0,
                    "tokensAfter": 0,
                });
                write_success(tx, id, "compact", Some(data)).await;
            }
            RpcCommand::Abort { id } => {
                session.read().await.abort().await;
                write_success::<()>(tx, id, "abort", None).await;
            }
            RpcCommand::NewSession { id, parent_session: _ } => {
                // Reset current session
                session.read().await.reset().await;
                write_success(tx, id, "new_session", Some(serde_json::json!({"cancelled": false}))).await;
            }
            RpcCommand::Prompt {
                id,
                message,
                images: _,
                streaming_behavior: _,
            } => {
                let session = session.clone();
                let tx = output_tx.clone();
                tokio::spawn(async move {
                    handle_async_prompt(session, tx, id, &message).await;
                });
            }
            RpcCommand::Steer { id, message, .. } => {
                let session = session.clone();
                let tx = output_tx.clone();
                tokio::spawn(async move {
                    handle_async_prompt(session, tx, id, &message).await;
                });
            }
            RpcCommand::FollowUp { id, message, .. } => {
                let session = session.clone();
                let tx = output_tx.clone();
                tokio::spawn(async move {
                    handle_async_prompt(session, tx, id, &message).await;
                });
            }
        }
    }
}

/// Handle an async prompt command, streaming events through the output channel.
async fn handle_async_prompt(
    session: Arc<tokio::sync::RwLock<AgentSession>>,
    tx: mpsc::UnboundedSender<String>,
    id: Option<String>,
    message: &str,
) {
    // Send immediate success response
    let resp = RpcResponse::success(id.clone(), "prompt", None);
    let _ = tx.send(resp.to_json_line());

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Forward events from event_rx to output_tx
    let tx_clone = tx.clone();
    let _event_forwarder = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            let _ = tx_clone.send(event.to_json_line());
        }
    });

    // Run the prompt
    let session_read = session.read().await;
    let _result = session_read.prompt(message, event_tx).await;
    drop(session_read);
}

async fn write_success<T: serde::Serialize>(
    tx: mpsc::UnboundedSender<String>,
    id: Option<String>,
    command: &str,
    data: Option<T>,
) {
    let data = data.map(|d| serde_json::to_value(d).unwrap_or_default());
    let resp = RpcResponse::success(id, command, data);
    let _ = tx.send(resp.to_json_line());
}

async fn write_error(
    tx: mpsc::UnboundedSender<String>,
    id: Option<String>,
    command: &str,
    message: String,
) {
    let resp = RpcResponse::error(id, command, message);
    let _ = tx.send(resp.to_json_line());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::session::AgentSession;
    use crate::provider::openai::OpenAIConfig;
    use tokio::sync::mpsc;

    fn create_test_handler() -> RpcHandler {
        let config = OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
            reasoning: false,
        };
        let session = AgentSession::from_config(config);
        RpcHandler::new(session)
    }

    async fn handle_and_collect(
        handler: &RpcHandler,
        command: RpcCommand,
    ) -> Vec<RpcResponse> {
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        handler.handle(command, tx).await;
        let mut responses = Vec::new();
        loop {
            tokio::select! {
                Some(line) = rx.recv() => {
                    if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
                        responses.push(resp);
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    break;
                }
            }
        }
        responses
    }

    #[tokio::test]
    async fn test_ping() {
        let handler = create_test_handler();
        let cmd = RpcCommand::Ping { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        assert_eq!(responses.len(), 1);
        assert!(responses[0].success);
        assert_eq!(responses[0].command, "ping");
    }

    #[tokio::test]
    async fn test_get_state() {
        let handler = create_test_handler();
        let cmd = RpcCommand::GetState { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        assert_eq!(responses.len(), 1);
        assert!(responses[0].success);
        assert!(responses[0].data.is_some());
    }

    #[tokio::test]
    async fn test_get_state_has_model() {
        let handler = create_test_handler();
        let cmd = RpcCommand::GetState { id: None };
        let responses = handle_and_collect(&handler, cmd).await;
        let data = responses[0].data.as_ref().unwrap();
        assert!(data.get("model").is_some());
    }

    #[tokio::test]
    async fn test_set_model_unknown_provider() {
        let handler = create_test_handler();
        let cmd = RpcCommand::SetModel {
            id: Some("req_1".into()),
            provider: "anthropic".into(),
            model_id: "claude-3".into(),
        };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(!responses[0].success);
        assert!(responses[0].error.as_deref().unwrap().contains("Unknown provider"));
    }

    #[tokio::test]
    async fn test_set_model_success() {
        let handler = create_test_handler();
        let cmd = RpcCommand::SetModel {
            id: Some("req_1".into()),
            provider: "openai-compatible".into(),
            model_id: "gpt-4".into(),
        };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);
        assert_eq!(responses[0].command, "set_model");
    }

    #[tokio::test]
    async fn test_get_available_models() {
        let handler = create_test_handler();
        let cmd = RpcCommand::GetAvailableModels { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        let data = responses[0].data.as_ref().unwrap();
        let models = data["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["provider"], "openai-compatible");
    }

    #[tokio::test]
    async fn test_cycle_model() {
        let handler = create_test_handler();
        let cmd = RpcCommand::CycleModel { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);
        // null data means only one model available, nothing to cycle to
    }

    #[tokio::test]
    async fn test_set_thinking_level() {
        let handler = create_test_handler();
        let cmd = RpcCommand::SetThinkingLevel {
            id: Some("req_1".into()),
            level: "high".into(),
        };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);

        let state_cmd = RpcCommand::GetState { id: None };
        let state_responses = handle_and_collect(&handler, state_cmd).await;
        let data = state_responses[0].data.as_ref().unwrap();
        assert_eq!(data["thinking_level"], "high");
    }

    #[tokio::test]
    async fn test_cycle_thinking_level() {
        let handler = create_test_handler();
        let cmd = RpcCommand::CycleThinkingLevel { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        let data = responses[0].data.as_ref().unwrap();
        assert_eq!(data["level"], "low");
    }

    #[tokio::test]
    async fn test_abort() {
        let handler = create_test_handler();
        let cmd = RpcCommand::Abort { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);
        assert_eq!(responses[0].command, "abort");
    }

    #[tokio::test]
    async fn test_new_session() {
        let handler = create_test_handler();
        let cmd = RpcCommand::NewSession {
            id: Some("req_1".into()),
            parent_session: None,
        };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);
        assert_eq!(responses[0].command, "new_session");
    }

    #[tokio::test]
    async fn test_get_messages_empty() {
        let handler = create_test_handler();
        let cmd = RpcCommand::GetMessages { id: Some("req_1".into()) };
        let responses = handle_and_collect(&handler, cmd).await;
        let data = responses[0].data.as_ref().unwrap();
        let messages = data["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 0);
    }

    #[tokio::test]
    async fn test_compact() {
        let handler = create_test_handler();
        let cmd = RpcCommand::Compact {
            id: Some("req_1".into()),
            custom_instructions: None,
        };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);
    }

    #[tokio::test]
    async fn test_set_auto_compaction() {
        let handler = create_test_handler();
        let cmd = RpcCommand::SetAutoCompaction {
            id: Some("req_1".into()),
            enabled: true,
        };
        let responses = handle_and_collect(&handler, cmd).await;
        assert!(responses[0].success);

        let state_cmd = RpcCommand::GetState { id: None };
        let state_responses = handle_and_collect(&handler, state_cmd).await;
        let data = state_responses[0].data.as_ref().unwrap();
        assert!(data["auto_compaction_enabled"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_prompt_async_response() {
        let handler = create_test_handler();
        let cmd = RpcCommand::Prompt {
            id: Some("req_1".into()),
            message: "Hello".into(),
            images: None,
            streaming_behavior: None,
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        handler.handle(cmd, tx).await;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let mut found_response = false;
        while let Ok(line) = rx.try_recv() {
            if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
                if resp.command == "prompt" {
                    found_response = true;
                    assert!(resp.success);
                    break;
                }
            }
        }
        assert!(found_response, "Should have received a prompt response");
    }

    #[tokio::test]
    async fn test_steer_and_follow_up() {
        let handler = create_test_handler();
        let (tx, _rx) = mpsc::unbounded_channel::<String>();

        let steer_cmd = RpcCommand::Steer {
            id: Some("req_1".into()),
            message: "Steer me".into(),
            images: None,
        };
        handler.handle(steer_cmd, tx.clone()).await;

        let follow_cmd = RpcCommand::FollowUp {
            id: Some("req_2".into()),
            message: "Follow up".into(),
            images: None,
        };
        handler.handle(follow_cmd, tx).await;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
