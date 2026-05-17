use serde::{Deserialize, Serialize};

/// Thinking level for model reasoning.
pub type ThinkingLevel = String;

/// Image content reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub data: Option<String>,
    pub url: Option<String>,
    pub media_type: Option<String>,
}

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub role: String,
    pub content: Vec<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<PromptCostData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCostData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cost: Option<f64>,
}

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub provider: String,
    pub id: String,
    pub context_window: u64,
    pub reasoning: bool,
}

/// RPC commands sent to stdin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RpcCommand {
    Ping {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    Prompt {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<ImageContent>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        streaming_behavior: Option<String>,
    },
    Steer {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<ImageContent>>,
    },
    FollowUp {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<ImageContent>>,
    },
    Abort {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    NewSession {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_session: Option<String>,
    },
    GetState {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    SetModel {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        provider: String,
        model_id: String,
    },
    CycleModel {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    GetAvailableModels {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    SetThinkingLevel {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        level: ThinkingLevel,
    },
    CycleThinkingLevel {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    Compact {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
    SetAutoCompaction {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        enabled: bool,
    },
    GetMessages {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    ListSessions {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
}

/// RPC responses emitted to stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub response_type: String,
    pub command: String,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl RpcResponse {
    pub fn success(id: Option<String>, command: &str, data: Option<serde_json::Value>) -> Self {
        RpcResponse {
            id,
            response_type: "response".to_string(),
            command: command.to_string(),
            success: true,
            data,
            error: None,
        }
    }

    pub fn error(id: Option<String>, command: &str, message: String) -> Self {
        RpcResponse {
            id,
            response_type: "response".to_string(),
            command: command.to_string(),
            success: false,
            data: None,
            error: Some(message),
        }
    }

    pub fn to_json_line(&self) -> String {
        serialize_json_line(self)
    }
}

/// Session state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelInfo>,
    pub thinking_level: ThinkingLevel,
    pub is_streaming: bool,
    pub is_compacting: bool,
    pub steering_mode: String,
    pub follow_up_mode: String,
    pub auto_compaction_enabled: bool,
    pub message_count: usize,
    pub pending_message_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
}

/// Agent event types streamed during prompt execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    GenerationId {
        id: String,
        timestamp: u64,
    },
    AgentStart {
        timestamp: u64,
    },
    TurnStart {
        timestamp: u64,
    },
    MessageStart {
        message: AgentMessage,
        timestamp: u64,
    },
    MessageUpdate {
        assistant_message_event: AssistantMessageEvent,
        timestamp: u64,
    },
    MessageEnd {
        message: AgentMessage,
        timestamp: u64,
    },
    TurnEnd {
        timestamp: u64,
    },
    AgentEnd {
        timestamp: u64,
    },
    ToolExecutionStart {
        tool_name: String,
        arguments: serde_json::Value,
        timestamp: u64,
    },
    ToolExecutionEnd {
        tool_name: String,
        result: String,
        timestamp: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    TextDelta { delta: String },
    ThinkingDelta { delta: String },
}

use super::jsonl::serialize_json_line;

impl AgentEvent {
    pub fn to_json_line(&self) -> String {
        serialize_json_line(self)
    }

    pub fn generation_id(id: String) -> Self {
        AgentEvent::GenerationId {
            id,
            timestamp: now_millis(),
        }
    }

    pub fn agent_start() -> Self {
        AgentEvent::AgentStart {
            timestamp: now_millis(),
        }
    }

    pub fn turn_start() -> Self {
        AgentEvent::TurnStart {
            timestamp: now_millis(),
        }
    }

    pub fn message_start(message: AgentMessage) -> Self {
        AgentEvent::MessageStart {
            message,
            timestamp: now_millis(),
        }
    }

    pub fn message_update(delta: String) -> Self {
        AgentEvent::MessageUpdate {
            assistant_message_event: AssistantMessageEvent::TextDelta { delta },
            timestamp: now_millis(),
        }
    }

    pub fn message_end(message: AgentMessage) -> Self {
        AgentEvent::MessageEnd {
            message,
            timestamp: now_millis(),
        }
    }

    pub fn turn_end() -> Self {
        AgentEvent::TurnEnd {
            timestamp: now_millis(),
        }
    }

    pub fn agent_end() -> Self {
        AgentEvent::AgentEnd {
            timestamp: now_millis(),
        }
    }

    pub fn tool_execution_start(tool_name: String, arguments: serde_json::Value) -> Self {
        AgentEvent::ToolExecutionStart {
            tool_name,
            arguments,
            timestamp: now_millis(),
        }
    }

    pub fn tool_execution_end(tool_name: String, result: String) -> Self {
        AgentEvent::ToolExecutionEnd {
            tool_name,
            result,
            timestamp: now_millis(),
        }
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_response_success() {
        let resp = RpcResponse::success(Some("1".into()), "ping", None);
        assert_eq!(resp.id.as_deref(), Some("1"));
        assert_eq!(resp.command, "ping");
        assert!(resp.success);
        assert!(resp.data.is_none());
    }

    #[test]
    fn test_rpc_response_success_with_data() {
        let data = serde_json::json!({"key": "value"});
        let resp = RpcResponse::success(Some("2".into()), "get_state", Some(data.clone()));
        assert_eq!(resp.data, Some(data));
    }

    #[test]
    fn test_rpc_response_error() {
        let resp = RpcResponse::error(Some("3".into()), "prompt", "Something went wrong".into());
        assert!(!resp.success);
        assert_eq!(resp.error.as_deref(), Some("Something went wrong"));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let resp = RpcResponse::success(None, "ping", None);
        let json = resp.to_json_line();
        let parsed: RpcResponse = serde_json::from_str(json.trim()).unwrap();
        assert_eq!(parsed.command, "ping");
        assert!(parsed.success);
    }

    #[test]
    fn test_deserialize_command_prompt() {
        let json = r#"{"type":"prompt","id":"req_1","message":"Hello","images":null}"#;
        let cmd: RpcCommand = serde_json::from_str(json).unwrap();
        match cmd {
            RpcCommand::Prompt { id, message, .. } => {
                assert_eq!(id, Some("req_1".into()));
                assert_eq!(message, "Hello");
            }
            _ => panic!("Expected Prompt command"),
        }
    }

    #[test]
    fn test_deserialize_command_get_state() {
        let json = r#"{"type":"get_state","id":"req_2"}"#;
        let cmd: RpcCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, RpcCommand::GetState { .. }));
    }

    #[test]
    fn test_deserialize_command_new_session() {
        let json = r#"{"type":"new_session","id":"req_3"}"#;
        let cmd: RpcCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, RpcCommand::NewSession { .. }));
    }

    #[test]
    fn test_deserialize_command_set_model() {
        let json = r#"{"type":"set_model","id":"req_4","provider":"openai-compatible","model_id":"gpt-4"}"#;
        let cmd: RpcCommand = serde_json::from_str(json).unwrap();
        match cmd {
            RpcCommand::SetModel { provider, model_id, .. } => {
                assert_eq!(provider, "openai-compatible");
                assert_eq!(model_id, "gpt-4");
            }
            _ => panic!("Expected SetModel command"),
        }
    }

    #[test]
    fn test_deserialize_command_abort() {
        let json = r#"{"type":"abort","id":"req_5"}"#;
        let cmd: RpcCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, RpcCommand::Abort { .. }));
    }

    #[test]
    fn test_agent_event_serialization() {
        let event = AgentEvent::agent_start();
        let json = event.to_json_line();
        let parsed: serde_json::Value = serde_json::from_str(json.trim()).unwrap();
        assert_eq!(parsed["type"], "agent_start");
        assert!(parsed["timestamp"].as_u64().is_some());
    }
}
