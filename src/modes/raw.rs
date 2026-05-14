use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::jsonl::serialize_json_line;
use crate::rpc::types::{AgentEvent, AssistantMessageEvent};

/// Run raw mode: prints each SSE delta as a JSON line to stdout,
/// reads user input from stdin interactively.
pub async fn run_raw(session: Arc<Mutex<AgentSession>>) {
    let mut stdin_reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    let _ = writeln!(stdout(), "rupi raw mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'.");
    let _ = stdout().flush();

    loop {
        let _ = write!(stdout(), "> ");
        let _ = stdout().flush();

        line.clear();
        match stdin_reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "/exit" || trimmed == "/quit" {
            break;
        }

        let mut final_input = trimmed;

        // Handle /goal command — set goal AND start working immediately
        if final_input.starts_with("/goal ") {
            let goal_text = final_input[6..].trim().to_string();
            if !goal_text.is_empty() {
                {
                    let sess = session.lock().await;
                    sess.set_goal(Some(goal_text.clone())).await;
                }
                let json = crate::rpc::jsonl::serialize_json_line(&serde_json::json!({
                    "type": "goal_set", "goal": goal_text
                }));
                let _ = write!(stdout(), "{}", json);
                let _ = stdout().flush();
                final_input = goal_text; // fall through to prompt handling
            } else {
                let _ = write!(stdout(), "{}", crate::rpc::jsonl::serialize_json_line(
                    &serde_json::json!({"type":"error","message":"Usage: /goal <description>"})
                ));
                let _ = stdout().flush();
                continue;
            }
        } else if final_input == "/goal" {
            let sess = session.lock().await;
            let goal = sess.get_goal().await;
            let json = crate::rpc::jsonl::serialize_json_line(&serde_json::json!({
                "type": "goal_info", "goal": goal
            }));
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
            continue;
        }

        // Handle /compact command
        if final_input == "/compact" {
            let sess = session.lock().await;
            match sess.compact().await {
                Ok(result) => {
                    let json = crate::rpc::jsonl::serialize_json_line(&serde_json::json!({
                        "type": "compaction_done", "tokens_before": result.tokens_before
                    }));
                    let _ = write!(stdout(), "{}", json);
                }
                Err(e) => {
                    let json = crate::rpc::jsonl::serialize_json_line(&serde_json::json!({
                        "type": "compaction_error", "error": e.to_string()
                    }));
                    let _ = write!(stdout(), "{}", json);
                }
            }
            let _ = stdout().flush();
            continue;
        }

        // Handle /model command
        if final_input.starts_with("/model ") || final_input == "/model" {
            let parts: Vec<&str> = final_input.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let model_spec = parts[1].trim();
                if !model_spec.is_empty() {
                    let sess = session.lock().await;
                    sess.set_model(model_spec.to_string());
                    let json = crate::rpc::jsonl::serialize_json_line(&serde_json::json!({
                        "type": "model_changed", "model": model_spec
                    }));
                    let _ = write!(stdout(), "{}", json);
                }
            } else {
                let sess = session.lock().await;
                let json = crate::rpc::jsonl::serialize_json_line(&serde_json::json!({
                    "type": "model_info", "model": sess.model()
                }));
                let _ = write!(stdout(), "{}", json);
            }
            let _ = stdout().flush();
            continue;
        }

        let session = session.clone();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

        let prompt_handle = tokio::spawn(async move {
            let sess = session.lock().await;
            let _ = sess.prompt(&final_input, event_tx).await;
        });

        while let Some(event) = event_rx.recv().await {
            let json = match &event {
                AgentEvent::MessageUpdate {
                    assistant_message_event: delta_event,
                    ..
                } => {
                    if let AssistantMessageEvent::TextDelta { delta } = delta_event {
                        serialize_json_line(&serde_json::json!({
                            "type": "delta",
                            "content": delta,
                        }))
                    } else {
                        continue;
                    }
                }
                AgentEvent::MessageEnd { message, .. } => {
                    let mut obj = serde_json::json!({
                        "type": "message_end",
                        "role": message.role,
                    });
                    // Include the message text content (especially useful for error messages)
                    let text_content: Vec<&str> = message.content.iter()
                        .filter_map(|c| c.text.as_deref())
                        .collect();
                    let text_content = text_content.join("\n");
                    if !text_content.is_empty() {
                        obj["content"] = serde_json::json!(text_content);
                    }
                    if let Some(reason) = &message.stop_reason {
                        obj["stop_reason"] = serde_json::json!(reason);
                    }
                    if let Some(usage) = &message.usage {
                        obj["usage"] = serde_json::json!({
                            "input_tokens": usage.input,
                            "output_tokens": usage.output,
                            "total_tokens": usage.total_tokens,
                        });
                        if let Some(cost) = &usage.cost {
                            let mut cost_obj = serde_json::Map::new();
                            if let Some(pc) = cost.prompt_cost {
                                cost_obj.insert("prompt_cost".into(), serde_json::json!(pc));
                            }
                            if let Some(cc) = cost.completion_cost {
                                cost_obj.insert("completion_cost".into(), serde_json::json!(cc));
                            }
                            if let Some(tc) = cost.total_cost {
                                cost_obj.insert("total_cost".into(), serde_json::json!(tc));
                            }
                            if !cost_obj.is_empty() {
                                obj["cost"] = serde_json::Value::Object(cost_obj);
                            }
                        }
                    }
                    serialize_json_line(&obj)
                }
                AgentEvent::GenerationId { id, .. } => {
                    serialize_json_line(&serde_json::json!({"type": "generation_id", "id": id}))
                }
                AgentEvent::AgentEnd { .. } => {
                    serialize_json_line(&serde_json::json!({"type": "agent_end"}))
                }
                AgentEvent::ToolExecutionStart { tool_name, arguments, .. } => {
                    serialize_json_line(&serde_json::json!({
                        "type": "tool_execution_start",
                        "tool": tool_name,
                        "arguments": arguments,
                    }))
                }
                AgentEvent::ToolExecutionEnd { tool_name, result, .. } => {
                    let truncated = if result.len() > 2000 {
                        format!("{}... [truncated]", &result[..2000])
                    } else {
                        result.clone()
                    };
                    serialize_json_line(&serde_json::json!({
                        "type": "tool_execution_end",
                        "tool": tool_name,
                        "result": truncated,
                    }))
                }
                _ => continue,
            };
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
        }

        let _ = prompt_handle.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAIConfig;

    #[tokio::test]
    async fn test_raw_mode_creates() {
        let config = OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
            timeout_secs: 0,
            reasoning: false,
        };
        let session = Arc::new(Mutex::new(AgentSession::from_config(config)));
        let sess = session.lock().await;
        assert_eq!(sess.model(), "gpt-4");
    }
}
