use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::jsonl::serialize_json_line;
use crate::rpc::types::{AgentEvent, AgentMessage, AssistantMessageEvent, MessageContent};

/// Run raw mode: prints each SSE delta as a JSON line to stdout,
/// reads user input from stdin interactively with steer support.
pub async fn run_raw(session: Arc<Mutex<AgentSession>>) {
    let _ = writeln!(stdout(), "rupi raw mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'.");
    let _ = stdout().flush();

    // Single persistent stdin reader feeding a channel
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        let mut reader = BufReader::new(tokio::io::stdin());
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let input = buf.trim().to_string();
                    if input.is_empty() { continue; }
                    let _ = stdin_tx.send(input).ok();
                }
            }
        }
    });

    loop {
        let line = match stdin_rx.recv().await {
            Some(l) => l,
            None => break,
        };

        if line == "exit" || line == "/exit" || line == "/quit" {
            break;
        }

        // Handle /steer command
        if line.starts_with("/steer ") {
            let steer_text = line[7..].trim().to_string();
            if !steer_text.is_empty() {
                let sess = session.lock().await;
                sess.steer(&steer_text).await;
                let json = serialize_json_line(&serde_json::json!({"type": "steer_queued", "message": steer_text}));
                let _ = write!(stdout(), "{}", json);
            } else {
                let _ = write!(stdout(), "{}", serialize_json_line(&serde_json::json!({"type":"error","message":"Usage: /steer <message>"})));
            }
            let _ = stdout().flush();
            continue;
        }

        // Handle /goal command
        if line.starts_with("/goal ") {
            let goal_text = line[6..].trim().to_string();
            if !goal_text.is_empty() {
                {
                    let sess = session.lock().await;
                    sess.set_goal(Some(goal_text.clone())).await;
                }
                let json = serialize_json_line(&serde_json::json!({"type": "goal_set", "goal": goal_text}));
                let _ = write!(stdout(), "{}", json);
                let _ = stdout().flush();
            } else {
                let _ = write!(stdout(), "{}", serialize_json_line(
                    &serde_json::json!({"type":"error","message":"Usage: /goal <description>"})
                ));
                let _ = stdout().flush();
                continue;
            }
        } else if line == "/goal" {
            let sess = session.lock().await;
            let goal = sess.get_goal().await;
            let json = serialize_json_line(&serde_json::json!({"type": "goal_info", "goal": goal}));
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
            continue;
        }

        // Handle /compact command
        if line == "/compact" {
            let sess = session.lock().await;
            match sess.compact().await {
                Ok(result) => {
                    let json = serialize_json_line(&serde_json::json!({
                        "type": "compaction_done", "tokens_before": result.tokens_before
                    }));
                    let _ = write!(stdout(), "{}", json);
                }
                Err(e) => {
                    let json = serialize_json_line(&serde_json::json!({
                        "type": "compaction_error", "error": e.to_string()
                    }));
                    let _ = write!(stdout(), "{}", json);
                }
            }
            let _ = stdout().flush();
            continue;
        }

        // Handle /model command
        if line.starts_with("/model ") || line == "/model" {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let model_spec = parts[1].trim();
                if !model_spec.is_empty() {
                    let sess = session.lock().await;
                    sess.set_model(model_spec.to_string());
                    let json = serialize_json_line(&serde_json::json!({"type": "model_changed", "model": model_spec}));
                    let _ = write!(stdout(), "{}", json);
                }
            } else {
                let sess = session.lock().await;
                let json = serialize_json_line(&serde_json::json!({"type": "model_info", "model": sess.model()}));
                let _ = write!(stdout(), "{}", json);
            }
            let _ = stdout().flush();
            continue;
        }

        // Process prompt
        process_with_steer_raw(&session, &line, &mut stdin_rx).await;
    }
}

async fn process_with_steer_raw(
    session: &Arc<Mutex<AgentSession>>,
    initial_input: &str,
    stdin_rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) {
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

    let sess = session.clone();
    let tx = event_tx.clone();
    let prompt_msg = initial_input.to_string();

    let prompt_handle = tokio::spawn(async move {
        let sess_lock = sess.lock().await;
        if let Err(e) = sess_lock.prompt(&prompt_msg, tx.clone()).await {
            let _ = tx.send(AgentEvent::message_end(AgentMessage {
                role: "assistant".to_string(),
                content: vec![MessageContent {
                    content_type: "text".to_string(),
                    text: Some(format!("Error: {}", e)),
                }],
                model: None,
                usage: None,
                stop_reason: Some("error".to_string()),
            }));
            let _ = tx.send(AgentEvent::turn_end());
            let _ = tx.send(AgentEvent::agent_end());
        }
    });

    loop {
        tokio::select! {
            event = event_rx.recv() => {
                let json = match event {
                    Some(AgentEvent::MessageUpdate { assistant_message_event, .. }) => {
                        if let AssistantMessageEvent::TextDelta { delta } = &assistant_message_event {
                            serialize_json_line(&serde_json::json!({"type": "delta", "content": delta}))
                        } else { continue; }
                    }
                    Some(AgentEvent::MessageEnd { message, .. }) => {
                        let mut obj = serde_json::json!({"type": "message_end", "role": message.role});
                        let text_content: Vec<&str> = message.content.iter()
                            .filter_map(|c| c.text.as_deref()).collect();
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
                                if let Some(pc) = cost.prompt_cost { cost_obj.insert("prompt_cost".into(), serde_json::json!(pc)); }
                                if let Some(cc) = cost.completion_cost { cost_obj.insert("completion_cost".into(), serde_json::json!(cc)); }
                                if let Some(tc) = cost.total_cost { cost_obj.insert("total_cost".into(), serde_json::json!(tc)); }
                                if !cost_obj.is_empty() { obj["cost"] = serde_json::Value::Object(cost_obj); }
                            }
                        }
                        serialize_json_line(&obj)
                    }
                    Some(AgentEvent::GenerationId { id, .. }) => {
                        serialize_json_line(&serde_json::json!({"type": "generation_id", "id": id}))
                    }
                    Some(AgentEvent::AgentEnd { .. }) | None => { break; }
                    Some(AgentEvent::ToolExecutionStart { tool_name, arguments, .. }) => {
                        serialize_json_line(&serde_json::json!({"type": "tool_execution_start", "tool": tool_name, "arguments": arguments}))
                    }
                    Some(AgentEvent::ToolExecutionEnd { tool_name, result, .. }) => {
                        let truncated = if result.len() > 2000 {
                            format!("{}... [truncated]", &result[..2000])
                        } else { result.clone() };
                        serialize_json_line(&serde_json::json!({"type": "tool_execution_end", "tool": tool_name, "result": truncated}))
                    }
                    _ => continue,
                };
                let _ = write!(stdout(), "{}", json);
                let _ = stdout().flush();
            }
            line = stdin_rx.recv() => {
                if let Some(input) = line {
                    let sess = session.lock().await;
                    if sess.is_streaming().await {
                        sess.follow_up(&input).await;
                        let _ = write!(stdout(), "{}", serialize_json_line(&serde_json::json!({"type": "queued"})));
                        let _ = stdout().flush();
                    }
                }
            }
        }
    }

    let _ = prompt_handle.await;
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
            reasoning: false,
            timeout_secs: 0,
        };
        let session = Arc::new(Mutex::new(AgentSession::from_config(config)));
        let sess = session.lock().await;
        assert_eq!(sess.model(), "gpt-4");
    }
}
