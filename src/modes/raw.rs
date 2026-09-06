use std::io::{stdout, Write};
use std::sync::Arc;

use crate::agent::session::AgentSession;
use crate::modes::stdin::{read_line_edited, spawn_streaming_reader, CANCEL_LOOP_SIG, EOF_SIG};
use crate::rpc::jsonl::serialize_json_line;
use crate::rpc::types::{AgentEvent, AgentMessage, AssistantMessageEvent, MessageContent};

pub async fn run_raw(session: Arc<AgentSession>) {
    let _ = writeln!(stdout(), "rupi raw mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'. Double-Esc to cancel loop.");
    let _ = stdout().flush();

    loop {
        let line = tokio::task::spawn_blocking(move || read_line_edited("> "))
            .await
            .ok()
            .flatten();

        let line = match line {
            Some(l) => l,
            None => break,
        };

        if line == "exit" || line == "/exit" || line == "/quit" || line == EOF_SIG {
            break;
        }

        if let Some(text) = line.strip_prefix("/steer ") {
            let steer_text = text.trim().to_string();
            if !steer_text.is_empty() {
                session.steer(&steer_text).await;
                session.abort().await;
                let json = serialize_json_line(
                    &serde_json::json!({"type": "steer_queued", "message": steer_text}),
                );
                let _ = write!(stdout(), "{}", json);
            } else {
                let _ = write!(
                    stdout(),
                    "{}",
                    serialize_json_line(
                        &serde_json::json!({"type":"error","message":"Usage: /steer <message>"})
                    )
                );
            }
            let _ = stdout().flush();
            continue;
        }

        if let Some(text) = line.strip_prefix("/goal ") {
            let goal_text = text.trim().to_string();
            if !goal_text.is_empty() {
                session.set_goal(Some(goal_text.clone())).await;
                let json = serialize_json_line(
                    &serde_json::json!({"type": "goal_set", "goal": goal_text}),
                );
                let _ = write!(stdout(), "{}", json);
                let _ = stdout().flush();
                process_prompt_raw(&session, &goal_text).await;
                continue;
            }
            let _ = write!(
                stdout(),
                "{}",
                serialize_json_line(
                    &serde_json::json!({"type":"error","message":"Usage: /goal <description>"})
                )
            );
            let _ = stdout().flush();
            continue;
        }
        if line == "/goal" {
            let goal = session.get_goal().await;
            let json = serialize_json_line(&serde_json::json!({"type": "goal_info", "goal": goal}));
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
            continue;
        }

        if line == "/compact" {
            match session.compact().await {
                Ok(result) => {
                    let json = serialize_json_line(
                        &serde_json::json!({"type": "compaction_done", "tokens_before": result.tokens_before}),
                    );
                    let _ = write!(stdout(), "{}", json);
                }
                Err(e) => {
                    let json = serialize_json_line(
                        &serde_json::json!({"type": "compaction_error", "error": e.to_string()}),
                    );
                    let _ = write!(stdout(), "{}", json);
                }
            }
            let _ = stdout().flush();
            continue;
        }

        if line.starts_with("/model ") || line == "/model" {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let model_spec = parts[1].trim();
                if !model_spec.is_empty() {
                    session.set_model(model_spec.to_string());
                    let json = serialize_json_line(
                        &serde_json::json!({"type": "model_changed", "model": model_spec}),
                    );
                    let _ = write!(stdout(), "{}", json);
                }
            } else {
                let json = serialize_json_line(
                    &serde_json::json!({"type": "model_info", "model": session.model()}),
                );
                let _ = write!(stdout(), "{}", json);
            }
            let _ = stdout().flush();
            continue;
        }

        if line == CANCEL_LOOP_SIG || line == EOF_SIG {
            session.cancel_loop().await;
            let json = serialize_json_line(&serde_json::json!({"type": "loop_cancelled"}));
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
            continue;
        }

        if line == "/stop" {
            session.cancel_loop().await;
            let json = serialize_json_line(&serde_json::json!({"type": "loop_cancelled"}));
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
            continue;
        }

        if let Some(text) = line.strip_prefix("/loop ") {
            let loop_text = text.trim().to_string();
            if !loop_text.is_empty() {
                session.set_loop(Some(loop_text.clone())).await;
                let json = serialize_json_line(
                    &serde_json::json!({"type": "loop_set", "message": loop_text}),
                );
                let _ = write!(stdout(), "{}", json);
                let _ = stdout().flush();
                process_prompt_raw(&session, &loop_text).await;
                continue;
            }
            let _ = write!(
                stdout(),
                "{}",
                serialize_json_line(
                    &serde_json::json!({"type":"error","message":"Usage: /loop <prompt>"})
                )
            );
            let _ = stdout().flush();
            continue;
        }

        if line == "/session" {
            let path = session.session_path().await;
            let id = path
                .as_ref()
                .and_then(|p| p.file_stem().and_then(|s| s.to_str()));
            let json =
                serialize_json_line(&serde_json::json!({"type": "session_id", "session_id": id}));
            let _ = write!(stdout(), "{}", json);
            let _ = stdout().flush();
            continue;
        }

        process_prompt_raw(&session, &line).await;
    }
}

async fn process_prompt_raw(session: &Arc<AgentSession>, initial_input: &str) {
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let sess = session.clone();
    let tx = event_tx.clone();
    let prompt_msg = initial_input.to_string();

    let prompt_handle = tokio::spawn(async move {
        if let Err(e) = sess.prompt(&prompt_msg, tx.clone()).await {
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

    // Spawn streaming reader for steer/follow-up during this prompt
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let streaming_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    spawn_streaming_reader(stdin_tx, streaming_stop.clone());

    loop {
        // Poll stdin non-blockingly
        while let Ok(input) = stdin_rx.try_recv() {
            if input == CANCEL_LOOP_SIG || input == EOF_SIG {
                session.cancel_loop().await;
                let _ = write!(
                    stdout(),
                    "{}",
                    serialize_json_line(&serde_json::json!({"type": "loop_cancelled"}))
                );
                let _ = stdout().flush();
                if input == EOF_SIG {
                    streaming_stop.store(true, std::sync::atomic::Ordering::SeqCst);
                    return;
                }
                continue;
            }
            if session.is_streaming().await {
                if input.starts_with("/steer ") {
                    session.abort().await;
                    let _ = write!(
                        stdout(),
                        "{}",
                        serialize_json_line(&serde_json::json!({"type": "steer_queued"}))
                    );
                } else {
                    session.follow_up(&input).await;
                    let _ = write!(
                        stdout(),
                        "{}",
                        serialize_json_line(&serde_json::json!({"type": "queued"}))
                    );
                }
                let _ = stdout().flush();
            }
        }

        tokio::select! {
            event = event_rx.recv() => {
                let json = match event {
                    Some(AgentEvent::MessageUpdate { assistant_message_event, .. }) => {
                        match &assistant_message_event {
                            AssistantMessageEvent::TextDelta { delta } =>
                                serialize_json_line(&serde_json::json!({"type": "delta", "content": delta})),
                            AssistantMessageEvent::ThinkingDelta { delta } =>
                                serialize_json_line(&serde_json::json!({"type": "thinking_delta", "content": delta})),
                        }
                    }
                    Some(AgentEvent::MessageEnd { message, .. }) => {
                        let mut obj = serde_json::json!({"type": "message_end", "role": message.role});
                        let text_content: Vec<&str> = message.content.iter()
                            .filter_map(|c| c.text.as_deref()).collect();
                        if !text_content.is_empty() {
                            obj["content"] = serde_json::json!(text_content.join("\n"));
                        }
                        if let Some(ref reason) = message.stop_reason {
                            obj["stop_reason"] = serde_json::json!(reason);
                        }
                        serialize_json_line(&obj)
                    }
                    Some(AgentEvent::ToolExecutionStart { tool_name, arguments, .. }) =>
                        serialize_json_line(&serde_json::json!({"type": "tool_execution_start", "tool": tool_name, "arguments": arguments})),
                    Some(AgentEvent::ToolExecutionEnd { tool_name, result, .. }) => {
                        let truncated = if result.len() > 2000 {
                            let end = result.floor_char_boundary(2000);
                            format!("{}... [truncated]", &result[..end])
                        } else { result.clone() };
                        serialize_json_line(&serde_json::json!({"type": "tool_execution_end", "tool": tool_name, "result": truncated}))
                    }
                    Some(AgentEvent::AgentStart { .. }) =>
                        serialize_json_line(&serde_json::json!({"type": "agent_start"})),
                    Some(AgentEvent::AgentEnd { .. }) | None => { break; }
                    _ => continue,
                };
                let _ = write!(stdout(), "{}", json);
                let _ = stdout().flush();
            }
        }
    }

    streaming_stop.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = prompt_handle.await;
}
