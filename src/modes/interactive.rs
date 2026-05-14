use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::types::{AgentEvent, AgentMessage, AssistantMessageEvent, MessageContent};

/// Run the interactive REPL mode.
pub async fn run_interactive(session: Arc<Mutex<AgentSession>>) {
    let mut stdin_reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    let _ = writeln!(stdout(), "rupi interactive mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'.");
    let _ = stdout().flush();

    loop {
        let _ = write!(stdout(), "> ");
        let _ = stdout().flush();

        // Simple REPL: type a line, press Enter, it sends.
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

        // Handle /goal command — sets goal AND starts working immediately
        if final_input.starts_with("/goal ") {
            let goal_text = final_input[6..].trim().to_string();
            if !goal_text.is_empty() {
                {
                    let sess = session.lock().await;
                    sess.set_goal(Some(goal_text.clone())).await;
                }
                let _ = writeln!(stdout(), "Goal set and starting work: {}", goal_text);
                let _ = stdout().flush();
                final_input = goal_text; // fall through to prompt handling
            } else {
                let _ = writeln!(stdout(), "Usage: /goal <description of what to achieve>");
                let _ = stdout().flush();
                continue;
            }
        } else if final_input == "/goal" {
            let sess = session.lock().await;
            match sess.get_goal().await {
                Some(g) => { let _ = writeln!(stdout(), "Current goal: {}", g); }
                None => { let _ = writeln!(stdout(), "No goal set."); }
            }
            let _ = stdout().flush();
            continue;
        }

        // Handle /compact command
        if final_input == "/compact" {
            let sess = session.lock().await;
            match sess.compact().await {
                Ok(result) => {
                    let _ = writeln!(stdout(), "Compaction complete: {} tokens before", result.tokens_before);
                }
                Err(e) => {
                    let _ = writeln!(stdout(), "Compaction skipped: {}", e);
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
                    let _ = writeln!(stdout(), "Switched to model: {}", model_spec);
                }
            } else {
                let sess = session.lock().await;
                let _ = writeln!(stdout(), "Current model: {}", sess.model());
            }
            let _ = stdout().flush();
            continue;
        }

        let session = session.clone();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

        let prompt_handle = tokio::spawn(async move {
            let sess = session.lock().await;
            if let Err(e) = sess.prompt(&final_input, event_tx.clone()).await {
                // Ensure an error event is always sent so the UI doesn't hang
                let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                    role: "assistant".to_string(),
                    content: vec![MessageContent {
                        content_type: "text".to_string(),
                        text: Some(format!("Error: {}", e)),
                    }],
                    model: None,
                    usage: None,
                    stop_reason: Some("error".to_string()),
                }));
                let _ = event_tx.send(AgentEvent::turn_end());
                let _ = event_tx.send(AgentEvent::agent_end());
            }
        });

        let _ = stdout().flush();
        let mut got_text = false;
        while let Some(event) = event_rx.recv().await {
            match &event {
                AgentEvent::MessageUpdate {
                    assistant_message_event: delta_event,
                    ..
                } => {
                    if let AssistantMessageEvent::TextDelta { delta } = delta_event {
                        got_text = true;
                        let _ = write!(stdout(), "{}", delta);
                        let _ = stdout().flush();
                    }
                }
                AgentEvent::MessageEnd { message, .. } => {
                    // Print message content (error text, etc.)
                    for c in &message.content {
                        if let Some(text) = &c.text {
                            let _ = write!(stdout(), "{}", text);
                            got_text = true;
                        }
                    }
                    // Print error info
                    if let Some(reason) = &message.stop_reason {
                        if reason == "error" || reason == "timeout" {
                            if !got_text {
                                let _ = write!(stdout(), "[Request failed: {}]", reason);
                            } else {
                                let _ = writeln!(stdout(), "\n[Request failed: {}]", reason);
                            }
                        }
                    }
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    let _ = writeln!(stdout(), "\n[Tool: {}]", tool_name);
                }
                AgentEvent::ToolExecutionEnd { tool_name, .. } => {
                    let _ = writeln!(stdout(), "[{} completed]", tool_name);
                }
                AgentEvent::AgentEnd { .. } => {
                    break;
                }
                _ => {}
            }
            let _ = stdout().flush();
        }

        let _ = prompt_handle.await;
        let _ = writeln!(stdout());
        let _ = stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAIConfig;

    #[tokio::test]
    async fn test_interactive_mode_creates() {
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
