use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::types::{AgentEvent, AgentMessage, AssistantMessageEvent, MessageContent};

/// Run the interactive REPL mode with concurrent stdin + event reading.
/// While the agent generates output, the prompt stays active for steer/follow-up.
pub async fn run_interactive(session: Arc<Mutex<AgentSession>>) {
    let mut stdin_reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    let _ = writeln!(stdout(), "rupi interactive mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'.");
    let _ = stdout().flush();

    loop {
        // Read input (blocks until Enter or Ctrl+D)
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

        // Handle /steer command — interrupts current generation
        if trimmed.starts_with("/steer ") {
            let steer_text = trimmed[7..].trim().to_string();
            if !steer_text.is_empty() {
                let sess = session.lock().await;
                sess.steer(&steer_text).await;
                let _ = writeln!(stdout(), "Steer queued: {}", steer_text);
            } else {
                let _ = writeln!(stdout(), "Usage: /steer <message to interrupt with>");
            }
            let _ = stdout().flush();
            continue;
        }

        // Handle /goal command
        if trimmed.starts_with("/goal ") {
            let goal_text = trimmed[6..].trim().to_string();
            if !goal_text.is_empty() {
                {
                    let sess = session.lock().await;
                    sess.set_goal(Some(goal_text.clone())).await;
                }
                let _ = writeln!(stdout(), "Goal set and starting work: {}", goal_text);
                let _ = stdout().flush();
                // Fall through — the goal text becomes the prompt
            } else {
                let _ = writeln!(stdout(), "Usage: /goal <description of what to achieve>");
                let _ = stdout().flush();
                continue;
            }
        } else if trimmed == "/goal" {
            let sess = session.lock().await;
            match sess.get_goal().await {
                Some(g) => { let _ = writeln!(stdout(), "Current goal: {}", g); }
                None => { let _ = writeln!(stdout(), "No goal set."); }
            }
            let _ = stdout().flush();
            continue;
        }

        // Handle /compact command
        if trimmed == "/compact" {
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
        if trimmed.starts_with("/model ") || trimmed == "/model" {
            let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
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

        // Process the prompt with concurrent input
        process_with_steer(&session, &trimmed).await;
    }
}

async fn process_with_steer(session: &Arc<Mutex<AgentSession>>, initial_input: &str) {
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Clone session and send for the main prompt task
    let sess = session.clone();
    let tx = event_tx.clone();
    let prompt_msg = initial_input.to_string();

    // Spawn the main prompt
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

    // Spawn stdin reader for steer/follow-up during streaming
    let stdin_session = session.clone();
    let stdin_tx_clone = stdin_tx.clone();
    let _stdin_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(tokio::io::stdin());
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let input = buf.trim().to_string();
                    if input.is_empty() {
                        continue;
                    }
                    // Check if agent is still streaming
                    if stdin_session.lock().await.is_streaming().await {
                        // Normal Enter during streaming → queue as follow-up
                        stdin_session.lock().await.follow_up(&input).await;
                        let _ = stdin_tx_clone.send(input);
                    } else {
                        // If not streaming, send to main channel for processing
                        let _ = stdin_tx_clone.send(input);
                        break; // Exit stdin reader, main loop will restart
                    }
                }
            }
        }
    });

    // Main event loop — reads events AND stdin concurrently
    let mut got_text = false;

    loop {
        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(AgentEvent::MessageUpdate { assistant_message_event, .. }) => {
                        if let AssistantMessageEvent::TextDelta { delta } = &assistant_message_event {
                            got_text = true;
                            let _ = write!(stdout(), "{}", delta);
                            let _ = stdout().flush();
                        }
                    }
                    Some(AgentEvent::MessageEnd { message, .. }) => {
                        for c in &message.content {
                            if let Some(text) = &c.text {
                                let _ = write!(stdout(), "{}", text);
                                got_text = true;
                            }
                        }
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
                    Some(AgentEvent::ToolExecutionStart { tool_name, .. }) => {
                        let _ = writeln!(stdout(), "\n[Tool: {}]", tool_name);
                    }
                    Some(AgentEvent::ToolExecutionEnd { tool_name, .. }) => {
                        let _ = writeln!(stdout(), "[{} completed]", tool_name);
                    }
                    Some(AgentEvent::AgentEnd { .. }) | None => {
                        break;
                    }
                    _ => {}
                }
                let _ = stdout().flush();
            }
            _ = stdin_rx.recv() => {
                // A steer or follow-up was queued during streaming.
                // Display a brief acknowledgment.
                let _ = write!(stdout(), "\n[queued]\n");
                let _ = stdout().flush();
            }
        }
    }

    let _ = prompt_handle.await;
    let _ = writeln!(std::io::stdout());
    let _ = stdout().flush();
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
            reasoning: false,
            timeout_secs: 0,
        };
        let session = Arc::new(Mutex::new(AgentSession::from_config(config)));
        let sess = session.lock().await;
        assert_eq!(sess.model(), "gpt-4");
    }
}
