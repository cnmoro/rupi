use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::types::{AgentEvent, AgentMessage, AssistantMessageEvent, MessageContent};

/// Run the interactive REPL mode with concurrent stdin + event reading.
/// While the agent generates output, the prompt stays active for steer/follow-up.
pub async fn run_interactive(session: Arc<Mutex<AgentSession>>) {
    let _ = writeln!(stdout(), "rupi interactive mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'.");
    let _ = stdout().flush();

    // Spawn a single persistent stdin reader that feeds lines through a channel.
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
                    if input.is_empty() {
                        continue;
                    }
                    let _ = stdin_tx.send(input).ok();
                }
            }
        }
    });

    loop {
        let _ = write!(stdout(), "> ");
        let _ = stdout().flush();
        let line = match stdin_rx.recv().await {
            Some(l) => l,
            None => break,
        };

        match handle_command(&session, &line).await {
            CommandAction::Continue => continue,
            CommandAction::Break => break,
            CommandAction::Prompt => {
                process_prompt(&session, &line, &mut stdin_rx).await;
            }
        }
    }
}

enum CommandAction {
    Continue,
    Break,
    Prompt,
}

async fn handle_command(session: &Arc<Mutex<AgentSession>>, line: &str) -> CommandAction {
    if line == "exit" || line == "/exit" || line == "/quit" {
        return CommandAction::Break;
    }

    if line.starts_with("/steer ") {
        let steer_text = line[7..].trim().to_string();
        if !steer_text.is_empty() {
            let sess = session.lock().await;
            sess.steer(&steer_text).await;
            sess.abort().await;
            let _ = writeln!(stdout(), "Steer queued: {}", steer_text);
        } else {
            let _ = writeln!(stdout(), "Usage: /steer <message to interrupt with>");
        }
        let _ = stdout().flush();
        return CommandAction::Continue;
    }

    if line.starts_with("/goal ") {
        let goal_text = line[6..].trim().to_string();
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
            return CommandAction::Continue;
        }
    } else if line == "/goal" {
        let sess = session.lock().await;
        match sess.get_goal().await {
            Some(g) => { let _ = writeln!(stdout(), "Current goal: {}", g); }
            None => { let _ = writeln!(stdout(), "No goal set."); }
        }
        let _ = stdout().flush();
        return CommandAction::Continue;
    }

    if line == "/compact" {
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
        return CommandAction::Continue;
    }

    if line.starts_with("/model ") || line == "/model" {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
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
        return CommandAction::Continue;
    }

    CommandAction::Prompt
}

async fn process_prompt(
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

    // Event loop: print events, poll stdin non-blockingly for steer/follow-up
    let mut got_text = false;
    let mut interrupted: Option<String> = None;

    loop {
        // Poll stdin non-blockingly before blocking on event_rx.
        // Only consume from the channel when still streaming — if streaming
        // already finished, leave the line in the channel for the main loop.
        let sess_check = session.lock().await;
        let still_streaming = sess_check.is_streaming().await;
        drop(sess_check);
        if still_streaming {
            loop {
                match stdin_rx.try_recv() {
                    Ok(input) => {
                        eprintln!("rupi: stdin during prompt: {:?}", &input[..input.len().min(60)]);
                        let sess = session.lock().await;
                        if input.starts_with("/steer ") {
                            let steer_text = input[7..].trim().to_string();
                            sess.abort().await;
                            interrupted = Some(steer_text);
                            let _ = writeln!(stdout(), "\n[interrupted]");
                        } else {
                            sess.follow_up(&input).await;
                            let _ = writeln!(stdout(), "\n[queued]");
                        }
                        let _ = stdout().flush();
                    }
                    Err(_) => break,
                }
            }
        }

        // Block on next event
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
                        if !got_text {
                            for c in &message.content {
                                if let Some(text) = &c.text {
                                    let _ = write!(stdout(), "{}", text);
                                    got_text = true;
                                }
                            }
                        }
                        if let Some(reason) = &message.stop_reason {
                            if reason == "error" || reason == "timeout" {
                                if !got_text {
                                    let _ = write!(stdout(), "[Request failed: {}", reason);
                                } else {
                                    let _ = writeln!(stdout(), "\n[Request failed: {}", reason);
                                }
                            }
                        }
                        let _ = stdout().flush();
                    }
                    Some(AgentEvent::ToolExecutionStart { tool_name, .. }) => {
                        let _ = writeln!(stdout(), "\n[Tool: {}]", tool_name);
                        let _ = stdout().flush();
                    }
                    Some(AgentEvent::ToolExecutionEnd { tool_name, .. }) => {
                        let _ = writeln!(stdout(), "[{} completed]", tool_name);
                        let _ = stdout().flush();
                    }
                    Some(AgentEvent::AgentEnd { .. }) | None => {
                        break;
                    }
                    _ => { let _ = stdout().flush(); }
                }
            }
        }
    }

    let _ = prompt_handle.await;

    // If interrupted by steer, immediately start a new prompt
    if let Some(steer_text) = interrupted {
        let _ = writeln!(stdout(), "[steer: {}]", steer_text);
        let _ = stdout().flush();
        let sess2 = session.clone();
        let tx2 = event_tx.clone();
        let handle2 = tokio::spawn(async move {
            let sl = sess2.lock().await;
            let _ = sl.prompt(&steer_text, tx2).await;
        });
        let mut got2 = false;
        loop {
            // Poll stdin for any steer during this second prompt too
            while let Ok(input) = stdin_rx.try_recv() {
                let sess = session.lock().await;
                if sess.is_streaming().await && input.starts_with("/steer ") {
                    let t = input[7..].trim().to_string();
                    sess.abort().await;
                    let _ = writeln!(stdout(), "\n[interrupted]");
                    // Replace the steer text
                    interrupted = Some(t);
                    let _ = stdout().flush();
                }
            }
            match event_rx.recv().await {
                Some(AgentEvent::MessageUpdate { assistant_message_event, .. }) => {
                    if let AssistantMessageEvent::TextDelta { delta } = &assistant_message_event {
                        got2 = true;
                        let _ = write!(stdout(), "{}", delta);
                        let _ = stdout().flush();
                    }
                }
                Some(AgentEvent::MessageEnd { message, .. }) => {
                    if !got2 {
                        for c in &message.content {
                            if let Some(text) = &c.text {
                                let _ = write!(stdout(), "{}", text);
                                got2 = true;
                            }
                        }
                    }
                    let _ = stdout().flush();
                }
                Some(AgentEvent::AgentEnd { .. }) | None => { break; }
                _ => {}
            }
        }
        let _ = handle2.await;
    }

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
