use std::io::{stdout, Write};
use std::sync::Arc;

use crate::agent::session::AgentSession;
use crate::modes::stdin::{spawn_stdin_reader, CANCEL_LOOP_SIG, EOF_SIG};
use crate::rpc::types::{AgentEvent, AgentMessage, AssistantMessageEvent, MessageContent};

/// Run the interactive REPL mode with concurrent stdin + event reading.
/// While the agent generates output, the prompt stays active for steer/follow-up.
pub async fn run_interactive(session: Arc<AgentSession>) {
    let _ = writeln!(stdout(), "rupi interactive mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'. Double-Esc to cancel loop.");
    let _ = stdout().flush();

    // Spawn a single persistent stdin reader with double-Esc detection
    let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    spawn_stdin_reader(stdin_tx);

    loop {
        let _ = write!(stdout(), "> ");
        let _ = stdout().flush();
        let line = match stdin_rx.recv().await {
            Some(l) => l,
            None => break,
        };

        if line == CANCEL_LOOP_SIG {
            session.cancel_loop().await;
            let _ = writeln!(stdout(), "\n[loop cancelled]");
            let _ = stdout().flush();
            continue;
        }

        if line == EOF_SIG {
            // Cancel any active loop, then exit
            session.cancel_loop().await;
            break;
        }

        match handle_command(&session, &line).await {
            CommandAction::Continue => continue,
            CommandAction::Break => break,
            CommandAction::Prompt(prompt_line) => {
                process_prompt(&session, &prompt_line, &mut stdin_rx).await;
            }
        }
    }
}

enum CommandAction {
    Continue,
    Break,
    Prompt(String),
}

async fn handle_command(session: &Arc<AgentSession>, line: &str) -> CommandAction {
    if line == "exit" || line == "/exit" || line == "/quit" {
        return CommandAction::Break;
    }

    if line.starts_with("/steer ") {
        let steer_text = line[7..].trim().to_string();
        if !steer_text.is_empty() {
            session.steer(&steer_text).await;
            session.abort().await;
            let _ = writeln!(stdout(), "Steer queued: {}", steer_text);
        } else {
            let _ = writeln!(stdout(), "Usage: /steer <message to interrupt with>");
        }
        let _ = stdout().flush();
        return CommandAction::Continue;
    }

    if line == "/stop" || line == "/stop" {
        session.cancel_loop().await;
        let _ = writeln!(stdout(), "Loop cancelled.");
        let _ = stdout().flush();
        return CommandAction::Continue;
    }

    if line.starts_with("/loop ") {
        let loop_text = line[6..].trim().to_string();
        if !loop_text.is_empty() {
            session.set_loop(Some(loop_text.clone())).await;
            let _ = writeln!(stdout(), "Loop started: {}", loop_text);
            let _ = stdout().flush();
            return CommandAction::Prompt(loop_text);
        } else {
            let _ = writeln!(stdout(), "Usage: /loop <prompt to repeat>");
            let _ = stdout().flush();
            return CommandAction::Continue;
        }
    }

    if line.starts_with("/goal ") {
        let goal_text = line[6..].trim().to_string();
        if !goal_text.is_empty() {
            session.set_goal(Some(goal_text.clone())).await;
            let _ = writeln!(stdout(), "Goal set and starting work: {}", goal_text);
            let _ = stdout().flush();
            return CommandAction::Prompt(goal_text);
        } else {
            let _ = writeln!(stdout(), "Usage: /goal <description of what to achieve>");
            let _ = stdout().flush();
            return CommandAction::Continue;
        }
    } else if line == "/goal" {
        match session.get_goal().await {
            Some(g) => { let _ = writeln!(stdout(), "Current goal: {}", g); }
            None => { let _ = writeln!(stdout(), "No goal set."); }
        }
        let _ = stdout().flush();
        return CommandAction::Continue;
    }

    if line == "/compact" {
        match session.compact().await {
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
                session.set_model(model_spec.to_string());
                let _ = writeln!(stdout(), "Switched to model: {}", model_spec);
            }
        } else {
            let _ = writeln!(stdout(), "Current model: {}", session.model());
        }
        let _ = stdout().flush();
        return CommandAction::Continue;
    }

    CommandAction::Prompt(line.to_string())
}

async fn process_prompt(
    session: &Arc<AgentSession>,
    initial_input: &str,
    stdin_rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) {
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

    // Event loop: print events, poll stdin non-blockingly for steer/follow-up.
    // IMPORTANT: Do NOT acquire the session lock here — prompt() holds it for
    // the entire duration, and we'd deadlock waiting for it. All session
    // operations use interior mutability (AtomicBool, tokio RwLock, etc.) and
    // don't need the outer Mutex.
    let mut got_text = false;
    let mut saw_reasoning = false;
    let mut interrupted: Option<String> = None;

    loop {
        // Poll stdin non-blockingly. Lines entered during streaming are
        // queued as steer or follow-up. Lines entered after the prompt has
        // finished will be left in the channel for the main REPL loop.
        loop {
            match stdin_rx.try_recv() {
                Ok(input) => {
                    if input == CANCEL_LOOP_SIG || input == EOF_SIG {
                        session.cancel_loop().await;
                        let _ = writeln!(stdout(), "\n[loop cancelled]");
                        let _ = stdout().flush();
                        continue;
                    }
                    eprintln!("rupi: stdin during prompt: {:?}", &input[..input.len().min(60)]);
                    if input.starts_with("/steer ") {
                        let steer_text = input[7..].trim().to_string();
                        session.abort().await;
                        interrupted = Some(steer_text);
                        let _ = writeln!(stdout(), "\n[interrupted]");
                    } else {
                        session.follow_up(&input).await;
                        let _ = writeln!(stdout(), "\n[queued]");
                    }
                    let _ = stdout().flush();
                }
                Err(_) => break,
            }
        }

        // Block on next event
        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(AgentEvent::MessageUpdate { assistant_message_event, .. }) => {
                        match &assistant_message_event {
                            AssistantMessageEvent::TextDelta { delta } => {
                                if saw_reasoning {
                                    let _ = write!(stdout(), "\n");
                                    saw_reasoning = false;
                                }
                                got_text = true;
                                let _ = write!(stdout(), "{}", delta);
                            }
                            AssistantMessageEvent::ThinkingDelta { delta } => {
                                saw_reasoning = true;
                                let _ = write!(stdout(), "\x1b[2m{}\x1b[22m", delta);
                            }
                        }
                        let _ = stdout().flush();
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
                    Some(AgentEvent::ToolExecutionStart { tool_name, arguments, .. }) => {
                        let args_str = serde_json::to_string(&arguments).unwrap_or_default();
                        let cmd = if args_str.len() > 120 {
                            format!("{}...", &args_str[..117])
                        } else {
                            args_str
                        };
                        let _ = writeln!(stdout(), "\n[Tool: {} {}]", tool_name, cmd);
                        let _ = stdout().flush();
                    }
                    Some(AgentEvent::ToolExecutionEnd { tool_name, result, .. }) => {
                        let truncated = if result.len() > 800 {
                            format!("{}...\n[+ {} more chars]", &result[..797], result.len() - 797)
                        } else {
                            result.clone()
                        };
                        let _ = writeln!(stdout(), "[{} completed]\n---\n{}\n---", tool_name, truncated);
                        let _ = stdout().flush();
                    }
                    Some(AgentEvent::AgentStart { .. }) => {
                        let _ = write!(stdout(), "\n[running...]");
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
            let _ = sess2.prompt(&steer_text, tx2).await;
        });
        let mut got2 = false;
        loop {
            // Poll stdin for any steer during this second prompt too
            while let Ok(input) = stdin_rx.try_recv() {
                if session.is_streaming().await && input.starts_with("/steer ") {
                    let t = input[7..].trim().to_string();
                    session.abort().await;
                    let _ = writeln!(stdout(), "\n[interrupted]");
                    interrupted = Some(t);
                    let _ = stdout().flush();
                }
            }
            match event_rx.recv().await {
                Some(AgentEvent::MessageUpdate { assistant_message_event, .. }) => {
                    match &assistant_message_event {
                        AssistantMessageEvent::TextDelta { delta } => {
                            if saw_reasoning {
                                let _ = write!(stdout(), "\n");
                                saw_reasoning = false;
                            }
                            got2 = true;
                            let _ = write!(stdout(), "{}", delta);
                        }
                        AssistantMessageEvent::ThinkingDelta { delta } => {
                            saw_reasoning = true;
                            let _ = write!(stdout(), "\x1b[2m{}\x1b[22m", delta);
                        }
                    }
                    let _ = stdout().flush();
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
        let session = Arc::new(AgentSession::from_config(config));
        assert_eq!(session.model(), "gpt-4");
    }
}
