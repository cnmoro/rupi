use std::io::{stdout, Write};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::types::{AgentEvent, AssistantMessageEvent};

/// Run the interactive REPL mode.
pub async fn run_interactive(session: Arc<Mutex<AgentSession>>) {
    let mut stdin_reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    let _ = writeln!(stdout(), "rupi interactive mode. Type your prompts. Exit: Ctrl+D, /exit, /quit, or 'exit'.");
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

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "/exit" || input == "/quit" {
            break;
        }

        // Handle /model command
        if input.starts_with("/model ") || input == "/model" {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
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
            let _ = sess.prompt(&input, event_tx).await;
        });

        let _ = stdout().flush();
        while let Some(event) = event_rx.recv().await {
            match &event {
                AgentEvent::MessageUpdate {
                    assistant_message_event: delta_event,
                    ..
                } => {
                    if let AssistantMessageEvent::TextDelta { delta } = delta_event {
                        let _ = write!(stdout(), "{}", delta);
                        let _ = stdout().flush();
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    break;
                }
                _ => {}
            }
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
            reasoning: false,
        };
        let session = Arc::new(Mutex::new(AgentSession::from_config(config)));
        let sess = session.lock().await;
        assert_eq!(sess.model(), "gpt-4");
    }
}
