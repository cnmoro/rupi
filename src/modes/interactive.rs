use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, stdin, stdout};
use tokio::sync::Mutex;

use crate::agent::session::AgentSession;
use crate::rpc::types::{AgentEvent, AssistantMessageEvent};

/// Run the interactive REPL mode.
/// Reads user input from stdin, sends to the model, prints streaming deltas to stdout.
pub async fn run_interactive(session: Arc<Mutex<AgentSession>>) {
    let mut out = stdout();
    let mut stdin_reader = BufReader::new(stdin());
    let mut line = String::new();

    let _ = out.write_all(b"rupi interactive mode. Type your prompts. Exit with Ctrl+C or 'exit'.\n").await;
    let _ = out.flush().await;

    loop {
        let _ = out.write_all(b"\n> ").await;
        let _ = out.flush().await;

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
        if input == "exit" || input == "/exit" {
            break;
        }

        let session = session.clone();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

        // Spawn the prompt
        let prompt_handle = tokio::spawn(async move {
            let sess = session.lock().await;
            let _ = sess.prompt(&input, event_tx).await;
        });

        // Read and print events
        let mut full_response = String::new();
        let _ = out.write_all(b"\n").await;
        while let Some(event) = event_rx.recv().await {
            match &event {
                AgentEvent::MessageUpdate {
                    assistant_message_event: delta_event,
                    ..
                } => {
                    if let AssistantMessageEvent::TextDelta { delta } = delta_event {
                        full_response.push_str(delta);
                        let _ = out.write_all(delta.as_bytes()).await;
                        let _ = out.flush().await;
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    break;
                }
                _ => {}
            }
        }

        let _ = prompt_handle.await;
        let _ = out.write_all(b"\n").await;
        let _ = out.flush().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAIConfig;
    use tokio::sync::Mutex;

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
        // Just verify the session is accessible
        let sess = session.lock().await;
        assert_eq!(sess.model(), "gpt-4");
    }
}
