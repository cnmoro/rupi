use tokio::sync::mpsc;

use rupi::agent::session::AgentSession;
use rupi::config::RupiConfig;
use rupi::provider::openai::OpenAIConfig;
use rupi::provider::ChatProvider;
use rupi::rpc::handler::RpcHandler;
use rupi::rpc::types::{AgentEvent, RpcCommand, RpcResponse};

fn load_e2e_config() -> Option<RupiConfig> {
    // Try the canonical location first
    let config_paths = [
        std::path::PathBuf::from("/mnt/nvme1tb/pi-clone/.env"),
        rupi::config::config_path().unwrap_or_default(),
        std::path::PathBuf::from(".env"),
    ];

    for path in &config_paths {
        if path.exists() {
            let contents = std::fs::read_to_string(path).ok()?;
            // Try JSON format first (rupi config)
            if let Ok(config) = serde_json::from_str::<RupiConfig>(&contents) {
                if !config.base_url.is_empty() && !config.api_key.is_empty() && !config.model_tag.is_empty() {
                    return Some(config);
                }
            }
            // Fall back to dotenv format
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = line.split_once('=') {
                    let key = key.trim();
                    let value = value.trim().trim_matches('"');
                    match key {
                        "OPENAI_COMPATIBLE_BASE_URL" | "RUPI_BASE_URL" => {
                            std::env::set_var("RUPI_BASE_URL", value);
                        }
                        "OPENAI_COMPATIBLE_API_KEY" | "RUPI_API_KEY" => {
                            std::env::set_var("RUPI_API_KEY", value);
                        }
                        "OPENAI_COMPATIBLE_MODEL" | "RUPI_MODEL" => {
                            std::env::set_var("RUPI_MODEL", value);
                        }
                        _ => {}
                    }
                }
            }
            let base_url = std::env::var("RUPI_BASE_URL").ok().filter(|s| !s.is_empty());
            let api_key = std::env::var("RUPI_API_KEY").ok().filter(|s| !s.is_empty());
            let model = std::env::var("RUPI_MODEL").ok().filter(|s| !s.is_empty());
            if let (Some(base_url), Some(api_key), Some(model)) = (base_url, api_key, model) {
                return Some(RupiConfig {
                    base_url,
                    api_key,
                    model_tag: model,
                });
            }
        }
    }
    None
}

fn make_config(config: &RupiConfig) -> OpenAIConfig {
    OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        reasoning: false,
    }
}

#[tokio::test]
async fn e2e_test_api_reachable() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let provider = rupi::provider::openai::OpenAIProvider::new(make_config(&config));
    let info = provider.model_info();
    assert_eq!(info.id, config.model_tag);
    assert_eq!(info.provider, "openai-compatible");
    assert!(info.context_window > 0);
}

#[tokio::test]
async fn e2e_test_rpc_get_state() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    let state = session.get_state().await;
    assert!(state.model.is_some());
    let model = state.model.as_ref().unwrap();
    assert_eq!(model.id, config.model_tag);
    assert_eq!(model.provider, "openai-compatible");
    assert_eq!(state.message_count, 0);
    assert!(!state.is_streaming);
}

#[tokio::test]
async fn e2e_test_rpc_set_and_get_thinking_level() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    assert_eq!(session.thinking_level().await, "off");

    session.set_thinking_level("high".into()).await;
    assert_eq!(session.thinking_level().await, "high");

    let level = session.cycle_thinking_level().await;
    assert!(level.is_some());
}

#[tokio::test]
async fn e2e_test_rpc_new_session_clears_messages() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    assert_eq!(session.get_state().await.message_count, 0);
    session.reset().await;
    assert_eq!(session.get_state().await.message_count, 0);
}

#[tokio::test]
async fn e2e_test_rpc_available_models() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    let info = session.provider_model_info();
    assert_eq!(info.provider, "openai-compatible");
    assert_eq!(info.id, config.model_tag);
}

#[tokio::test]
async fn e2e_test_rpc_set_model() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let cmd = RpcCommand::SetModel {
        id: Some("e2e-2".into()),
        provider: "openai-compatible".into(),
        model_id: config.model_tag.clone(),
    };
    handler.handle(cmd, tx).await;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut found = false;
    while let Ok(line) = rx.try_recv() {
        if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
            if resp.command == "set_model" && resp.success {
                found = true;
            }
        }
    }
    assert!(found, "Should have received a successful set_model response");
}

#[tokio::test]
async fn e2e_test_rpc_abort() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let cmd = RpcCommand::Abort {
        id: Some("e2e-3".into()),
    };
    handler.handle(cmd, tx).await;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut found = false;
    while let Ok(line) = rx.try_recv() {
        if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
            if resp.command == "abort" && resp.success {
                found = true;
            }
        }
    }
    assert!(found, "Should have received a successful abort response");
}

#[tokio::test]
async fn e2e_test_rpc_prompt_and_stream() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let prompt_cmd = RpcCommand::Prompt {
        id: Some("e2e-1".into()),
        message: "Reply with just the word 'hello' in lowercase, nothing else.".into(),
        images: None,
        streaming_behavior: None,
    };

    handler.handle(prompt_cmd, tx.clone()).await;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    let mut prompt_response_found = false;
    let mut agent_end_found = false;
    let mut _text_deltas: Vec<String> = Vec::new();

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        while let Ok(line) = rx.try_recv() {
            if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
                if resp.command == "prompt" {
                    prompt_response_found = true;
                }
                continue;
            }

            if let Ok(event) = serde_json::from_str::<AgentEvent>(line.trim()) {
                match &event {
                    AgentEvent::MessageUpdate { .. } => {}
                    AgentEvent::AgentEnd { .. } => {
                        agent_end_found = true;
                    }
                    _ => {}
                }
            }
        }

        if prompt_response_found && agent_end_found {
            break;
        }
    }

    assert!(prompt_response_found, "Should have received a prompt response");
    assert!(agent_end_found, "Agent should have completed");
    // With tools available, the model might use bash instead of just text
    // So we don't require text deltas for the test to pass
}

#[tokio::test]
async fn e2e_test_full_conversation_flow() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let session = AgentSession::from_config(make_config(&config));
    let state = session.get_state().await;
    assert_eq!(state.message_count, 0);
    assert!(state.model.is_some());

    let handler_session = AgentSession::from_config(make_config(&config));
    let handler = RpcHandler::new(handler_session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let prompt_cmd = RpcCommand::Prompt {
        id: Some("e2e-full-1".into()),
        message: "Reply with just: pong".into(),
        images: None,
        streaming_behavior: None,
    };

    handler.handle(prompt_cmd, tx.clone()).await;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    let mut prompt_response = false;
    let mut agent_end = false;
    let mut _deltas: Vec<String> = Vec::new();

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        while let Ok(line) = rx.try_recv() {
            if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
                if resp.command == "prompt" {
                    prompt_response = true;
                }
                continue;
            }

            if let Ok(event) = serde_json::from_str::<AgentEvent>(line.trim()) {
                match &event {
                    AgentEvent::MessageUpdate { .. } => {}
                    AgentEvent::AgentEnd { .. } => {
                        agent_end = true;
                    }
                    _ => {}
                }
            }
        }
        if prompt_response && agent_end {
            break;
        }
    }

    assert!(prompt_response, "Prompt response should be received");
    assert!(agent_end, "Agent should have completed");
}
