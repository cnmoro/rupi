use tokio::sync::mpsc;

use rupi::agent::session::AgentSession;

use rupi::provider::openai::OpenAIConfig;

use rupi::provider::ChatProvider;

use rupi::rpc::handler::RpcHandler;

use rupi::rpc::types::{AgentEvent, RpcCommand, RpcResponse};

struct E2EConfig {
    base_url: String,
    api_key: String,
    model: String,
}

fn load_e2e_config() -> Option<E2EConfig> {
    dotenvy::from_filename("/mnt/nvme1tb/pi-clone/.env").ok();

    let api_key = std::env::var("OPENAI_COMPATIBLE_API_KEY").ok()?;
    let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL").ok()?;
    let model = std::env::var("OPENAI_COMPATIBLE_MODEL").ok()?;

    if api_key.is_empty() || base_url.is_empty() || model.is_empty() {
        return None;
    }

    Some(E2EConfig {
        base_url,
        api_key,
        model,
    })
}

fn make_config(config: &E2EConfig) -> OpenAIConfig {
    OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model.clone(),
        context_window: 128000,
        reasoning: false,
    }
}

#[tokio::test]
async fn e2e_test_api_reachable() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: OPENAI_COMPATIBLE_API_KEY/BASE_URL/MODEL not set");
            return;
        }
    };

    let provider =
        rupi::provider::openai::OpenAIProvider::new(make_config(&config));
    let info = provider.model_info();
    assert_eq!(info.id, config.model);
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
    assert_eq!(model.id, config.model);
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
    assert_eq!(info.id, config.model);
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
        model_id: config.model.clone(),
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

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut prompt_response_found = false;
    let mut agent_end_found = false;
    let mut text_deltas: Vec<String> = Vec::new();

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(line) = rx.try_recv() {
            if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
                if resp.command == "prompt" {
                    prompt_response_found = true;
                }
                continue;
            }

            if let Ok(event) = serde_json::from_str::<AgentEvent>(line.trim()) {
                match &event {
                    AgentEvent::MessageUpdate {
                        assistant_message_event: delta_event,
                        ..
                    } => {
                        if let rupi::rpc::types::AssistantMessageEvent::TextDelta {
                            delta,
                        } = delta_event
                        {
                            text_deltas.push(delta.clone());
                        }
                    }
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
    assert!(!text_deltas.is_empty(), "Should have received text deltas");

    let full_text: String = text_deltas.iter().flat_map(|s| s.chars()).collect();
    assert!(!full_text.is_empty(), "Response text should not be empty");
    assert!(
        full_text.to_lowercase().contains("hello"),
        "Response should contain 'hello', got: {}",
        full_text
    );
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

    // Step 1: Create session and verify initial state
    let session = AgentSession::from_config(make_config(&config));
    let state = session.get_state().await;
    assert_eq!(state.message_count, 0);
    assert!(state.model.is_some());

    // Step 2: Create a separate session for the handler (session can't be shared easily)
    let handler_session = AgentSession::from_config(make_config(&config));
    let handler = RpcHandler::new(handler_session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Step 3: Send a prompt and stream the response
    let prompt_cmd = RpcCommand::Prompt {
        id: Some("e2e-full-1".into()),
        message: "Reply with just: pong".into(),
        images: None,
        streaming_behavior: None,
    };

    handler.handle(prompt_cmd, tx.clone()).await;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut prompt_response = false;
    let mut agent_end = false;
    let mut deltas: Vec<String> = Vec::new();

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(line) = rx.try_recv() {
            if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
                if resp.command == "prompt" {
                    prompt_response = true;
                }
                continue;
            }

            if let Ok(event) = serde_json::from_str::<AgentEvent>(line.trim()) {
                match &event {
                    AgentEvent::MessageUpdate {
                        assistant_message_event: delta_event,
                        ..
                    } => {
                        if let rupi::rpc::types::AssistantMessageEvent::TextDelta {
                            delta,
                        } = delta_event
                        {
                            deltas.push(delta.clone());
                        }
                    }
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
    assert!(!deltas.is_empty(), "Should have streaming text deltas");

    let response_text: String = deltas.iter().flat_map(|s| s.chars()).collect();
    assert!(!response_text.is_empty(), "Response should not be empty");
}
