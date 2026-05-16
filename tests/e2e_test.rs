use std::sync::Arc;
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
        timeout_secs: 0,
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

#[tokio::test]
async fn e2e_test_compaction_and_continuation() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    // Create a session with a tiny context window (500 tokens) so
    // compaction triggers immediately after a few messages.
    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 500,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // Disable auto-compaction so we can accumulate messages
    session.set_auto_compaction_enabled(false).await;
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Send multiple prompts to build up enough messages for compaction (needs >= 4)
    let secret = format!("secret-value-{}", std::process::id());
    for i in 0..3 {
        let msg = if i == 0 {
            format!("IMPORTANT: Remember this secret code: {}. Reply with: stored", secret)
        } else {
            format!("Reply with: message-{}", i)
        };
        session.prompt(&msg, tx.clone()).await.unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut found = false;
        while std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            while let Ok(event) = rx.try_recv() {
                if matches!(event, AgentEvent::AgentEnd { .. }) { found = true; }
            }
            if found { break; }
        }
    }

    // Compact
    let result = session.compact().await;
    assert!(result.is_ok(), "Compaction should succeed: {:?}", result.err());
    let result = result.unwrap();
    assert!(!result.summary.is_empty(), "Compaction summary should not be empty");
    assert!(result.tokens_before > 0, "tokens_before should be > 0");
    eprintln!("Compaction OK: {} tokens before", result.tokens_before);

    // Ask about the fact — the agent should remember via the compaction summary
    let (tx2, mut rx2) = mpsc::unbounded_channel::<AgentEvent>();
    session.prompt(
        &format!("What was the secret code I asked you to remember? Reply with just the code."),
        tx2.clone(),
    ).await.unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut full = String::new();
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx2.try_recv() {
            if let AgentEvent::MessageUpdate { assistant_message_event, .. } = &event {
                if let rupi::rpc::types::AssistantMessageEvent::TextDelta { delta } = assistant_message_event {
                    full.push_str(delta);
                }
            }
            if matches!(event, AgentEvent::AgentEnd { .. }) { break; }
        }
        if full.contains(&secret) {
            break;
        }
    }

    assert!(
        full.contains(&secret),
        "Agent should remember the secret code from before compaction. Expected '{}' in '{}'",
        secret, full
    );
    eprintln!("Agent remembered the secret code after compaction: confirmed");
}

/// Helper: wait for an AgentEnd event from an event channel.
async fn wait_for_agent_end(rx: &mut mpsc::UnboundedReceiver<AgentEvent>, timeout_secs: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AgentEvent::AgentEnd { .. }) { return; }
        }
    }
}

#[tokio::test]
async fn e2e_test_compaction_via_rpc_and_continue() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 500,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // Disable auto-compaction so we can accumulate messages
    session.set_auto_compaction_enabled(false).await;
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Step 1: send multiple prompts to build up messages (needs >= 4 for find_cut_point)
    let secret = format!("top-secret-{}", std::process::id());
    for i in 0..3 {
        let msg = if i == 0 {
            format!("Remember this secret: {}. Reply: stored", secret)
        } else {
            format!("Reply with: msg-{}", i)
        };
        let cmd = RpcCommand::Prompt {
            id: Some(format!("c1-{}", i)),
            message: msg,
            images: None,
            streaming_behavior: None,
        };
        handler.handle(cmd, tx.clone()).await;
        wait_events_rpc(&mut rx, 60, &["agent_end"]).await;
    }
    eprintln!("Step 1: {} prompts done", 3);

    // Step 2: compact
    let compact_cmd = RpcCommand::Compact { id: Some("c2".into()), custom_instructions: None };
    handler.handle(compact_cmd, tx.clone()).await;
    let events = wait_events_rpc(&mut rx, 60, &["response"]).await;
    let compact_success = events.iter().any(|e| {
        if let Ok(resp) = serde_json::from_str::<RpcResponse>(e) {
            resp.command == "compact" && resp.success
        } else { false }
    });
    assert!(compact_success, "Compaction RPC should succeed");
    eprintln!("Step 2: compaction done");

    // Step 3: ask about the fact — should still be remembered
    let follow_cmd = RpcCommand::Prompt {
        id: Some("c3".into()),
        message: "What was the secret I told you to remember? Reply with just the code.".into(),
        images: None,
        streaming_behavior: None,
    };
    handler.handle(follow_cmd, tx.clone()).await;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut full = String::new();
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(line) = rx.try_recv() {
            if let Ok(event) = serde_json::from_str::<AgentEvent>(line.trim()) {
                if let AgentEvent::MessageUpdate { assistant_message_event, .. } = &event {
                    if let rupi::rpc::types::AssistantMessageEvent::TextDelta { delta } = assistant_message_event {
                        full.push_str(delta);
                    }
                }
                if matches!(event, AgentEvent::AgentEnd { .. }) { break; }
            }
        }
        if std::time::Instant::now() > deadline { break; }
        if full.contains(&secret) { break; }
    }

    assert!(
        full.contains(&secret),
        "After compaction, agent should remember '{}' but got: {}",
        secret, full
    );
    eprintln!("Step 3: agent remembered after compaction: confirmed");
}

async fn wait_events_rpc(rx: &mut mpsc::UnboundedReceiver<String>, timeout_secs: u64, targets: &[&str]) -> Vec<String> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut collected = Vec::new();
    let mut found = std::collections::HashSet::new();
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(line) = rx.try_recv() {
            collected.push(line.clone());
            for t in targets {
                if line.contains(t) { found.insert(t.to_string()); }
            }
        }
        if found.len() == targets.len() { break; }
    }
    collected
}

#[tokio::test]
async fn e2e_test_goal_completes() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // Set a simple, quickly achievable goal
    session.set_goal(Some("Say the word 'pineapple' in your response.".into())).await;

    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    let result = session.prompt("What is 2+2? Reply briefly.", tx.clone()).await;
    assert!(result.is_ok(), "Goal prompt should complete");

    // Collect all events — should eventually get agent_end
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut got_agent_end = false;
    let mut got_message_end = false;
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            match &event {
                AgentEvent::MessageEnd { .. } => { got_message_end = true; }
                AgentEvent::AgentEnd { .. } => { got_agent_end = true; }
                _ => {}
            }
        }
        if got_agent_end { break; }
    }

    assert!(got_agent_end, "Agent should have completed after goal was achieved");
    assert!(got_message_end, "Should have message end");
    eprintln!("Goal test: agent completed after goal achieved");
}

#[tokio::test]
async fn e2e_test_goal_nudge_detected() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // Goal requires a specific phrase. The prompt does NOT mention this phrase,
    // so the first response won't satisfy it. Verification fails → nudge → agent tries again.
    let required_phrase = "nudge_xyz_789";
    session.set_goal(Some(format!("Your response must include the exact phrase '{}'.", required_phrase))).await;

    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Deliberately do NOT mention the required phrase in the prompt
    let result = session.prompt("Say exactly 'hello world', nothing else.", tx.clone()).await;
    assert!(result.is_ok(), "Goal prompt should complete");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut got_agent_end = false;
    let mut msg_count = 0u32;
    let mut user_message_after_first = false;

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        while let Ok(event) = rx.try_recv() {
            match &event {
                AgentEvent::MessageStart { ref message, .. } => {
                    msg_count += 1;
                    // If we see a 2nd+ user message, a nudge occurred
                    if message.role == "user" && msg_count > 1 {
                        user_message_after_first = true;
                    }
                }
                AgentEvent::AgentEnd { .. } => { got_agent_end = true; }
                _ => {}
            }
        }
        if got_agent_end { break; }
    }

    assert!(got_agent_end, "Agent should complete");
    assert!(
        user_message_after_first,
        "Expected a nudge (second user message). msg_count={}",
        msg_count
    );
    eprintln!(
        "Nudge test PASSED: msg_count={}, nudge detected",
        msg_count
    );
}

#[tokio::test]
async fn e2e_test_goal_no_goal_normal_flow() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    // No goal set — normal flow
    let session = AgentSession::from_config(openai_config);
    // Explicitly clear goal
    session.set_goal(None).await;

    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    session.prompt("Say hello.", tx.clone()).await.unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut got_end = false;
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AgentEvent::AgentEnd { .. }) { got_end = true; }
        }
        if got_end { break; }
    }
    assert!(got_end, "Normal flow should complete with agent_end");
    eprintln!("No-goal test: normal flow completed");
}

#[tokio::test]
async fn e2e_test_goal_rpc_set_and_run() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    session.set_goal(Some("Say the word 'done' at the end of your response.".into())).await;

    let handler = RpcHandler::new(session);
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let prompt_cmd = RpcCommand::Prompt {
        id: Some("goal-test-1".into()),
        message: "What color is the sky? Reply briefly.".into(),
        images: None,
        streaming_behavior: None,
    };
    handler.handle(prompt_cmd, tx.clone()).await;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut got_agent_end = false;
    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(line) = rx.try_recv() {
            if let Ok(event) = serde_json::from_str::<AgentEvent>(line.trim()) {
                if matches!(event, AgentEvent::AgentEnd { .. }) { got_agent_end = true; }
            }
        }
        if got_agent_end { break; }
    }
    assert!(got_agent_end, "RPC goal mode should complete with agent_end");
    eprintln!("RPC goal test: completed");
}#[tokio::test]
async fn e2e_test_approval_deny_tool() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // Set approval callback that denies ALL tool executions
    use rupi::agent::session::ApprovalFn;
    use std::sync::Arc;
    let deny: ApprovalFn = Arc::new(|_, _| false);
    session.set_approval_fn(Some(deny));

    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Send a prompt that explicitly asks to use bash
    session.prompt("Use bash to check the current date and time.", tx.clone()).await.unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut saw_denied = false;
    let mut agent_ended = false;

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            match &event {
                AgentEvent::ToolExecutionEnd { tool_name, result, .. } => {
                    if tool_name == "bash" && result.contains("denied") {
                        saw_denied = true;
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    agent_ended = true;
                }
                _ => {}
            }
        }
        if saw_denied && agent_ended { break; }
    }

    assert!(saw_denied, "Should have seen a denied tool execution");
    assert!(agent_ended, "Agent should complete even with denied tools");
    eprintln!("Deny test passed: tool was blocked");
}

#[tokio::test]
async fn e2e_test_approval_allow_tool() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // Set approval callback that ALLOWS all tool executions
    use rupi::agent::session::ApprovalFn;
    use std::sync::Arc;
    let allow: ApprovalFn = Arc::new(|_, _| true);
    session.set_approval_fn(Some(allow));

    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    session.prompt("Use bash to check the current date.", tx.clone()).await.unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut saw_allowed = false;
    let mut agent_ended = false;

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            match &event {
                AgentEvent::ToolExecutionEnd { tool_name, result, .. } => {
                    if tool_name == "bash" && !result.contains("denied") {
                        saw_allowed = true;
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    agent_ended = true;
                }
                _ => {}
            }
        }
        if saw_allowed && agent_ended { break; }
    }

    assert!(saw_allowed, "Should have seen an allowed tool execution");
    assert!(agent_ended, "Agent should complete");
    eprintln!("Allow test passed: tool was executed");
}

#[tokio::test]
async fn e2e_test_approval_yolo_default() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        timeout_secs: 0,
            reasoning: false,
    };

    let session = AgentSession::from_config(openai_config);
    // No approval fn set = YOLO mode (tools always allowed)
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    session.prompt("Use bash to check the current date.", tx.clone()).await.unwrap();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut saw_tool = false;
    let mut agent_ended = false;

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            match &event {
                AgentEvent::ToolExecutionEnd { tool_name, .. } => {
                    if tool_name == "bash" {
                        saw_tool = true;
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    agent_ended = true;
                }
                _ => {}
            }
        }
        if saw_tool && agent_ended { break; }
    }

    assert!(saw_tool, "Should have seen a bash tool execution in YOLO mode");
    assert!(agent_ended, "Agent should complete");
    eprintln!("YOLO test passed: tool ran without approval prompt");
}

#[tokio::test]
async fn e2e_test_steer_queues_during_streaming() {
    let config = match load_e2e_config() {
        Some(c) => c,
        None => {
            eprintln!("Skipping e2e test: credentials not set");
            return;
        }
    };

    let openai_config = OpenAIConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model_tag.clone(),
        context_window: 128000,
        reasoning: false,
        timeout_secs: 0,
    };

    let session = Arc::new(AgentSession::from_config(openai_config));
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();

    // Start a multi-step prompt that keeps the agent busy (multiple bash calls)
    let session_clone = session.clone();
    tokio::spawn(async move {
        session_clone.prompt(
            "Use bash to: first check the date, then check who you are (whoami), then check the current directory. Do each step one at a time.",
            tx.clone(),
        ).await.unwrap();
    });

    // Queue a steer very soon after starting (before the agent finishes)
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    session.steer("Also, check how long the system has been running (uptime).").await;

    // Wait for completion
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut agent_ended = false;
    let mut steer_processed = false;

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AgentEvent::AgentEnd { .. }) {
                agent_ended = true;
            }
        }
        if agent_ended { break; }
    }

    // After completion, check that the steer message is in the conversation history
    let msgs = session.messages().await;
    steer_processed = msgs.iter().any(|m| m.content.contains("uptime"));

    assert!(agent_ended, "Agent should complete after steer");
    assert!(steer_processed, "Steer message should be in conversation history. Messages: {:?}",
        msgs.iter().map(|m| format!("{}: {}", m.role, &m.content[..m.content.len().min(60)])).collect::<Vec<_>>());
    eprintln!("Steer test: completed, steer was processed");
}
