use std::collections::HashSet;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use rupi::agent::session::AgentSession;
use rupi::provider::openai::{OpenAIConfig, OpenAIProvider};
use rupi::provider::ChatProvider;
use rupi::rpc::handler::RpcHandler;
use rupi::rpc::jsonl::serialize_json_line;
use rupi::rpc::types::{RpcCommand, RpcResponse};

/// Create a mock OpenAI-compatible server that returns a streaming response.
async fn start_mock_server(
    response_chunks: Vec<String>,
    status_code: u16,
) -> (u16, Arc<Mutex<Vec<String>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let requests_clone = requests.clone();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };

            let mut buf = Vec::new();
            let body_start = loop {
                let mut chunk = [0; 4096];
                let n = socket.read(&mut chunk).await.unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&chunk[..n]);
                if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..body_start]);
            let len: usize = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            while buf.len() < body_start + len {
                let mut chunk = [0; 4096];
                let n = socket.read(&mut chunk).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            requests_clone
                .lock()
                .await
                .push(String::from_utf8_lossy(&buf[body_start..]).into_owned());

            // Build chunked SSE response
            let mut response = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: text/event-stream\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
                status_code,
                if status_code == 200 { "OK" } else { "Error" }
            );

            for chunk in &response_chunks {
                response.push_str(chunk);
            }

            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        }
    });

    (port, requests)
}

fn sse_chunk(content: &str) -> String {
    format!("data: {}\n\n", content)
}

#[tokio::test]
async fn test_mock_server_streaming_response() {
    let chunks = vec![
        sse_chunk(r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#),
        sse_chunk(r#"{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}"#),
        sse_chunk(
            r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}"#,
        ),
        "data: [DONE]\n\n".to_string(),
    ];

    let (port, _requests) = start_mock_server(chunks, 200).await;

    let config = OpenAIConfig {
        base_url: format!("http://127.0.0.1:{}", port),
        api_key: "test-key".into(),
        model: "gpt-4".into(),
        context_window: 8192,
        timeout_secs: 0,
        reasoning: false,
    };

    let provider = OpenAIProvider::new(config);
    let (_cancel, signal) = tokio::sync::watch::channel(false);
    let mut stream = provider.stream_chat("gpt-4", &[], signal).await.unwrap();
    let mut text = String::new();
    let mut usage = None;
    while let Some(event) = stream.recv().await {
        match event {
            rupi::provider::StreamEvent::Delta(delta) => text.push_str(&delta),
            rupi::provider::StreamEvent::Done(result) => {
                usage = Some((result.input_tokens, result.output_tokens))
            }
            rupi::provider::StreamEvent::Error(error) => panic!("{error}"),
            _ => {}
        }
    }
    assert_eq!(text, "Hello world");
    assert_eq!(usage, Some((10, 3)));
    assert_eq!(_requests.lock().await.len(), 1);
}

#[tokio::test]
async fn test_mock_server_error_response() {
    let chunks = vec![];
    let (port, _requests) = start_mock_server(chunks, 401).await;

    let config = OpenAIConfig {
        base_url: format!("http://127.0.0.1:{}", port),
        api_key: "bad-key".into(),
        model: "gpt-4".into(),
        context_window: 8192,
        timeout_secs: 0,
        reasoning: false,
    };

    let provider = OpenAIProvider::new(config);
    let (_cancel, signal) = tokio::sync::watch::channel(false);
    let error = provider
        .stream_chat("gpt-4", &[], signal)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        rupi::error::AgentError::Api {
            status_code: 401,
            ..
        }
    ));
    assert_eq!(_requests.lock().await.len(), 1);
}

#[tokio::test]
async fn test_rpc_ping_through_mock_server() {
    let (port, _requests) = start_mock_server(vec![], 200).await;

    let config = OpenAIConfig {
        base_url: format!("http://127.0.0.1:{}", port),
        api_key: "test-key".into(),
        model: "gpt-4".into(),
        context_window: 8192,
        timeout_secs: 0,
        reasoning: false,
    };

    let session = AgentSession::from_config(config);
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let cmd = RpcCommand::Ping {
        id: Some("req_1".into()),
    };
    handler.handle(cmd, tx).await;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut found = false;
    while let Ok(line) = rx.try_recv() {
        if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
            if resp.command == "ping" && resp.success {
                found = true;
            }
        }
    }
    assert!(found, "Should have received a ping response");
}

#[tokio::test]
async fn test_rpc_get_state_through_mock() {
    let (port, _requests) = start_mock_server(vec![], 200).await;

    let config = OpenAIConfig {
        base_url: format!("http://127.0.0.1:{}", port),
        api_key: "test-key".into(),
        model: "gpt-4".into(),
        context_window: 128000,
        timeout_secs: 0,
        reasoning: false,
    };

    let session = AgentSession::from_config(config);
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let cmd = RpcCommand::GetState { id: None };
    handler.handle(cmd, tx).await;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    while let Ok(line) = rx.try_recv() {
        if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
            if resp.command == "get_state" && resp.success {
                if let Some(data) = resp.data {
                    assert_eq!(data["model"]["id"], "gpt-4");
                    assert_eq!(data["model"]["context_window"], 128000);
                    return;
                }
            }
        }
    }
    panic!("Should have received a get_state response with model data");
}

#[tokio::test]
async fn test_full_rpc_command_flow() {
    let (port, _requests) = start_mock_server(vec![], 200).await;

    let config = OpenAIConfig {
        base_url: format!("http://127.0.0.1:{}", port),
        api_key: "test-key".into(),
        model: "gpt-4".into(),
        context_window: 8192,
        timeout_secs: 0,
        reasoning: false,
    };

    let session = AgentSession::from_config(config);
    let handler = RpcHandler::new(session);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let commands = vec![
        RpcCommand::Ping {
            id: Some("1".into()),
        },
        RpcCommand::GetState {
            id: Some("2".into()),
        },
        RpcCommand::SetThinkingLevel {
            id: Some("3".into()),
            level: "high".into(),
        },
    ];

    for cmd in &commands {
        handler.handle(cmd.clone(), tx.clone()).await;
    }

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let mut found_commands = HashSet::new();
    while let Ok(line) = rx.try_recv() {
        if let Ok(resp) = serde_json::from_str::<RpcResponse>(line.trim()) {
            if resp.success {
                found_commands.insert(resp.command);
            }
        }
    }

    assert!(found_commands.contains("ping"));
    assert!(found_commands.contains("get_state"));
    assert!(found_commands.contains("set_thinking_level"));
}

#[tokio::test]
async fn test_jsonl_protocol_roundtrip() {
    let sent_command = RpcCommand::GetState {
        id: Some("test-1".into()),
    };
    let jsonl = serialize_json_line(&sent_command);

    let parsed: RpcCommand = serde_json::from_str(jsonl.trim()).unwrap();
    match parsed {
        RpcCommand::GetState { id } => {
            assert_eq!(id, Some("test-1".into()));
        }
        _ => panic!("Expected GetState command"),
    }

    let response = RpcResponse::success(
        Some("test-1".into()),
        "get_state",
        Some(serde_json::json!({"model": {"provider": "openai-compatible", "id": "gpt-4"}})),
    );
    let response_jsonl = serialize_json_line(&response);
    let parsed_response: RpcResponse = serde_json::from_str(response_jsonl.trim()).unwrap();
    assert!(parsed_response.success);
    assert_eq!(parsed_response.command, "get_state");
    assert!(parsed_response.data.is_some());
}
