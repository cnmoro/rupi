//! Deterministic runtime regressions: no credentials or model downloads.
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, Once,
};
use std::time::Duration;

use async_trait::async_trait;
use rupi::agent::session::{AgentSession, Message};
use rupi::error::AgentError;
use rupi::provider::openai::{OpenAIConfig, OpenAIProvider};
use rupi::provider::{ChatProvider, StreamEvent, StreamResult};
use rupi::rpc::{
    handler::RpcHandler,
    types::{AgentEvent, ModelInfo},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, watch, Notify};

fn config(url: String) -> OpenAIConfig {
    OpenAIConfig {
        base_url: url,
        api_key: "test".into(),
        model: "test".into(),
        context_window: 128000,
        reasoning: false,
        timeout_secs: 0,
    }
}

fn init_sessions() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        rupi::sessions::set_sessions_dir(
            std::env::temp_dir().join(format!("rupi-runtime-tests-{}", std::process::id())),
        )
    });
}

#[tokio::test]
async fn retries_http_429_before_streaming() {
    init_sessions();
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(429)
        .with_body("rate limited")
        .expect(3)
        .create_async()
        .await;
    let session = AgentSession::from_config(config(server.url()));
    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::time::timeout(Duration::from_secs(8), session.prompt("hello", tx))
        .await
        .unwrap()
        .unwrap();
    mock.assert_async().await;
    assert!(std::iter::from_fn(|| rx.try_recv().ok())
        .any(|event| matches!(event, AgentEvent::AgentEnd { .. })));
}

#[tokio::test]
async fn preserves_utf8_split_across_network_chunks_and_stops_at_done() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let header_end = loop {
            let mut bytes = [0; 8192];
            let n = socket.read(&mut bytes).await.unwrap();
            assert!(n > 0);
            request.extend_from_slice(&bytes[..n]);
            if let Some(i) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
        };
        let len: usize = String::from_utf8_lossy(&request[..header_end])
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().unwrap())
            })
            .unwrap();
        while request.len() < header_end + len {
            let mut bytes = [0; 8192];
            let n = socket.read(&mut bytes).await.unwrap();
            assert!(n > 0);
            request.extend_from_slice(&bytes[..n]);
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        let data = "data: {\"choices\":[{\"delta\":{\"content\":\"Olá 🌍\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n".as_bytes();
        let split = data.iter().position(|b| *b == 0xc3).unwrap() + 1;
        socket.write_all(&data[..split]).await.unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        socket.write_all(&data[split..]).await.unwrap();
        socket.flush().await.unwrap();
        // Keep the connection open: [DONE] must finish without waiting for EOF.
        tokio::time::sleep(Duration::from_secs(5)).await;
    });
    let provider = OpenAIProvider::new(config(url));
    let (_cancel, signal) = watch::channel(false);
    let mut rx = provider.stream_chat("test", &[], signal).await.unwrap();
    let text = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Done(result) = event {
                return result.content;
            }
        }
        panic!("missing completion")
    })
    .await
    .unwrap();
    assert_eq!(text, "Olá 🌍");
    server.abort();
}

#[derive(Default)]
struct ControlledProvider {
    calls: AtomicUsize,
    started: Notify,
    release: Notify,
    cancelled: AtomicBool,
    requests: Mutex<Vec<Vec<Message>>>,
}

#[async_trait]
impl ChatProvider for ControlledProvider {
    async fn stream_chat(
        &self,
        _: &str,
        messages: &[Message],
        mut signal: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        self.requests.lock().unwrap().push(messages.to_vec());
        let first = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
        if first {
            self.started.notify_one();
            tokio::select! {
                _ = self.release.notified() => {},
                _ = signal.changed() => {
                    self.cancelled.store(true, Ordering::SeqCst);
                    return Err(AgentError::Cancelled);
                }
            }
        }
        let (tx, rx) = mpsc::channel(2);
        tx.send(StreamEvent::Done(StreamResult {
            content: if first {
                "original response"
            } else {
                "new response"
            }
            .into(),
            ..Default::default()
        }))
        .await
        .unwrap();
        Ok(rx)
    }
    async fn complete(&self, _: &str, _: &[Message]) -> Result<String, AgentError> {
        Ok("summary".into())
    }
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            provider: "openai-compatible".into(),
            id: "test".into(),
            context_window: 128000,
            reasoning: false,
        }
    }
}

fn session(provider: Arc<ControlledProvider>) -> AgentSession {
    init_sessions();
    AgentSession::new(provider, "test".into(), 128000, ".".into(), vec![], vec![])
}

async fn wait_response(rx: &mut mpsc::UnboundedReceiver<String>, id: &str) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(line) = rx.recv().await {
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            if value["type"] == "response" && value["id"] == id {
                return;
            }
        }
        panic!("missing response")
    })
    .await
    .unwrap();
}

async fn check_rpc_behavior(command: serde_json::Value, interrupts: bool) {
    let provider = Arc::new(ControlledProvider::default());
    let handler = RpcHandler::new(session(provider.clone()));
    let (tx, mut rx) = mpsc::unbounded_channel();
    handler
        .handle(
            serde_json::from_value(
                serde_json::json!({"type":"prompt","id":"first","message":"original"}),
            )
            .unwrap(),
            tx.clone(),
        )
        .await;
    tokio::time::timeout(Duration::from_secs(2), provider.started.notified())
        .await
        .unwrap();
    handler
        .handle(serde_json::from_value(command).unwrap(), tx.clone())
        .await;
    wait_response(&mut rx, "second").await;
    if !interrupts {
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!provider.cancelled.load(Ordering::SeqCst));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        provider.release.notify_one();
    }
    tokio::time::timeout(Duration::from_secs(3), async {
        while provider.calls.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(provider.cancelled.load(Ordering::SeqCst), interrupts);
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(line) = rx.recv().await {
            let event: serde_json::Value = serde_json::from_str(&line).unwrap();
            if event["type"] == "agent_end" {
                return;
            }
        }
        panic!("missing terminal event");
    })
    .await
    .unwrap();
    let requests = provider.requests.lock().unwrap();
    if !interrupts {
        assert!(
            requests[1]
                .iter()
                .any(|m| m.role == "assistant" && m.content == "original response"),
            "follow-up discarded the completed answer"
        );
    }
    assert!(requests[1]
        .iter()
        .any(|m| m.role == "user" && m.content == "changed"));
}

#[tokio::test]
async fn rpc_steer_interrupts_and_follow_up_waits() {
    check_rpc_behavior(
        serde_json::json!({"type":"steer","id":"second","message":"changed"}),
        true,
    )
    .await;
    check_rpc_behavior(
        serde_json::json!({"type":"follow_up","id":"second","message":"changed"}),
        false,
    )
    .await;
    check_rpc_behavior(serde_json::json!({"type":"prompt","id":"second","message":"changed","streamingBehavior":"steer"}), true).await;
    check_rpc_behavior(serde_json::json!({"type":"prompt","id":"second","message":"changed","streamingBehavior":"followUp"}), false).await;
}

#[tokio::test]
async fn reset_waits_for_generation_and_isolates_new_history() {
    let provider = Arc::new(ControlledProvider::default());
    let session = Arc::new(session(provider.clone()));
    let old_path = session.session_path().await;
    let running = session.clone();
    let (tx, _rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(async move { running.prompt("original", tx).await });
    tokio::time::timeout(Duration::from_secs(2), provider.started.notified())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), session.reset())
        .await
        .unwrap();
    task.await.unwrap().unwrap();
    assert!(provider.cancelled.load(Ordering::SeqCst));
    assert!(session.messages().await.is_empty());
    assert_ne!(session.session_path().await, old_path);
    let (tx, _rx) = mpsc::unbounded_channel();
    session.prompt("new task", tx).await.unwrap();
    let messages = session.messages().await;
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "new task");
    assert_eq!(messages[1].content, "new response");
}

#[tokio::test]
async fn blocking_bash_does_not_stall_runtime_and_is_cancellable() {
    use rupi::tools::{execute_tool_async, ToolCall, ToolContext};
    let context = ToolContext::new();
    let cancel = context.cancelled.clone();
    let call = ToolCall {
        id: "sleep".into(),
        name: "bash".into(),
        arguments: serde_json::json!({"command":"sleep 20"}),
        raw_arguments: None,
    };
    let task = tokio::spawn(execute_tool_async(call, context, None));
    tokio::time::sleep(Duration::from_millis(100)).await;
    cancel.store(true, Ordering::SeqCst);
    let result = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert!(result.contains("cancelled"), "{result}");
}

#[tokio::test]
async fn rpc_reset_cannot_overtake_a_just_submitted_prompt() {
    let provider = Arc::new(ControlledProvider::default());
    let handler = RpcHandler::new(session(provider));
    let (tx, mut rx) = mpsc::unbounded_channel();
    handler
        .handle(
            serde_json::from_value(serde_json::json!({
                "type":"prompt", "id":"first", "message":"old task"
            }))
            .unwrap(),
            tx.clone(),
        )
        .await;
    tokio::time::timeout(
        Duration::from_secs(2),
        handler.handle(
            serde_json::from_value(serde_json::json!({"type":"new_session", "id":"reset"}))
                .unwrap(),
            tx.clone(),
        ),
    )
    .await
    .unwrap();
    handler
        .handle(
            serde_json::from_value(serde_json::json!({"type":"get_messages", "id":"messages"}))
                .unwrap(),
            tx,
        )
        .await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(line) = rx.recv().await {
            let response: serde_json::Value = serde_json::from_str(&line).unwrap();
            if response["id"] == "messages" {
                assert_eq!(response["data"]["messages"], serde_json::json!([]));
                return;
            }
        }
        panic!("missing get_messages response");
    })
    .await
    .unwrap();
}
