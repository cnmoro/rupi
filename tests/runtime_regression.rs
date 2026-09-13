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

/// Read one `response` line for `id` and report whether it succeeded.
async fn response_success(rx: &mut mpsc::UnboundedReceiver<String>, id: &str) -> bool {
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(line) = rx.recv().await {
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            if value["type"] == "response" && value["id"] == id {
                return value["success"].as_bool().unwrap_or(false);
            }
        }
        panic!("missing response")
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn a_loop_refuses_a_plain_prompt_but_admits_a_steer() {
    // Refusing every `prompt` blocked the one steering path the README documents,
    // while the undocumented `steer` command went through untouched.
    let provider = Arc::new(ControlledProvider::default());
    let handler = RpcHandler::new(session(provider.clone()));
    let (tx, mut rx) = mpsc::unbounded_channel();
    handler
        .handle(
            serde_json::from_value(
                serde_json::json!({"type":"set_loop","id":"loop","message":"keep going"}),
            )
            .unwrap(),
            tx.clone(),
        )
        .await;
    assert!(response_success(&mut rx, "loop").await, "set_loop refused");

    handler
        .handle(
            serde_json::from_value(
                serde_json::json!({"type":"prompt","id":"plain","message":"unrelated"}),
            )
            .unwrap(),
            tx.clone(),
        )
        .await;
    assert!(
        !response_success(&mut rx, "plain").await,
        "a plain prompt was admitted while a loop was running"
    );

    handler
        .handle(
            serde_json::from_value(serde_json::json!({
                "type":"prompt","id":"steering","message":"fix the indentation",
                "streamingBehavior":"steer"
            }))
            .unwrap(),
            tx.clone(),
        )
        .await;
    assert!(
        response_success(&mut rx, "steering").await,
        "a documented steer was refused while a loop was running"
    );
}

/// A provider that streams one delta and then reports an idle timeout.
struct SilentAfterDelta;

#[async_trait]
impl ChatProvider for SilentAfterDelta {
    async fn stream_chat(
        &self,
        _: &str,
        _: &[Message],
        _: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        let (tx, rx) = mpsc::channel(4);
        tx.send(StreamEvent::Delta("partial answer".into()))
            .await
            .unwrap();
        tx.send(StreamEvent::Error(
            "idle timeout: the provider sent nothing for 2s".into(),
        ))
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

#[tokio::test]
async fn a_stream_that_goes_silent_keeps_what_it_already_said() {
    // The HTTP read timeout fires before the session's own idle limit, so this is
    // the path a real stall takes. Reported as an error, it threw away the text the
    // user had already seen and called the turn a decode failure.
    init_sessions();
    let session = AgentSession::new(
        Arc::new(SilentAfterDelta),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    );
    let (tx, mut rx) = mpsc::unbounded_channel();
    let collector = tokio::spawn(async move {
        let mut seen = None;
        while let Some(event) = rx.recv().await {
            let value = serde_json::to_value(&event).unwrap();
            if value["type"] == "message_end" {
                seen = value["message"]["stop_reason"].as_str().map(str::to_string);
            }
        }
        seen
    });
    let outcome = session.prompt("say something", tx).await;
    assert!(outcome.is_ok(), "a stall was reported as a failed turn");
    let stop_reason = collector.await.unwrap();
    assert_eq!(stop_reason.as_deref(), Some("timeout"), "stop reason");
    let kept = session.messages().await;
    assert!(
        kept.iter()
            .any(|m| m.role == "assistant" && m.content == "partial answer"),
        "the text the user already saw was dropped from the conversation"
    );
}

/// A provider that ignores cancellation and answers long after it was told to stop.
struct Unstoppable;

#[async_trait]
impl ChatProvider for Unstoppable {
    async fn stream_chat(
        &self,
        _: &str,
        _: &[Message],
        _: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        tokio::time::sleep(Duration::from_millis(2500)).await;
        let (tx, rx) = mpsc::channel(2);
        tx.send(StreamEvent::Done(StreamResult {
            content: "answer from the old conversation".into(),
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

#[tokio::test]
async fn a_generation_that_outlives_reset_cannot_write_into_the_new_session() {
    // `reset` waits for the generation, but the wait is bounded. A tool that
    // ignores cancellation outlives it, and the task used to come back and append
    // its reply to the cleared history and the new session file.
    init_sessions();
    let session = Arc::new(AgentSession::new(
        Arc::new(Unstoppable),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    ));
    session.set_reset_wait_secs(1);
    let running = session.clone();
    let generation = tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let _ = running.prompt("the old question", tx).await;
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    session.reset().await;
    assert!(
        session.messages().await.is_empty(),
        "reset left the old conversation behind"
    );

    let _ = tokio::time::timeout(Duration::from_secs(6), generation).await;
    let kept = session.messages().await;
    assert!(
        !kept.iter().any(|m| m.content.contains("old conversation")),
        "a reset conversation wrote into the new one: {:?}",
        kept.iter().map(|m| m.content.clone()).collect::<Vec<_>>()
    );
}

/// A provider whose generation never finishes and never honours cancellation.
struct NeverFinishes;

#[async_trait]
impl ChatProvider for NeverFinishes {
    async fn stream_chat(
        &self,
        _: &str,
        _: &[Message],
        _: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        std::future::pending::<()>().await;
        unreachable!()
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

#[tokio::test]
async fn reset_returns_even_when_a_generation_never_finishes() {
    // The bounded wait was decorative. An ordinary generation held a read lock on
    // the loop prompt for its whole duration, so `reset` blocked in `set_loop(None)`
    // right after giving up on the wait it does bound.
    init_sessions();
    let session = Arc::new(AgentSession::new(
        Arc::new(NeverFinishes),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    ));
    session.set_reset_wait_secs(1);
    let running = session.clone();
    tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let _ = running.prompt("start work that never ends", tx).await;
    });
    tokio::time::sleep(Duration::from_millis(300)).await;

    tokio::time::timeout(Duration::from_secs(8), session.reset())
        .await
        .expect("reset hung on a generation that never finishes");
}

/// A provider that answers each turn after a short, cancellable pause.
#[derive(Default)]
struct SlowAnswers {
    calls: AtomicUsize,
    started: Notify,
}

#[async_trait]
impl ChatProvider for SlowAnswers {
    async fn stream_chat(
        &self,
        _: &str,
        _: &[Message],
        mut signal: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(300)) => {}
            _ = signal.changed() => return Err(AgentError::Cancelled),
        }
        let (tx, rx) = mpsc::channel(2);
        tx.send(StreamEvent::Done(StreamResult {
            content: "round done".into(),
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

#[tokio::test]
async fn steering_a_loop_keeps_the_loop_running() {
    // A steer cancels the round it interrupts. That used to end the loop silently
    // while `is_loop_active` still said it was running, so plain prompts stayed
    // refused for a loop that would never run another round.
    init_sessions();
    let provider = Arc::new(SlowAnswers::default());
    let session = Arc::new(AgentSession::new(
        provider.clone(),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    ));
    session.set_loop(Some("keep going".into())).await;
    let running = session.clone();
    tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let _ = running.prompt("start the loop", tx).await;
    });
    tokio::time::timeout(Duration::from_secs(2), provider.started.notified())
        .await
        .unwrap();

    session.steer("please also check the tests").await;
    let before = provider.calls.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(1200)).await;

    assert!(
        provider.calls.load(Ordering::SeqCst) > before + 1,
        "the loop stopped cycling after a steer"
    );
    assert!(
        session.is_loop_active().await,
        "the loop was reported inactive while it kept running"
    );
    session.cancel_loop().await;
}

#[tokio::test]
async fn stopping_a_loop_beats_a_steer_that_arrived_first() {
    // `cancel_loop` must end the loop even when a steer is already pending. The
    // steered message is still processed as one ordinary turn, because dropping
    // what the user typed would be worse than running it.
    init_sessions();
    let provider = Arc::new(SlowAnswers::default());
    let session = Arc::new(AgentSession::new(
        provider.clone(),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    ));
    session.set_loop(Some("keep going".into())).await;
    let running = session.clone();
    let driver = tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let _ = running.prompt("start the loop", tx).await;
    });
    tokio::time::timeout(Duration::from_secs(2), provider.started.notified())
        .await
        .unwrap();

    session.steer("one more thing").await;
    session.cancel_loop().await;

    tokio::time::timeout(Duration::from_secs(5), driver)
        .await
        .expect("the loop kept running after it was stopped")
        .unwrap();
    assert!(!session.is_loop_active().await, "the loop is still active");
    assert!(
        !session.is_streaming().await,
        "a generation is still running"
    );
    let said = session.messages().await;
    let steered = said
        .iter()
        .filter(|m| m.content == "one more thing")
        .count();
    assert_eq!(steered, 1, "the steered message was lost or duplicated");
    let rounds = said.iter().filter(|m| m.content == "keep going").count();
    assert_eq!(rounds, 0, "a loop round started after the loop was stopped");
}

/// A provider that streams part of an answer and then closes the stream.
struct CutsTheStream;

#[async_trait]
impl ChatProvider for CutsTheStream {
    async fn stream_chat(
        &self,
        _: &str,
        _: &[Message],
        _: watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        let (tx, rx) = mpsc::channel(4);
        tx.send(StreamEvent::Delta("partial thought that got cut".into()))
            .await
            .unwrap();
        tx.send(StreamEvent::Error(format!(
            "{}: the provider closed the stream after 28 characters",
            rupi::provider::openai::TRUNCATED_PREFIX
        )))
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

#[tokio::test]
async fn a_cut_stream_is_never_reported_as_a_finished_answer() {
    // A dropped connection used to look exactly like a completed answer: the text
    // was kept, the turn said "stop", and a subagent printed half a sentence and
    // exited zero. The text is still kept, but the turn says what happened.
    init_sessions();
    let session = Arc::new(AgentSession::new(
        Arc::new(CutsTheStream),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    ));
    let (tx, mut rx) = mpsc::unbounded_channel();
    let collector = tokio::spawn(async move {
        let mut seen = None;
        while let Some(event) = rx.recv().await {
            let value = serde_json::to_value(&event).unwrap();
            if value["type"] == "message_end" {
                seen = value["message"]["stop_reason"].as_str().map(str::to_string);
            }
        }
        seen
    });
    let outcome = session.prompt("tell me something", tx).await;
    assert!(outcome.is_ok(), "a cut stream failed the whole turn");
    assert_eq!(
        collector.await.unwrap().as_deref(),
        Some("truncated"),
        "a cut stream was reported as a clean stop"
    );

    // One shot must refuse it rather than print a fragment as the answer.
    let fresh = Arc::new(AgentSession::new(
        Arc::new(CutsTheStream),
        "test".into(),
        128000,
        ".".into(),
        vec![],
        vec![],
    ));
    let answer = rupi::modes::once::run_once(fresh, "tell me something").await;
    assert!(
        answer.is_err(),
        "a fragment was printed as a finished answer: {:?}",
        answer
    );
}
