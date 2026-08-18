//! End-to-end checks for the task anchor, the cache-aligned summarizer, tool
//! pairing across a compaction, and spill.
//!
//! These run against a mock provider, so they need no network and no model. That is
//! the point: the properties under test are properties of the harness, and a live
//! model would only add noise to them.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use tokio::sync::mpsc;

use rupi::agent::session::{recover_anchor, AgentSession, Message};
use rupi::error::AgentError;
use rupi::provider::{ChatProvider, StreamEvent};
use rupi::rpc::types::{AgentEvent, ModelInfo};
use rupi::tools::ToolCall;

/// The exact request every test anchors on. Distinctive on purpose: the assertions
/// check for these bytes, not for a paraphrase a summarizer might produce.
const ORIGINAL_REQUEST: &str =
    "REFACTOR the zephyr parser in src/zephyr.rs and keep the QUUX-7 invariant intact";

/// One shared sessions directory for this test binary.
///
/// `set_sessions_dir` is backed by a `OnceLock`, so only the first call wins. Every
/// test routes through this helper to make sure they all agree on the location.
fn test_sessions_dir() -> std::path::PathBuf {
    static DIR: OnceLock<std::path::PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("rupi-anchor-tests-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        rupi::sessions::set_sessions_dir(dir.clone());
        dir
    })
    .clone()
}

/// What one mocked turn sends back.
enum Turn {
    /// Call `bash` with a command whose output is `output_bytes` long.
    ToolCall { output_bytes: usize },
    /// Call the `goal` tool, naming `round` as the round the call belongs to.
    GoalDecision { operation: &'static str, round: u32 },
    /// Call `todo_write` with a whole list.
    TodoWrite(&'static str),
    /// Finish with plain text.
    Text(&'static str),
}

/// A scripted provider that records what the summarizer was actually sent.
struct MockProvider {
    script: Mutex<Vec<Turn>>,
    /// Every message list handed to `complete_aligned`, in call order.
    aligned_requests: Mutex<Vec<Vec<Message>>>,
    /// Every message list handed to `stream_chat`, in call order.
    stream_requests: Mutex<Vec<Vec<Message>>>,
    summary_text: String,
}

impl MockProvider {
    fn new(script: Vec<Turn>, summary_text: &str) -> Self {
        MockProvider {
            script: Mutex::new(script.into_iter().rev().collect()),
            aligned_requests: Mutex::new(Vec::new()),
            stream_requests: Mutex::new(Vec::new()),
            summary_text: summary_text.to_string(),
        }
    }

    fn aligned_requests(&self) -> Vec<Vec<Message>> {
        self.aligned_requests.lock().unwrap().clone()
    }

    fn stream_requests(&self) -> Vec<Vec<Message>> {
        self.stream_requests.lock().unwrap().clone()
    }
}

/// A comparable projection of a message: everything that reaches the wire.
fn wire_shape(msg: &Message) -> (String, String, Option<String>, Vec<String>) {
    (
        msg.role.clone(),
        msg.content.clone(),
        msg.tool_call_id.clone(),
        msg.tool_calls
            .as_ref()
            .map(|calls| calls.iter().map(|c| format!("{}:{}", c.id, c.arguments)).collect())
            .unwrap_or_default(),
    )
}

#[async_trait]
impl ChatProvider for MockProvider {
    async fn stream_chat(
        &self,
        _model: &str,
        messages: &[Message],
        _signal: tokio::sync::watch::Receiver<bool>,
    ) -> Result<mpsc::Receiver<StreamEvent>, AgentError> {
        self.stream_requests.lock().unwrap().push(messages.to_vec());
        let turn = self.script.lock().unwrap().pop();
        let (tx, rx) = mpsc::channel(16);
        match turn {
            Some(Turn::ToolCall { output_bytes }) => {
                let call = ToolCall {
                    id: format!("call_{}", output_bytes),
                    name: "bash".to_string(),
                    arguments: serde_json::json!({
                        "command": format!("printf 'z%.0s' $(seq 1 {})", output_bytes)
                    }),
                    raw_arguments: None,
                };
                let _ = tx
                    .send(StreamEvent::ToolCalls {
                        calls: vec![call],
                        content: "inspecting the parser".to_string(),
                        input_tokens: 10,
                        output_tokens: 5,
                        cost: None,
                        finish_reason: Some("tool_calls".to_string()),
                        reasoning_content: String::new(),
                    })
                    .await;
            }
            Some(Turn::GoalDecision { operation, round }) => {
                let call = ToolCall {
                    id: format!("goal_{}_{}", operation, round),
                    name: "goal".to_string(),
                    arguments: serde_json::json!({
                        "operation": operation,
                        "round": round,
                        "reason": "a dependency is missing"
                    }),
                    raw_arguments: None,
                };
                let _ = tx
                    .send(StreamEvent::ToolCalls {
                        calls: vec![call],
                        content: "deciding the goal".to_string(),
                        input_tokens: 10,
                        output_tokens: 5,
                        cost: None,
                        finish_reason: Some("tool_calls".to_string()),
                        reasoning_content: String::new(),
                    })
                    .await;
            }
            Some(Turn::TodoWrite(payload)) => {
                let call = ToolCall {
                    id: "todo_1".to_string(),
                    name: "todo_write".to_string(),
                    arguments: serde_json::from_str(payload).expect("valid todo payload"),
                    raw_arguments: None,
                };
                let _ = tx
                    .send(StreamEvent::ToolCalls {
                        calls: vec![call],
                        content: "planning".to_string(),
                        input_tokens: 10,
                        output_tokens: 5,
                        cost: None,
                        finish_reason: Some("tool_calls".to_string()),
                        reasoning_content: String::new(),
                    })
                    .await;
            }
            other => {
                let text = match other {
                    Some(Turn::Text(t)) => t,
                    _ => "done",
                };
                let _ = tx.send(StreamEvent::Delta(text.to_string())).await;
                let _ = tx
                    .send(StreamEvent::Done(rupi::provider::StreamResult {
                        content: text.to_string(),
                        input_tokens: 10,
                        output_tokens: 5,
                        cost: None,
                        reasoning_content: String::new(),
                    }))
                    .await;
            }
        }
        Ok(rx)
    }

    async fn complete(&self, _model: &str, messages: &[Message]) -> Result<String, AgentError> {
        self.aligned_requests.lock().unwrap().push(messages.to_vec());
        Ok(self.summary_text.clone())
    }

    async fn complete_aligned(
        &self,
        _model: &str,
        messages: &[Message],
    ) -> Result<String, AgentError> {
        self.aligned_requests.lock().unwrap().push(messages.to_vec());
        Ok(self.summary_text.clone())
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            provider: "mock".to_string(),
            id: "mock-model".to_string(),
            context_window: 2000,
            reasoning: false,
        }
    }
}

/// A session wired to `provider`, with a context window small enough that any real
/// conversation trips compaction.
fn session_with(provider: Arc<MockProvider>) -> AgentSession {
    session_with_window(provider, 2000)
}

/// A session wired to `provider` with an explicit context window.
///
/// Tests about goal rounds and anchor re-emission need a window wide enough that
/// compaction does not fire and rewrite the history they are counting.
fn session_with_window(provider: Arc<MockProvider>, context_window: u64) -> AgentSession {
    test_sessions_dir();
    AgentSession::new(
        provider as Arc<dyn ChatProvider>,
        "mock-model".to_string(),
        context_window,
        std::env::temp_dir().to_string_lossy().to_string(),
        Vec::new(),
        Vec::new(),
    )
}

/// How many goal-round blocks the conversation carries.
fn goal_rounds(messages: &[Message]) -> usize {
    messages.iter().filter(|m| m.content.starts_with("<goal_round>")).count()
}

/// Run one prompt to completion, discarding the event stream.
async fn run_prompt(session: &AgentSession, prompt: &str) {
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let _ = session.prompt(prompt, tx).await;
    drop(drain);
}

/// Assert that a message list is a request every OpenAI-compatible endpoint accepts.
///
/// Two rules: a `tool` message must answer an immediately preceding assistant call,
/// and an assistant call must be answered. A compaction that breaks either one is a
/// hard 400 on the very next turn.
fn assert_well_formed(messages: &[Message], label: &str) {
    let mut expecting: Vec<String> = Vec::new();
    for (index, msg) in messages.iter().enumerate() {
        if msg.role == "tool" {
            let id = msg.tool_call_id.clone().unwrap_or_default();
            assert!(
                expecting.contains(&id),
                "{}: message {} is an orphan tool result (id {:?})",
                label,
                index,
                id
            );
            expecting.retain(|open| open != &id);
            continue;
        }
        assert!(
            expecting.is_empty(),
            "{}: message {} follows {} unanswered tool call(s)",
            label,
            index,
            expecting.len()
        );
        if let Some(ref calls) = msg.tool_calls {
            expecting = calls.iter().map(|c| c.id.clone()).collect();
        }
    }
    assert!(
        expecting.is_empty(),
        "{}: the list ends with {} unanswered tool call(s)",
        label,
        expecting.len()
    );
}

// ---------------------------------------------------------------------------
// The core fix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_original_request_survives_compaction_verbatim() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("finished the refactor"),
        ],
        // A summary that deliberately does NOT restate the request. Before the
        // anchor, this is exactly the case where the prompt was lost for good.
        "## Primary Request and Intent\n- (the model forgot to restate it)\n",
    ));
    let session = session_with(provider.clone());

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    assert!(
        messages[0].content.starts_with("[Compacted conversation history]"),
        "expected a checkpoint at the head, got: {}",
        &messages[0].content.chars().take(80).collect::<String>()
    );

    // The anchor sits directly below the checkpoint and carries the exact bytes.
    let anchor = &messages[1];
    assert_eq!(
        rupi::anchor::extract_request(&anchor.content).as_deref(),
        Some(ORIGINAL_REQUEST),
        "the anchor must carry the request verbatim"
    );
    assert_eq!(session.task_anchor().await.as_deref(), Some(ORIGINAL_REQUEST));

    // And the request text is present in the conversation the model will see,
    // even though the summary dropped it.
    let visible = messages
        .iter()
        .map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(visible.contains(ORIGINAL_REQUEST));
}

#[tokio::test]
async fn the_request_survives_repeated_compactions() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("first pass done"),
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("second pass done"),
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("third pass done"),
        ],
        "## Primary Request and Intent\n- (still not restated)\n",
    ));
    let session = session_with(provider.clone());

    run_prompt(&session, ORIGINAL_REQUEST).await;
    // Follow-ups are the user speaking, so they re-anchor. A host-written block
    // never does; that is what the next two turns exercise.
    run_prompt(&session, ORIGINAL_REQUEST).await;
    run_prompt(&session, ORIGINAL_REQUEST).await;

    assert!(
        provider.aligned_requests().len() >= 2,
        "expected more than one compaction, got {}",
        provider.aligned_requests().len()
    );

    let messages = session.messages().await;
    let anchor = messages
        .iter()
        .find(|m| rupi::anchor::is_anchor(&m.content))
        .expect("an anchor must survive every compaction");
    assert_eq!(
        rupi::anchor::extract_request(&anchor.content).as_deref(),
        Some(ORIGINAL_REQUEST),
        "the request must not degrade across compactions"
    );
}

#[tokio::test]
async fn a_checkpoint_never_becomes_the_anchor() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("done"),
        ],
        "## Primary Request and Intent\n- something else entirely\n",
    ));
    let session = session_with(provider);
    run_prompt(&session, ORIGINAL_REQUEST).await;

    // The checkpoint and the anchor both arrive on the `user` role, because
    // providers accept one system message. Only the real request may anchor.
    assert_eq!(session.task_anchor().await.as_deref(), Some(ORIGINAL_REQUEST));

    session.set_task_anchor("[Compacted conversation history]\nsummary text").await;
    session.set_task_anchor(&rupi::anchor::render("a re-emission", 2)).await;
    session.set_task_anchor("   ").await;
    assert_eq!(
        session.task_anchor().await.as_deref(),
        Some(ORIGINAL_REQUEST),
        "host-written blocks must not overwrite the anchor"
    );
}

#[tokio::test]
async fn a_resumed_session_recovers_the_anchor_from_disk() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("done"),
        ],
        "## Primary Request and Intent\n- (not restated)\n",
    ));
    let session = session_with(provider);
    run_prompt(&session, ORIGINAL_REQUEST).await;

    let path = session.session_path().await.expect("a session file");
    let replayed = rupi::sessions::load_session(&path).expect("the transcript must load");

    // Replay drops everything above the compaction record, so the anchor is only
    // recoverable because it is persisted after that record.
    assert!(
        replayed.iter().any(|m| rupi::anchor::is_anchor(&m.content)),
        "the anchor must be persisted below the compaction record"
    );
    assert_eq!(
        recover_anchor(&replayed).as_deref(),
        Some(ORIGINAL_REQUEST),
        "resume must restore the exact request"
    );
}

#[tokio::test]
async fn recover_anchor_falls_back_to_the_first_real_user_message() {
    // A transcript from before this change: no anchor block anywhere.
    let messages = vec![
        Message::new("user", ORIGINAL_REQUEST),
        Message::new("assistant", "working on it"),
    ];
    assert_eq!(recover_anchor(&messages).as_deref(), Some(ORIGINAL_REQUEST));

    // Checkpoints are host-written and must be skipped.
    let compacted = vec![
        Message::new("user", "[Compacted conversation history]\nsummary"),
        Message::new("user", ORIGINAL_REQUEST),
    ];
    assert_eq!(recover_anchor(&compacted).as_deref(), Some(ORIGINAL_REQUEST));

    assert_eq!(recover_anchor(&[]), None);
}

// ---------------------------------------------------------------------------
// Cache-aligned summarization
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_summarizer_call_is_a_prefix_of_the_real_request() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("done"),
        ],
        "## Summary\n- did the work\n",
    ));
    let session = session_with(provider.clone());
    run_prompt(&session, ORIGINAL_REQUEST).await;

    let requests = provider.aligned_requests();
    let request = requests.first().expect("compaction must have run");

    // The conversation's own system prompt leads, so the provider can match the
    // cached prefix instead of re-prefilling the whole span.
    assert_eq!(request[0].role, "system");
    assert!(
        request[0].content.contains("expert coding agent operating inside rupi"),
        "the summarizer must reuse the conversation's own system prompt"
    );

    // The instruction is LAST, after the replayed region — never a separate
    // summarizer system prompt, which would break the prefix match.
    let last = request.last().unwrap();
    assert_eq!(last.role, "user");
    assert_eq!(last.content, rupi::compaction::COMPACTION_INSTRUCTION);

    // The region is replayed verbatim: the real user message is still in there.
    assert!(
        request[1..request.len() - 1]
            .iter()
            .any(|m| m.content.contains(ORIGINAL_REQUEST)),
        "the region must be replayed verbatim, not serialized into a blob"
    );

    // And what was replayed is itself a valid request.
    assert_well_formed(&request[..request.len() - 1], "summarizer request");

    // The strongest form of the claim: everything before the trailing instruction
    // is a LITERAL prefix of the request the model was last streamed. Anything less
    // than literal — a reordered field, a snipped tool result, a different system
    // prompt — and the provider re-prefills the whole span instead of serving it
    // from cache.
    let streamed = provider.stream_requests();
    let last_streamed = streamed.last().expect("at least one streamed request");
    let replayed = &request[..request.len() - 1];
    assert!(
        replayed.len() <= last_streamed.len(),
        "the replayed region cannot be longer than the request it came from"
    );
    for (index, (replay, sent)) in replayed.iter().zip(last_streamed.iter()).enumerate() {
        assert_eq!(
            wire_shape(replay),
            wire_shape(sent),
            "message {} diverges from the streamed request, so the cached prefix is lost",
            index
        );
    }
}

#[tokio::test]
async fn the_checkpoint_is_framed_and_tagged() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("done"),
        ],
        "## Summary\n- did the work\n",
    ));
    let session = session_with(provider);
    run_prompt(&session, ORIGINAL_REQUEST).await;

    let head = session.messages().await[0].content.clone();
    assert!(head.contains(rupi::compaction::CHECKPOINT_PREAMBLE));
    assert!(head.contains(rupi::compaction::SUMMARY_OPEN_TAG));
    assert!(head.contains(rupi::compaction::SUMMARY_CLOSE_TAG));
    assert!(head.contains("did the work"));
}

#[tokio::test]
async fn a_later_compaction_sees_the_prior_checkpoint() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("one"),
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("two"),
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("three"),
        ],
        "## Summary\n- did the work\n",
    ));
    let session = session_with(provider.clone());
    run_prompt(&session, ORIGINAL_REQUEST).await;
    run_prompt(&session, "keep going").await;
    run_prompt(&session, "keep going").await;

    let requests = provider.aligned_requests();
    assert!(requests.len() >= 2, "expected at least two compactions");

    // A later region carries the earlier checkpoint, so the merge rule in the
    // instruction has something to act on rather than re-condensing blindly.
    let later = requests.last().unwrap();
    assert!(
        rupi::compaction::region_contains_checkpoint(&later[..later.len() - 1]),
        "the second compaction must replay the first checkpoint"
    );
}

// ---------------------------------------------------------------------------
// Tool pairing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compaction_leaves_a_valid_request_on_both_sides() {
    // One tool turn never reaches a compaction, and that is the point: the loop
    // covers both the declining case and every case that does compact.
    for tool_turns in 1..=6 {
        let mut script: Vec<Turn> = (0..tool_turns)
            .map(|_| Turn::ToolCall { output_bytes: 5000 })
            .collect();
        script.push(Turn::Text("done"));

        let provider = Arc::new(MockProvider::new(script, "## Summary\n- work\n"));
        let session = session_with(provider.clone());
        run_prompt(&session, ORIGINAL_REQUEST).await;

        let label = format!("{} tool turns", tool_turns);
        assert_well_formed(&session.messages().await, &format!("tail after {}", label));
        for request in provider.aligned_requests() {
            assert_well_formed(&request[..request.len() - 1], &format!("head after {}", label));
        }
    }
}

// ---------------------------------------------------------------------------
// Spill
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oversized_tool_output_stays_reachable() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 9000 },
            Turn::Text("done"),
        ],
        "## Summary\n- work\n",
    ));
    let session = session_with(provider);
    run_prompt(&session, ORIGINAL_REQUEST).await;

    // The transcript holds the truncated copy plus a path, not the whole payload.
    let path = session.session_path().await.expect("a session file");
    let transcript = std::fs::read_to_string(&path).expect("the transcript must be readable");
    assert!(
        transcript.contains("were saved to"),
        "a truncated tool result must name its spill artifact"
    );

    // The named file exists and holds more than the truncated copy did.
    let spilled = transcript
        .split("were saved to ")
        .nth(1)
        .and_then(|rest| rest.split(". Use the read tool").next())
        .expect("the hint must contain a path");
    let full = std::fs::read_to_string(spilled).expect("the spill artifact must exist");
    assert!(
        full.len() > 8000,
        "the spill artifact must hold the full output, got {} bytes",
        full.len()
    );
}

// ---------------------------------------------------------------------------
// Goal rounds and completion authority
// ---------------------------------------------------------------------------

#[tokio::test]
async fn every_goal_round_states_its_number() {
    let provider = Arc::new(MockProvider::new(
        (0..6).map(|_| Turn::Text("still working")).collect(),
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;
    session.set_goal(Some("make the tests pass".to_string())).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    // Five rounds, including the first: the model needs the round number from the
    // start, because the goal tool rejects a decision that names any other round.
    assert_eq!(goal_rounds(&messages), 5);
    assert!(messages.iter().any(|m| m.content.contains("Round: 1/5")));
    assert!(messages.iter().any(|m| m.content.contains("Round: 5/5")));
    assert!(messages.iter().any(|m| m.content.contains("authoritative")));
}

#[tokio::test]
async fn a_goal_completed_in_its_own_round_stops_the_driver() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::GoalDecision { operation: "complete", round: 1 },
            Turn::Text("the objective is met"),
            Turn::Text("this turn must never run"),
        ],
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;
    session.set_goal(Some("make the tests pass".to_string())).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    assert_eq!(goal_rounds(&messages), 1, "an accepted completion ends the run");
    let tool_result = messages
        .iter()
        .find(|m| m.role == "tool")
        .expect("the goal tool must have run");
    assert!(
        tool_result.content.contains("Goal marked complete in round 1"),
        "unexpected goal tool result: {}",
        tool_result.content
    );
}

#[tokio::test]
async fn a_completion_from_the_wrong_round_is_rejected() {
    let mut script = vec![Turn::GoalDecision { operation: "complete", round: 99 }];
    script.extend((0..8).map(|_| Turn::Text("still working")));
    let provider = Arc::new(MockProvider::new(script, "NO"));

    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;
    session.set_goal(Some("make the tests pass".to_string())).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    let tool_result = messages
        .iter()
        .find(|m| m.role == "tool")
        .expect("the goal tool must have run");
    assert!(
        tool_result.content.contains("round 1 is the open one"),
        "a call from outside the open round must be rejected: {}",
        tool_result.content
    );
    // Rejected, so the driver kept going instead of stopping on the model's word.
    assert_eq!(goal_rounds(&messages), 5);
}

#[tokio::test]
async fn a_blocked_goal_stops_the_driver_and_records_the_reason() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::GoalDecision { operation: "block", round: 1 },
            Turn::Text("cannot proceed"),
            Turn::Text("this turn must never run"),
        ],
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;
    session.set_goal(Some("make the tests pass".to_string())).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    assert_eq!(goal_rounds(&messages), 1);
    assert!(messages
        .iter()
        .any(|m| m.role == "tool" && m.content.contains("Goal marked blocked in round 1")));
}

#[tokio::test]
async fn the_goal_tool_is_rejected_when_no_goal_is_set() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::GoalDecision { operation: "complete", round: 1 },
            Turn::Text("done"),
        ],
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    assert_eq!(goal_rounds(&messages), 0);
    assert!(messages
        .iter()
        .any(|m| m.role == "tool" && m.content.contains("no goal is set")));
}

// ---------------------------------------------------------------------------
// Anchor re-emission at the tail
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_long_tool_run_re_states_the_request_at_the_tail() {
    // Forty-one tool turns: past the re-emission threshold, with no compaction to
    // do the job instead.
    let mut script: Vec<Turn> = (0..41).map(|_| Turn::ToolCall { output_bytes: 4 }).collect();
    script.push(Turn::Text("done"));
    let provider = Arc::new(MockProvider::new(script, "NO"));

    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let messages = session.messages().await;
    let anchors: Vec<&Message> = messages
        .iter()
        .filter(|m| rupi::anchor::is_anchor(&m.content))
        .collect();
    assert_eq!(anchors.len(), 1, "exactly one re-emission at 41 tool results");
    assert_eq!(
        rupi::anchor::extract_request(&anchors[0].content).as_deref(),
        Some(ORIGINAL_REQUEST)
    );
    assert_well_formed(&messages, "after a long tool run");
}

#[tokio::test]
async fn a_short_tool_run_does_not_re_state_the_request() {
    let mut script: Vec<Turn> = (0..5).map(|_| Turn::ToolCall { output_bytes: 4 }).collect();
    script.push(Turn::Text("done"));
    let provider = Arc::new(MockProvider::new(script, "NO"));

    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    assert!(
        !session
            .messages()
            .await
            .iter()
            .any(|m| rupi::anchor::is_anchor(&m.content)),
        "a short run must not pay for a reminder"
    );
}

// ---------------------------------------------------------------------------
// Todo list
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_todo_list_is_whole_list_replacement() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::TodoWrite(
                r#"{"todos":[{"content":"read the parser","status":"in_progress"},
                             {"content":"write the fix","status":"pending"}]}"#,
            ),
            Turn::TodoWrite(
                r#"{"todos":[{"content":"read the parser","status":"completed"},
                             {"content":"write the fix","status":"in_progress"}]}"#,
            ),
            Turn::Text("done"),
        ],
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let results: Vec<String> = session
        .messages()
        .await
        .iter()
        .filter(|m| m.role == "tool")
        .map(|m| m.content.clone())
        .collect();
    assert_eq!(results.len(), 2);
    assert!(results[0].contains("0/2 complete"));
    assert!(results[0].contains("[~] read the parser"));
    assert!(results[1].contains("1/2 complete"));
    assert!(results[1].contains("[x] read the parser"));
    assert!(results[1].contains("[~] write the fix"));
}

#[tokio::test]
async fn a_second_active_todo_is_rejected() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::TodoWrite(
                r#"{"todos":[{"content":"one","status":"in_progress"},
                             {"content":"two","status":"in_progress"}]}"#,
            ),
            Turn::Text("done"),
        ],
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let result = session
        .messages()
        .await
        .into_iter()
        .find(|m| m.role == "tool")
        .expect("the todo tool must have run")
        .content;
    assert!(result.contains("AT MOST ONE"), "unexpected result: {}", result);
}

#[tokio::test]
async fn the_anchor_reminder_carries_the_current_plan() {
    let mut script: Vec<Turn> = vec![Turn::TodoWrite(
        r#"{"todos":[{"content":"finish the refactor","status":"in_progress"}]}"#,
    )];
    script.extend((0..40).map(|_| Turn::ToolCall { output_bytes: 4 }));
    script.push(Turn::Text("done"));
    let provider = Arc::new(MockProvider::new(script, "NO"));

    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;

    let reminder = session
        .messages()
        .await
        .into_iter()
        .find(|m| rupi::anchor::is_anchor(&m.content))
        .expect("a reminder must have been emitted");
    // The anchor states the goal; the plan states where the work stands.
    assert!(reminder.content.contains(ORIGINAL_REQUEST));
    assert!(reminder.content.contains("<todo-list>"));
    assert!(reminder.content.contains("[~] finish the refactor"));
}

#[tokio::test]
async fn snipping_does_not_corrupt_the_replayed_region() {
    // Eight exchanges, each with a tool result well over the snip threshold. That
    // is enough history for snip to fire on the oldest turns.
    let mut script: Vec<Turn> = Vec::new();
    for _ in 0..8 {
        script.push(Turn::ToolCall { output_bytes: 600 });
        script.push(Turn::Text("step done"));
    }
    let provider = Arc::new(MockProvider::new(script, "## Summary\n- work\n"));

    let session = session_with_window(provider.clone(), 2000);
    session.set_auto_compaction_enabled(false).await;
    for _ in 0..8 {
        run_prompt(&session, ORIGINAL_REQUEST).await;
    }

    let before = session.messages().await;

    // Confirm the setup really does give snip something to remove. Without this the
    // test could pass on a history that was never snippable.
    let mut probe = before.clone();
    let removed = rupi::compaction::snip_old_tool_results(&mut probe, 6);
    assert!(removed > 0, "the setup must produce snippable history");

    session.compact().await.expect("compaction must run");

    let requests = provider.aligned_requests();
    let request = requests.last().expect("compaction must have called the summarizer");
    let replayed = &request[1..request.len() - 1]; // drop the system prompt and the instruction

    // The region is the pristine text, not the snipped text. Replaying the snipped
    // copy would still summarize correctly, but it would no longer match the prefix
    // the provider cached, and the whole span would be re-prefilled.
    for (index, msg) in replayed.iter().enumerate() {
        assert!(
            !msg.content.contains("[snip:"),
            "replayed message {} carries a snip marker, so the cached prefix is lost",
            index
        );
        assert_eq!(
            wire_shape(msg),
            wire_shape(&before[index]),
            "replayed message {} diverges from the stored history",
            index
        );
    }

    // The kept tail still benefits from the snip, which is the point of running it.
    let after = session.messages().await;
    assert!(after.len() < before.len());
    assert_eq!(
        rupi::anchor::extract_request(&after[1].content).as_deref(),
        Some(ORIGINAL_REQUEST)
    );
}

#[tokio::test]
async fn a_steer_joins_the_anchor_instead_of_replacing_it() {
    let provider = Arc::new(MockProvider::new(
        vec![
            Turn::ToolCall { output_bytes: 6000 },
            Turn::ToolCall { output_bytes: 6000 },
            Turn::Text("done"),
        ],
        "## Primary Request and Intent\n- (not restated)\n",
    ));
    let session = session_with(provider);

    session.set_task_anchor(ORIGINAL_REQUEST).await;
    session.steer("also keep the public API unchanged").await;

    let anchor = session.task_anchor().await.expect("an anchor");
    assert!(anchor.contains(ORIGINAL_REQUEST), "the original request must survive a steer");
    assert!(anchor.contains("also keep the public API unchanged"));

    // A follow-up is the user too, so it joins as well.
    session.follow_up("and add a regression test").await;
    let anchor = session.task_anchor().await.expect("an anchor");
    assert!(anchor.contains(ORIGINAL_REQUEST));
    assert!(anchor.contains("also keep the public API unchanged"));
    assert!(anchor.contains("and add a regression test"));

    // Host-written blocks still never touch it.
    session.append_task_anchor("[Compacted conversation history]\nsummary").await;
    session.append_task_anchor(&rupi::anchor::render("a re-emission", 1)).await;
    let after = session.task_anchor().await.expect("an anchor");
    assert!(!after.contains("[Compacted conversation history]"));
    assert!(!after.contains("a re-emission"));
}

#[tokio::test]
async fn a_steered_task_is_carried_through_a_compaction() {
    let mut script: Vec<Turn> = Vec::new();
    for _ in 0..3 {
        script.push(Turn::ToolCall { output_bytes: 600 });
        script.push(Turn::Text("step done"));
    }
    let provider = Arc::new(MockProvider::new(script, "## Primary Request and Intent\n- (not restated)\n"));

    let session = session_with_window(provider, 2000);
    session.set_auto_compaction_enabled(false).await;
    for _ in 0..3 {
        run_prompt(&session, ORIGINAL_REQUEST).await;
    }

    // The user redirects the work in flight. A new top-level prompt would start a
    // new task and replace the anchor; a steer refines this one, so it joins.
    session.steer("also keep the public API unchanged").await;
    session.follow_up("and add a regression test").await;

    session.compact().await.expect("compaction must run");

    let messages = session.messages().await;
    let anchor = messages
        .iter()
        .find(|m| rupi::anchor::is_anchor(&m.content))
        .expect("an anchor must survive the compaction");
    assert!(anchor.content.contains(ORIGINAL_REQUEST));
    assert!(anchor.content.contains("also keep the public API unchanged"));
    assert!(anchor.content.contains("and add a regression test"));
}

#[tokio::test]
async fn a_new_top_level_prompt_starts_a_new_task() {
    let provider = Arc::new(MockProvider::new(
        vec![Turn::Text("one"), Turn::Text("two")],
        "NO",
    ));
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;

    run_prompt(&session, ORIGINAL_REQUEST).await;
    assert_eq!(session.task_anchor().await.as_deref(), Some(ORIGINAL_REQUEST));

    // A prompt sent while the agent is idle is a new task, not a refinement of the
    // last one, so it replaces rather than accumulates.
    run_prompt(&session, "now document the module").await;
    assert_eq!(session.task_anchor().await.as_deref(), Some("now document the module"));
}

#[tokio::test]
async fn many_follow_ups_do_not_grow_the_anchor_without_bound() {
    let provider = Arc::new(MockProvider::new(vec![Turn::Text("done")], "NO"));
    let session = session_with_window(provider, 400_000);

    session.set_task_anchor(ORIGINAL_REQUEST).await;
    for i in 0..400 {
        session
            .follow_up(&format!("refinement number {} with some padding text to add bulk", i))
            .await;
    }

    let anchor = session.task_anchor().await.expect("an anchor");
    assert!(
        anchor.len() <= rupi::anchor::MAX_ANCHOR_CHARS + 200,
        "the stored anchor grew to {} chars",
        anchor.len()
    );
    // Head and tail both survive: the request that started the work, and the
    // instruction that is most current.
    assert!(anchor.contains(ORIGINAL_REQUEST));
    assert!(anchor.contains("refinement number 399"));
}

// ---------------------------------------------------------------------------
// Prompt caching
// ---------------------------------------------------------------------------

/// Where `later` stops matching `earlier`, or `None` when it extends it cleanly.
fn first_divergence(earlier: &[Message], later: &[Message]) -> Option<usize> {
    if later.len() < earlier.len() {
        return Some(later.len());
    }
    earlier
        .iter()
        .zip(later.iter())
        .position(|(a, b)| wire_shape(a) != wire_shape(b))
}

#[tokio::test]
async fn every_request_extends_the_previous_one() {
    // The invariant behind automatic prefix caching on vLLM, SGLang, OpenAI, and
    // DeepSeek: nothing already sent may ever change. Editing one old message
    // invalidates the cache from that point to the end of the conversation, which is
    // a full re-prefill of everything after it.
    let mut script: Vec<Turn> = (0..25).map(|_| Turn::ToolCall { output_bytes: 300 }).collect();
    script.push(Turn::Text("done"));
    let provider = Arc::new(MockProvider::new(script, "NO"));

    let session = session_with_window(provider.clone(), 400_000);
    session.set_auto_compaction_enabled(false).await;
    run_prompt(&session, ORIGINAL_REQUEST).await;

    let requests = provider.stream_requests();
    assert!(requests.len() >= 25, "expected a long tool run, got {}", requests.len());
    for (turn, pair) in requests.windows(2).enumerate() {
        assert_eq!(
            first_divergence(&pair[0], &pair[1]),
            None,
            "turn {} rewrote history instead of appending to it",
            turn
        );
    }
}

#[tokio::test]
async fn the_system_prompt_is_byte_identical_on_every_turn() {
    let mut script: Vec<Turn> = (0..6).map(|_| Turn::ToolCall { output_bytes: 300 }).collect();
    script.push(Turn::Text("done"));
    let provider = Arc::new(MockProvider::new(script, "NO"));

    let session = session_with_window(provider.clone(), 400_000);
    session.set_auto_compaction_enabled(false).await;
    run_prompt(&session, ORIGINAL_REQUEST).await;
    run_prompt(&session, "and now the other thing").await;

    let requests = provider.stream_requests();
    let first = requests[0][0].content.clone();
    assert_eq!(requests[0][0].role, "system");
    for (turn, request) in requests.iter().enumerate() {
        assert_eq!(request[0].role, "system");
        assert_eq!(request[0].content, first, "the system prompt changed on turn {}", turn);
    }
}

#[tokio::test]
async fn a_declined_compaction_leaves_the_history_untouched() {
    // compact() runs a snip pass to size the kept tail. It used to apply that snip to
    // the live history BEFORE the guards that can decline, so a declined run rewrote
    // message bodies the provider had cached and bought nothing for it.
    let mut script: Vec<Turn> = Vec::new();
    for _ in 0..8 {
        script.push(Turn::ToolCall { output_bytes: 600 });
        script.push(Turn::Text("step done"));
    }
    let provider = Arc::new(MockProvider::new(script, "NO"));

    // A window wide enough that should_compact says no.
    let session = session_with_window(provider, 400_000);
    session.set_auto_compaction_enabled(false).await;
    for _ in 0..8 {
        run_prompt(&session, ORIGINAL_REQUEST).await;
    }

    let before = session.messages().await;
    // Confirm the history really is snippable, so this cannot pass vacuously.
    let mut probe = before.clone();
    assert!(rupi::compaction::snip_old_tool_results(&mut probe, 6) > 0);

    assert!(session.compact().await.is_err(), "compaction must decline here");

    let after = session.messages().await;
    assert_eq!(after.len(), before.len());
    for (index, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(
            wire_shape(a),
            wire_shape(b),
            "a declined compaction rewrote message {}, invalidating the cache",
            index
        );
    }
}

#[tokio::test]
async fn only_a_compaction_breaks_the_prefix() {
    let mut script: Vec<Turn> = Vec::new();
    for _ in 0..6 {
        script.push(Turn::ToolCall { output_bytes: 3000 });
        script.push(Turn::Text("step done"));
    }
    let provider = Arc::new(MockProvider::new(script, "## Summary\n- work\n"));

    let session = session_with_window(provider.clone(), 2000);
    for _ in 0..6 {
        run_prompt(&session, ORIGINAL_REQUEST).await;
    }

    let compactions = provider.aligned_requests().len();
    assert!(compactions > 0, "this run must compact");

    let requests = provider.stream_requests();
    let breaks = requests
        .windows(2)
        .filter(|pair| first_divergence(&pair[0], &pair[1]).is_some())
        .count();
    // A compaction rewrites the head, so the prefix legitimately dies there. Nothing
    // else may break it, so the count can never exceed the number of compactions.
    assert!(
        breaks <= compactions,
        "the prefix broke {} times for {} compactions",
        breaks,
        compactions
    );
}
