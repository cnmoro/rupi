//! One prompt, one answer, then exit.
//!
//! This is the mode one agent uses to start another. The parent runs the binary
//! with `-p "<task>"`, reads the answer off stdout, and never has to speak the
//! event protocol. The subagent gets its own process, its own context and its own
//! session file, so nothing it does can disturb the conversation that started it.

use std::sync::Arc;

use crate::agent::session::AgentSession;
use crate::rpc::types::AgentEvent;
use tokio::sync::mpsc;

/// Stop reasons that mean the turn did not finish.
///
/// An answer from one of these is a fragment: a stream that went silent, one the
/// provider cut off, a turn that hit the token limit, or one that was stopped. A
/// caller reading stdout cannot tell a short answer from half of one, so a turn
/// that ended this way must fail instead of printing what it had.
const INCOMPLETE: [&str; 6] = [
    "timeout",
    "truncated",
    "error",
    "aborted",
    "length",
    "cancelled",
];

/// Run `message` to completion and return the agent's final answer.
///
/// Events are drained rather than printed: stdout carries the answer alone, so a
/// caller can read it straight out of a pipe. Progress still reaches stderr
/// through the messages the agent and the tools already write there.
pub async fn run_once(session: Arc<AgentSession>, message: &str) -> Result<String, String> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    // The last stop reason the turn reported. It is the only place that says
    // whether the text in the conversation is a whole answer.
    let watcher = tokio::spawn(async move {
        let mut last: Option<String> = None;
        while let Some(event) = rx.recv().await {
            if let AgentEvent::MessageEnd { message, .. } = &event {
                if let Some(reason) = &message.stop_reason {
                    last = Some(reason.clone());
                }
            }
        }
        last
    });
    let outcome = session.prompt(message, tx).await;
    let stop_reason = watcher.await.ok().flatten();
    if let Err(e) = outcome {
        return Err(e.to_string());
    }
    if let Some(reason) = stop_reason.as_deref() {
        if INCOMPLETE.contains(&reason) {
            return Err(format!(
                "the turn did not finish ({reason}); the partial text is not reported as an answer"
            ));
        }
    }

    // The last assistant message that is not itself a tool call. A message that
    // carries calls holds the narration around them — "let me check the tests" —
    // which is the work, not the report of it. A turn that ends on one ended
    // early, and printing that narration as the answer would read as a finished
    // report to whatever started this agent.
    let answer = session
        .messages()
        .await
        .into_iter()
        .rev()
        .find(|m| {
            m.role == "assistant"
                && m.tool_calls.as_ref().is_none_or(|calls| calls.is_empty())
                && !m.content.trim().is_empty()
        })
        .map(|m| m.content);
    match answer {
        Some(text) => Ok(text),
        // The turn can end without an answer when the provider failed mid-run: the
        // tool loop reports that on stderr and still returns. Say where to look,
        // rather than leaving a caller with an empty stdout and no reason.
        None => Err("the agent produced no answer; the cause is on stderr".to_string()),
    }
}
