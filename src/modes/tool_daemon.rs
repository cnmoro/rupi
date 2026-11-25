//! Tool calls executed here, for an agent running somewhere else.
//!
//! The remote half of "the agent runs here, the tools run there". Pi ships this
//! as a separate Rust daemon deployed over SSH; rupi is already a single Rust
//! binary that speaks JSONL on stdin and stdout, so it is its own daemon.
//!
//! One JSON object per line in, one per line out:
//!
//! ```text
//! {"id":"1","name":"read","arguments":{"file_path":"src/main.rs"}}
//! {"id":"1","result":"...the file..."}
//! ```
//!
//! The daemon holds no conversation, no provider and no credentials. It runs
//! tools with the privileges of wherever it runs, which is the whole point and
//! also the whole risk: run it inside the boundary you want the tools to have.

use crate::tools::{self, ToolCall, ToolContext};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// One request from an agent.
#[derive(serde::Deserialize)]
struct ToolRequest {
    id: String,
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

/// Serve tool calls on stdin and stdout until stdin closes.
pub async fn serve_stdio() {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = tokio::io::stdout();
    // One context for the whole daemon: the plan and the goal belong to the agent
    // that drives it, not here, and the tools that use them are rejected below.
    let context = ToolContext::new();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<ToolRequest>(line) {
            Ok(request) => handle(request, &context).await,
            // A malformed line gets an answer too. Silence would leave the agent
            // waiting for a response that is never coming.
            Err(e) => serde_json::json!({
                "id": serde_json::Value::Null,
                "error": format!("could not read the request: {}", e),
            }),
        };
        let encoded = crate::rpc::jsonl::serialize_json_line(&answer);
        if stdout.write_all(encoded.as_bytes()).await.is_err() {
            break;
        }
        let _ = stdout.flush().await;
    }
}

/// Tools that mean nothing without the conversation that owns them.
const LOCAL_ONLY: [&str; 2] = ["todo_write", "goal"];

async fn handle(request: ToolRequest, context: &ToolContext) -> serde_json::Value {
    if LOCAL_ONLY.contains(&request.name.as_str()) {
        return serde_json::json!({
            "id": request.id,
            "error": format!("{} belongs to the agent, not to the tool host", request.name),
        });
    }
    let call = ToolCall {
        id: request.id.clone(),
        name: request.name,
        arguments: request.arguments,
        raw_arguments: None,
    };
    // No approval callback: approval is the agent's decision, and it is taken
    // before the call is sent.
    let result = tools::execute_tool_async(call, context.clone(), None).await;
    serde_json::json!({ "id": request.id, "result": result })
}
