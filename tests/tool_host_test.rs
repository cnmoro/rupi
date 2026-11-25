//! Tools executed by another process.
//!
//! Driven through the real binary, because the protocol between the two halves is
//! the thing under test. `CARGO_BIN_EXE_rupi` is the binary cargo just built.

use std::sync::Arc;

use rupi::tools::{execute_tool_async, ToolCall, ToolContext};

fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: format!("call_{}", name),
        name: name.to_string(),
        arguments,
        raw_arguments: None,
    }
}

/// Where the tools run is one setting for the whole process, so these tests take
/// turns. Without this they overwrite each other's host and fail at random.
///
/// An async mutex, because each test holds the turn across its awaits.
fn turn() -> &'static tokio::sync::Mutex<()> {
    static TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    &TURN
}

/// The host is this same binary, in daemon mode.
fn use_daemon() {
    rupi::tool_host::set_tool_command(Some(format!(
        "{} --tool-daemon",
        env!("CARGO_BIN_EXE_rupi")
    )));
}

#[tokio::test]
async fn a_tool_runs_on_the_host_and_answers() {
    let _turn = turn().lock().await;
    let dir = std::env::temp_dir().join(format!("rupi-host-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("hello.txt");
    std::fs::write(&file, "written on the host side\n").unwrap();

    use_daemon();
    let context = ToolContext::new();

    let read = execute_tool_async(
        call(
            "read",
            serde_json::json!({ "file_path": file.display().to_string() }),
        ),
        context.clone(),
        None,
    )
    .await;
    assert!(
        read.contains("written on the host side"),
        "the host did not read the file: {}",
        read
    );

    // A second call reuses the same process.
    let shell = execute_tool_async(
        call(
            "bash",
            serde_json::json!({ "command": "echo from the host" }),
        ),
        context.clone(),
        None,
    )
    .await;
    assert!(shell.contains("from the host"), "{}", shell);

    rupi::tool_host::set_tool_command(None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn the_conversation_tools_stay_here() {
    let _turn = turn().lock().await;
    // The plan and the goal belong to the agent. Sending them away would record
    // them in a process that is thrown away after the call.
    use_daemon();
    let context = ToolContext::new();
    let result = execute_tool_async(
        call(
            "todo_write",
            serde_json::json!({ "todos": [{ "content": "first step", "status": "pending" }] }),
        ),
        context.clone(),
        None,
    )
    .await;
    assert!(
        !result.contains("Error from the tool host"),
        "the plan was sent to the host: {}",
        result
    );
    assert!(
        context
            .todos
            .snapshot()
            .iter()
            .any(|item| item.content.contains("first step")),
        "the plan did not land in this process"
    );
    rupi::tool_host::set_tool_command(None);
}

#[tokio::test]
async fn a_host_that_cannot_start_is_reported_not_silent() {
    let _turn = turn().lock().await;
    // A tool that did not run must never read as a tool that returned nothing.
    rupi::tool_host::set_tool_command(Some(
        "exit 7 # this host refuses to speak the protocol".to_string(),
    ));
    let result = execute_tool_async(
        call("bash", serde_json::json!({ "command": "echo nope" })),
        ToolContext::new(),
        None,
    )
    .await;
    assert!(
        result.contains("did not answer") && result.contains("did not run"),
        "a dead host was not reported: {}",
        result
    );
    rupi::tool_host::set_tool_command(None);
}

#[tokio::test]
async fn a_declined_call_is_never_sent_to_the_host() {
    let _turn = turn().lock().await;
    // Approval is decided here, before anything leaves the process.
    use_daemon();
    let approval: rupi::agent::session::ApprovalFn = Arc::new(|_name, _args| false);
    let result = execute_tool_async(
        call(
            "bash",
            serde_json::json!({ "command": "echo should not run" }),
        ),
        ToolContext::new(),
        Some(approval),
    )
    .await;
    assert!(
        result.contains("denied"),
        "the approval gate was bypassed: {}",
        result
    );
    rupi::tool_host::set_tool_command(None);
}

/// A fake host, driven by a tiny shell script, for the failures a real daemon
/// does not produce.
fn fake_host(body: &str) -> String {
    let dir = std::env::temp_dir().join(format!("rupi-fake-host-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let python = body.trim_start().starts_with("import");
    let path = dir.join(format!(
        "host-{}.{}",
        body.len(),
        if python { "py" } else { "sh" }
    ));
    std::fs::write(&path, body).unwrap();
    if python {
        format!("python3 {}", path.display())
    } else {
        format!("sh {}", path.display())
    }
}

#[tokio::test]
async fn an_endless_answer_is_cut_off_instead_of_filling_memory() {
    // A host that writes a line and never ends it. `BufRead::lines` has no cap:
    // this took the agent to 20 GB in 41 seconds, and no deadline could fire
    // because nothing was waiting on time.
    let _turn = turn().lock().await;
    rupi::tool_host::set_tool_command(Some(fake_host(
        "read line; printf '{\"id\":\"x\",\"result\":\"'; while :; do printf 'xxxxxxxxxxxxxxxx'; done",
    )));
    let started = std::time::Instant::now();
    let result = execute_tool_async(
        call("bash", serde_json::json!({ "command": "echo hi" })),
        ToolContext::new(),
        None,
    )
    .await;
    let took = started.elapsed();
    assert!(
        result.contains("did not answer") && result.contains("bytes"),
        "an endless answer was not cut off: {}",
        &result[..result.len().min(200)]
    );
    assert!(
        took < std::time::Duration::from_secs(60),
        "it took {:?} to give up",
        took
    );
    rupi::tool_host::set_tool_command(None);
}

#[tokio::test]
async fn an_abort_reaches_a_tool_running_on_the_host() {
    // One blocking wait made every remote call uncancellable for up to the whole
    // limit: an abort left a remote build running and the caller waiting.
    let _turn = turn().lock().await;
    use_daemon();
    let context = ToolContext::new();
    let flag = context.cancelled.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let started = std::time::Instant::now();
    let result = execute_tool_async(
        call(
            "bash",
            serde_json::json!({ "command": "sleep 20", "timeout": 25 }),
        ),
        context,
        None,
    )
    .await;
    let took = started.elapsed();
    assert!(result.contains("cancelled"), "{}", result);
    assert!(
        took < std::time::Duration::from_secs(5),
        "the abort took {:?} to free the caller",
        took
    );
    rupi::tool_host::set_tool_command(None);
}

#[tokio::test]
async fn a_queued_call_is_not_sent_after_an_abort() {
    // Calls wait their turn behind one another. Without a check after the queue,
    // every queued `write` and `edit` still ran on the host after an abort.
    let _turn = turn().lock().await;
    use_daemon();
    let context = ToolContext::new();
    context
        .cancelled
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("rupi-queued-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let marker = dir.join("written.txt");
    let result = execute_tool_async(
        call(
            "write",
            serde_json::json!({
                "file_path": marker.display().to_string(),
                "content": "this must not reach the host"
            }),
        ),
        context,
        None,
    )
    .await;
    assert!(result.contains("cancelled"), "{}", result);
    assert!(!marker.exists(), "a cancelled write still ran on the host");
    rupi::tool_host::set_tool_command(None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_stale_answer_is_never_given_to_the_next_call() {
    // A host that answers every request twice. With a provider that reuses a call
    // id, the leftover answer used to be handed to the following call.
    let _turn = turn().lock().await;
    rupi::tool_host::set_tool_command(Some(fake_host(
        r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    for result in ("answer for " + request["id"], "stale duplicate"):
        sys.stdout.write(json.dumps({"id": request["id"], "result": result}) + "\n")
        sys.stdout.flush()
"#,
    )));
    let first = execute_tool_async(
        call("bash", serde_json::json!({ "command": "one" })),
        ToolContext::new(),
        None,
    )
    .await;
    assert!(first.starts_with("answer for call_bash#"), "{}", first);
    for round in 0..3 {
        let next = execute_tool_async(
            call("bash", serde_json::json!({ "command": "two" })),
            ToolContext::new(),
            None,
        )
        .await;
        assert_ne!(
            next, "stale duplicate",
            "round {} was answered with a leftover from an earlier call",
            round
        );
    }
    rupi::tool_host::set_tool_command(None);
}

#[tokio::test]
async fn the_plan_does_not_wait_behind_remote_work() {
    // The plan and the goal are in-memory state. A batch of slow remote calls, all
    // serialized, used to starve them of a worker for minutes.
    let _turn = turn().lock().await;
    rupi::tool_host::set_tool_command(Some(fake_host(
        "while read line; do sleep 3; printf '{\"id\":\"call_bash\",\"result\":\"slow\"}\\n'; done",
    )));
    let context = ToolContext::new();
    let mut slow = Vec::new();
    for _ in 0..4 {
        let ctx = context.clone();
        slow.push(tokio::spawn(async move {
            execute_tool_async(
                call("bash", serde_json::json!({ "command": "slow" })),
                ctx,
                None,
            )
            .await
        }));
    }
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let started = std::time::Instant::now();
    let plan = execute_tool_async(
        call(
            "todo_write",
            serde_json::json!({ "todos": [{ "content": "a step", "status": "pending" }] }),
        ),
        context.clone(),
        None,
    )
    .await;
    let took = started.elapsed();
    assert!(!plan.contains("Error"), "{}", plan);
    assert!(
        took < std::time::Duration::from_secs(2),
        "the plan waited {:?} behind remote work",
        took
    );
    context
        .cancelled
        .store(true, std::sync::atomic::Ordering::SeqCst);
    for task in slow {
        let _ = task.await;
    }
    rupi::tool_host::set_tool_command(None);
}

#[tokio::test]
async fn ending_the_host_ends_what_it_started() {
    // Killing the shell alone left its children running: a timeout on
    // `ssh box rupi --tool-daemon` orphaned the remote command, and a
    // `docker exec` host left the container process alive.
    let _turn = turn().lock().await;
    let marker = format!("rupi-orphan-probe-{}", std::process::id());
    // The host starts a long-lived grandchild, then never answers.
    rupi::tool_host::set_tool_command(Some(format!(
        "sh -c 'sleep 240 # {} & cat > /dev/null'",
        marker
    )));
    let context = ToolContext::new();
    let flag = context.cancelled.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let result = execute_tool_async(
        call("bash", serde_json::json!({ "command": "echo hi" })),
        context,
        None,
    )
    .await;
    assert!(result.contains("cancelled"), "{}", result);

    // The grandchild must go with the host. Polled rather than sampled once: under
    // a loaded machine the process table lags the kill by more than a moment.
    let probe = || {
        std::process::Command::new("pgrep")
            .arg("-f")
            .arg(&marker)
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let mut alive = probe();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(8);
    while !alive.is_empty() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        alive = probe();
    }
    assert!(
        alive.is_empty(),
        "the host's child outlived it: pids {}",
        alive
    );
    rupi::tool_host::set_tool_command(None);
}
