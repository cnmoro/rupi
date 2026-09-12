use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Bounds for the bash tool's per-command timeout, in seconds.
///
/// Two minutes and a 30s default suit interactive coding. Embedders whose tools
/// shell out to longer jobs (media conversion, crawls, browser automation)
/// raise them with `--bash-timeout-max` / `--bash-timeout-default`.
///
/// Both are advertised to the model in the tool schema: it picks a timeout from
/// what the description says is allowed, so raising the ceiling without saying
/// so changes nothing in practice.
static BASH_TIMEOUT_MAX: AtomicU64 = AtomicU64::new(120);
static BASH_TIMEOUT_DEFAULT: AtomicU64 = AtomicU64::new(30);

pub fn set_bash_timeout_max(seconds: u64) {
    BASH_TIMEOUT_MAX.store(seconds.max(1), Ordering::Relaxed);
}

pub fn set_bash_timeout_default(seconds: u64) {
    BASH_TIMEOUT_DEFAULT.store(seconds.max(1), Ordering::Relaxed);
}

fn bash_timeout_max() -> u64 {
    BASH_TIMEOUT_MAX.load(Ordering::Relaxed)
}

fn bash_timeout_default() -> u64 {
    bash_timeout_default_raw().min(bash_timeout_max())
}

fn bash_timeout_default_raw() -> u64 {
    BASH_TIMEOUT_DEFAULT.load(Ordering::Relaxed)
}

/// A tool definition sent to the API.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
}

/// A tool call from the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    /// Raw JSON string when arguments parsing failed (truncation recovery).
    pub raw_arguments: Option<String>,
}

/// Session-scoped state that the stateful tools read and write.
///
/// Most tools are pure functions of their arguments and the filesystem. `todo_write`
/// and `goal` are not: they own conversation state. Passing that state in keeps it
/// owned by the session instead of by the process, so two sessions in one process
/// cannot overwrite each other's plan or objective.
#[derive(Debug, Default, Clone)]
pub struct ToolContext {
    pub cancelled: Arc<AtomicBool>,
    pub todos: std::sync::Arc<crate::todo::TodoStore>,
    pub goal: std::sync::Arc<crate::goal::GoalRegistry>,
}

impl ToolContext {
    /// A context with an empty plan and no goal.
    pub fn new() -> Self {
        ToolContext::default()
    }
}

/// Get all available tool definitions.
pub fn all_tools() -> Vec<ToolDef> {
    vec![
        bash_tool(),
        read_tool(),
        write_tool(),
        edit_tool(),
        grep_tool(),
        find_tool(),
        ls_tool(),
        search_code_tool(),
        todo_write_tool(),
        goal_tool(),
    ]
}

fn bash_tool() -> ToolDef {
    ToolDef {
        name: "bash",
        description: "Execute a bash command. Use it to run shell commands, scripts, compilers, curl, git, or anything else in the terminal. The command runs in the current working directory.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The bash command to execute" },
                "description": { "type": "string", "description": "A short description (for logging)" },
                "timeout": {
                    "type": "number",
                    "description": format!("Timeout in seconds (max {})", bash_timeout_max()),
                    "default": bash_timeout_default()
                }
            },
            "required": ["command"]
        }),
    }
}

fn read_tool() -> ToolDef {
    ToolDef {
        name: "read",
        description: "Read the contents of a file. Supports optional offset and limit to read specific portions. Use this to examine source code, configuration files, logs, and other text files.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The absolute or relative path to the file to read" },
                "offset": { "type": "integer", "description": "Line number to start reading from (1-indexed)", "default": 1 },
                "limit": { "type": "integer", "description": "Maximum number of lines to read", "default": 2000 }
            },
            "required": ["file_path"]
        }),
    }
}

/// Schema for the optional follow-up command carried by `edit` and `write`.
///
/// A file mutation is nearly always followed by a command that checks it: run the
/// tests, build, restart the server. Splitting that pair across two turns costs a
/// full round trip — a whole response, a tool result, and a new request carrying
/// the entire conversation again — for a decision the model has already made.
/// Carrying the command with the mutation removes that turn.
fn then_run_schema(mutation: &str) -> Value {
    serde_json::json!({
        "type": "object",
        "description": format!(
            "Optional command to run immediately after the {} succeeds, in the same call. Use it for the check you would run next: tests, a build, a linter, a restart. The command is SKIPPED if the {} fails. A non-zero exit is reported but the {} is KEPT.",
            mutation, mutation, mutation
        ),
        "properties": {
            "command": { "type": "string", "description": "The bash command to run" },
            "timeout": {
                "type": "number",
                "description": format!("Timeout in seconds (max {})", bash_timeout_max()),
                "default": bash_timeout_default()
            }
        },
        "required": ["command"]
    })
}

fn write_tool() -> ToolDef {
    ToolDef {
        name: "write",
        description: "Create a NEW file with the given content. REFUSES if the file already exists — use edit to modify existing files. Creates parent directories automatically. Pass then_run to check the result in the same call instead of spending another turn on it.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The absolute file path for the new file" },
                "content": { "type": "string", "description": "The full content to write" },
                "then_run": then_run_schema("write")
            },
            "required": ["file_path", "content"]
        }),
    }
}

fn edit_tool() -> ToolDef {
    ToolDef {
        name: "edit",
        description: "Replace exact text in a file. Uses exact string matching (not regex). Each old_text must be found exactly once in the file. Supports batch edits via the edits array. Prefer this over write for any change to an existing file. Pass then_run to check the result in the same call instead of spending another turn on it.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The absolute or relative path to the file to edit" },
                "old_text": { "type": "string", "description": "(Deprecated) Use edits array instead. The exact text to find and replace (must match exactly once)" },
                "new_text": { "type": "string", "description": "(Deprecated) Use edits array instead. The replacement text" },
                "edits": {
                    "type": "array",
                    "description": "Array of edits to apply. Each edit's old_text is matched against the ORIGINAL file content (not after other edits). Edits must not overlap.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": { "type": "string", "description": "Exact text to find (must match exactly once in the file)" },
                            "new_text": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["old_text", "new_text"]
                    }
                },
                "then_run": then_run_schema("edit")
            }
        }),
    }
}

fn grep_tool() -> ToolDef {
    ToolDef {
        name: "grep",
        description: "Search file contents with a regular expression. Start broad with output_mode \"files_with_matches\" to see WHERE the matches are, then search again narrowly, or read the file. A content search that returns hundreds of lines is a question that was too broad: narrow the pattern, the path, or the include glob instead of asking for everything.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "The regex pattern to search for" },
                "include": { "type": "string", "description": "Glob pattern for files to include (e.g. '*.rs', '*.{ts,js}')", "default": "" },
                "path": { "type": "string", "description": "Directory or file to search in (default: current directory)", "default": "." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive search", "default": false },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content: matching lines with line numbers. files_with_matches: just the file paths, much cheaper for a broad search. count: matches per file.",
                    "default": "content"
                },
                "head_limit": {
                    "type": "number",
                    "description": format!("Return at most this many lines (default {}). Raise it only after a narrower search has proved the matches are all wanted.", GREP_DEFAULT_HEAD)
                },
                "context": {
                    "type": "number",
                    "description": "Lines of context around each match, like grep -C. Only applies to output_mode \"content\".",
                    "default": 0
                }
            },
            "required": ["pattern"]
        }),
    }
}

fn find_tool() -> ToolDef {
    ToolDef {
        name: "find",
        description: "Find files matching a glob pattern. Use this to locate files by name pattern, discover project structure, or find configuration files.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern to match (e.g. '**/*.rs', 'src/**/*.ts', '*.json')" },
                "path": { "type": "string", "description": "Directory to search in (default: current directory)", "default": "." }
            },
            "required": ["pattern"]
        }),
    }
}

fn search_code_tool() -> ToolDef {
    ToolDef {
        name: "search_code",
        description: "Search code using natural language queries. Finds relevant code across the codebase by understanding what it does, not just matching keywords. Use this instead of grep when you need to find code by its purpose or behavior. Indexes the codebase on first call (takes ~1-2 seconds), subsequent calls are instant.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "A natural language description of the code you want to find (e.g. 'how is authentication handled', 'database connection code', 'the function that saves files')" },
                "path": { "type": "string", "description": "Directory to search (default: current directory)", "default": "." },
                "top_k": { "type": "integer", "description": "Number of results to return (default: 5)", "default": 5 }
            },
            "required": ["query"]
        }),
    }
}

fn ls_tool() -> ToolDef {
    ToolDef {
        name: "ls",
        description: "List files and directories in a given path. Use this to explore project structure, see what files are in a directory, or verify file existence.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path to list (default: current directory)", "default": "." }
            },
            "required": []
        }),
    }
}

/// Serialize tool definitions to the OpenAI API format.
pub fn serialize_tools(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            })
        })
        .collect()
}

// ---- action fusion ----

/// The fused command ran and exited zero.
pub const THEN_RUN_SUCCEEDED: &str = "[then_run:succeeded]";
/// The fused command ran and did not exit zero. The file mutation is still applied.
pub const THEN_RUN_FAILED: &str = "[then_run:failed]";
/// The fused command was not run, because the mutation it depends on did not happen.
pub const THEN_RUN_SKIPPED: &str = "[then_run:skipped]";
/// Opens the fused command's own output.
pub const THEN_RUN_OUTPUT_OPEN: &str = "--- then_run output ---";
/// Closes the fused command's own output.
pub const THEN_RUN_OUTPUT_CLOSE: &str = "--- end then_run output ---";

/// The exact messages `execute_write` and `execute_edit` use to report a clean,
/// complete mutation. Nothing else counts.
const MUTATION_SUCCESS_PREFIXES: [&str; 2] = ["Successfully wrote ", "Successfully applied "];

/// Whether a mutation message reports a change that reached the disk intact.
///
/// This matches success positively and treats everything else as failure, rather
/// than the other way round. The tempting inverse — anything that does not start
/// with `Error` succeeded — is wrong: `execute_edit` reports an unmatched
/// `old_text` as `Edit failed:`, and a truncated `write` reports `WARNING:` over a
/// half-written file. Both would have been read as success, and the follow-up
/// command would have run against a file that is unchanged or knowingly broken.
///
/// Failing closed also means a future message this function does not recognize
/// skips the command rather than running it on a false premise.
/// `mutation_classification_is_exhaustive` pins every path of both functions.
fn mutation_succeeded(message: &str) -> bool {
    MUTATION_SUCCESS_PREFIXES
        .iter()
        .any(|prefix| message.starts_with(prefix))
}

/// The shell command a mutation tool call would fuse, if it carries one.
///
/// Exposed so the session can put a fused command through the shell approval gate
/// rather than the file-mutation one. The two are not the same privilege.
pub fn fused_command<'a>(tool_name: &str, args: &'a Value) -> Option<&'a str> {
    if tool_name != "edit" && tool_name != "write" {
        return None;
    }
    args.get("then_run")?.get("command")?.as_str()
}

/// A follow-up command carried by a mutation tool call.
#[derive(Debug)]
struct ThenRun {
    command: String,
    timeout: Option<u64>,
}

/// Read the optional `then_run` object from a mutation tool call.
///
/// A malformed `then_run` returns `Err` rather than being ignored. Silently
/// dropping it would report a bare mutation as complete while the model believes
/// its check ran.
fn parse_then_run(args: &Value) -> Result<Option<ThenRun>, String> {
    let Some(value) = args.get("then_run") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(object) = value.as_object() else {
        return Err(format!(
            "{} `then_run` must be an object such as {{\"command\": \"cargo test\"}}.",
            THEN_RUN_SKIPPED
        ));
    };
    let Some(command) = object.get("command").and_then(|c| c.as_str()) else {
        return Err(format!(
            "{} `then_run` needs a `command` string.",
            THEN_RUN_SKIPPED
        ));
    };
    if command.trim().is_empty() {
        return Err(format!("{} `then_run.command` is empty.", THEN_RUN_SKIPPED));
    }
    // A timeout must be a positive whole number of seconds. A float or a negative
    // value used to cast to 0, and a 0 means "kill before the command runs" — so
    // `{"timeout": -1}` killed the command and reported the test suite as failed.
    // The plain `bash` tool takes the looser route and falls back to its default
    // for the same values; here the model is told, because a fused command it
    // believes ran is worse than one it is told was skipped.
    let timeout = match object.get("timeout") {
        None | Some(Value::Null) => None,
        Some(value) => match value.as_u64() {
            Some(seconds) if seconds >= 1 => Some(seconds),
            _ => {
                return Err(format!(
                    "{} `then_run.timeout` must be a whole number of seconds, 1 or more. Got {}.",
                    THEN_RUN_SKIPPED, value
                ))
            }
        },
    };
    Ok(Some(ThenRun {
        command: command.to_string(),
        timeout,
    }))
}

/// Apply a file mutation and, when the model asked for one, run its follow-up
/// command, returning both as a single observation.
///
/// There is no per-file lock here and no re-hash of the target, unlike SoL-Pi.
/// `run_tool_loop` drives tool calls one at a time with no yield between the
/// mutation and the command, so nothing in a rupi session can interleave, and
/// reading the file twice on every fused call to narrow a microsecond window
/// against an external writer is not worth the I/O.
///
/// That reasoning is about rupi's own loop, not about this function. `execute_tool`
/// is public, so an embedder driving two sessions over the same working directory
/// concurrently gets no such guarantee and must serialize its own mutations.
fn mutation_then_run(
    tool_call: &ToolCall,
    mutation: String,
    mutation_kind: &str,
    ctx: &ToolContext,
) -> String {
    let then_run = match parse_then_run(&tool_call.arguments) {
        Ok(Some(then_run)) => then_run,
        Ok(None) => return mutation_without_command(tool_call, mutation),
        Err(problem) => return format!("{}\n\n{}", mutation, problem),
    };

    if !mutation_succeeded(&mutation) {
        return format!(
            "{}\n\n{} The {} did not complete cleanly, so `{}` was not run.",
            mutation, THEN_RUN_SKIPPED, mutation_kind, then_run.command
        );
    }

    let mut bash_args = serde_json::json!({ "command": then_run.command });
    if let Some(timeout) = then_run.timeout {
        bash_args["timeout"] = serde_json::json!(timeout);
    }
    let outcome = run_bash_blocking(&bash_args, ctx.cancelled.clone());

    // A failed command never undoes the mutation. Saying so explicitly stops the
    // model from re-applying an edit that is already on disk.
    let marker = if outcome.succeeded {
        format!("{} `{}`", THEN_RUN_SUCCEEDED, then_run.command)
    } else {
        format!(
            "{} `{}` failed. The {} is still applied.",
            THEN_RUN_FAILED, then_run.command, mutation_kind
        )
    };
    // Fence the output. The markers are bare bracketed words, and a command such as
    // `grep then_run src/tools.rs` prints them as ordinary data — so without a
    // delimiter there is nothing separating what the harness reports from what the
    // command happened to echo.
    format!(
        "{}\n\n{}\n{}\n{}\n{}",
        mutation, marker, THEN_RUN_OUTPUT_OPEN, outcome.text, THEN_RUN_OUTPUT_CLOSE
    )
}

/// Report a `then_run` that was present in the raw call but could not be read.
///
/// A provider that truncates a streamed tool call leaves `arguments` as null and
/// the partial text in `raw_arguments`. `write` recovers its path and content from
/// that text, so the mutation still happens — but `then_run` is not recovered, and
/// returning the bare mutation message would tell the model its check ran when no
/// command was ever parsed, let alone executed.
fn mutation_without_command(tool_call: &ToolCall, mutation: String) -> String {
    if !tool_call.arguments.is_null() {
        return mutation;
    }
    let Some(raw) = tool_call.raw_arguments.as_ref() else {
        return mutation;
    };
    if !raw.contains("then_run") {
        return mutation;
    }
    format!(
        "{}\n\n{} The call was truncated before `then_run` could be read, so no command ran. \
Re-issue it if you still need the check.",
        mutation, THEN_RUN_SKIPPED
    )
}

/// Execute a tool call and return the result.
pub fn execute_tool(tool_call: &ToolCall, ctx: &ToolContext) -> String {
    match tool_call.name.as_str() {
        "bash" => execute_bash_with_cancel(&tool_call.arguments, ctx.cancelled.clone()),
        "read" => execute_read(&tool_call.arguments),
        "write" => mutation_then_run(tool_call, execute_write(tool_call), "write", ctx),
        "edit" => mutation_then_run(tool_call, execute_edit(&tool_call.arguments), "edit", ctx),
        "grep" => execute_grep(&tool_call.arguments),
        "find" => execute_find(&tool_call.arguments),
        "ls" => execute_ls(&tool_call.arguments),
        "search_code" => execute_search_code(&tool_call.arguments),
        "todo_write" => execute_todo_write(&tool_call.arguments, ctx),
        "goal" => execute_goal(&tool_call.arguments, ctx),
        _ => format!("Unknown tool: {}", tool_call.name),
    }
}

/// Keep blocking tools and approval callbacks off Tokio's IO workers. The global
/// permit bounds concurrent filesystem/indexing work across embedded sessions.
pub async fn execute_tool_async(
    mut call: ToolCall,
    context: ToolContext,
    approval: Option<crate::agent::session::ApprovalFn>,
) -> String {
    static WORKERS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
    let cancellation = async {
        while !context.cancelled.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    };
    let permit = tokio::select! {
        permit = WORKERS.acquire() => permit.expect("tool semaphore remains open"),
        _ = cancellation => return "[cancelled]".into(),
    };
    tokio::task::spawn_blocking(move || {
        // Keep the slot occupied until the actual blocking work finishes, even
        // if the async caller is dropped.
        let _permit = permit;
        if context.cancelled.load(Ordering::SeqCst) {
            return "[cancelled]".into();
        }
        let mut declined_command: Option<String> = None;
        if let Some(approval) = approval {
            let args = serde_json::to_string(&call.arguments).unwrap_or_default();
            if !approval(&call.name, &args) {
                return format!("[User denied execution of tool '{}']", call.name);
            }
            // A fused `then_run` is shell execution, so it is approved as shell
            // execution. Approving it under the `edit` label would let a model reach
            // the shell through the one tool a user has learned is safe to wave
            // through, which is the whole thing approval mode exists to prevent.
            //
            // Named plainly `bash`, so a callback keying an allowlist on the tool
            // name sees it for what it is.
            if let Some(command) = fused_command(&call.name, &call.arguments) {
                if !approval("bash", command) {
                    // Declining the command is not declining the change. Drop only
                    // what was refused.
                    declined_command = Some(command.to_string());
                    if let Some(object) = call.arguments.as_object_mut() {
                        object.remove("then_run");
                    }
                }
            }
        }
        if context.cancelled.load(Ordering::SeqCst) {
            return "[cancelled]".into();
        }
        let mut result = execute_tool(&call, &context);
        // Say that the command was blocked. Stripping it silently left a result
        // byte-identical to one where no command was ever asked for, so the model
        // could report a check as done that a human had explicitly refused.
        if let Some(command) = declined_command {
            result.push_str(&format!(
                "\n\n{} The user declined to run `{}`. The change was applied; \
nothing was verified.",
                THEN_RUN_SKIPPED, command
            ));
        }
        result
    })
    .await
    .unwrap_or_else(|e| format!("Tool execution failed: {e}"))
}

fn get_arg<'a>(args: &'a Value, name: &str) -> Option<&'a str> {
    args.get(name).and_then(|v| v.as_str())
}

fn get_arg_i64(args: &Value, name: &str) -> Option<i64> {
    args.get(name).and_then(|v| {
        // Models emit `1` and `1.0` interchangeably, and some emit `"1"`. All three
        // mean the same number, so accept all three rather than rejecting the call.
        v.as_i64()
            .or_else(|| v.as_f64().map(|f| f as i64))
            .or_else(|| v.as_str().and_then(|s| s.trim().parse::<i64>().ok()))
    })
}

fn get_arg_bool(args: &Value, name: &str, default: bool) -> bool {
    args.get(name).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn resolve_path(path_str: &str) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    }
}

// ---- bash ----
#[cfg(test)]
fn execute_bash(args: &Value) -> String {
    execute_bash_with_cancel(args, Arc::new(AtomicBool::new(false)))
}

fn execute_bash_with_cancel(args: &Value, cancelled: Arc<AtomicBool>) -> String {
    run_bash_blocking(args, cancelled).text
}

/// What one bash run produced, plus whether it actually succeeded.
///
/// `execute_bash_with_cancel` flattens this to the text the model sees. Action
/// Fusion needs the distinction too: it must label a fused command `succeeded` or
/// `failed`, and reading that back out of the assembled text would mean parsing
/// prose the output filter is free to rewrite.
pub struct BashOutcome {
    pub text: String,
    /// True only when the command ran to completion and exited zero. A spawn
    /// failure, a timeout, a cancellation, and a non-zero exit are all false.
    pub succeeded: bool,
}

fn run_bash_blocking(args: &Value, cancelled: Arc<AtomicBool>) -> BashOutcome {
    // Called on a blocking worker; a local IO runtime lets us cancel pipe reads
    // portably without leaving threads parked behind background children.
    std::thread::scope(|scope| {
        scope
            .spawn(move || {
                match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime.block_on(run_bash(args, cancelled)),
                    Err(e) => BashOutcome {
                        text: format!("Failed to create bash IO runtime: {e}"),
                        succeeded: false,
                    },
                }
            })
            .join()
            .unwrap_or_else(|_| BashOutcome {
                text: "Error: bash execution panicked".into(),
                succeeded: false,
            })
    })
}

const MAX_CAPTURE_BYTES: usize = 1024 * 1024;

async fn drain_pipe<R: tokio::io::AsyncRead + Unpin>(mut pipe: R, sink: Arc<Mutex<Vec<u8>>>) {
    use tokio::io::AsyncReadExt;
    let mut chunk = [0; 8192];
    while let Ok(n) = pipe.read(&mut chunk).await {
        if n == 0 {
            break;
        }
        let mut buffer = sink.lock().unwrap_or_else(|e| e.into_inner());
        let remaining = MAX_CAPTURE_BYTES.saturating_sub(buffer.len());
        buffer.extend_from_slice(&chunk[..n.min(remaining)]);
        // Continue draining excess output so a noisy command cannot deadlock.
    }
}

async fn kill_process_tree(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        #[cfg(unix)]
        // The child is the leader of the private process group created below.
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
        #[cfg(windows)]
        {
            let _ = tokio::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .output()
                .await;
        }
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

async fn run_bash(args: &Value, cancelled: Arc<AtomicBool>) -> BashOutcome {
    let Some(command) = get_arg(args, "command") else {
        return BashOutcome {
            text: "Error: missing 'command' argument".into(),
            succeeded: false,
        };
    };
    if cancelled.load(Ordering::SeqCst) {
        return BashOutcome {
            text: "[cancelled]".into(),
            succeeded: false,
        };
    }
    // A timeout below one second means "kill before the command can run", which no
    // caller intends. `0` is a plausible way for a model to say "no timeout", so it
    // falls back to the default rather than killing the command instantly.
    let timeout_secs = args
        .get("timeout")
        .and_then(Value::as_u64)
        .filter(|seconds| *seconds >= 1)
        .unwrap_or_else(bash_timeout_default)
        .min(bash_timeout_max());
    let mut process = tokio::process::Command::new("bash");
    process
        .arg("-c")
        .arg(command)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        process.as_std_mut().process_group(0);
    }
    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(e) => {
            return BashOutcome {
                text: format!("Failed to spawn bash: {e}"),
                succeeded: false,
            }
        }
    };
    let out = Arc::new(Mutex::new(Vec::new()));
    let err = Arc::new(Mutex::new(Vec::new()));
    let mut readers = tokio::task::JoinSet::new();
    if let Some(pipe) = child.stdout.take() {
        readers.spawn(drain_pipe(pipe, out.clone()));
    }
    if let Some(pipe) = child.stderr.take() {
        readers.spawn(drain_pipe(pipe, err.clone()));
    }
    let cancellation = async {
        while !cancelled.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    };
    let (status, notice) = tokio::select! {
        status = child.wait() => (status.ok().and_then(|s| s.code()), String::new()),
        _ = tokio::time::sleep(std::time::Duration::from_secs(timeout_secs)) => {
            kill_process_tree(&mut child).await;
            (None, format!("[timed out after {timeout_secs}s]\n"))
        },
        _ = cancellation => {
            kill_process_tree(&mut child).await;
            (None, "[cancelled]\n".into())
        },
    };
    // Drain normal exits, but close our pipe handles when descendants outlive
    // their shell. The tasks are joined before returning, even on cancellation.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(300), async {
        while readers.join_next().await.is_some() {}
    })
    .await;
    readers.shutdown().await;
    let out = out.lock().unwrap_or_else(|e| e.into_inner());
    let err = err.lock().unwrap_or_else(|e| e.into_inner());
    let mut result = format!(
        "{notice}{}",
        assemble_bash_result(command, &out, &err, status)
    );
    if out.len() == MAX_CAPTURE_BYTES || err.len() == MAX_CAPTURE_BYTES {
        result.push_str("\n[output capture capped at 1 MiB per stream]");
    }
    BashOutcome {
        succeeded: notice.is_empty() && status == Some(0),
        text: result,
    }
}

fn assemble_bash_result(command: &str, out: &[u8], err: &[u8], exit_code: Option<i32>) -> String {
    let mut result = String::new();
    if !out.is_empty() {
        result.push_str(&String::from_utf8_lossy(out));
    }
    if !err.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&String::from_utf8_lossy(err));
    }
    match exit_code {
        Some(0) => {}
        Some(code) => result.push_str(&format!("\n[exit code: {}]", code)),
        // No exit code means the process died to a signal — a segfault, an OOM
        // kill, a `kill -9`. Printing nothing here made a crash that had already
        // written output indistinguishable from a clean success.
        None => result.push_str("\n[no exit code: the command was killed or did not finish]"),
    }
    if result.trim().is_empty() {
        return format!(
            "[command completed with exit code {}]",
            exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "unknown".into())
        );
    }
    // Apply RTK-style output compression
    let original_len = result.len();
    result = crate::rtk_filter::filter_output(command, &result);
    let new_len = result.len();
    if new_len < original_len && original_len > 200 {
        let pct = (original_len - new_len) * 100 / original_len;
        if pct > 0 {
            result.push_str(&format!("\n[rtk: {}% token savings]", pct));
        }
    }
    result
}

// ---- read ----
fn execute_read(args: &Value) -> String {
    let file_path = match get_arg(args, "file_path") {
        Some(p) => p,
        None => return "Error: missing 'file_path' argument".to_string(),
    };
    let offset = get_arg_i64(args, "offset").unwrap_or(1).max(1) as usize;
    let limit = get_arg_i64(args, "limit").unwrap_or(2000).max(1) as usize;

    let path = resolve_path(file_path);
    if !path.exists() {
        return format!("Error: file not found: {}", path.display());
    }
    if !path.is_file() {
        return format!("Error: not a file: {}", path.display());
    }

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let start = (offset - 1).min(lines.len());
            let end = (start + limit).min(lines.len());
            let selected = &lines[start..end];
            let mut result = String::new();
            for (i, line) in selected.iter().enumerate() {
                result.push_str(&format!("{:6}  {}\n", start + i + 1, line));
            }
            if end < lines.len() {
                result.push_str(&format!("... {} more lines\n", lines.len() - end));
            }
            if result.is_empty() {
                result = "[empty file]".to_string();
            }
            result
        }
        Err(e) => format!("Error reading file: {}", e),
    }
}

// ---- write (with guard + truncation recovery) ----
const TRUNCATION_MARKER: &str = "\n<!-- RUPI_TRUNCATED: Response was cut off before completion. File is incomplete. Continue writing from where this marker ends. -->\n";

fn execute_write(tool_call: &ToolCall) -> String {
    let args = &tool_call.arguments;
    let mut file_path = get_arg(args, "file_path").map(|s| s.to_string());
    let mut content = get_arg(args, "content").map(|s| s.to_string());
    let mut truncated = false;

    // If content is missing but raw_arguments exist, attempt truncation recovery
    if content.is_none() {
        if let Some(ref raw) = tool_call.raw_arguments {
            // Try to extract file_path and content from raw JSON
            if let Ok(raw_val) = serde_json::from_str::<Value>(raw) {
                file_path = file_path.or_else(|| {
                    raw_val
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                });
                content = raw_val
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                truncated = true;
            } else {
                // Even raw JSON is malformed — try to extract file_path with regex-like approach
                if let Some(fp_start) = raw.find("\"file_path\"") {
                    let after_key = &raw[fp_start + 11..];
                    if let Some(q1) = after_key.find('"') {
                        let after_open = &after_key[q1 + 1..];
                        if let Some(q2) = after_open.find('"') {
                            file_path = Some(after_open[..q2].to_string());
                        }
                    }
                }
                // Try to extract content — find the last complete segment
                if let Some(c_start) = raw.find("\"content\"") {
                    let after_key = &raw[c_start + 9..];
                    if let Some(q1) = after_key.find('"') {
                        let after_open = &after_key[q1 + 1..];
                        // Find the last unescaped quote, or take everything if none (truncated)
                        let mut last_end = None;
                        let mut chars = after_open.char_indices();
                        while let Some((i, ch)) = chars.next() {
                            if ch == '\\' {
                                chars.next(); // skip escaped char
                                continue;
                            }
                            if ch == '"' {
                                last_end = Some(i);
                                break;
                            }
                        }
                        let end = last_end.unwrap_or(after_open.len());
                        if end > 0 {
                            content = Some(after_open[..end].to_string());
                            truncated = true;
                        }
                    }
                }
            }
        }
    }

    let file_path = match file_path {
        Some(p) => p,
        None => return "Error: missing 'file_path' argument".to_string(),
    };
    let mut content = match content {
        Some(c) => c,
        None => return "Error: missing 'content' argument. If the response was truncated, try again with a shorter file.".to_string(),
    };

    let path = resolve_path(&file_path);

    // Write guard: refuse if file already exists (skip for truncated recovery)
    if path.exists() && !truncated {
        return format!(
            "Error: Write refused — {} already exists.\n\
             \n\
             Write is only for creating NEW files. To change an existing file, use Edit:\n\
             {{\"name\":\"edit\",\"input\":{{\"file_path\":\"{}\",\"edits\":[{{\"old_text\":\"<exact text>\",\"new_text\":\"<replacement>\"}}]}}}}\n\
             \n\
             If you do not already know the file's current content, Read it first to get the exact text. \
             Do NOT retry Write — it will be refused again.",
            path.display(),
            path.display(),
        );
    }

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return format!("Error creating parent directory: {}", e);
            }
        }
    }

    // Append truncation marker if response was cut off
    if truncated {
        content.push_str(TRUNCATION_MARKER);
    }

    match std::fs::write(&path, &content) {
        Ok(_) => {
            let line_count = content.lines().count();
            if truncated {
                format!(
                    "WARNING: Response was truncated. Wrote {} lines (incomplete) to {}. \
                     The file has a RUPI_TRUNCATED marker. Use Edit to continue writing from where it left off.",
                    line_count,
                    path.display()
                )
            } else {
                format!(
                    "Successfully wrote {} lines to {}",
                    line_count,
                    path.display()
                )
            }
        }
        Err(e) => format!("Error writing file: {}", e),
    }
}

// ---- edit (multi-edit support) ----
fn execute_edit(args: &Value) -> String {
    let file_path = match get_arg(args, "file_path") {
        Some(p) => p,
        None => return "Error: missing 'file_path' argument".to_string(),
    };

    let path = resolve_path(file_path);
    if !path.exists() {
        return format!("Error: file not found: {}", path.display());
    }

    // Collect edits from either the singular old_text/new_text or the edits array
    // `edits` present but not an array used to fall through to the deprecated
    // single-edit path, applying `old_text`/`new_text` and reporting success while
    // ignoring what the model actually asked for.
    if let Some(value) = args.get("edits") {
        if !value.is_null() && !value.is_array() {
            return "Error: `edits` must be an array of {old_text, new_text} objects. \
No edits were applied."
                .to_string();
        }
    }
    let edits: Vec<(String, String)> = if let Some(edits_val) =
        args.get("edits").and_then(|v| v.as_array())
    {
        if edits_val.is_empty() {
            return "Error: edits array is empty".to_string();
        }
        // A malformed entry is rejected, not dropped. Dropping one and applying the
        // rest reported "Successfully applied edit" for a half-applied request —
        // and with a fused then_run, the check then passed against a file carrying
        // only part of the intended change.
        let mut parsed = Vec::with_capacity(edits_val.len());
        for (index, entry) in edits_val.iter().enumerate() {
            let old = match entry.get("old_text").and_then(|v| v.as_str()) {
                Some(text) => text.to_string(),
                None => {
                    return format!(
                        "Error: edit {} has no `old_text` string. No edits were applied.",
                        index
                    )
                }
            };
            let new = match entry.get("new_text").and_then(|v| v.as_str()) {
                Some(text) => text.to_string(),
                None => {
                    return format!(
                        "Error: edit {} has no `new_text` string. No edits were applied.",
                        index
                    )
                }
            };
            parsed.push((old, new));
        }
        parsed
    } else {
        let old_text = match get_arg(args, "old_text") {
            Some(t) => t,
            None => {
                return "Error: missing 'old_text' argument (or provide 'edits' array)".to_string()
            }
        };
        let new_text = match get_arg(args, "new_text") {
            Some(t) => t,
            None => {
                return "Error: missing 'new_text' argument (or provide 'edits' array)".to_string()
            }
        };
        vec![(old_text.to_string(), new_text.to_string())]
    };

    if edits.is_empty() {
        return "Error: no valid edits provided".to_string();
    }

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            // Check all old_text exist exactly once in the ORIGINAL content
            let mut results: Vec<String> = Vec::new();
            let mut has_error = false;

            for (old_text, _new_text) in &edits {
                let count = content.matches(old_text).count();
                if count == 0 {
                    results.push(format!("old_text not found: {:?}", old_text));
                    has_error = true;
                } else if count > 1 {
                    results.push(format!("old_text found {} times: {:?}", count, old_text));
                    has_error = true;
                }
            }

            if has_error {
                let mut msg = "Edit failed:\n".to_string();
                for r in &results {
                    msg.push_str(&format!("  - {}\n", r));
                }
                msg.push_str("\nRecovery: Read the file to get the exact current content, then retry with the exact text.");
                return msg;
            }

            // Check for overlapping edits by finding positions in original content
            let mut positions: Vec<(usize, usize, &str, &str)> = Vec::new(); // (start, end, old_text, new_text)
            for (old_text, new_text) in &edits {
                if let Some(pos) = content.find(old_text) {
                    positions.push((pos, pos + old_text.len(), old_text, new_text));
                }
            }

            // Sort by position (ascending)
            positions.sort_by_key(|&(start, _, _, _)| start);

            // Check for overlaps
            for i in 1..positions.len() {
                if positions[i].0 < positions[i - 1].1 {
                    return format!(
                        "Error: edits overlap. First edit ends at position {} but next edit starts at {}. \
                         Edits must not overlap — each old_text is matched against the original file content.",
                        positions[i - 1].1, positions[i].0
                    );
                }
            }

            // Apply edits in reverse position order to preserve positions
            let mut new_content = content.clone();
            for &(start, end, _old, new) in positions.iter().rev() {
                new_content.replace_range(start..end, new);
            }

            match std::fs::write(&path, &new_content) {
                Ok(_) => {
                    if positions.len() == 1 {
                        format!("Successfully applied edit to {}", path.display())
                    } else {
                        format!(
                            "Successfully applied {} edits to {}",
                            positions.len(),
                            path.display()
                        )
                    }
                }
                Err(e) => format!("Error writing file: {}", e),
            }
        }
        Err(e) => format!("Error reading file: {}", e),
    }
}

// ---- grep ----
/// Lines a content search returns unless the model asks for more.
///
/// A limit the model can raise beats a cap it cannot see. The old grep took no
/// limit at all, returned everything, cut the result at 10000 bytes, and then had
/// the storage cap cut it again at 4000 — so the model could neither ask for less
/// nor reach what was removed.
const GREP_DEFAULT_HEAD: usize = 100;

/// Upper bound on what a single search can return, whatever it asks for.
const GREP_MAX_HEAD: usize = 2000;

fn execute_grep(args: &Value) -> String {
    let pattern = match get_arg(args, "pattern") {
        Some(p) => p,
        None => return "Error: missing 'pattern' argument".to_string(),
    };
    let include = get_arg(args, "include").unwrap_or("");
    let search_path = get_arg(args, "path").unwrap_or(".");
    let ignore_case = get_arg_bool(args, "ignore_case", false);
    let mode = get_arg(args, "output_mode").unwrap_or("content");
    if !matches!(mode, "content" | "files_with_matches" | "count") {
        return format!(
            "Error: output_mode {:?} is not one of content, files_with_matches, count.",
            mode
        );
    }
    let head = get_arg_i64(args, "head_limit")
        .filter(|n| *n > 0)
        .map(|n| (n as usize).min(GREP_MAX_HEAD))
        .unwrap_or(GREP_DEFAULT_HEAD);
    let context = get_arg_i64(args, "context").unwrap_or(0).clamp(0, 20) as usize;

    let mut cmd = std::process::Command::new("rg");
    cmd.arg("--color").arg("never");
    match mode {
        "files_with_matches" => {
            cmd.arg("--files-with-matches");
        }
        "count" => {
            cmd.arg("--count");
        }
        _ => {
            cmd.arg("--line-number");
            if context > 0 {
                cmd.arg("--context").arg(context.to_string());
            }
        }
    }
    if !include.is_empty() {
        cmd.arg("--glob").arg(include);
    }
    if ignore_case {
        cmd.arg("-i");
    }
    cmd.arg(pattern);
    cmd.arg(search_path);

    match cmd.output() {
        Ok(output) => {
            let text = if !output.stdout.is_empty() {
                String::from_utf8_lossy(&output.stdout).to_string()
            } else if !output.stderr.is_empty() {
                String::from_utf8_lossy(&output.stderr).to_string()
            } else {
                String::new()
            };
            grep_result(&text, head, mode)
        }
        Err(e) => {
            // Fall back to grep when ripgrep is absent.
            let mut cmd = std::process::Command::new("grep");
            match mode {
                "files_with_matches" => {
                    cmd.arg("-rl");
                }
                "count" => {
                    cmd.arg("-rc");
                }
                _ => {
                    cmd.arg("-rn");
                    if context > 0 {
                        cmd.arg(format!("-C{}", context));
                    }
                }
            }
            if ignore_case {
                cmd.arg("-i");
            }
            if !include.is_empty() {
                cmd.arg("--include").arg(include);
            }
            cmd.arg(pattern);
            cmd.arg(search_path);

            match cmd.output() {
                Ok(o) => {
                    let text = String::from_utf8_lossy(&o.stdout).to_string();
                    grep_result(&text, head, mode)
                }
                Err(e2) => format!("Error running grep: {} (rg also unavailable: {})", e2, e),
            }
        }
    }
}

/// Trim a search result to `head` lines and say what was left out.
///
/// Counting lines rather than bytes is what makes the limit something the model can
/// reason about. The previous byte truncation also used `String::truncate`, which
/// panics when the cut lands inside a multi-byte character — so a search over any
/// file with non-ASCII content could take down the tool call.
fn grep_result(text: &str, head: usize, mode: &str) -> String {
    let trimmed = text.trim_end_matches('\n');
    if trimmed.is_empty() {
        return "No matches found.".to_string();
    }
    let total = trimmed.lines().count();
    if total <= head {
        return trimmed.to_string();
    }
    let kept: Vec<&str> = trimmed.lines().take(head).collect();
    let advice = if mode == "content" {
        " Narrow the pattern or the path, use output_mode \"files_with_matches\", or raise head_limit."
    } else {
        " Narrow the pattern or the path, or raise head_limit."
    };
    format!(
        "{}\n... [{} of {} lines shown.{}]",
        kept.join("\n"),
        head,
        total,
        advice
    )
}

// ---- find ----
fn execute_find(args: &Value) -> String {
    let pattern = match get_arg(args, "pattern") {
        Some(p) => p,
        None => return "Error: missing 'pattern' argument".to_string(),
    };
    let search_path = get_arg(args, "path").unwrap_or(".");

    // Try fd first, fallback to find
    let mut cmd = std::process::Command::new("fd");
    cmd.arg("--glob").arg(pattern).arg(search_path);

    match cmd.output() {
        Ok(output) => {
            let mut result = String::new();
            if !output.stdout.is_empty() {
                result.push_str(&String::from_utf8_lossy(&output.stdout));
            }
            if result.is_empty() {
                result = "No files found.".to_string();
            }
            if result.len() > 5000 {
                result.truncate(5000);
                result.push_str(&format!(
                    "\n... [{} results total, truncated]",
                    result.matches('\n').count()
                ));
            }
            result
        }
        Err(_) => {
            // Fallback to find
            let mut cmd = std::process::Command::new("find");
            cmd.arg(search_path).arg("-name").arg(pattern);

            match cmd.output() {
                Ok(o) => {
                    let mut r = String::new();
                    if !o.stdout.is_empty() {
                        r.push_str(&String::from_utf8_lossy(&o.stdout));
                    }
                    if r.is_empty() {
                        r = "No files found.".to_string();
                    }
                    if r.len() > 5000 {
                        r.truncate(5000);
                        r.push_str("\n... [truncated]");
                    }
                    r
                }
                Err(e) => format!("Error running find: {}", e),
            }
        }
    }
}

// ---- ls ----
fn execute_ls(args: &Value) -> String {
    let dir_path = get_arg(args, "path").unwrap_or(".");
    let path = resolve_path(dir_path);

    if !path.exists() {
        return format!("Error: path not found: {}", path.display());
    }
    if !path.is_dir() {
        return format!("Error: not a directory: {}", path.display());
    }

    match std::fs::read_dir(&path) {
        Ok(entries) => {
            let mut files = Vec::new();
            let mut dirs = Vec::new();
            let mut total = 0;

            for entry in entries.flatten() {
                total += 1;
                if total > 200 {
                    break;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') {
                    continue;
                }
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    dirs.push(format!("{}/", name));
                } else {
                    files.push(name);
                }
            }

            files.sort();
            dirs.sort();

            let mut result = String::new();
            for d in &dirs {
                result.push_str(&format!("  {}\n", d));
            }
            for f in &files {
                result.push_str(&format!("  {}\n", f));
            }
            if result.is_empty() {
                result = "(empty directory)".to_string();
            }
            result
        }
        Err(e) => format!("Error reading directory: {}", e),
    }
}

// ---- search_code ----
fn execute_search_code(args: &Value) -> String {
    let query = match get_arg(args, "query") {
        Some(q) => q,
        None => return "Error: missing 'query' argument".to_string(),
    };
    let search_path = get_arg(args, "path").unwrap_or(".");
    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .min(50) as usize;

    let path = resolve_path(search_path);
    if !path.exists() {
        return format!("Error: path not found: {}", path.display());
    }
    if !path.is_dir() {
        return format!("Error: not a directory: {}", path.display());
    }

    // Try semantic search first (requires model), fall back to keyword search
    let start = std::time::Instant::now();
    match crate::code_search::cached_index(&path) {
        Ok(index) => {
            let results = index.search(query, top_k);
            let elapsed = start.elapsed();
            let mut out = crate::code_search::format_results(query, &results);
            out.push_str(&format!(
                "[{} chunks indexed, total search time {:?}]",
                index.len(),
                elapsed
            ));
            out
        }
        Err(e) => {
            // Fall back to keyword search
            let chunks = crate::code_search::index_path(&path);
            if chunks.is_empty() {
                return format!("Error: {}", e);
            }
            let results = crate::code_search::search_keyword(&chunks, query, top_k);
            let mut out = crate::code_search::format_results(query, &results);
            out.push_str(&format!(
                "[keyword search, {} chunks, model unavailable: {}]",
                chunks.len(),
                e
            ));
            out
        }
    }
}

#[cfg(test)]
mod timeout_tests {
    use super::*;

    #[test]
    fn the_schema_advertises_the_configured_bounds() {
        // The model chooses a timeout from what the schema says is allowed, so a
        // raised ceiling that is not advertised buys nothing: it keeps asking
        // for the old maximum.
        set_bash_timeout_max(900);
        set_bash_timeout_default(300);

        let schema = bash_tool().parameters;
        let timeout = &schema["properties"]["timeout"];
        assert_eq!(timeout["default"].as_u64(), Some(300));
        assert!(
            timeout["description"].as_str().unwrap().contains("900"),
            "ceiling not advertised: {}",
            timeout["description"]
        );

        // A default above the ceiling must not be handed out.
        set_bash_timeout_max(60);
        assert_eq!(bash_timeout_default(), 60);

        set_bash_timeout_max(120);
        set_bash_timeout_default(30);
    }
}

// ---- todo_write ----

fn todo_write_tool() -> ToolDef {
    ToolDef {
        name: "todo_write",
        description: "Record and update a structured task list for the current work. Send the ENTIRE list on every call — it REPLACES the previous list. There are no partial updates and no per-item edits. Use it to plan multi-step work and to show progress: add one todo per concrete step before you start. Keep AT MOST ONE todo in_progress at a time, and while work remains, exactly one task should be in_progress. Mark a todo completed the moment it is done — do not batch completions. Skip the list for trivial single-step tasks. Statuses: pending (not started), in_progress (being worked on now), completed (finished).",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete task list, in order. Replaces the previous list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string", "description": "What the step does, in the imperative" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current state of this step"
                            }
                        },
                        "required": ["content", "status"]
                    }
                }
            },
            "required": ["todos"]
        }),
    }
}

fn execute_todo_write(args: &Value, ctx: &ToolContext) -> String {
    let raw = match args.get("todos") {
        Some(value) => value,
        None => return "Error: `todos` is required. Send the entire list.".to_string(),
    };
    match crate::todo::parse_items(raw).and_then(|items| ctx.todos.replace(items)) {
        Ok(rendered) => rendered,
        Err(message) => message,
    }
}

// ---- goal ----

fn goal_tool() -> ToolDef {
    ToolDef {
        name: "goal",
        description: "Read or decide the durable goal that drives this session. Use operation \"read\" to see the objective, the open round, and the status. Use operation \"complete\" once you have gathered evidence that the WHOLE objective is achieved. Use operation \"block\" when you cannot proceed, and give the reason. A decision is accepted only from inside the round shown in the current <goal_round> block, so pass that exact round number.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "complete", "block"],
                    "description": "What to do with the goal"
                },
                "round": {
                    "type": "number",
                    "description": "The round number from the current <goal_round> block. Required for complete and block."
                },
                "reason": {
                    "type": "string",
                    "description": "Why the goal is blocked. Required for block."
                }
            },
            "required": ["operation"]
        }),
    }
}

fn execute_goal(args: &Value, ctx: &ToolContext) -> String {
    let operation = get_arg(args, "operation").unwrap_or("read");
    let result = match operation {
        "read" => ctx.goal.read_goal(),
        "complete" | "block" => {
            let round = match get_arg_i64(args, "round") {
                Some(r) if r > 0 => r as u32,
                _ => {
                    return "Rejected: `round` is required and must be the round number shown in \
the current <goal_round> block."
                        .to_string()
                }
            };
            if operation == "complete" {
                ctx.goal.complete(round)
            } else {
                ctx.goal.block(round, get_arg(args, "reason").unwrap_or(""))
            }
        }
        other => Err(format!(
            "Rejected: unknown operation {:?}. Use read, complete, or block.",
            other
        )),
    };
    match result {
        Ok(message) => message,
        Err(message) => message,
    }
}

#[cfg(test)]
mod grep_tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rupi-grep-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn multibyte_output_does_not_panic() {
        // `String::truncate` panics when the cut is not on a character boundary, so
        // the old byte-based limit crashed on any search over non-ASCII content.
        let body: String = (0..4000)
            .map(|i| format!("行 {} のマッチ target\n", i))
            .collect();
        let out = grep_result(&body, 100, "content");
        assert!(out.contains("行"), "{}", &out[..60]);
        assert!(out.contains("of 4000 lines shown"), "{}", out);
    }

    #[test]
    fn the_limit_counts_lines_and_says_what_was_left_out() {
        let body: String = (0..500)
            .map(|i| format!("src/f.rs:{}:match\n", i))
            .collect();
        let out = grep_result(&body, 10, "content");
        assert_eq!(out.lines().count(), 11, "10 results plus one notice");
        assert!(out.contains("10 of 500 lines shown"), "{}", out);
        assert!(
            out.contains("files_with_matches"),
            "the notice must name the cheaper mode"
        );
    }

    #[test]
    fn a_result_within_the_limit_is_untouched() {
        let body = "src/a.rs:1:hit\nsrc/b.rs:2:hit\n";
        assert_eq!(
            grep_result(body, 100, "content"),
            "src/a.rs:1:hit\nsrc/b.rs:2:hit"
        );
    }

    #[test]
    fn an_empty_result_says_so() {
        assert_eq!(grep_result("", 100, "content"), "No matches found.");
        assert_eq!(grep_result("\n\n", 100, "content"), "No matches found.");
    }

    #[test]
    fn output_mode_is_validated() {
        let out = execute_grep(&serde_json::json!({"pattern": "x", "output_mode": "everything"}));
        assert!(out.starts_with("Error: output_mode"), "{}", out);
    }

    #[test]
    fn files_with_matches_returns_paths_not_content() {
        let dir = scratch("modes");
        std::fs::write(dir.join("a.rs"), "fn unique_marker_alpha() {}\n").unwrap();
        std::fs::write(dir.join("b.rs"), "fn unique_marker_alpha() {}\n").unwrap();

        let paths = execute_grep(&serde_json::json!({
            "pattern": "unique_marker_alpha",
            "path": dir.display().to_string(),
            "output_mode": "files_with_matches"
        }));
        assert!(paths.contains("a.rs"), "{}", paths);
        assert!(!paths.contains("fn unique_marker_alpha() {}"), "{}", paths);

        let content = execute_grep(&serde_json::json!({
            "pattern": "unique_marker_alpha",
            "path": dir.display().to_string()
        }));
        assert!(content.contains("fn unique_marker_alpha"), "{}", content);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn head_limit_is_bounded_and_positive() {
        let dir = scratch("head");
        let body: String = (0..300).map(|i| format!("line {} needle\n", i)).collect();
        std::fs::write(dir.join("a.txt"), body).unwrap();

        let two = execute_grep(&serde_json::json!({
            "pattern": "needle", "path": dir.display().to_string(), "head_limit": 2
        }));
        assert_eq!(two.lines().count(), 3, "{}", two);

        // A nonsense limit falls back to the default rather than returning nothing.
        for bad in [serde_json::json!(0), serde_json::json!(-5)] {
            let out = execute_grep(&serde_json::json!({
                "pattern": "needle", "path": dir.display().to_string(), "head_limit": bad
            }));
            assert!(
                out.lines().count() > 2,
                "limit {:?} returned {}",
                bad,
                out.lines().count()
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_schema_offers_the_limits() {
        let grep = all_tools().into_iter().find(|t| t.name == "grep").unwrap();
        let props = &grep.parameters["properties"];
        assert!(!props["output_mode"].is_null());
        assert!(!props["head_limit"].is_null());
        assert!(!props["context"].is_null());
        // Only `pattern` stays required, so an existing caller is unaffected.
        assert_eq!(grep.parameters["required"].as_array().unwrap().len(), 1);
    }
}

#[cfg(test)]
mod action_fusion_tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rupi-fusion-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "t1".into(),
            name: name.into(),
            arguments: args,
            raw_arguments: None,
        }
    }

    fn run(name: &str, args: Value) -> String {
        execute_tool(&call(name, args), &ToolContext::new())
    }

    // ---- the invariant the fusion rests on ----

    #[test]
    fn mutation_classification_is_exhaustive() {
        let dir = scratch("classify");
        let existing = dir.join("existing.txt");
        std::fs::write(&existing, "hello world").unwrap();
        let fresh = dir.join("fresh.txt");

        // Every failure path of write and edit, in one place. If a future change
        // returns a failure that does not start with `Error`, the fusion would run
        // a validation command against a file that never changed.
        let failures = vec![
            execute_write(&call("write", serde_json::json!({"content": "x"}))),
            execute_write(&call(
                "write",
                serde_json::json!({"file_path": fresh.display().to_string()}),
            )),
            execute_write(&call(
                "write",
                serde_json::json!({
                    "file_path": existing.display().to_string(), "content": "x"
                }),
            )),
            execute_edit(&serde_json::json!({"old_text": "a", "new_text": "b"})),
            execute_edit(&serde_json::json!({
                "file_path": dir.join("missing.txt").display().to_string(),
                "old_text": "a", "new_text": "b"
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(), "edits": []
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(), "new_text": "b"
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(),
                "old_text": "not present anywhere", "new_text": "b"
            })),
            // The paths the first version of this test missed, while its comment
            // claimed to cover every one.
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(), "old_text": "hello"
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(),
                "edits": [{"old_text": "hello"}]
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(),
                "edits": [{"new_text": "x"}]
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(),
                "edits": [
                    {"old_text": "hello world", "new_text": "a"},
                    {"old_text": "world", "new_text": "b"}
                ]
            })),
            execute_edit(&serde_json::json!({
                "file_path": dir.display().to_string(), "old_text": "a", "new_text": "b"
            })),
            execute_write(&call(
                "write",
                serde_json::json!({
                    "file_path": dir.join("sub").join("x").join("..").display().to_string(),
                    "content": "x"
                }),
            )),
        ];
        for failure in &failures {
            assert!(
                !mutation_succeeded(failure),
                "classified as success: {}",
                failure
            );
        }

        // The two paths that do not start with `Error` and would have been read as
        // success by a naive inverse check.
        let edit_failed = execute_edit(&serde_json::json!({
            "file_path": existing.display().to_string(),
            "old_text": "not present anywhere", "new_text": "b"
        }));
        assert!(edit_failed.starts_with("Edit failed:"), "{}", edit_failed);
        assert!(!mutation_succeeded(&edit_failed));

        assert!(!mutation_succeeded(
            "WARNING: Response was truncated. Wrote 3 lines (incomplete) to /tmp/x"
        ));

        // Fail closed: an unrecognized message must not run the command.
        assert!(!mutation_succeeded("something new a future change returns"));
        assert!(!mutation_succeeded(""));

        // And the success paths must classify the other way.
        let successes = vec![
            execute_write(&call(
                "write",
                serde_json::json!({
                    "file_path": fresh.display().to_string(), "content": "one\ntwo"
                }),
            )),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(), "old_text": "hello", "new_text": "goodbye"
            })),
            execute_edit(&serde_json::json!({
                "file_path": existing.display().to_string(),
                "edits": [{"old_text": "goodbye", "new_text": "hi"}, {"old_text": "world", "new_text": "there"}]
            })),
        ];
        for success in &successes {
            assert!(
                mutation_succeeded(success),
                "classified as failure: {}",
                success
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- the existing contract is untouched ----

    #[test]
    fn a_call_without_then_run_is_unchanged() {
        let dir = scratch("plain");
        let target = dir.join("a.txt");
        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(), "content": "body"
            }),
        );
        assert_eq!(
            result,
            format!("Successfully wrote 1 lines to {}", target.display())
        );
        assert!(!result.contains("then_run"));

        let edited = run(
            "edit",
            serde_json::json!({
                "file_path": target.display().to_string(), "old_text": "body", "new_text": "new body"
            }),
        );
        assert_eq!(
            edited,
            format!("Successfully applied edit to {}", target.display())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_explicit_null_then_run_is_not_a_request() {
        let dir = scratch("null");
        let target = dir.join("a.txt");
        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(), "content": "body", "then_run": null
            }),
        );
        assert!(!result.contains("then_run"), "{}", result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- the happy path ----

    #[test]
    fn a_successful_write_runs_its_command_in_the_same_call() {
        let dir = scratch("happy");
        let target = dir.join("a.txt");
        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "content": "body",
                "then_run": {"command": format!("cat {}", target.display())}
            }),
        );
        assert!(result.contains("Successfully wrote"), "{}", result);
        assert!(result.contains(THEN_RUN_SUCCEEDED), "{}", result);
        assert!(
            result.contains("body"),
            "the command output must be included: {}",
            result
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_successful_edit_runs_its_command_in_the_same_call() {
        let dir = scratch("happy-edit");
        let target = dir.join("a.txt");
        std::fs::write(&target, "alpha").unwrap();
        let result = run(
            "edit",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "old_text": "alpha", "new_text": "omega",
                "then_run": {"command": format!("cat {}", target.display())}
            }),
        );
        assert!(result.contains("Successfully applied edit"), "{}", result);
        assert!(result.contains(THEN_RUN_SUCCEEDED), "{}", result);
        assert!(result.contains("omega"), "{}", result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- a failed mutation must not run the command ----

    #[test]
    fn a_refused_write_skips_the_command() {
        let dir = scratch("refused");
        let target = dir.join("a.txt");
        std::fs::write(&target, "already here").unwrap();
        let sentinel = dir.join("sentinel");

        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "content": "body",
                "then_run": {"command": format!("touch {}", sentinel.display())}
            }),
        );
        assert!(result.contains("Write refused"), "{}", result);
        assert!(result.contains(THEN_RUN_SKIPPED), "{}", result);
        assert!(
            !sentinel.exists(),
            "the command must not run after a refused write"
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "already here");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_edit_skips_the_command() {
        let dir = scratch("failed-edit");
        let target = dir.join("a.txt");
        std::fs::write(&target, "alpha").unwrap();
        let sentinel = dir.join("sentinel");

        let result = run(
            "edit",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "old_text": "text that is not there", "new_text": "omega",
                "then_run": {"command": format!("touch {}", sentinel.display())}
            }),
        );
        assert!(result.starts_with("Edit failed:"), "{}", result);
        assert!(result.contains(THEN_RUN_SKIPPED), "{}", result);
        assert!(
            !sentinel.exists(),
            "the command must not run after a failed edit"
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "alpha");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- a failed command keeps the mutation ----

    #[test]
    fn a_failing_command_keeps_the_edit_and_says_so() {
        let dir = scratch("cmd-fails");
        let target = dir.join("a.txt");
        std::fs::write(&target, "alpha").unwrap();

        let result = run(
            "edit",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "old_text": "alpha", "new_text": "omega",
                "then_run": {"command": "exit 7"}
            }),
        );
        assert!(result.contains(THEN_RUN_FAILED), "{}", result);
        assert!(result.contains("still applied"), "{}", result);
        assert!(!result.contains(THEN_RUN_SUCCEEDED), "{}", result);
        // The edit really is on disk. A model told otherwise would redo it.
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "omega");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- malformed requests are reported, never silently dropped ----

    #[test]
    fn a_malformed_then_run_is_reported() {
        let dir = scratch("malformed");
        let target = dir.join("a.txt");

        let not_an_object = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(), "content": "body",
                "then_run": "cargo test"
            }),
        );
        assert!(
            not_an_object.contains("Successfully wrote"),
            "{}",
            not_an_object
        );
        assert!(
            not_an_object.contains(THEN_RUN_SKIPPED),
            "{}",
            not_an_object
        );
        assert!(
            not_an_object.contains("must be an object"),
            "{}",
            not_an_object
        );

        let second = dir.join("b.txt");
        let no_command = run(
            "write",
            serde_json::json!({
                "file_path": second.display().to_string(), "content": "body",
                "then_run": {"timeout": 5}
            }),
        );
        assert!(
            no_command.contains("needs a `command` string"),
            "{}",
            no_command
        );

        let third = dir.join("c.txt");
        let empty = run(
            "write",
            serde_json::json!({
                "file_path": third.display().to_string(), "content": "body",
                "then_run": {"command": "   "}
            }),
        );
        assert!(empty.contains("is empty"), "{}", empty);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_timeout_must_be_a_positive_whole_number() {
        let accepted = parse_then_run(&serde_json::json!({
            "then_run": {"command": "true", "timeout": 5}
        }))
        .unwrap()
        .unwrap();
        assert_eq!(accepted.timeout, Some(5));

        let absent = parse_then_run(&serde_json::json!({"then_run": {"command": "true"}}))
            .unwrap()
            .unwrap();
        assert_eq!(absent.timeout, None);
        let null = parse_then_run(&serde_json::json!({
            "then_run": {"command": "true", "timeout": null}
        }))
        .unwrap()
        .unwrap();
        assert_eq!(null.timeout, None);

        // A float or a negative used to cast to 0, which run_bash reads as "kill
        // immediately" — so the command died before it ran and the model was told
        // its test suite failed.
        for bad in [
            serde_json::json!(-1),
            serde_json::json!(0),
            serde_json::json!(0.5),
            serde_json::json!(5.0),
            serde_json::json!("5"),
            serde_json::json!(true),
        ] {
            let error = parse_then_run(&serde_json::json!({
                "then_run": {"command": "true", "timeout": bad}
            }))
            .unwrap_err();
            assert!(
                error.contains("whole number of seconds"),
                "{:?} gave {}",
                bad,
                error
            );
        }
    }

    #[test]
    fn a_negative_timeout_does_not_kill_the_command() {
        let dir = scratch("neg-timeout");
        let target = dir.join("a.txt");
        let started = std::time::Instant::now();
        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(), "content": "body",
                "then_run": {"command": "echo ran", "timeout": -1}
            }),
        );
        // Rejected outright, so nothing is killed and nothing is misreported.
        assert!(result.contains(THEN_RUN_SKIPPED), "{}", result);
        assert!(!result.contains("timed out"), "{}", result);
        assert!(!result.contains(THEN_RUN_FAILED), "{}", result);
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_malformed_edit_entry_applies_nothing() {
        let dir = scratch("partial-edits");
        let target = dir.join("a.txt");
        std::fs::write(&target, "alpha beta").unwrap();
        let sentinel = dir.join("sentinel");

        // One good edit and one malformed. Applying the good one and calling it
        // success let a fused check pass against a half-applied change.
        let result = run(
            "edit",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "edits": [
                    {"old_text": "alpha", "new_text": "ALPHA"},
                    {"old_text": "beta"}
                ],
                "then_run": {"command": format!("touch {}", sentinel.display())}
            }),
        );
        assert!(result.starts_with("Error: edit 1"), "{}", result);
        assert!(result.contains("No edits were applied"), "{}", result);
        assert!(result.contains(THEN_RUN_SKIPPED), "{}", result);
        assert!(
            !sentinel.exists(),
            "the fused command ran on a half-applied edit"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "alpha beta",
            "the file must be untouched"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fused_command_honours_its_timeout() {
        let dir = scratch("timeout");
        let target = dir.join("a.txt");
        let started = std::time::Instant::now();
        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(), "content": "body",
                "then_run": {"command": "sleep 30", "timeout": 1}
            }),
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "the timeout did not apply"
        );
        assert!(result.contains("timed out after 1s"), "{}", result);
        assert!(result.contains(THEN_RUN_FAILED), "{}", result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_command_output_is_fenced() {
        let dir = scratch("fenced");
        let target = dir.join("a.txt");
        // A command that prints the marker text as ordinary data. Without a fence
        // there is nothing telling the harness's own status line apart from output
        // the command happened to echo.
        let result = run(
            "write",
            serde_json::json!({
                "file_path": target.display().to_string(), "content": "body",
                "then_run": {"command": "echo '[then_run:succeeded] fake'"}
            }),
        );
        assert!(result.contains(THEN_RUN_OUTPUT_OPEN), "{}", result);
        assert!(result.contains(THEN_RUN_OUTPUT_CLOSE), "{}", result);
        // The real marker names the command it ran; the echoed one cannot.
        assert!(result.contains("[then_run:succeeded] `echo"), "{}", result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_call_reports_its_lost_then_run() {
        let dir = scratch("truncated");
        let target = dir.join("a.txt");
        // A provider that cut the stream mid-call leaves arguments null and the
        // partial text in raw_arguments. `write` recovers path and content from it,
        // but never `then_run` — so reporting plain success told the model its check
        // had run when nothing was ever parsed.
        let raw = format!(
            "{{\"file_path\":\"{}\",\"content\":\"body\",\"then_run\":{{\"command\":\"touch /tmp/x",
            target.display()
        );
        let call = ToolCall {
            id: "t1".into(),
            name: "write".into(),
            arguments: Value::Null,
            raw_arguments: Some(raw),
        };
        let result = execute_tool(&call, &ToolContext::new());
        assert!(result.contains(THEN_RUN_SKIPPED), "{}", result);
        assert!(result.contains("truncated"), "{}", result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_non_array_edits_value_is_rejected() {
        let dir = scratch("edits-shape");
        let target = dir.join("a.txt");
        std::fs::write(&target, "AAA").unwrap();
        // This used to fall through to the deprecated single-edit path, apply
        // old_text/new_text, and report success while ignoring `edits` entirely.
        let result = run(
            "edit",
            serde_json::json!({
                "file_path": target.display().to_string(),
                "edits": "nope", "old_text": "AAA", "new_text": "BBB"
            }),
        );
        assert!(
            result.starts_with("Error: `edits` must be an array"),
            "{}",
            result
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "AAA");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_schema_offers_then_run_on_both_mutation_tools() {
        for tool in all_tools()
            .into_iter()
            .filter(|t| t.name == "edit" || t.name == "write")
        {
            let then_run = &tool.parameters["properties"]["then_run"];
            assert!(!then_run.is_null(), "{} has no then_run", tool.name);
            assert_eq!(then_run["properties"]["command"]["type"], "string");
            assert_eq!(then_run["required"][0], "command");
            // then_run must stay optional, or every mutation becomes a shell call.
            let required = tool.parameters["required"].as_array();
            if let Some(required) = required {
                assert!(!required.iter().any(|r| r == "then_run"), "{}", tool.name);
            }
        }
    }
}

#[cfg(test)]
mod stateful_tool_tests {
    use super::*;

    fn call(name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: "t1".into(),
            name: name.into(),
            arguments: args,
            raw_arguments: None,
        }
    }

    fn context_with_open_round(round: u32) -> ToolContext {
        let ctx = ToolContext::new();
        ctx.goal.set(Some("ship the fix".into()), 5);
        ctx.goal.admit_round(round);
        ctx
    }

    #[test]
    fn goal_round_accepts_every_spelling_of_the_number() {
        // Models emit 1, 1.0, and "1" interchangeably. All three name round 1.
        for round in [
            serde_json::json!(1),
            serde_json::json!(1.0),
            serde_json::json!("1"),
        ] {
            let ctx = context_with_open_round(1);
            let result = execute_tool(
                &call(
                    "goal",
                    serde_json::json!({"operation": "complete", "round": round}),
                ),
                &ctx,
            );
            assert!(
                result.contains("Goal marked complete in round 1"),
                "round {:?} was not accepted: {}",
                round,
                result
            );
        }
    }

    #[test]
    fn goal_requires_a_round_for_a_decision() {
        let ctx = context_with_open_round(1);
        let result = execute_tool(
            &call("goal", serde_json::json!({"operation": "complete"})),
            &ctx,
        );
        assert!(result.contains("`round` is required"), "{}", result);
        assert!(!ctx.goal.is_decided());
    }

    #[test]
    fn goal_rejects_an_unknown_operation() {
        let ctx = context_with_open_round(1);
        let result = execute_tool(
            &call(
                "goal",
                serde_json::json!({"operation": "finish", "round": 1}),
            ),
            &ctx,
        );
        assert!(result.contains("unknown operation"), "{}", result);
    }

    #[test]
    fn goal_read_needs_no_round() {
        let ctx = context_with_open_round(2);
        let result = execute_tool(
            &call("goal", serde_json::json!({"operation": "read"})),
            &ctx,
        );
        assert!(result.contains("Objective: ship the fix"));
        assert!(result.contains("Round: 2/5"));
    }

    #[test]
    fn todo_write_requires_the_list() {
        let ctx = ToolContext::new();
        let result = execute_tool(&call("todo_write", serde_json::json!({})), &ctx);
        assert!(result.contains("`todos` is required"), "{}", result);
    }

    #[test]
    fn todo_write_updates_the_context_it_was_given() {
        let ctx = ToolContext::new();
        let other = ToolContext::new();
        let result = execute_tool(
            &call(
                "todo_write",
                serde_json::json!({"todos": [{"content": "do the thing", "status": "in_progress"}]}),
            ),
            &ctx,
        );
        assert!(result.contains("[~] do the thing"), "{}", result);
        assert_eq!(ctx.todos.snapshot().len(), 1);
        assert!(
            other.todos.snapshot().is_empty(),
            "contexts must stay independent"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn pipe_capture_is_bounded_and_drains_excess() {
        use tokio::io::AsyncWriteExt;
        let (mut writer, reader) = tokio::io::duplex(8192);
        let sink = Arc::new(Mutex::new(Vec::new()));
        let task = tokio::spawn(drain_pipe(reader, sink.clone()));
        writer
            .write_all(&vec![b'x'; MAX_CAPTURE_BYTES * 3])
            .await
            .unwrap();
        drop(writer);
        task.await.unwrap();
        assert_eq!(sink.lock().unwrap().len(), MAX_CAPTURE_BYTES);
    }

    #[test]
    #[cfg(unix)]
    fn bash_timeout_kills_descendants() {
        let marker = std::env::temp_dir().join(format!("rupi-descendant-{}", uuid::Uuid::new_v4()));
        let command = format!("(sleep 2; echo survived > '{}') & wait", marker.display());
        let result = execute_bash(&serde_json::json!({"command":command,"timeout":1}));
        assert!(result.contains("timed out"));
        std::thread::sleep(std::time::Duration::from_millis(1300));
        assert!(!marker.exists(), "descendant survived the timeout");
    }

    #[test]
    fn a_zero_timeout_does_not_kill_the_command() {
        // `0` is a plausible way for a model to mean "no timeout". It used to be
        // read as "kill on the first loop iteration".
        let result = execute_bash(&serde_json::json!({
            "command": "echo SHOULD_HAVE_RUN", "timeout": 0
        }));
        assert!(result.contains("SHOULD_HAVE_RUN"), "{}", result);
        assert!(!result.contains("timed out"), "{}", result);
    }

    #[test]
    fn a_signal_killed_command_is_not_reported_as_clean() {
        // No exit code means the process died to a signal. Printing nothing left a
        // crash that had already written output looking exactly like a success.
        let result = execute_bash(&serde_json::json!({
            "command": "echo PRINTED_BEFORE_DEATH; kill -9 $$"
        }));
        assert!(result.contains("PRINTED_BEFORE_DEATH"), "{}", result);
        assert!(result.contains("killed or did not finish"), "{}", result);
    }

    #[test]
    fn test_bash_echo() {
        let args = serde_json::json!({"command": "echo hello"});
        let result = execute_bash(&args);
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn test_read_file() {
        let dir = std::env::temp_dir().join("rupi-tools-test-read-file");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        fs::write(&path, "line1\nline2\nline3\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy()});
        let result = execute_read(&args);
        assert!(result.contains("line1"));
        assert!(result.contains("line2"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_file_with_offset() {
        let dir = std::env::temp_dir().join("rupi-tools-test-read-offset");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        fs::write(&path, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        let args =
            serde_json::json!({"file_path": path.to_string_lossy(), "offset": 3, "limit": 2});
        let result = execute_read(&args);
        assert!(result.contains("line3"));
        assert!(result.contains("line4"));
        assert!(!result.contains("line1"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_creates_new_file() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-file");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("new_file.txt");

        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": path.to_string_lossy(), "content": "hello\nworld"}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("Successfully wrote"));
        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello\nworld");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_refuses_on_existing_file() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-refuse");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("existing.txt");
        fs::write(&path, "original content").unwrap();

        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": path.to_string_lossy(), "content": "new content"}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("Write refused"));
        // Content should be unchanged
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "original content");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_guard_returns_edit_recipe() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-recipe");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("existing.txt");
        fs::write(&path, "content").unwrap();

        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": path.to_string_lossy(), "content": "new"}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("use Edit"));
        assert!(result.contains("old_text"));
        assert!(result.contains("new_text"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_singular() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-file");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "old_text": "hello", "new_text": "goodbye"});
        let result = execute_edit(&args);
        assert!(result.contains("Successfully applied edit"));
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "goodbye world\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multi_edits() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-multi");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi.txt");
        fs::write(&path, "AAA line\nBBB line\nCCC line\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "edits": [
                {"old_text": "AAA", "new_text": "XXX"},
                {"old_text": "CCC", "new_text": "ZZZ"}
            ]
        });
        let result = execute_edit(&args);
        assert!(result.contains("2 edits"));
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "XXX line\nBBB line\nZZZ line\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_not_found() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-not-found");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "old_text": "nonexistent", "new_text": "foo"});
        let result = execute_edit(&args);
        assert!(result.contains("not found"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multiple_matches_singular() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-multi-singular");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        fs::write(&path, "hello hello\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "old_text": "hello", "new_text": "foo"});
        let result = execute_edit(&args);
        assert!(result.contains("2 times"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multi_overlap_detected() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-overlap");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overlap.txt");
        fs::write(&path, "hello world foo\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "edits": [
                {"old_text": "hello world", "new_text": "goodbye"},
                {"old_text": "world foo", "new_text": "moon"}
            ]
        });
        let result = execute_edit(&args);
        assert!(result.contains("overlap"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multi_one_fails_all_fail() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-partial");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial.txt");
        fs::write(&path, "AAA line\nBBB line\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "edits": [
                {"old_text": "AAA", "new_text": "XXX"},
                {"old_text": "NONEXISTENT", "new_text": "YYY"}
            ]
        });
        let result = execute_edit(&args);
        assert!(result.contains("Edit failed"));
        // File should be unchanged
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "AAA line\nBBB line\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_grep() {
        let dir = std::env::temp_dir().join("rupi-tools-test-grep");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("test.txt"), "hello world\nfoo bar\n").unwrap();

        let args = serde_json::json!({"pattern": "hello", "path": dir.to_string_lossy()});
        let result = execute_grep(&args);
        assert!(result.contains("hello"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_ls() {
        let dir = std::env::temp_dir().join("rupi-tools-test-ls");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(dir.join("b.txt"), "b").unwrap();

        let args = serde_json::json!({"path": dir.to_string_lossy()});
        let result = execute_ls(&args);
        assert!(result.contains("a.txt"));
        assert!(result.contains("b.txt"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_serialize_tools() {
        let tools = all_tools();
        let serialized = serialize_tools(&tools);
        assert_eq!(serialized.len(), all_tools().len());
        assert_eq!(serialized[0]["function"]["name"], "bash");
        assert_eq!(serialized[1]["function"]["name"], "read");
        assert_eq!(serialized[2]["function"]["name"], "write");
        assert_eq!(serialized[3]["function"]["name"], "edit");
        assert_eq!(serialized[4]["function"]["name"], "grep");
        assert_eq!(serialized[5]["function"]["name"], "find");
        assert_eq!(serialized[6]["function"]["name"], "ls");
        assert_eq!(serialized[7]["function"]["name"], "search_code");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let tc = ToolCall {
            id: "call_1".into(),
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
            raw_arguments: None,
        };
        let result = execute_tool(&tc, &ToolContext::new());
        assert!(result.contains("Unknown tool"));
    }

    #[test]
    fn test_read_file_not_found() {
        let args = serde_json::json!({"file_path": "/nonexistent/path/file.txt"});
        let result = execute_read(&args);
        assert!(result.contains("not found"));
    }

    #[test]
    fn test_read_missing_path() {
        let args = serde_json::json!({});
        let result = execute_read(&args);
        assert!(result.contains("missing"));
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-parent");
        let nested = dir.join("nested").join("deep").join("file.txt");
        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": nested.to_string_lossy(), "content": "test"}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("Successfully wrote"));
        assert!(nested.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_creates_new_even_if_parent_doesnt_exist() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-deep");
        let nested = dir.join("a").join("b").join("c").join("f.txt");
        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": nested.to_string_lossy(), "content": "hi"}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("Successfully wrote"));
        assert!(nested.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_truncation_recovery_valid_json() {
        let dir = std::env::temp_dir().join("rupi-tools-test-trunc-valid");
        let _ = fs::remove_dir_all(&dir);
        let file_path = dir.join("truncated.html");
        // Simulate truncated JSON: valid JSON but with raw_arguments containing partial content
        let raw = r#"{"file_path":"/tmp/rupi-tools-test-trunc-valid/truncated.html","content":"<html><body><h1>Hello</h1>"}"#;
        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": file_path.to_string_lossy()}), // content missing
            raw_arguments: Some(raw.to_string()),
        };
        let result = execute_write(&tc);
        assert!(
            result.contains("truncated"),
            "Expected truncation warning, got: {}",
            result
        );
        assert!(file_path.exists());
        let written = fs::read_to_string(&file_path).unwrap();
        assert!(written.contains("<html>"));
        assert!(written.contains("RUPI_TRUNCATED"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_truncation_recovery_malformed_json() {
        let dir = std::env::temp_dir().join("rupi-tools-test-trunc-malformed");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("broken.html");
        // Simulate severely truncated JSON: content string cut off, JSON is invalid
        let raw = format!(
            r#"{{"file_path":"{}","content":"<html><body> Hel"#,
            file_path.to_string_lossy()
        );
        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({}), // empty from failed parse
            raw_arguments: Some(raw),
        };
        let result = execute_write(&tc);
        assert!(
            result.contains("truncated"),
            "Expected truncation warning, got: {}",
            result
        );
        assert!(file_path.exists());
        let written = fs::read_to_string(&file_path).unwrap();
        assert!(written.contains("<html>"));
        assert!(written.contains("RUPI_TRUNCATED"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_no_content_no_raw_returns_error() {
        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({"file_path": "/tmp/test.txt"}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("missing 'content'"));
    }

    #[test]
    fn test_tool_def_names() {
        for tool in all_tools() {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty());
            assert!(
                tool.parameters.get("properties").is_some()
                    || tool.parameters.get("type").is_some()
            );
        }
    }

    #[test]
    fn test_edit_empty_edits_array() {
        let dir = std::env::temp_dir().join("rupi-tools-test-empty-edits");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.txt");
        fs::write(&path, "content").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "edits": []});
        let result = execute_edit(&args);
        assert!(result.contains("empty"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_missing_file_path() {
        let args = serde_json::json!({"old_text": "foo", "new_text": "bar"});
        let result = execute_edit(&args);
        assert!(result.contains("missing"));
    }

    #[test]
    fn test_edit_file_not_found() {
        let args = serde_json::json!({"file_path": "/tmp/rupi-nonexistent-12345", "edits": [{"old_text": "foo", "new_text": "bar"}]});
        let result = execute_edit(&args);
        assert!(result.contains("not found"));
    }

    #[test]
    fn test_write_missing_args() {
        let tc = ToolCall {
            id: "test".into(),
            name: "write".into(),
            arguments: serde_json::json!({}),
            raw_arguments: None,
        };
        let result = execute_write(&tc);
        assert!(result.contains("missing"));
    }

    #[test]
    fn a_backgrounded_child_does_not_stall_the_tool() {
        // The command exits immediately but leaves a process holding its stdout.
        // Reading to EOF waits for *that* process, so the tool used to block
        // forever — past its own timeout, since the timeout only covers the
        // command itself. This is the normal shape of "start a browser/server
        // and leave it running", which is exactly what an embedder's helper
        // scripts do.
        let args = serde_json::json!({"command": "sleep 30 & echo STARTED", "timeout": 5});
        let began = std::time::Instant::now();
        let result = execute_bash(&args);
        let elapsed = began.elapsed();

        assert!(result.contains("STARTED"), "output lost: {result}");
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "tool waited {elapsed:?} for a process that outlived the command"
        );
    }

    #[test]
    fn a_timeout_still_returns_what_was_printed() {
        let args = serde_json::json!({"command": "echo PARCIAL; sleep 30", "timeout": 1});
        let result = execute_bash(&args);
        assert!(result.contains("timed out"), "{result}");
        assert!(
            result.contains("PARCIAL"),
            "partial output was discarded: {result}"
        );
    }

    #[test]
    fn test_bash_missing_command() {
        let args = serde_json::json!({});
        let result = execute_bash(&args);
        assert!(result.contains("missing"));
    }

    #[test]
    fn test_ls_nonexistent_dir() {
        let args = serde_json::json!({"path": "/tmp/rupi-nonexistent-dir-99999"});
        let result = execute_ls(&args);
        assert!(result.contains("not found"));
    }

    #[test]
    fn test_search_code_missing_query() {
        let tc = ToolCall {
            id: "e1".into(),
            name: "search_code".into(),
            arguments: serde_json::json!({"path": "."}),
            raw_arguments: None,
        };
        let result = execute_tool(&tc, &ToolContext::new());
        assert!(result.contains("missing"));
    }
}
