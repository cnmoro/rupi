use std::sync::Arc;

pub type ApprovalFn = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;
use tokio::sync::mpsc;
use tokio::sync::{watch, Mutex, RwLock};

use crate::error::AgentError;
use crate::provider::openai::{OpenAIConfig, OpenAIProvider};
use crate::provider::{ChatProvider, StreamEvent};
use crate::rpc::types::*;

/// A message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
    pub tool_call_id: Option<String>,
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn tool_call(assistant_content: &str, calls: Vec<crate::tools::ToolCall>) -> Self {
        Message {
            role: "assistant".to_string(),
            content: assistant_content.to_string(),
            tool_calls: Some(calls),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Message {
            role: "tool".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            reasoning_content: None,
        }
    }

    pub fn with_reasoning(mut self, reasoning: Option<String>) -> Self {
        self.reasoning_content = reasoning;
        self
    }
}

use crate::compaction::{self, CompactionResult};
use crate::output_parser;
use crate::quality;
use crate::sessions;
use crate::skills::Skill;
use crate::tools::{self, ToolCall};

fn memory_file_path() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let dir = home.join(".config").join("rupi");
    let _ = std::fs::create_dir_all(&dir);
    Some(dir.join("MEMORY.md"))
}

fn ensure_memory_file() -> String {
    let path = match memory_file_path() {
        Some(p) => p,
        None => return String::new(),
    };
    if path.exists() {
        std::fs::read_to_string(&path).unwrap_or_default()
    } else {
        String::new()
    }
}

/// Maximum characters to store for a tool result in the conversation.
/// Larger results are truncated to reduce KV cache prefill work in subsequent turns.
/// The full result is still sent to the event stream; only the stored message is capped.
const MAX_TOOL_RESULT_CHARS: usize = 4000;

/// Truncate a tool result for storage in the conversation, keeping KV cache cost bounded.
/// Keeps the first ~70% and last ~30% of long results so the model can still see both
/// the beginning and end of tool output.
fn capped_tool_result(result: &str) -> String {
    if result.len() <= MAX_TOOL_RESULT_CHARS {
        return result.to_string();
    }
    let keep_beginning = MAX_TOOL_RESULT_CHARS * 7 / 10;
    let keep_end = MAX_TOOL_RESULT_CHARS - keep_beginning;

    // Cut on line boundaries. Tool output is lines — a stack trace, a compiler
    // error, a directory listing — and a cut in the middle of one leaves a
    // fragment that reads as a different, shorter message than it is.
    //
    // Line snapping is only worth it when it keeps most of the budget. One short
    // framing line followed by one enormous line — `curl | jq -c`, a webpack log,
    // a `docker inspect` — would otherwise pin the head at that framing line and
    // throw the rest of the budget away, which is far worse than a clean cut
    // mid-line. Below `MIN_LINE_CUT_RATIO` of the budget, take the characters.
    let head = best_excerpt(result, keep_beginning, true);
    let tail = best_excerpt(result, keep_end, false);

    let omitted = result.len().saturating_sub(head.len() + tail.len());
    let rendered = format!("{}... [truncated: {} bytes]\n...{}", head, omitted, tail);
    // Truncation must never inflate. Just over the cap, head plus tail can cover
    // the whole input and the marker is pure overhead.
    if rendered.len() >= result.len() {
        return result.to_string();
    }
    rendered
}

/// Smallest share of a budget a line-aligned excerpt must fill to be worth taking.
const MIN_LINE_CUT_RATIO: usize = 2;

/// The better of a line-aligned and a character-aligned excerpt for one budget.
fn best_excerpt(text: &str, budget: usize, from_start: bool) -> &str {
    let lines = if from_start {
        head_lines(text, budget)
    } else {
        tail_lines(text, budget)
    };
    if lines.len() * MIN_LINE_CUT_RATIO >= budget {
        return lines;
    }
    if from_start {
        head_chars(text, budget)
    } else {
        tail_chars(text, budget)
    }
}

/// Whole lines from the start of `text`, within `budget` bytes.
fn head_lines(text: &str, budget: usize) -> &str {
    let mut end = 0;
    for line in text.split_inclusive('\n') {
        if end + line.len() > budget {
            break;
        }
        end += line.len();
    }
    &text[..end]
}

/// Whole lines from the end of `text`, within `budget` bytes.
///
/// Walks backwards over newline positions instead of materializing every line.
/// Collecting first cost a vector proportional to the whole input to select the
/// last few lines of it, and tool output reaches hundreds of megabytes.
fn tail_lines(text: &str, budget: usize) -> &str {
    let bytes = text.as_bytes();
    let mut start = text.len();
    loop {
        // The newline that ends the line before `start`.
        let search_end = start.saturating_sub(1);
        let previous = bytes[..search_end].iter().rposition(|b| *b == b'\n');
        let candidate = match previous {
            Some(index) => index + 1,
            None => 0,
        };
        if text.len() - candidate > budget {
            break;
        }
        start = candidate;
        if start == 0 {
            break;
        }
    }
    &text[start..]
}

/// Largest prefix of `text` within `budget` bytes, cut on a character boundary.
fn head_chars(text: &str, budget: usize) -> &str {
    let mut end = budget.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Largest suffix of `text` within `budget` bytes, cut on a character boundary.
fn tail_chars(text: &str, budget: usize) -> &str {
    let mut start = text.len().saturating_sub(budget);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// The date to put in the system prompt.
///
/// Deliberately day-granular. Providers cache on an exact token prefix, and
/// this sits ~200 characters into the *first* message — so a clock that ticks
/// makes every process a cache miss for the whole conversation, not just for
/// this line. Two runs a second apart shared 226 characters before this.
///
/// The exact time is a `date` call away when a task actually needs it.
fn current_date_for_prompt() -> String {
    chrono::Local::now()
        .format("%A, %B %d, %Y (%Z)")
        .to_string()
}

/// Build the system prompt describing available tools, skills, context files, and memory.
/// If `datetime` is provided, it is used as the current time (for KV cache stability).
/// Otherwise, `chrono::Local::now()` is used (for one-shot prompts like goal verification).
fn build_system_prompt(
    skills: &[Skill],
    context_files: &[ContextFile],
    memory_enabled: bool,
    datetime: Option<&str>,
) -> String {
    let time_str = match datetime {
        Some(d) => d.to_string(),
        None => current_date_for_prompt(),
    };
    let mut prompt = format!(
        "You are an expert coding agent operating inside rupi, a coding agent harness. \
        You help users by reading files, executing commands, editing code, and writing new files.

Current date: {}\nThe exact time of day is not given here — run `date` if a task needs it.

Available tools:
- bash: Execute bash commands (ls, grep, find, curl, git, compilers, etc.). Returns stdout and stderr. Optionally provide a timeout in seconds.
- read: Read file contents with optional line offset/limit.
- write: Create a NEW file. REFUSES if the file already exists — use edit to modify existing files instead. Creates parent directories if needed.
- edit: Replace exact text in a file. Supports batch edits via the edits array. Each old_text is matched against the ORIGINAL file content (not after other edits). Edits must not overlap. Prefer this over write for any change to an existing file.
- edit and write both accept an optional then_run: {{\"command\": \"...\"}}. It runs that command in the SAME call, right after the change lands. Use it whenever you already know what you would run next to check the change — the tests, a build, a linter, a restart. It is skipped if the change fails, and a non-zero exit is reported without undoing the change.
- grep: Search file contents for patterns (uses ripgrep, respects .gitignore, falls back to grep). Takes output_mode, head_limit and context — start with output_mode \"files_with_matches\" for a broad search, then narrow.
- find: Find files by glob pattern (uses fd, respects .gitignore, falls back to find).
- ls: List directory contents.
- search_code: Search code using natural language queries. Uses a local AI model (Model2Vec with potion-code-16M) to find relevant code by what it does, not just by keyword matching. Describe what you are looking for in plain English. Falls back to keyword search if the model is unavailable.
- todo_write: Record and update the task list for multi-step work. Send the ENTIRE list every call — it replaces the previous one. Keep at most one task in_progress. Skip it for trivial single-step tasks.
- goal: Read or decide the durable goal, when one is set. Call it with operation \"complete\" once evidence shows the whole objective is met, or \"block\" with a reason when you cannot proceed. Pass the round number from the current <goal_round> block.

Guidelines:
- Be concise in your responses
- Show file paths clearly when working with files
- Use bash to explore when you are unsure about the project structure or when you need to gather information
- When a command fails, read the error output and try a different approach rather than giving up
- If you don't have enough information to complete a task, use bash, read, grep, or find to get the necessary context
- For work of more than a few steps, plan it with todo_write first and keep the list current as you go
- Fuse the check into the change: pass then_run on edit or write instead of spending a separate turn on the command that verifies it
- Ask narrow questions of the tools. A whole-repository `git diff` runs to tens of thousands of characters and is truncated before you see it: run `git diff --stat` first, then diff the specific paths. The same applies to grep — find the files first, then read what matters
- A message inside <active-task> tags re-states the request that started this session. It is context, not a new instruction — do not restart finished work when you see it
- A message inside <compacted-summary> tags is a checkpoint of earlier context. Treat it as established background and continue from the messages after it",
        time_str
    );

    // Append skills as XML block
    if !skills.is_empty() {
        prompt.push_str("\n\nSkills available in this environment:");
        for skill in skills {
            prompt.push_str(&format!(
                "\n<skill>\n  <name>{}</name>\n  <description>{}</description>\n  <location>{}</location>\n</skill>",
                skill.name, skill.description, skill.file_path.display()
            ));
        }
    }

    // Append memory section if enabled
    if memory_enabled {
        if let Some(path) = memory_file_path() {
            prompt.push_str(&format!(
                "\n\nPersistent memory: you have a MEMORY.md file at {}. \
                At the START of each response, read it with the read tool if it exists. \
                During your work, if you discover important facts, decisions, or progress \
                that should be remembered across sessions, OVERWRITE the file with an updated version \
                using the write tool. Keep it concise — bullet points of key facts and decisions only.\n\
                The file may be empty if nothing has been saved yet.",
                path.display()
            ));
            // Read existing memory content and append it if present
            let existing = ensure_memory_file();
            if !existing.is_empty() {
                prompt.push_str("\n\nCurrent MEMORY.md contents:\n");
                prompt.push_str(&existing);
            }
        }
    }

    // Append context files content
    for cf in context_files {
        prompt.push_str(&format!(
            "\n\n<context-file name=\"{}\">\n{}\n</context-file>",
            cf.name, cf.content
        ));
    }

    prompt
}

/// A context file (CLAUDE.md, AGENTS.md) loaded from cwd or ancestors.
#[derive(Debug, Clone)]
pub struct ContextFile {
    pub name: String,
    pub content: String,
}

/// Load context files from cwd and ancestor directories.
pub fn load_context_files(cwd: &str) -> Vec<ContextFile> {
    let mut files = Vec::new();
    let mut current = std::path::PathBuf::from(cwd);

    // Normalize and make absolute
    if !current.is_absolute() {
        if let Ok(cwd) = std::env::current_dir() {
            current = cwd.join(&current);
        }
    }

    // Collect ancestor dirs including cwd
    let mut dirs = Vec::new();
    dirs.push(current.clone());
    while let Some(parent) = current.parent() {
        dirs.push(parent.to_path_buf());
        current = parent.to_path_buf();
    }

    let names = ["CLAUDE.md", "AGENTS.md"];
    for dir in dirs {
        for name in &names {
            let path = dir.join(name);
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    // Avoid duplicates (if parent and child have same file)
                    if !files
                        .iter()
                        .any(|f: &ContextFile| f.name == *name && f.content == content)
                    {
                        files.push(ContextFile {
                            name: name.to_string(),
                            content,
                        });
                    }
                }
            }
        }
    }

    files
}

/// Recover the originating request from a transcript loaded off disk.
///
/// Resume rebuilds a session with no in-memory state, so the anchor has to be
/// readable out of the history itself. A previously emitted anchor block is the
/// best source because it is exact. Failing that, the first real user message is
/// the request, skipping checkpoints and anchors, which are host-written.
pub fn recover_anchor(messages: &[Message]) -> Option<String> {
    // Only `user` messages are considered. rupi writes its anchors on that role,
    // and the restriction is what keeps untrusted content out: a tool result is
    // role `tool`, so a full anchor block sitting inside a file the agent read, or
    // a page it fetched, can no longer be adopted as the session's own request and
    // re-emitted under framing that tells the model to trust it.
    for msg in messages.iter().rev() {
        if msg.role != "user" {
            continue;
        }
        if crate::anchor::is_anchor(&msg.content) {
            if let Some(request) = crate::anchor::extract_request(&msg.content) {
                return Some(request);
            }
        }
    }
    messages
        .iter()
        .find(|m| {
            m.role == "user"
                && !crate::anchor::is_anchor(&m.content)
                && !m.content.starts_with(sessions::COMPACTION_PREFIX)
                // Goal rounds are host-written and arrive on the user role too.
                // Without this they could be recovered as "the request that
                // started this session" once a compaction dropped the real one.
                && !m.content.trim_start().starts_with("<goal_round>")
                && !m.content.trim().is_empty()
        })
        .map(|m| m.content.clone())
}

/// Agent session manages conversation state and model interaction.
pub struct AgentSession {
    provider: Arc<dyn ChatProvider>,
    lifecycle: RwLock<()>,
    admission: Mutex<()>,
    resetting: std::sync::atomic::AtomicBool,
    approval_fn: RwLock<Option<ApprovalFn>>,
    memory_enabled: bool,
    model: std::sync::RwLock<String>,
    pending_steer: RwLock<Vec<String>>,
    pending_follow_up: RwLock<Vec<String>>,
    context_window: u64,
    #[allow(dead_code)]
    cwd: String,
    skills: Vec<Skill>,
    context_files: Vec<ContextFile>,
    messages: RwLock<Vec<Message>>,
    session_path: RwLock<Option<std::path::PathBuf>>,
    is_streaming: std::sync::atomic::AtomicBool,
    is_compacting: Mutex<bool>,
    abort_signal: Mutex<Option<watch::Sender<bool>>>,
    abort_requested: Mutex<bool>,
    empty_response_retries: Mutex<u32>,
    thinking_level: RwLock<String>,
    auto_compaction_enabled: RwLock<bool>,
    recent_tool_calls: RwLock<Vec<Vec<crate::tools::ToolCall>>>,
    consecutive_quality_issues: RwLock<u32>,
    loop_prompt: RwLock<Option<String>>,
    loop_cancelled: std::sync::atomic::AtomicBool,
    system_prompt: RwLock<String>,
    /// The verbatim request that started the current task.
    ///
    /// Compaction deletes everything outside the recent tail, so the originating
    /// user message does not survive a long run. Holding it here makes it a value
    /// the session owns rather than history that a summarizer might restate.
    task_anchor: RwLock<Option<String>>,
    /// How many times the anchor has been re-emitted into the conversation.
    anchor_emissions: RwLock<u32>,
    /// Tool results stored since the last user-role message.
    ///
    /// Drives the tail re-emission: appending at the end costs nothing in cache
    /// terms, while inserting in the middle would invalidate every token after it.
    tool_results_since_user: RwLock<u32>,
    /// Session-scoped state for the stateful tools (`todo_write` and `goal`).
    tool_context: tools::ToolContext,
}

impl AgentSession {
    pub fn new(
        provider: Arc<dyn ChatProvider>,
        model: String,
        context_window: u64,
        cwd: String,
        skills: Vec<Skill>,
        context_files: Vec<ContextFile>,
    ) -> Self {
        Self::new_with_session_path(
            provider,
            model,
            context_window,
            cwd,
            skills,
            context_files,
            None,
        )
    }

    fn new_with_session_path(
        provider: Arc<dyn ChatProvider>,
        model: String,
        context_window: u64,
        cwd: String,
        skills: Vec<Skill>,
        context_files: Vec<ContextFile>,
        existing_path: Option<std::path::PathBuf>,
    ) -> Self {
        let session_path = existing_path.or_else(|| sessions::create_session(&model).ok());
        // Freeze the timestamp at session creation so the system prompt stays
        // byte-identical across turns — critical for server-side KV prefix caching.
        let frozen_time = current_date_for_prompt();
        let system_prompt = build_system_prompt(&skills, &context_files, false, Some(&frozen_time));
        AgentSession {
            provider,
            lifecycle: RwLock::new(()),
            admission: Mutex::new(()),
            resetting: std::sync::atomic::AtomicBool::new(false),
            approval_fn: RwLock::new(None),
            memory_enabled: false,
            model: std::sync::RwLock::new(model),
            pending_steer: RwLock::new(Vec::new()),
            pending_follow_up: RwLock::new(Vec::new()),
            context_window,
            cwd,
            skills,
            context_files,
            messages: RwLock::new(Vec::new()),
            session_path: RwLock::new(session_path),
            is_streaming: std::sync::atomic::AtomicBool::new(false),
            is_compacting: Mutex::new(false),
            abort_signal: Mutex::new(None),
            abort_requested: Mutex::new(false),
            empty_response_retries: Mutex::new(0),
            thinking_level: RwLock::new("off".to_string()),
            auto_compaction_enabled: RwLock::new(true),
            recent_tool_calls: RwLock::new(Vec::new()),
            consecutive_quality_issues: RwLock::new(0),
            loop_prompt: RwLock::new(None),
            loop_cancelled: std::sync::atomic::AtomicBool::new(false),
            system_prompt: RwLock::new(system_prompt),
            task_anchor: RwLock::new(None),
            anchor_emissions: RwLock::new(0),
            tool_results_since_user: RwLock::new(0),
            tool_context: tools::ToolContext::new(),
        }
    }

    pub fn from_config(config: OpenAIConfig) -> Self {
        Self::from_config_with(
            config,
            std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            Vec::new(),
            Vec::new(),
            false,
        )
    }

    pub fn from_config_with(
        config: OpenAIConfig,
        cwd: String,
        skills: Vec<Skill>,
        context_files: Vec<ContextFile>,
        memory_enabled: bool,
    ) -> Self {
        let context_window = config.context_window;
        let model = config.model.clone();
        let provider = Arc::new(OpenAIProvider::new(config));
        let mut session = Self::new(
            provider as Arc<dyn ChatProvider>,
            model,
            context_window,
            cwd,
            skills,
            context_files,
        );
        session.memory_enabled = memory_enabled;
        if memory_enabled {
            ensure_memory_file();
            // Rebuild system prompt with memory content included
            let frozen_time = current_date_for_prompt();
            let prompt = build_system_prompt(
                &session.skills,
                &session.context_files,
                memory_enabled,
                Some(&frozen_time),
            );
            *session.system_prompt.get_mut() = prompt;
        }
        session
    }

    /// Resume a session from an existing session file.
    /// Loads all messages from the file and reuses it for further appends.
    pub async fn from_session(
        config: OpenAIConfig,
        session_path: std::path::PathBuf,
        cwd: String,
        skills: Vec<Skill>,
        context_files: Vec<ContextFile>,
        memory_enabled: bool,
    ) -> Result<Self, String> {
        let messages = crate::sessions::load_session(&session_path)?;
        let context_window = config.context_window;
        let model = config.model.clone();
        let provider = Arc::new(OpenAIProvider::new(config));
        let mut session = Self::new_with_session_path(
            provider as Arc<dyn ChatProvider>,
            model,
            context_window,
            cwd,
            skills,
            context_files,
            Some(session_path.clone()),
        );
        session.memory_enabled = memory_enabled;
        if memory_enabled {
            ensure_memory_file();
            let frozen_time = current_date_for_prompt();
            let prompt = build_system_prompt(
                &session.skills,
                &session.context_files,
                memory_enabled,
                Some(&frozen_time),
            );
            *session.system_prompt.get_mut() = prompt;
        }
        // Load existing messages into the session
        for msg in &messages {
            session.messages.write().await.push(msg.clone());
        }
        // Restore the anchor so a resumed session that compacts again still carries
        // the request it started from.
        *session.task_anchor.write().await = recover_anchor(&messages);
        *session.session_path.write().await = Some(session_path);
        Ok(session)
    }

    /// Get the session file path, if any.
    pub async fn session_path(&self) -> Option<std::path::PathBuf> {
        self.session_path.read().await.clone()
    }

    /// The session identity used to scope spill artifacts.
    async fn session_id(&self) -> String {
        self.session_path
            .read()
            .await
            .as_ref()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .unwrap_or_else(|| "unscoped".to_string())
    }

    /// Store one tool result in the conversation, spilling the full text first.
    ///
    /// Truncation used to destroy the discarded bytes outright. Writing them to a
    /// session-scoped file first costs one write and keeps the whole result
    /// reachable through the `read` tool the agent already has, so the context stays
    /// small without losing data.
    async fn store_tool_result(&self, tool_name: &str, result: &str) -> String {
        if result.len() <= MAX_TOOL_RESULT_CHARS {
            return result.to_string();
        }
        let mut stored = capped_tool_result(result);
        let session_id = self.session_id().await;

        // Archive off the runtime. Hashing and writing a large result is hundreds
        // of milliseconds of synchronous work, and doing it inline holds a worker
        // thread for the whole time — on a one or two core host that stalls the
        // interactive loop.
        let owned_tool = tool_name.to_string();
        let owned_result = result.to_string();
        let spill = tokio::task::spawn_blocking(move || {
            crate::spill::save_text(&session_id, &owned_tool, &owned_result)
        })
        .await
        .unwrap_or(None);

        match spill {
            Some(spill) => stored.push_str(&crate::spill::retrieval_hint(&spill)),
            // Say so in the result itself. The truncated bytes really are gone at
            // this point, and letting the agent believe it can read them back is
            // worse than telling it they are lost.
            None => stored.push_str(
                "\n[The full result could not be archived, so the truncated part is not \
recoverable. Re-run the command if you need it.]",
            ),
        }
        stored
    }

    pub fn set_model(&self, new_model: String) {
        if let Ok(mut m) = self.model.write() {
            *m = new_model;
        }
    }

    pub fn model(&self) -> String {
        self.model.read().map(|m| m.clone()).unwrap_or_default()
    }

    pub async fn messages(&self) -> Vec<Message> {
        self.messages.read().await.clone()
    }

    pub async fn thinking_level(&self) -> String {
        self.thinking_level.read().await.clone()
    }

    pub async fn set_thinking_level(&self, level: String) {
        *self.thinking_level.write().await = level;
    }

    pub async fn cycle_thinking_level(&self) -> Option<String> {
        let mut level = self.thinking_level.write().await;
        *level = match level.as_str() {
            "off" => "low".to_string(),
            "low" => "medium".to_string(),
            "medium" => "high".to_string(),
            "high" => "off".to_string(),
            _ => "off".to_string(),
        };
        Some(level.clone())
    }

    pub async fn auto_compaction_enabled(&self) -> bool {
        *self.auto_compaction_enabled.read().await
    }

    pub async fn set_auto_compaction_enabled(&self, enabled: bool) {
        *self.auto_compaction_enabled.write().await = enabled;
    }

    /// Whether the agent is currently streaming a response.
    pub async fn is_streaming(&self) -> bool {
        self.is_streaming.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub async fn message_count(&self) -> u64 {
        self.messages.read().await.len() as u64
    }

    /// Persist a message to the session file.
    async fn persist_message(&self, msg: &Message) {
        if let Some(ref path) = *self.session_path.read().await {
            if let Err(e) = sessions::append_message(path, msg) {
                eprintln!("rupi: failed to persist message: {}", e);
            }
        }
    }

    /// Persist a compaction record to the session file.
    async fn persist_compaction(&self, summary: &str, tokens_before: u64) {
        if let Some(ref path) = *self.session_path.read().await {
            if let Err(e) = sessions::append_compaction(path, summary, tokens_before) {
                eprintln!("rupi: failed to persist compaction: {}", e);
            }
        }
    }

    /// Persist a provider generation id so cost can be reconciled later.
    async fn persist_generation_id(&self, generation_id: &str) {
        if let Some(ref path) = *self.session_path.read().await {
            if let Err(e) = sessions::append_generation_id(path, generation_id) {
                eprintln!("rupi: failed to persist generation id: {}", e);
            }
        }
    }

    /// Reset the session (clear messages, create new session file).
    pub async fn reset(&self) {
        self.resetting
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.loop_cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.abort().await;
        // Wait for the generation (including blocking tools) to finish before
        // changing either history or its transcript destination.
        let _lifecycle = self.lifecycle.write().await;
        // Wait for any in-progress compaction to finish
        let waited_from = std::time::Instant::now();
        loop {
            let c = *self.is_compacting.lock().await;
            if !c {
                break;
            }
            // Bounded. `is_compacting` is cleared by hand at each exit of
            // `compact`, so a panic in between leaves it set for the life of the
            // process, and an unbounded poll here turns that into a reset that
            // never returns. Force it instead, and say so.
            if waited_from.elapsed() > std::time::Duration::from_secs(30) {
                eprintln!("rupi: compaction did not finish within 30s; resetting anyway");
                *self.is_compacting.lock().await = false;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        self.is_streaming
            .store(false, std::sync::atomic::Ordering::SeqCst);
        *self.is_compacting.lock().await = false;
        self.messages.write().await.clear();
        self.recent_tool_calls.write().await.clear();
        *self.consecutive_quality_issues.write().await = 0;
        self.pending_steer.write().await.clear();
        self.pending_follow_up.write().await.clear();
        self.set_goal(None).await;
        self.set_loop(None).await;
        *self.task_anchor.write().await = None;
        *self.anchor_emissions.write().await = 0;
        *self.tool_results_since_user.write().await = 0;
        self.tool_context.todos.reset();
        *self.auto_compaction_enabled.write().await = true;
        *self.thinking_level.write().await = "off".to_string();

        // Rebuild system prompt with a fresh frozen timestamp for the new session
        let frozen_time = current_date_for_prompt();
        let prompt = build_system_prompt(
            &self.skills,
            &self.context_files,
            self.memory_enabled,
            Some(&frozen_time),
        );
        *self.system_prompt.write().await = prompt;

        *self.abort_signal.lock().await = None;
        let new_path = sessions::create_session(&self.model()).ok();
        *self.session_path.write().await = new_path;
        *self.abort_requested.lock().await = false;
        *self.empty_response_retries.lock().await = 0;
        self.tool_context
            .cancelled
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.resetting
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Stream a prompt to the model. Events are sent to the event_tx channel.
    /// Handles multi-turn tool execution (bash, etc.) internally.
    /// If a goal is set, loops until the goal is verified.
    /// If streaming and `streaming_behavior` is "steer" or "followUp", queues instead.
    /// Run a turn, guaranteeing the caller is told when it ends.
    ///
    /// The turn body has several error exits (retry caps, provider failures,
    /// aborts) and some returned without emitting `agent_end`. A caller driving
    /// rupi over RPC waits on that event, so those paths left it hanging until
    /// its own timeout — indistinguishable from a model that is simply slow.
    /// Emitting it here means every exit is terminal, whatever the body does.
    pub async fn prompt(
        &self,
        message: &str,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        self.prompt_with_behavior(message, event_tx, None).await
    }

    pub async fn prompt_with_behavior(
        &self,
        message: &str,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
        streaming_behavior: Option<&str>,
    ) -> Result<(), AgentError> {
        let _lifecycle = self.lifecycle.read().await;
        if self.resetting.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(AgentError::Cancelled);
        }
        let result = self
            .prompt_inner(message, event_tx.clone(), streaming_behavior)
            .await;
        if let Err(ref e) = result {
            eprintln!("rupi: turn ended with error: {}", e);
            let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                role: "assistant".to_string(),
                content: vec![MessageContent {
                    content_type: "text".to_string(),
                    text: Some(format!("Error: {}", e)),
                }],
                model: None,
                usage: None,
                stop_reason: Some("error".to_string()),
            }));
            let _ = event_tx.send(AgentEvent::turn_end());
            let _ = event_tx.send(AgentEvent::agent_end());
        }
        result
    }

    async fn prompt_inner(
        &self,
        message: &str,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
        streaming_behavior: Option<&str>,
    ) -> Result<(), AgentError> {
        let admission = self.admission.lock().await;
        if self.is_streaming.load(std::sync::atomic::Ordering::SeqCst) {
            if streaming_behavior == Some("steer") {
                self.steer(message).await;
            } else {
                self.follow_up(message).await;
            }
            return Ok(());
        }
        self.is_streaming
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *self.abort_requested.lock().await = false;
        self.tool_context
            .cancelled
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let terminal_tx = event_tx;
        let (event_tx, mut forwarded) = mpsc::unbounded_channel();
        let output = terminal_tx.clone();
        let forwarder = tokio::spawn(async move {
            while let Some(event) = forwarded.recv().await {
                if !matches!(event, AgentEvent::AgentEnd { .. }) {
                    let _ = output.send(event);
                }
            }
        });

        // Keep admission locked until the initial request is recorded, so a
        // concurrently queued refinement cannot be overwritten by this anchor.
        self.set_task_anchor(message).await;
        *self.tool_results_since_user.write().await = 0;

        // Add user message to history
        let user_msg = Message::new("user", message);
        self.persist_message(&user_msg).await;
        self.messages.write().await.push(user_msg);
        drop(admission);

        let _ = event_tx.send(AgentEvent::agent_start());
        let _ = event_tx.send(AgentEvent::turn_start());
        let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
            role: "user".to_string(),
            content: vec![MessageContent {
                content_type: "text".to_string(),
                text: Some(message.to_string()),
            }],
            model: None,
            usage: None,
            stop_reason: None,
        }));

        // Goal-aware execution loop. The objective comes from the goal registry, not
        // from the local field, so a goal the model already marked complete or
        // blocked does not drive another round on the next prompt.
        let goal_text = self.tool_context.goal.active_objective();

        if let Some(g) = goal_text {
            // Goal mode: suppress agent_end events during the loop.
            // Collect them and only forward the last one after completion.
            let (wrapped_tx, mut wrapped_rx) = mpsc::unbounded_channel::<AgentEvent>();
            let (held_tx, mut held_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

            let tx_clone = event_tx.clone();
            let fwd = tokio::spawn(async move {
                while let Some(ev) = wrapped_rx.recv().await {
                    if matches!(ev, AgentEvent::AgentEnd { .. }) {
                        let _ = held_tx.send(ev);
                    } else {
                        let _ = tx_clone.send(ev);
                    }
                }
            });

            let max_rounds = self
                .tool_context
                .goal
                .current()
                .map(|state| state.max_rounds)
                .unwrap_or(crate::goal::DEFAULT_MAX_ROUNDS);

            for round in 1..=max_rounds {
                // Open the round before the model runs. The goal tool accepts a
                // decision only from inside the open round, so a model that calls
                // `goal complete` from anywhere else is rejected rather than allowed
                // to declare victory on its own.
                self.tool_context.goal.admit_round(round);

                // Every round gets the block, including the first. The round number
                // is what the model has to echo back, so it must be visible from the
                // start, and the "inspect, do not assume" framing matters most right
                // after a compaction has degraded the narration.
                let round_text = crate::goal::render_round_prompt(&g, round, max_rounds);
                let round_msg = Message::new("user", &round_text);
                self.persist_message(&round_msg).await;
                self.messages.write().await.push(round_msg);
                if round > 1 {
                    // Round 1 rides the turn that prompt_inner already opened.
                    let _ = event_tx.send(AgentEvent::turn_start());
                    let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                        role: "user".to_string(),
                        content: vec![MessageContent {
                            content_type: "text".to_string(),
                            text: Some(round_text),
                        }],
                        model: None,
                        usage: None,
                        stop_reason: None,
                    }));
                }

                if let Err(e) = self.run_tool_loop(wrapped_tx.clone()).await {
                    eprintln!("rupi: tool loop error in goal round {}: {}", round, e);
                    self.tool_context
                        .goal
                        .conclude(&format!("the tool loop failed in round {}", round));
                    break;
                }

                // The working model decided inside its own round. No extra call.
                if self.tool_context.goal.is_decided() {
                    let status = self.tool_context.goal.current().map(|state| state.status);
                    eprintln!("rupi: goal decided in round {} ({:?})", round, status);
                    break;
                }

                // The model ignored the goal tool. Fall back to the out-of-band
                // check so an older model still terminates instead of burning
                // every round.
                if self.verify_goal(&g).await {
                    eprintln!("rupi: goal verified out of band in round {}", round);
                    self.tool_context
                        .goal
                        .conclude(&format!("verified out of band in round {}", round));
                    break;
                }
            }

            // Rounds can also simply run out. Leaving the goal active then made the
            // next unrelated prompt re-enter goal mode and start again.
            self.tool_context
                .goal
                .conclude("the driver ran out of rounds without a decision");

            drop(wrapped_tx);
            let _ = fwd.await;
            let mut last_ae = None;
            while let Ok(ae) = held_rx.try_recv() {
                last_ae = Some(ae);
            }
            if let Some(ae) = last_ae {
                let _ = event_tx.send(ae);
            } else {
                let _ = event_tx.send(AgentEvent::agent_end());
            }
        } else if let Some(loop_msg) = self.loop_prompt.read().await.clone() {
            // Loop mode: re-send the loop prompt after each agent_end until cancelled.
            // Suppress agent_end events during the loop like goal mode.
            //
            // Read once. Checking `is_some()` and then unwrapping a second read
            // panics if `stop_loop` clears the field in between, which it can: the
            // RPC handler takes only a read lock for both.
            let (wrapped_tx, mut wrapped_rx) = mpsc::unbounded_channel::<AgentEvent>();
            let (held_tx, mut held_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

            let tx_clone = event_tx.clone();
            let fwd = tokio::spawn(async move {
                while let Some(ev) = wrapped_rx.recv().await {
                    if matches!(ev, AgentEvent::AgentEnd { .. }) {
                        let _ = held_tx.send(ev);
                    } else {
                        let _ = tx_clone.send(ev);
                    }
                }
            });

            loop {
                if self
                    .loop_cancelled
                    .load(std::sync::atomic::Ordering::SeqCst)
                {
                    break;
                }
                if let Err(e) = self.run_tool_loop(wrapped_tx.clone()).await {
                    eprintln!("rupi: tool loop error in loop mode: {}", e);
                    break;
                }
                if self
                    .loop_cancelled
                    .load(std::sync::atomic::Ordering::SeqCst)
                {
                    break;
                }
                // Re-send the loop prompt
                let msg = Message::new("user", &loop_msg);
                self.persist_message(&msg).await;
                self.messages.write().await.push(msg);
                let _ = event_tx.send(AgentEvent::turn_start());
                let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                    role: "user".to_string(),
                    content: vec![MessageContent {
                        content_type: "text".to_string(),
                        text: Some(loop_msg.clone()),
                    }],
                    model: None,
                    usage: None,
                    stop_reason: None,
                }));
            }

            drop(wrapped_tx);
            let _ = fwd.await;
            let mut last_ae = None;
            while let Ok(ae) = held_rx.try_recv() {
                last_ae = Some(ae);
            }
            if let Some(ae) = last_ae {
                let _ = event_tx.send(ae);
            } else {
                let _ = event_tx.send(AgentEvent::agent_end());
            }
        } else {
            // No goal, no loop: normal flow
            if let Err(e) = self.run_tool_loop(event_tx.clone()).await {
                eprintln!("rupi: tool loop error: {}", e);
            }
        }

        // Drain any messages queued during streaming (e.g., steer that aborted the loop)
        loop {
            let admission = self.admission.lock().await;
            if self.resetting.load(std::sync::atomic::Ordering::SeqCst) {
                self.is_streaming
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                break;
            }
            let pending = self.drain_pending().await;
            if pending.is_empty() {
                self.is_streaming
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                break;
            }
            drop(admission);
            for msg in &pending {
                self.messages.write().await.push(msg.clone());
            }
            // Send turn_start + message_start for the queued messages
            let _ = event_tx.send(AgentEvent::turn_start());
            let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                role: "user".to_string(),
                content: vec![],
                model: None,
                usage: None,
                stop_reason: None,
            }));
            // Run tool loop to process queued messages
            if let Err(e) = self.run_tool_loop(event_tx.clone()).await {
                eprintln!("rupi: tool loop error after drain: {}", e);
            }
        }

        drop(event_tx);
        let _ = forwarder.await;
        let _ = terminal_tx.send(AgentEvent::agent_end());
        Ok(())
    }

    /// Internal tool loop: keeps sending messages + executing tools until final response.
    async fn run_tool_loop(
        &self,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        for _turn_num in 0.. {
            // Check abort signal; also check persistent abort_requested flag
            let abort_now = std::mem::take(&mut *self.abort_requested.lock().await);
            if abort_now {
                let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                    role: "assistant".to_string(),
                    content: vec![],
                    model: None,
                    usage: None,
                    stop_reason: Some("aborted".to_string()),
                }));
                let _ = event_tx.send(AgentEvent::turn_end());
                let _ = event_tx.send(AgentEvent::agent_end());
                return Err(AgentError::Cancelled);
            }

            self.tool_context
                .cancelled
                .store(false, std::sync::atomic::Ordering::SeqCst);
            // Carry cancellations arriving between rounds into the new stream.
            let (abort_tx, abort_rx) = watch::channel(false);
            {
                let mut signal = self.abort_signal.lock().await;
                if *self.abort_requested.lock().await {
                    let _ = abort_tx.send(true);
                }
                *signal = Some(abort_tx);
            }

            // Drain queued steer/follow-up messages, persist them, and add to history
            let drained = self.drain_pending().await;
            if !drained.is_empty() {
                eprintln!("rupi: processing {} queued message(s)", drained.len());
                for msg in &drained {
                    // Persist as well as push. The comment here claimed this
                    // happened for a long time while it did not, so a steer
                    // delivered mid-turn never reached the transcript and was gone
                    // on resume.
                    self.persist_message(msg).await;
                    self.messages.write().await.push(msg.clone());
                }
            }

            // Only compact at a completed tool boundary, before the next request.
            if let Err(e) = self.check_auto_compaction().await {
                eprintln!("rupi: compaction before request failed: {}", e);
            }

            // Build messages: use the cached system prompt (frozen at session creation)
            // to keep the token prefix identical across turns — critical for server-side KV cache reuse.
            let prompt_text = self.system_prompt.read().await.clone();
            let system_msg = Message::new("system", &prompt_text);
            let msgs_guard = self.messages.read().await;
            let mut messages_for_api = Vec::with_capacity(msgs_guard.len() + 1);
            messages_for_api.push(system_msg);
            messages_for_api.extend(msgs_guard.iter().cloned());
            drop(msgs_guard);

            let current_model = self.model();

            // Retry loop with exponential backoff
            let mut last_error = None;
            let mut rx = None;
            for attempt in 0..3 {
                let stream_result = self
                    .provider
                    .stream_chat(&current_model, &messages_for_api, abort_rx.clone())
                    .await;

                match stream_result {
                    Ok(rx_inner) => {
                        rx = Some(rx_inner);
                        break;
                    }
                    Err(e) => {
                        // Retry on transient errors: timeouts, connection errors, 5xx, 429
                        let retryable = match &e {
                            AgentError::Http(_) => true, // timeouts, connection refused, DNS, TLS
                            AgentError::Timeout => true,
                            AgentError::Api { status_code, .. } => {
                                *status_code == 429 || *status_code >= 500
                            }
                            _ => false,
                        };
                        if !retryable || attempt == 2 {
                            last_error = Some(e);
                            break;
                        }
                        let delay = std::time::Duration::from_secs(1 << attempt); // 1s, 2s, 4s
                        let mut retry_signal = abort_rx.clone();
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {},
                            _ = retry_signal.changed() => {
                                last_error = Some(AgentError::Cancelled);
                                break;
                            }
                        }
                    }
                }
            }

            let mut rx = match rx {
                Some(rx) => rx,
                None => {
                    let err = last_error
                        .unwrap_or(AgentError::Config("Request failed after retries".into()));
                    if matches!(err, AgentError::Cancelled) {
                        *self.abort_requested.lock().await = false;
                    }
                    let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                        role: "assistant".to_string(),
                        content: vec![MessageContent {
                            content_type: "text".into(),
                            text: Some(err.to_string()),
                        }],
                        model: None,
                        usage: None,
                        stop_reason: Some("error".to_string()),
                    }));
                    let _ = event_tx.send(AgentEvent::turn_end());
                    let _ = event_tx.send(AgentEvent::agent_end());
                    return Err(err);
                }
            };

            let mut full_content = String::new();
            let mut reasoning_content = String::new();
            let mut input_tokens = 0;
            let mut output_tokens = 0;
            let mut prompt_cost: Option<f64> = None;
            let mut completion_cost: Option<f64> = None;
            let mut total_cost: Option<f64> = None;

            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut finish_reason: Option<String> = None;
            let mut had_stream_events = false;
            // 5-minute timeout between stream events (accommodates reasoning models)
            loop {
                let event = match tokio::time::timeout(
                    std::time::Duration::from_secs(300),
                    rx.recv(),
                )
                .await
                {
                    Ok(Some(event)) => event,
                    Ok(None) => break,
                    Err(_) => {
                        // Stream timeout — treat as done
                        break;
                    }
                };
                match event {
                    StreamEvent::GenerationId(id) => {
                        self.persist_generation_id(&id).await;
                        let _ = event_tx.send(AgentEvent::generation_id(id));
                    }
                    StreamEvent::Delta(delta) => {
                        had_stream_events = true;
                        full_content.push_str(&delta);
                        let _ = event_tx.send(AgentEvent::message_update(delta));
                    }
                    StreamEvent::Reasoning(rc) => {
                        had_stream_events = true;
                        let _ = event_tx.send(AgentEvent::reasoning_update(rc));
                    }
                    StreamEvent::Done(result) => {
                        had_stream_events = true;
                        full_content = result.content;
                        reasoning_content = result.reasoning_content;
                        input_tokens = result.input_tokens;
                        output_tokens = result.output_tokens;
                        if let Some(c) = result.cost {
                            prompt_cost = Some(c.prompt_cost);
                            completion_cost = Some(c.completion_cost);
                            total_cost = Some(c.total_cost);
                        }
                        finish_reason = Some("stop".to_string());
                    }
                    StreamEvent::ToolCalls {
                        calls,
                        content,
                        reasoning_content: rc,
                        input_tokens: it,
                        output_tokens: ot,
                        cost,
                        finish_reason: fr,
                    } => {
                        had_stream_events = true;
                        full_content = content;
                        reasoning_content = rc;
                        input_tokens = it;
                        output_tokens = ot;
                        if let Some(c) = cost {
                            prompt_cost = Some(c.prompt_cost);
                            completion_cost = Some(c.completion_cost);
                            total_cost = Some(c.total_cost);
                        }
                        finish_reason = fr;
                        tool_calls = calls;
                    }
                    StreamEvent::Error(err) => {
                        let error_text = if err == "cancelled" {
                            "Request cancelled".to_string()
                        } else {
                            format!("Error: {}", err)
                        };
                        let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                            role: "assistant".to_string(),
                            content: vec![MessageContent {
                                content_type: "text".to_string(),
                                text: Some(error_text),
                            }],
                            model: Some(current_model.clone()),
                            usage: Some(Usage {
                                input: input_tokens,
                                output: output_tokens,
                                total_tokens: input_tokens + output_tokens,
                                cost: None,
                            }),
                            stop_reason: Some("error".to_string()),
                        }));
                        let _ = event_tx.send(AgentEvent::turn_end());
                        let _ = event_tx.send(AgentEvent::agent_end());

                        return if err == "cancelled" {
                            *self.abort_requested.lock().await = false;
                            Err(AgentError::Cancelled)
                        } else {
                            Err(AgentError::Api {
                                message: err,
                                status_code: 0,
                            })
                        };
                    }
                }
            }

            // If no native tool calls, try to extract embedded tool calls from text
            if tool_calls.is_empty() && !full_content.is_empty() {
                let embedded = output_parser::extract_tool_calls_from_text(&full_content);
                if !embedded.is_empty() {
                    eprintln!(
                        "rupi: output parser extracted {} tool call(s) from text",
                        embedded.len()
                    );
                    tool_calls = embedded;
                }
            }

            // Truncation detection: if finish_reason is "length", the response was cut off
            if finish_reason.as_deref() == Some("length") {
                eprintln!("rupi: response truncated (finish_reason=length)");
                let mut cc = self.consecutive_quality_issues.write().await;
                let max_corrections: u32 = 2;
                if *cc < max_corrections {
                    let correction =
                        quality::build_correction_message(&quality::QualityIssue::Truncated);
                    self.pending_follow_up.write().await.push(correction);
                    *cc += 1;
                } else {
                    eprintln!(
                        "rupi: truncation correction suppressed after {} corrections",
                        *cc
                    );
                }
            }

            // Quality check: assess the response before proceeding
            if !tool_calls.is_empty() || !full_content.is_empty() {
                let known = quality::known_tool_names();
                // Newest first. The buffer is appended to, so it is stored
                // oldest-first, while `consecutive_repeat_count` walks backwards
                // from the current turn. Passing it unreversed both missed real
                // repeat runs and raised false alarms on benign alternating
                // patterns — and those alarms bypass the correction cap.
                let recent: Vec<Vec<ToolCall>> = self
                    .recent_tool_calls
                    .read()
                    .await
                    .iter()
                    .rev()
                    .cloned()
                    .collect();
                let verdict = quality::assess_response(&full_content, &tool_calls, &recent, known);
                if !verdict.ok {
                    let issue = verdict.reason.as_ref().unwrap();
                    let mut cc = self.consecutive_quality_issues.write().await;
                    let max_corrections: u32 = 2;
                    // The cap exists to stop a correction loop. The repeat ladder
                    // cannot loop — it fires once per threshold for one run of
                    // identical calls — so capping it would only silence the
                    // escalation that was doing the work.
                    let exempt = quality::is_self_limiting(issue);
                    if exempt || *cc < max_corrections {
                        let correction = quality::build_correction_message(issue);
                        eprintln!(
                            "rupi: quality issue detected: {:?} — queuing correction",
                            issue
                        );
                        self.pending_follow_up.write().await.push(correction);
                        if !exempt {
                            *cc += 1;
                        }
                    } else {
                        eprintln!("rupi: quality issue suppressed after {} corrections", *cc);
                    }
                } else {
                    // Reset counter on a clean response
                    *self.consecutive_quality_issues.write().await = 0;
                }
            }

            // Update recent tool calls tracking
            if !tool_calls.is_empty() {
                let mut recent = self.recent_tool_calls.write().await;
                recent.push(tool_calls.clone());
                if recent.len() > 8 {
                    recent.remove(0);
                }
            }

            // If tool calls were made, execute them and continue to next turn
            if !tool_calls.is_empty() {
                // Add assistant message with tool calls to history
                let rc = if reasoning_content.is_empty() {
                    None
                } else {
                    Some(reasoning_content.clone())
                };
                let assistant_msg =
                    Message::tool_call(&full_content, tool_calls.clone()).with_reasoning(rc);
                self.persist_message(&assistant_msg).await;
                self.messages.write().await.push(assistant_msg);

                // Emit message_end + turn_end for this turn (matching Pi's event flow)
                let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                    role: "assistant".to_string(),
                    content: vec![MessageContent {
                        content_type: "text".to_string(),
                        text: Some(full_content.clone()),
                    }],
                    model: Some(current_model.clone()),
                    usage: Some(Usage {
                        input: input_tokens,
                        output: output_tokens,
                        total_tokens: input_tokens + output_tokens,
                        cost: None,
                    }),
                    stop_reason: Some("tool_calls".to_string()),
                }));
                let _ = event_tx.send(AgentEvent::turn_end());

                // Execute each tool, emit events, and add results to history
                for tc in &tool_calls {
                    let _ = event_tx.send(AgentEvent::tool_execution_start(
                        tc.name.clone(),
                        tc.arguments.clone(),
                    ));

                    let approval = self.approval_fn.read().await.clone();
                    let call = tc.clone();
                    let context = self.tool_context.clone();
                    let result = tools::execute_tool_async(call, context, approval).await;

                    let _ = event_tx.send(AgentEvent::tool_execution_end(
                        tc.name.clone(),
                        result.clone(),
                    ));
                    // Cap the stored tool result to keep KV cache prefill bounded.
                    // The full result is still sent to the event stream above, and
                    // spilled to disk; only the conversation-history copy is cut.
                    let stored_result = self.store_tool_result(&tc.name, &result).await;
                    let result_msg = Message::tool_result(&tc.id, &stored_result);
                    self.persist_message(&result_msg).await;
                    self.messages.write().await.push(result_msg);
                    *self.tool_results_since_user.write().await += 1;
                }

                // Re-state the request when it has drifted far back. Appending at
                // the tail keeps the cached prefix intact, so this is free.
                if let Some(reminder) = self.maybe_reemit_anchor().await {
                    eprintln!("rupi: re-stated the task anchor after a long tool run");
                    let _ = event_tx.send(AgentEvent::turn_start());
                    let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                        role: "user".to_string(),
                        content: vec![MessageContent {
                            content_type: "text".to_string(),
                            text: Some(reminder.content.clone()),
                        }],
                        model: None,
                        usage: None,
                        stop_reason: None,
                    }));
                    continue;
                }

                // Start a new turn for the next LLM call (matching Pi's flow)
                let _ = event_tx.send(AgentEvent::turn_start());
                let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                    role: "user".to_string(),
                    content: vec![],
                    model: None,
                    usage: None,
                    stop_reason: None,
                }));

                continue;
            }

            // No tool calls. If the response is empty (stream failed silently), retry with cap.
            if full_content.is_empty() && !had_stream_events {
                let mut retries = self.empty_response_retries.lock().await;
                if *retries >= 5 {
                    return Err(AgentError::Api {
                        message: "Model returned empty response 5 consecutive times".to_string(),
                        status_code: 0,
                    });
                }
                let delay = 500u64 * (1 << *retries);
                *retries += 1;
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                continue;
            }

            // No tool calls — this is the final response.
            // Reset empty response retry counter on success
            *self.empty_response_retries.lock().await = 0;
            let cost_data =
                if prompt_cost.is_some() || completion_cost.is_some() || total_cost.is_some() {
                    Some(PromptCostData {
                        prompt_cost,
                        completion_cost,
                        total_cost,
                    })
                } else {
                    None
                };

            // Add assistant message to history
            let rc = if reasoning_content.is_empty() {
                None
            } else {
                Some(reasoning_content.clone())
            };
            let assistant_msg = Message::new("assistant", &full_content).with_reasoning(rc);
            self.persist_message(&assistant_msg).await;
            self.messages.write().await.push(assistant_msg);

            let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                role: "assistant".to_string(),
                content: vec![MessageContent {
                    content_type: "text".to_string(),
                    text: Some(full_content.clone()),
                }],
                model: Some(current_model.clone()),
                usage: Some(Usage {
                    input: input_tokens,
                    output: output_tokens,
                    total_tokens: input_tokens + output_tokens,
                    cost: cost_data,
                }),
                stop_reason: Some(finish_reason.unwrap_or_else(|| {
                    if had_stream_events {
                        "stop".to_string()
                    } else {
                        "error".to_string()
                    }
                })),
            }));
            let _ = event_tx.send(AgentEvent::turn_end());
            let _ = event_tx.send(AgentEvent::agent_end());

            return Ok(());
        }
        unreachable!()
    }

    /// Run compaction on the conversation history.
    /// Summarizes older messages to free up context window space.
    pub async fn compact(&self) -> Result<CompactionResult, AgentError> {
        {
            let mut compacting = self.is_compacting.lock().await;
            if *compacting {
                return Err(AgentError::Config("Already compacting".into()));
            }
            *compacting = true;
        }

        // Work on a copy. Nothing below this line touches the live history until the
        // compaction is certain to commit.
        //
        // Snipping used to run here, in place, BEFORE the two guards that can return
        // early. Both guards leave the conversation alive, so a run that snipped and
        // then declined rewrote message bodies the provider had already cached and
        // bought nothing for it: the next turn re-prefilled the whole conversation at
        // full price. Even when it did avoid an LLM call, that was a bad trade —
        // cached input is roughly a tenth the price of fresh input, so saving ~13k
        // tokens by invalidating ~112k cached ones loses badly.
        let pristine = self.messages.read().await.clone();
        let total_tokens = compaction::estimate_total_tokens(&pristine);

        if !compaction::should_compact(total_tokens, self.context_window) {
            {
                let mut compacting = self.is_compacting.lock().await;
                *compacting = false;
            }
            return Err(AgentError::Config(
                "Context not full enough to compact".into(),
            ));
        }

        // Snip a copy. It sizes the kept tail — more messages survive the same token
        // budget once the old tool results are trimmed — and it becomes the tail that
        // is actually stored, but only if this compaction commits. Snipping never
        // adds, removes, or reorders messages, so one index addresses the same
        // message in both copies.
        let mut snipped = pristine.clone();
        let removed = compaction::snip_old_tool_results(&mut snipped, 6);
        let snipped_len = snipped.len();

        let keep_recent = self.context_window.saturating_div(10).clamp(1, 20000);
        let cut_index = match compaction::find_cut_point(&snipped, keep_recent) {
            Some(i) => i,
            None => {
                {
                    let mut compacting = self.is_compacting.lock().await;
                    *compacting = false;
                }
                return Err(AgentError::Config("Nothing to compact".into()));
            }
        };

        // The summarizer replays the PRISTINE head. It is byte-identical to the
        // prefix of the last routed request, so the provider serves it from cache
        // instead of re-prefilling it, and the summarizer sees full-fidelity input.
        let region = &pristine[..cut_index.min(pristine.len())];
        let compact_model = self.model();
        // Replay the conversation's own system prompt so the summarization call is a
        // genuine prefix of the last routed request. The provider then serves the
        // whole replayed span from its KV cache instead of re-prefilling it. The old
        // shape sent a different system prompt and a serialized text blob, which was
        // a guaranteed cache miss that re-billed the entire history every time.
        let system_prompt = self.system_prompt.read().await.clone();
        if compaction::region_contains_checkpoint(region) {
            eprintln!("rupi: compacting over a prior checkpoint — merging it into one summary");
        }
        let raw_summary = match compaction::generate_summary(
            &self.provider,
            &compact_model,
            Some(&system_prompt),
            region,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                *self.is_compacting.lock().await = false;
                return Err(e);
            }
        };

        // Never store a response that is not a checkpoint. Whatever comes back
        // replaces the whole region, so a refusal, a filtered stub, or a stream cut
        // off at the token limit would delete the context silently and leave the
        // agent continuing from nothing.
        let summary = match compaction::validate_summary(&raw_summary) {
            Ok(()) => raw_summary,
            Err(problem) => {
                eprintln!("rupi: the summarizer returned {} — retrying once", problem);
                let retried = compaction::generate_summary(
                    &self.provider,
                    &compact_model,
                    Some(&system_prompt),
                    region,
                )
                .await
                .ok()
                .filter(|s| compaction::validate_summary(s).is_ok());
                match retried {
                    Some(good) => good,
                    None => {
                        // Refusing to compact would leave the next request over the
                        // provider's limit. A mechanical checkpoint states only what
                        // the messages show, so it invents nothing and always works.
                        eprintln!("rupi: falling back to a mechanical checkpoint");
                        compaction::mechanical_checkpoint(region)
                    }
                }
            }
        };

        // Replace summarized messages with a compaction checkpoint message.
        // Use role "user" (not "system") because OpenAI-compatible endpoints
        // expect at most one system message — the one built fresh each turn.
        let summary_msg = Message::new("user", &compaction::build_checkpoint_body(&summary));
        self.persist_message(&summary_msg).await;

        // Re-emit the anchor directly below the checkpoint. This is the position that
        // makes the fix structural: the exact request the user typed is present in
        // every request after every compaction, whatever the summarizer chose to
        // write. The prefix is already invalid here, so the anchor costs nothing.
        let anchor_msg = self
            .next_anchor_block()
            .await
            .map(|block| Message::new("user", &block));

        {
            let mut all_messages = self.messages.write().await;
            let cut = cut_index.min(all_messages.len());
            // Take the tail from the snipped copy. The checkpoint sits at position
            // zero, so every message after it is at a new offset and the whole
            // request is a fresh prefix regardless — the snip is free here.
            // Anything appended while the summary was in flight is taken from the
            // live list, because the copy predates it.
            let mut rebuilt = Vec::with_capacity(all_messages.len() - cut + 2);
            rebuilt.push(summary_msg);
            if let Some(ref anchor) = anchor_msg {
                rebuilt.push(anchor.clone());
            }
            rebuilt.extend(snipped.drain(cut.min(snipped.len())..));
            if all_messages.len() > snipped_len {
                rebuilt.extend(all_messages[snipped_len..].to_vec());
            }
            *all_messages = rebuilt;
        }
        if removed > 0 {
            eprintln!("rupi: snipped {} chars from old tool results", removed);
        }
        *self.tool_results_since_user.write().await = 0;

        self.persist_compaction(&summary, total_tokens).await;
        // Persist the anchor AFTER the compaction record. Replay clears everything
        // above that record, so an anchor written before it would be dropped on
        // resume and the fix would hold only until the process restarted.
        if let Some(ref anchor) = anchor_msg {
            self.persist_message(anchor).await;
        }

        // Released only now. `reset()` polls this flag and then swaps
        // `session_path`, so clearing it before the writes above let a concurrent
        // new-session command redirect this session's compaction record and anchor
        // into the fresh session's file.
        {
            let mut compacting = self.is_compacting.lock().await;
            *compacting = false;
        }

        Ok(CompactionResult {
            summary,
            tokens_before: total_tokens,
        })
    }

    /// Check if auto-compaction is needed and run it.
    /// Called after each prompt completes.
    /// Queue a steer message (interrupts current generation).
    ///
    /// The anchor grows here rather than in the drain, because by drain time a user
    /// steer and a host-written quality correction share one queue and cannot be
    /// told apart. Corrections push to the queue directly and never reach this path.
    pub async fn steer(&self, message: &str) {
        self.append_task_anchor(message).await;
        self.pending_steer.write().await.push(message.to_string());
        self.abort().await;
    }

    /// Queue a follow-up message (processed after current generation finishes).
    pub async fn follow_up(&self, message: &str) {
        self.append_task_anchor(message).await;
        self.pending_follow_up
            .write()
            .await
            .push(message.to_string());
    }

    /// Check if there are pending steer messages.
    pub async fn has_pending_steer(&self) -> bool {
        !self.pending_steer.read().await.is_empty()
    }

    /// Check if there are pending follow-up messages.
    pub async fn has_pending_follow_up(&self) -> bool {
        !self.pending_follow_up.read().await.is_empty()
    }

    /// Drain all queued messages for the next LLM turn.
    /// Steer messages come first, then follow-ups.
    async fn drain_pending(&self) -> Vec<Message> {
        let mut all = Vec::new();
        // Drain steers first (highest priority)
        {
            let mut steer = self.pending_steer.write().await;
            for msg in steer.drain(..) {
                let m = Message::new("user", &msg);
                self.persist_message(&m).await;
                all.push(m);
            }
        }
        // Then drain follow-ups
        {
            let mut fu = self.pending_follow_up.write().await;
            for msg in fu.drain(..) {
                let m = Message::new("user", &msg);
                self.persist_message(&m).await;
                all.push(m);
            }
        }
        all
    }

    pub async fn check_auto_compaction(&self) -> Result<Option<CompactionResult>, AgentError> {
        let enabled = *self.auto_compaction_enabled.read().await;
        if !enabled {
            return Ok(None);
        }

        let messages = self.messages.read().await.clone();
        let total_tokens = compaction::estimate_total_tokens(&messages);

        if !compaction::should_compact(total_tokens, self.context_window) {
            return Ok(None);
        }

        let result = self.compact().await?;
        Ok(Some(result))
    }

    /// Abort the current streaming operation.
    /// Signals the tool loop to stop. Does NOT clear is_streaming — the tool
    /// loop itself sets is_streaming=false when it exits. This prevents a race
    /// where is_streaming=false lets a new prompt() start before the old tool
    /// loop has finished cleaning up.
    pub async fn abort(&self) {
        self.tool_context
            .cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *self.abort_requested.lock().await = true;
        if let Some(tx) = self.abort_signal.lock().await.as_ref() {
            let _ = tx.send(true);
        }
    }

    // ---- task anchor ----

    /// Record the request that the current task is anchored to.
    ///
    /// Host-written blocks are rejected on purpose. A checkpoint, an anchor
    /// re-emission, and a goal round all arrive on the `user` role because the
    /// providers accept at most one system message, and letting any of them become
    /// the anchor would overwrite the real request with rupi's own prose.
    pub async fn set_task_anchor(&self, request: &str) {
        let trimmed = request.trim();
        if trimmed.is_empty()
            || crate::anchor::is_anchor(trimmed)
            || trimmed.starts_with(sessions::COMPACTION_PREFIX)
            || trimmed.starts_with("<goal_round>")
        {
            return;
        }
        *self.task_anchor.write().await = Some(trimmed.to_string());
        *self.anchor_emissions.write().await = 0;
    }

    /// Add a later user instruction to the anchor.
    ///
    /// A steer or a follow-up refines the task in flight rather than starting a new
    /// one, so it joins the anchor instead of replacing it. Replacing would drop the
    /// request the work is actually for; ignoring it would leave the anchor stating
    /// a task the user has already redirected. The stored value is clamped, so a
    /// session with many follow-ups keeps the original request and the most recent
    /// instruction without growing without bound.
    pub async fn append_task_anchor(&self, addition: &str) {
        let trimmed = addition.trim();
        if trimmed.is_empty()
            || crate::anchor::is_anchor(trimmed)
            || trimmed.starts_with(sessions::COMPACTION_PREFIX)
            || trimmed.starts_with("<goal_round>")
        {
            return;
        }
        {
            let mut guard = self.task_anchor.write().await;
            match guard.as_mut() {
                Some(existing) => {
                    existing.push_str(crate::anchor::LATER_INSTRUCTION_SEPARATOR);
                    existing.push_str(trimmed);
                    *existing = crate::anchor::clamp(existing);
                }
                None => *guard = Some(trimmed.to_string()),
            }
        }
        *self.anchor_emissions.write().await = 0;
    }

    /// The request the current task is anchored to.
    pub async fn task_anchor(&self) -> Option<String> {
        self.task_anchor.read().await.clone()
    }

    /// Render the next anchor emission and count it.
    async fn next_anchor_block(&self) -> Option<String> {
        let request = self.task_anchor.read().await.clone()?;
        let mut emissions = self.anchor_emissions.write().await;
        *emissions += 1;
        // The anchor states the goal. The todo list states where the work stands.
        // They answer different questions, so a reminder carries both — but the plan
        // goes INSIDE the block. Appending it after the closing tag left a message
        // that no longer parsed as an anchor, so a re-emitted reminder could not be
        // recovered on resume.
        let plan = self.tool_context.todos.render_current();
        Some(crate::anchor::render_with_plan(
            &request,
            *emissions,
            plan.as_deref(),
        ))
    }

    /// Tool results allowed between anchor re-emissions.
    ///
    /// The anchor is free at the tail and free right after a compaction, because
    /// neither position invalidates a cached prefix. It is expensive anywhere else,
    /// so it goes at the end and only after the request has drifted genuinely far
    /// out of the model's attention.
    const ANCHOR_REEMIT_AFTER_TOOL_RESULTS: u32 = 40;

    /// Append an anchor reminder when the request has drifted far enough back.
    ///
    /// Returns the message that was appended, for the caller to announce.
    async fn maybe_reemit_anchor(&self) -> Option<Message> {
        {
            let count = self.tool_results_since_user.read().await;
            if *count < Self::ANCHOR_REEMIT_AFTER_TOOL_RESULTS {
                return None;
            }
        }
        let block = self.next_anchor_block().await?;
        *self.tool_results_since_user.write().await = 0;
        let msg = Message::new("user", &block);
        self.persist_message(&msg).await;
        self.messages.write().await.push(msg.clone());
        Some(msg)
    }

    /// Set a goal for durable execution. When set, the agent will loop until
    /// an internal verification prompt confirms the goal is met.
    pub async fn set_goal(&self, goal: Option<String>) {
        self.tool_context
            .goal
            .set(goal, crate::goal::DEFAULT_MAX_ROUNDS);
    }

    /// The objective still driving the session, or `None` once it is decided.
    ///
    /// Reads the registry rather than a second copy. A goal the model completed or
    /// blocked is no longer current, and reporting it as current made `/goal` lie
    /// about what the agent was doing.
    pub async fn get_goal(&self) -> Option<String> {
        self.tool_context.goal.active_objective()
    }

    /// A one-line description of the goal for display, including its status.
    pub async fn goal_status(&self) -> Option<String> {
        let state = self.tool_context.goal.current()?;
        Some(format!(
            "{} [round {}/{}, {}]",
            state.objective,
            state.admitted_round,
            state.max_rounds,
            match state.status {
                crate::goal::GoalStatus::Active => "active",
                crate::goal::GoalStatus::Complete => "complete",
                crate::goal::GoalStatus::Blocked => "blocked",
                crate::goal::GoalStatus::Ended => "ended",
            }
        ))
    }

    /// Set a loop prompt. When set, the agent will re-send the prompt
    /// repeatedly after each agent_end until cancelled.
    pub async fn set_loop(&self, prompt: Option<String>) {
        *self.loop_prompt.write().await = prompt;
        self.loop_cancelled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Cancel the current loop (if any).
    pub async fn cancel_loop(&self) {
        self.loop_cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *self.loop_prompt.write().await = None;
        self.abort().await;
    }

    /// Check if loop mode is active.
    pub async fn is_loop_active(&self) -> bool {
        self.loop_prompt.read().await.is_some()
    }

    /// Verify whether the current goal has been achieved by asking the model.
    ///
    /// This is now the FALLBACK path. The primary route is the `goal` tool, which
    /// the working model calls from inside its own round with full context and no
    /// extra request. This judge sees six truncated messages and no tools, so it
    /// runs only when the model never used the tool.
    ///
    /// Returns true if the model confirms the goal is met.
    async fn verify_goal(&self, goal: &str) -> bool {
        // Include recent conversation so the model can actually check the assistant's output.
        // Truncate long tool results to avoid overflowing the verify context window.
        let msgs = self.messages.read().await.clone();
        let last_few: String = msgs
            .iter()
            .rev()
            .take(6)
            .map(|m| {
                let content = if m.content.len() > 1000 {
                    // Snap to a character boundary. A raw byte slice panics on any
                    // message whose byte 1000 lands inside a multi-byte character,
                    // and tool results routinely carry non-ASCII: a source file, a
                    // git log with an accented name, box-drawing progress output.
                    format!(
                        "{}... [truncated: {} bytes]",
                        head_chars(&m.content, 1000),
                        m.content.len()
                    )
                } else {
                    m.content.clone()
                };
                format!(
                    "<{}>\n{}\n</{}>",
                    m.role.to_uppercase(),
                    content,
                    m.role.to_uppercase()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let verify_prompt = format!(
            "You are verifying whether a specific CONDITION has been met.
Read the EXACT GOAL below and check the assistant's most recent response(s) in the conversation shown.

GOAL: {}

CONVERSATION (most recent first):
{}

Has the assistant's output satisfied this exact condition? Reply with only YES or NO.",
            goal, last_few
        );

        // Minimal system prompt — no tool descriptions needed for YES/NO verification
        let simple_system = format!(
            "You are a verification assistant. Current date: {}",
            chrono::Local::now().format("%A, %B %d, %Y")
        );
        let model_name = self.model();
        for attempt in 0..3 {
            let sys = Message::new("system", &simple_system);
            let vfy = Message::new("user", &verify_prompt);
            match self.provider.complete(&model_name, &[sys, vfy]).await {
                Ok(response) => {
                    return response.trim().to_uppercase().starts_with("Y");
                }
                Err(_) if attempt < 2 => {
                    tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                }
                _ => return false,
            }
        }
        false
    }

    /// Set an approval callback for tool execution.
    /// Called with (tool_name, args_json) before execution. Return true to allow.
    pub async fn set_approval_fn(&self, f: Option<ApprovalFn>) {
        *self.approval_fn.write().await = f;
    }

    /// Whether a human approves each tool call.
    ///
    /// Interactive mode consults this before it starts reading stdin for steering:
    /// the approval prompt reads stdin too, and two readers on one terminal means
    /// the prompt never receives the answer.
    pub async fn requires_approval(&self) -> bool {
        self.approval_fn.read().await.is_some()
    }

    pub fn provider_model_info(&self) -> ModelInfo {
        self.provider.model_info()
    }

    pub async fn get_state(&self) -> SessionState {
        let mut info = self.provider_model_info();
        info.id = self.model();
        SessionState {
            model: Some(info),
            thinking_level: self.thinking_level.read().await.clone(),
            is_streaming: self.is_streaming.load(std::sync::atomic::Ordering::SeqCst),
            is_compacting: *self.is_compacting.lock().await,
            steering_mode: "all".to_string(),
            follow_up_mode: "all".to_string(),
            auto_compaction_enabled: *self.auto_compaction_enabled.read().await,
            message_count: self.messages.read().await.len(),
            pending_message_count: self.pending_steer.read().await.len()
                + self.pending_follow_up.read().await.len(),
            session_file: self
                .session_path
                .read()
                .await
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        }
    }

    pub async fn get_messages_as_rpc(&self) -> Vec<AgentMessage> {
        self.messages
            .read()
            .await
            .iter()
            .map(|m| AgentMessage {
                role: m.role.clone(),
                content: vec![MessageContent {
                    content_type: "text".to_string(),
                    text: Some(m.content.clone()),
                }],
                model: None,
                usage: None,
                stop_reason: None,
            })
            .collect()
    }
}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    #[test]
    fn short_output_is_untouched() {
        let text = "line one\nline two\n";
        assert_eq!(capped_tool_result(text), text);
    }

    #[test]
    fn long_output_is_cut_on_line_boundaries() {
        // Distinct, numbered lines so a fragment is obvious.
        let body: String = (0..2000)
            .map(|i| format!("line {:05} of output\n", i))
            .collect();
        assert!(body.len() > MAX_TOOL_RESULT_CHARS);
        let capped = capped_tool_result(&body);

        assert!(capped.contains("[truncated:"), "{}", capped);
        assert!(capped.len() < body.len());

        // Every retained line is whole. A cut mid-line leaves a fragment that reads
        // as a different, shorter message than it is.
        let head = capped.split("... [truncated:").next().unwrap();
        for line in head.lines() {
            assert!(
                line.is_empty() || line.ends_with(" of output"),
                "head carries a partial line: {:?}",
                line
            );
        }
        let tail = capped.rsplit("bytes]\n...").next().unwrap();
        for line in tail.lines() {
            assert!(
                line.is_empty() || line.starts_with("line "),
                "tail carries a partial line: {:?}",
                line
            );
        }
    }

    #[test]
    fn the_beginning_and_the_end_both_survive() {
        let body: String = (0..2000)
            .map(|i| format!("line {:05} of output\n", i))
            .collect();
        let capped = capped_tool_result(&body);
        assert!(
            capped.contains("line 00000 of output"),
            "the first line must survive"
        );
        assert!(
            capped.contains("line 01999 of output"),
            "the last line must survive"
        );
    }

    #[test]
    fn output_without_line_breaks_still_truncates() {
        // Minified JSON: no line boundary exists inside the budget, so the cut
        // falls back to character boundaries rather than returning nothing.
        let body = format!("{{\"k\":\"{}\"}}", "v".repeat(MAX_TOOL_RESULT_CHARS * 2));
        let capped = capped_tool_result(&body);
        assert!(capped.contains("[truncated:"), "{}", capped);
        assert!(capped.len() < body.len());
        assert!(capped.starts_with("{\"k\""), "{}", &capped[..20]);
    }

    #[test]
    fn multibyte_output_never_panics() {
        // Three bytes per character, so naive byte offsets land mid-character.
        for body in [
            "の".repeat(MAX_TOOL_RESULT_CHARS),
            format!("{}\n", "の".repeat(MAX_TOOL_RESULT_CHARS)),
            (0..1000).map(|_| "のの\n").collect::<String>(),
        ] {
            let capped = capped_tool_result(&body);
            assert!(capped.contains("の"));
            assert!(capped.len() <= body.len());
        }
    }

    #[test]
    fn a_short_first_line_does_not_collapse_the_head_budget() {
        // The shape that broke line-aligned cutting: one framing line, then one
        // enormous line. Snapping to lines pins the head at the framing line and
        // throws the rest of the budget away, which is far worse than a clean cut
        // mid-line. This is `curl | jq -c`, a docker inspect, a webpack log.
        let body = format!("Running command...\n{}", "j".repeat(160_000));
        let capped = capped_tool_result(&body);

        assert!(
            capped.len() > MAX_TOOL_RESULT_CHARS / 2,
            "kept only {} bytes of a {} byte budget",
            capped.len(),
            MAX_TOOL_RESULT_CHARS
        );
        assert!(
            capped.contains("Running command..."),
            "the framing line must survive"
        );
        assert!(
            capped.contains("jjjj"),
            "the payload must not be thrown away"
        );
    }

    #[test]
    fn a_single_huge_line_still_uses_the_budget() {
        for body in [
            format!("\n{}", "x".repeat(50_000)),
            "y".repeat(50_000),
            format!("{}\n", "z".repeat(50_000)),
        ] {
            let capped = capped_tool_result(&body);
            assert!(
                capped.len() > MAX_TOOL_RESULT_CHARS / 2,
                "kept only {} bytes for a {} byte input",
                capped.len(),
                body.len()
            );
        }
    }

    #[test]
    fn truncation_never_inflates() {
        // Just over the cap, head plus tail can cover the whole input and the
        // marker is pure overhead. Growing a result while truncating it is a loss
        // on every axis the cap exists to protect.
        for size in [
            MAX_TOOL_RESULT_CHARS + 1,
            MAX_TOOL_RESULT_CHARS + 27,
            MAX_TOOL_RESULT_CHARS + 30,
            MAX_TOOL_RESULT_CHARS + 100,
        ] {
            for body in [
                "x".repeat(size),
                (0..size / 2).map(|_| "a\n").collect::<String>(),
            ] {
                let capped = capped_tool_result(&body);
                assert!(
                    capped.len() <= body.len(),
                    "a {} byte input grew to {} bytes",
                    body.len(),
                    capped.len()
                );
            }
        }
    }

    #[test]
    fn the_truncation_marker_names_its_unit() {
        let body: String = (0..2000)
            .map(|i| format!("line {:05} of output\n", i))
            .collect();
        let capped = capped_tool_result(&body);
        // The count is a byte count. Labelling it "chars" overstates it by up to
        // three times on non-ASCII, next to a spill hint that correctly says bytes.
        assert!(capped.contains("bytes]"), "{}", &capped[..80]);
        assert!(!capped.contains("chars]"));
    }

    #[test]
    fn the_cap_is_respected_within_a_line_of_slack() {
        let body: String = (0..2000)
            .map(|i| format!("line {:05} of output\n", i))
            .collect();
        let capped = capped_tool_result(&body);
        // Head and tail budgets plus the marker. Line snapping only ever removes
        // content, so the result cannot exceed the budget plus the marker text.
        assert!(
            capped.len() < MAX_TOOL_RESULT_CHARS + 100,
            "capped to {} chars, budget is {}",
            capped.len(),
            MAX_TOOL_RESULT_CHARS
        );
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    use super::*;

    #[test]
    fn the_system_prompt_is_identical_for_two_processes_started_moments_apart() {
        // Providers cache on an exact prefix. This string sits in the first
        // message, so anything that changes between runs — a clock, in
        // particular — throws away the cache for the whole conversation.
        let first = build_system_prompt(&[], &[], false, None);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let second = build_system_prompt(&[], &[], false, None);

        assert_eq!(
            first, second,
            "system prompt changed between runs — every cold start is now a cache miss"
        );
    }

    #[test]
    fn the_date_is_still_available_to_the_model() {
        let prompt = build_system_prompt(&[], &[], false, None);
        let today = chrono::Local::now().format("%Y").to_string();
        assert!(prompt.contains("Current date:"), "date line missing");
        assert!(
            prompt.contains(&today),
            "current year missing from the prompt"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::openai::OpenAIConfig;

    fn create_test_session() -> AgentSession {
        let config = OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test-key".into(),
            model: "gpt-4".into(),
            context_window: 8192,
            timeout_secs: 0,
            reasoning: false,
        };
        AgentSession::from_config(config)
    }

    #[tokio::test]
    async fn test_session_creation() {
        let session = create_test_session();
        assert_eq!(session.model(), "gpt-4");
        assert_eq!(session.thinking_level().await, "off");
        assert_eq!(session.message_count().await, 0);
        assert!(session.auto_compaction_enabled().await);
    }

    #[tokio::test]
    async fn test_cycle_thinking_level() {
        let session = create_test_session();
        assert_eq!(session.thinking_level().await, "off");
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("low"));
        assert_eq!(
            session.cycle_thinking_level().await.as_deref(),
            Some("medium")
        );
        assert_eq!(
            session.cycle_thinking_level().await.as_deref(),
            Some("high")
        );
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("off"));
    }

    #[tokio::test]
    async fn test_set_thinking_level() {
        let session = create_test_session();
        session.set_thinking_level("high".into()).await;
        assert_eq!(session.thinking_level().await, "high");
    }

    #[tokio::test]
    async fn test_auto_compaction() {
        let session = create_test_session();
        assert!(session.auto_compaction_enabled().await);
        session.set_auto_compaction_enabled(false).await;
        assert!(!session.auto_compaction_enabled().await);
        session.set_auto_compaction_enabled(true).await;
        assert!(session.auto_compaction_enabled().await);
    }

    #[tokio::test]
    async fn test_get_state() {
        let session = create_test_session();
        let state = session.get_state().await;
        assert!(state.model.is_some());
        assert_eq!(state.thinking_level, "off");
        assert_eq!(state.message_count, 0);
        assert!(!state.is_streaming);
    }

    #[tokio::test]
    async fn test_abort_when_not_streaming() {
        let session = create_test_session();
        session.abort().await;
    }

    #[tokio::test]
    async fn test_provider_model_info() {
        let session = create_test_session();
        let info = session.provider_model_info();
        assert_eq!(info.provider, "openai-compatible");
        assert_eq!(info.id, "gpt-4");
    }

    #[tokio::test]
    async fn test_reset() {
        let session = create_test_session();
        // Add a message directly
        session
            .messages
            .write()
            .await
            .push(Message::new("user", "hello"));

        assert_eq!(session.messages().await.len(), 1);
        session.reset().await;
        assert_eq!(session.messages().await.len(), 0);
        assert_eq!(session.message_count().await, 0);
    }

    #[tokio::test]
    async fn test_messages_rpc_conversion() {
        let session = create_test_session();
        session
            .messages
            .write()
            .await
            .push(Message::new("user", "hello"));
        let msgs = session.get_messages_as_rpc().await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content[0].text.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn test_loop_set_and_cancel() {
        let session = create_test_session();
        assert!(!session.is_loop_active().await);
        session.set_loop(Some("test prompt".to_string())).await;
        assert!(session.is_loop_active().await);
        assert_eq!(
            *session.loop_prompt.read().await,
            Some("test prompt".to_string())
        );
        assert!(!session
            .loop_cancelled
            .load(std::sync::atomic::Ordering::SeqCst));
        session.cancel_loop().await;
        assert!(session
            .loop_cancelled
            .load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_loop_reset_clears() {
        let session = create_test_session();
        session.set_loop(Some("loop text".to_string())).await;
        assert!(session.is_loop_active().await);
        session.reset().await;
        assert!(!session.is_loop_active().await);
        assert!(!session
            .loop_cancelled
            .load(std::sync::atomic::Ordering::SeqCst));
    }
}
