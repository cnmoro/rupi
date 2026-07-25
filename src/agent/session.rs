use std::sync::Arc;

pub type ApprovalFn = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;
use tokio::sync::{Mutex, RwLock, watch};
use tokio::sync::mpsc;

use crate::provider::openai::{OpenAIConfig, OpenAIProvider};
use crate::provider::{ChatProvider, StreamEvent};
use crate::rpc::types::*;
use crate::error::AgentError;

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
    let beginning = &result[..keep_beginning];
    let end = &result[result.len() - keep_end..];
    format!(
        "{}... [truncated: {} chars]\n...{}",
        beginning,
        result.len() - keep_beginning - keep_end,
        end
    )
}

/// Build the system prompt describing available tools, skills, context files, and memory.
/// If `datetime` is provided, it is used as the current time (for KV cache stability).
/// Otherwise, `chrono::Local::now()` is used (for one-shot prompts like goal verification).
fn build_system_prompt(skills: &[Skill], context_files: &[ContextFile], memory_enabled: bool, datetime: Option<&str>) -> String {
    let time_str = match datetime {
        Some(d) => d.to_string(),
        None => chrono::Local::now().format("%A, %B %d, %Y at %I:%M:%S %p %z (%Z)").to_string(),
    };
    let mut prompt = format!(
        "You are an expert coding agent operating inside rupi, a coding agent harness. \
        You help users by reading files, executing commands, editing code, and writing new files.

Current date and time: {}

Available tools:
- bash: Execute bash commands (ls, grep, find, curl, git, compilers, etc.). Returns stdout and stderr. Optionally provide a timeout in seconds.
- read: Read file contents with optional line offset/limit.
- write: Create a NEW file. REFUSES if the file already exists — use edit to modify existing files instead. Creates parent directories if needed.
- edit: Replace exact text in a file. Supports batch edits via the edits array. Each old_text is matched against the ORIGINAL file content (not after other edits). Edits must not overlap. Prefer this over write for any change to an existing file.
- grep: Search file contents for patterns (uses ripgrep, respects .gitignore, falls back to grep).
- find: Find files by glob pattern (uses fd, respects .gitignore, falls back to find).
- ls: List directory contents.
- search_code: Search code using natural language queries. Uses a local AI model (Model2Vec with potion-code-16M) to find relevant code by what it does, not just by keyword matching. Describe what you are looking for in plain English. Falls back to keyword search if the model is unavailable.

Guidelines:
- Be concise in your responses
- Show file paths clearly when working with files
- Use bash to explore when you are unsure about the project structure or when you need to gather information
- When a command fails, read the error output and try a different approach rather than giving up
- If you don't have enough information to complete a task, use bash, read, grep, or find to get the necessary context",
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
                    if !files.iter().any(|f: &ContextFile| f.name == *name && f.content == content) {
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

/// Agent session manages conversation state and model interaction.
pub struct AgentSession {
    provider: Arc<dyn ChatProvider>,
    approval_fn: RwLock<Option<ApprovalFn>>,
    goal: RwLock<Option<String>>,
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
            provider, model, context_window, cwd, skills, context_files, None,
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
        let frozen_time = chrono::Local::now().format("%A, %B %d, %Y at %I:%M:%S %p %z (%Z)").to_string();
        let system_prompt = build_system_prompt(&skills, &context_files, false, Some(&frozen_time));
        AgentSession {
            provider,
            approval_fn: RwLock::new(None),
            goal: RwLock::new(None),
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
        }
    }

    pub fn from_config(config: OpenAIConfig) -> Self {
        Self::from_config_with(
            config,
            std::env::current_dir().unwrap_or_default().to_string_lossy().to_string(),
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
        let mut session = Self::new(provider as Arc<dyn ChatProvider>, model, context_window, cwd, skills, context_files);
        session.memory_enabled = memory_enabled;
        if memory_enabled {
            ensure_memory_file();
            // Rebuild system prompt with memory content included
            let frozen_time = chrono::Local::now().format("%A, %B %d, %Y at %I:%M:%S %p %z (%Z)").to_string();
            let prompt = build_system_prompt(&session.skills, &session.context_files, memory_enabled, Some(&frozen_time));
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
            provider as Arc<dyn ChatProvider>, model, context_window, cwd, skills, context_files, Some(session_path.clone()),
        );
        session.memory_enabled = memory_enabled;
        if memory_enabled {
            ensure_memory_file();
            let frozen_time = chrono::Local::now().format("%A, %B %d, %Y at %I:%M:%S %p %z (%Z)").to_string();
            let prompt = build_system_prompt(&session.skills, &session.context_files, memory_enabled, Some(&frozen_time));
            *session.system_prompt.get_mut() = prompt;
        }
        // Load existing messages into the session
        for msg in &messages {
            session.messages.write().await.push(msg.clone());
        }
        *session.session_path.write().await = Some(session_path);
        Ok(session)
    }

    /// Get the session file path, if any.
    pub async fn session_path(&self) -> Option<std::path::PathBuf> {
        self.session_path.read().await.clone()
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
        // Wait for any in-progress compaction to finish
        loop {
            let c = *self.is_compacting.lock().await;
            if !c { break; }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        self.is_streaming.store(false, std::sync::atomic::Ordering::SeqCst);
        *self.is_compacting.lock().await = false;
        self.messages.write().await.clear();
        self.recent_tool_calls.write().await.clear();
        *self.consecutive_quality_issues.write().await = 0;
        self.pending_steer.write().await.clear();
        self.pending_follow_up.write().await.clear();
        self.set_goal(None).await;
        self.set_loop(None).await;
        *self.auto_compaction_enabled.write().await = true;
        *self.thinking_level.write().await = "off".to_string();

        // Rebuild system prompt with a fresh frozen timestamp for the new session
        let frozen_time = chrono::Local::now().format("%A, %B %d, %Y at %I:%M:%S %p %z (%Z)").to_string();
        let prompt = build_system_prompt(&self.skills, &self.context_files, self.memory_enabled, Some(&frozen_time));
        *self.system_prompt.write().await = prompt;

        *self.abort_signal.lock().await = None;
        let new_path = sessions::create_session(&self.model()).ok();
        *self.session_path.write().await = new_path;
    }

    /// Stream a prompt to the model. Events are sent to the event_tx channel.
    /// Handles multi-turn tool execution (bash, etc.) internally.
    /// If a goal is set, loops until the goal is verified.
    /// If streaming and `streaming_behavior` is "steer" or "followUp", queues instead.
    pub async fn prompt(
        &self,
        message: &str,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        // Check if already streaming (atomically set to true if currently false)
        if self.is_streaming.compare_exchange(
            false, true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        ).is_err() {
            // Already streaming — queue as steer
            self.pending_steer.write().await.push(message.to_string());
            return Ok(());
        }

        // Add user message to history
        let user_msg = Message::new("user", message);
        self.persist_message(&user_msg).await;
        self.messages.write().await.push(user_msg);

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

        // Goal-aware execution loop
        let goal_text = self.goal.read().await.clone();

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

            let max_goal_iterations: u32 = 5;
            for goal_iter in 0..max_goal_iterations {
                if let Err(e) = self.run_tool_loop(wrapped_tx.clone()).await {
                    eprintln!("rupi: tool loop error in goal iteration {}: {}", goal_iter, e);
                    break;
                }

                if self.verify_goal(&g).await {
                    break;
                }

                if goal_iter == max_goal_iterations - 1 {
                    break;
                }

                let nudge_text = format!("Continue working toward the goal. The goal is: {}. Do not stop until this goal is fully achieved. What is your next step?", g);
                let nudge_msg = Message::new("user", &nudge_text);
                self.persist_message(&nudge_msg).await;
                self.messages.write().await.push(nudge_msg);
                let _ = event_tx.send(AgentEvent::turn_start());
                let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                    role: "user".to_string(),
                    content: vec![MessageContent {
                        content_type: "text".to_string(),
                        text: Some(nudge_text),
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
        } else if self.loop_prompt.read().await.is_some() {
            // Loop mode: re-send the loop prompt after each agent_end until cancelled.
            // Suppress agent_end events during the loop like goal mode.
            let loop_msg = self.loop_prompt.read().await.clone().unwrap();
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
                if self.loop_cancelled.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                if let Err(e) = self.run_tool_loop(wrapped_tx.clone()).await {
                    eprintln!("rupi: tool loop error in loop mode: {}", e);
                    break;
                }
                if self.loop_cancelled.load(std::sync::atomic::Ordering::SeqCst) {
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
            let pending = self.drain_pending().await;
            if pending.is_empty() {
                break;
            }
            for msg in &pending {
                self.persist_message(msg).await;
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

        self.is_streaming.store(false, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    /// Internal tool loop: keeps sending messages + executing tools until final response.
    async fn run_tool_loop(
        &self,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        for _turn_num in 0.. {
            // Check abort signal; also check persistent abort_requested flag
            let abort_now = {
                let signal = self.abort_signal.lock().await;
                if let Some(ref tx) = *signal {
                    *tx.borrow()
                } else {
                    // Signal is None — check the persistent flag (abort was called between iterations)
                    let requested = *self.abort_requested.lock().await;
                    // Reset the flag so subsequent iterations aren't cancelled too
                    *self.abort_requested.lock().await = false;
                    requested
                }
            };
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

            // Create abort signal for this round
            // Reset the persistent flag now that we have a fresh signal path
            *self.abort_requested.lock().await = false;
            let (abort_tx, abort_rx) = watch::channel(false);
            {
                let mut signal = self.abort_signal.lock().await;
                *signal = Some(abort_tx);
            }

            // Drain queued steer/follow-up messages, persist them, and add to history
            let drained = self.drain_pending().await;
            if !drained.is_empty() {
                eprintln!("rupi: processing {} queued message(s)", drained.len());
                // Add drained messages to the in-memory conversation and persist
                for msg in &drained {
                    self.messages.write().await.push(msg.clone());
                }
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
                            AgentError::Http(_) => true,       // timeouts, connection refused, DNS, TLS
                            AgentError::Timeout => true,
                            AgentError::Api { status_code, .. } => *status_code == 429 || *status_code >= 500,
                            _ => false,
                        };
                        if !retryable || attempt == 2 {
                            last_error = Some(e);
                            break;
                        }
                        let delay = std::time::Duration::from_secs(1 << attempt); // 1s, 2s, 4s
                        tokio::time::sleep(delay).await;
                    }
                }
            }

            let mut rx = match rx {
                Some(rx) => rx,
                None => {
                    let err = last_error.unwrap_or(AgentError::Config("Request failed after retries".into()));
                    let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                        role: "assistant".to_string(),
                        content: vec![],
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
                ).await {
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
                    eprintln!("rupi: output parser extracted {} tool call(s) from text", embedded.len());
                    tool_calls = embedded;
                }
            }

            // Truncation detection: if finish_reason is "length", the response was cut off
            if finish_reason.as_deref() == Some("length") {
                eprintln!("rupi: response truncated (finish_reason=length)");
                let mut cc = self.consecutive_quality_issues.write().await;
                let max_corrections: u32 = 2;
                if *cc < max_corrections {
                    let correction = quality::build_correction_message(&quality::QualityIssue::Truncated);
                    self.pending_follow_up.write().await.push(correction);
                    *cc += 1;
                } else {
                    eprintln!("rupi: truncation correction suppressed after {} corrections", *cc);
                }
            }

            // Quality check: assess the response before proceeding
            if !tool_calls.is_empty() || !full_content.is_empty() {
                let known = quality::known_tool_names();
                let recent = self.recent_tool_calls.read().await.clone();
                let verdict = quality::assess_response(&full_content, &tool_calls, &recent, &known);
                if !verdict.ok {
                    let issue = verdict.reason.as_ref().unwrap();
                    let mut cc = self.consecutive_quality_issues.write().await;
                    let max_corrections: u32 = 2;
                    if *cc < max_corrections {
                        let correction = quality::build_correction_message(issue);
                        eprintln!("rupi: quality issue detected: {:?} — queuing correction", issue);
                        self.pending_follow_up.write().await.push(correction);
                        *cc += 1;
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

            // If quality corrections were queued but no tool calls, process them inline
            if tool_calls.is_empty() && !self.pending_follow_up.read().await.is_empty() {
                let drained = self.drain_pending().await;
                for msg in &drained {
                    self.messages.write().await.push(msg.clone());
                    let _ = event_tx.send(AgentEvent::turn_start());
                    let _ = event_tx.send(AgentEvent::message_start(AgentMessage {
                        role: "user".to_string(),
                        content: vec![MessageContent {
                            content_type: "text".to_string(),
                            text: Some(msg.content.clone()),
                        }],
                        model: None,
                        usage: None,
                        stop_reason: None,
                    }));
                }
                continue;
            }

            // If tool calls were made, execute them and continue to next turn
            if !tool_calls.is_empty() {
                // Add assistant message with tool calls to history
                let rc = if reasoning_content.is_empty() { None } else { Some(reasoning_content.clone()) };
                let assistant_msg = Message::tool_call(&full_content, tool_calls.clone()).with_reasoning(rc);
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

                    let allowed = {
                        let guard = self.approval_fn.read().await;
                        guard.as_ref().map(|f| {
                            let args_str = serde_json::to_string(&tc.arguments).unwrap_or_default();
                            f(&tc.name, &args_str)
                        }).unwrap_or(true)
                    };

                    let result = if allowed {
                        tools::execute_tool(tc)
                    } else {
                        format!("[User denied execution of tool '{}']", tc.name)
                    };

                    let _ = event_tx.send(AgentEvent::tool_execution_end(
                        tc.name.clone(),
                        result.clone(),
                    ));
                    // Cap the stored tool result to keep KV cache prefill bounded.
                    // The full result is still sent to the event stream above;
                    // only the conversation-history copy is truncated.
                    let stored_result = capped_tool_result(&result);
                    let result_msg = Message::tool_result(&tc.id, &stored_result);
                    self.persist_message(&result_msg).await;
                    self.messages.write().await.push(result_msg);
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
            if full_content.is_empty() && had_stream_events == false {
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
            // Run auto-compaction check in background (fire-and-forget)
            let auto_enabled = *self.auto_compaction_enabled.read().await;
            if auto_enabled {
                let msgs = self.messages.read().await.clone();
                let total = compaction::estimate_total_tokens(&msgs);
                if compaction::should_compact(total, self.context_window) {
                    // Say why when it doesn't happen. Swallowing the error made a
                    // failed compaction indistinguishable from one that never
                    // triggered: the context silently stays over budget and every
                    // later turn pays to summarize again.
                    match self.compact().await {
                        Ok(result) => eprintln!(
                            "rupi: compacted at ~{} tokens (window {})",
                            result.tokens_before, self.context_window
                        ),
                        Err(e) => eprintln!("rupi: compaction failed: {}", e),
                    }
                }
            }

            let cost_data = if prompt_cost.is_some() || completion_cost.is_some() || total_cost.is_some() {
                Some(PromptCostData {
                    prompt_cost,
                    completion_cost,
                    total_cost,
                })
            } else {
                None
            };

            // Add assistant message to history
            let rc = if reasoning_content.is_empty() { None } else { Some(reasoning_content.clone()) };
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
                    if had_stream_events { "stop".to_string() } else { "error".to_string() }
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

        // Step 1: Snip old tool results on the actual stored messages (rule-based, no API cost)
        {
            let mut stored = self.messages.write().await;
            let snipped = compaction::snip_old_tool_results(&mut stored, 6);
            if snipped > 0 {
                eprintln!("rupi: snipped {} chars from old tool results", snipped);
            }
        }

        let messages = self.messages.read().await.clone();
        let total_tokens = compaction::estimate_total_tokens(&messages);

        if !compaction::should_compact(total_tokens, self.context_window) {
            {
                let mut compacting = self.is_compacting.lock().await;
                *compacting = false;
            }
            return Err(AgentError::Config("Context not full enough to compact".into()));
        }

        let keep_recent = self.context_window.saturating_div(10).max(1).min(20000);
        let cut_index = match compaction::find_cut_point(&messages, keep_recent) {
            Some(i) => i,
            None => {
                {
                    let mut compacting = self.is_compacting.lock().await;
                    *compacting = false;
                }
                return Err(AgentError::Config("Nothing to compact".into()));
            }
        };

        let messages_to_summarize = &messages[..cut_index];
        let compact_model = self.model();
        let summary = match compaction::generate_summary(
            &self.provider,
            &compact_model,
            messages_to_summarize,
            None,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                *self.is_compacting.lock().await = false;
                return Err(e);
            }
        };

        // Replace summarized messages with a compaction summary message.
        // Use role "user" (not "system") because OpenAI-compatible endpoints
        // expect at most one system message — the one built fresh each turn.
        // Relies on the summary's ## Goal section to retain the task objective.
        let summary_msg = Message::new(
            "user",
            &format!("{}\n{}", crate::sessions::COMPACTION_PREFIX, summary),
        );
        self.persist_message(&summary_msg).await;
        let mut all_messages = self.messages.write().await;
        let keep: Vec<Message> = all_messages[cut_index..].to_vec();
        *all_messages = keep;
        all_messages.insert(0, summary_msg);

        {
            let mut compacting = self.is_compacting.lock().await;
            *compacting = false;
        }

        self.persist_compaction(&summary, total_tokens).await;

        Ok(CompactionResult {
            summary,
            tokens_before: total_tokens,
        })
    }

    /// Check if auto-compaction is needed and run it.
    /// Called after each prompt completes.
    /// Queue a steer message (interrupts current generation).
    pub async fn steer(&self, message: &str) {
        self.pending_steer.write().await.push(message.to_string());
        self.abort().await;
    }

    /// Queue a follow-up message (processed after current generation finishes).
    pub async fn follow_up(&self, message: &str) {
        self.pending_follow_up.write().await.push(message.to_string());
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
        let mut signal = self.abort_signal.lock().await;
        if let Some(tx) = signal.take() {
            let _ = tx.send(true);
            // Signal delivered through the watch channel — no need for the flag
            *self.abort_requested.lock().await = false;
        } else {
            // No active stream — set persistent flag so the next iteration
            // checks and aborts immediately.
            *self.abort_requested.lock().await = true;
        }
    }

    /// Set a goal for durable execution. When set, the agent will loop until
    /// an internal verification prompt confirms the goal is met.
    pub async fn set_goal(&self, goal: Option<String>) {
        *self.goal.write().await = goal;
    }

    pub async fn get_goal(&self) -> Option<String> {
        self.goal.read().await.clone()
    }

    /// Set a loop prompt. When set, the agent will re-send the prompt
    /// repeatedly after each agent_end until cancelled.
    pub async fn set_loop(&self, prompt: Option<String>) {
        *self.loop_prompt.write().await = prompt;
        self.loop_cancelled.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Cancel the current loop (if any).
    pub async fn cancel_loop(&self) {
        self.loop_cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
        *self.loop_prompt.write().await = None;
        self.abort().await;
    }

    /// Check if loop mode is active.
    pub async fn is_loop_active(&self) -> bool {
        self.loop_prompt.read().await.is_some()
    }

    /// Verify whether the current goal has been achieved by asking the model.
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
                    format!("{}... [truncated: {} chars]", &m.content[..1000], m.content.len())
                } else {
                    m.content.clone()
                };
                format!("<{}>\n{}\n</{}>", m.role.to_uppercase(), content, m.role.to_uppercase())
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
        let system_msg = Message::new("system", &simple_system);
        let verify_msg = Message::new("user", &verify_prompt);
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
            pending_message_count: self.pending_steer.read().await.len() + self.pending_follow_up.read().await.len(),
            session_file: self.session_path.read().await.as_ref().map(|p| p.to_string_lossy().to_string()),
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
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("medium"));
        assert_eq!(session.cycle_thinking_level().await.as_deref(), Some("high"));
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
        assert_eq!(*session.loop_prompt.read().await, Some("test prompt".to_string()));
        assert!(!session.loop_cancelled.load(std::sync::atomic::Ordering::SeqCst));
        session.cancel_loop().await;
        assert!(session.loop_cancelled.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_loop_reset_clears() {
        let session = create_test_session();
        session.set_loop(Some("loop text".to_string())).await;
        assert!(session.is_loop_active().await);
        session.reset().await;
        assert!(!session.is_loop_active().await);
        assert!(!session.loop_cancelled.load(std::sync::atomic::Ordering::SeqCst));
    }
}
