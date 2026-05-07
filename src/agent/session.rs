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
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Message {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_call(assistant_content: &str, calls: Vec<crate::tools::ToolCall>) -> Self {
        Message {
            role: "assistant".to_string(),
            content: assistant_content.to_string(),
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Message {
            role: "tool".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }
}

use crate::compaction::{self, CompactionResult};
use crate::sessions;
use crate::skills::Skill;
use crate::tools::{self, ToolCall};

/// Build the system prompt describing available tools, skills, and context files.
fn build_system_prompt(skills: &[Skill], context_files: &[ContextFile]) -> String {
    let now = chrono::Local::now();
    let mut prompt = format!(
        "You are rupi, an AI coding assistant running in a terminal.

Current date and time: {}

Available tools:
- bash: Execute a bash command in the terminal. Use for shell commands, scripts, curl, git, compilers.
- read: Read a file with optional line offset/limit.
- write: Write or overwrite a file. Creates parent directories.
- edit: Replace exact text in a file (one match only).
- grep: Search file contents with regex (uses ripgrep, falls back to grep).
- find: Find files by glob pattern (uses fd, falls back to find).
- ls: List directory contents.",
        now.format("%A, %B %d, %Y at %I:%M:%S %p %z (%Z)")
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
    approval_fn: std::sync::RwLock<Option<ApprovalFn>>,
    model: std::sync::RwLock<String>,
    context_window: u64,
    #[allow(dead_code)]
    cwd: String,
    skills: Vec<Skill>,
    context_files: Vec<ContextFile>,
    messages: RwLock<Vec<Message>>,
    session_path: RwLock<Option<std::path::PathBuf>>,
    is_streaming: Mutex<bool>,
    is_compacting: Mutex<bool>,
    abort_signal: Mutex<Option<watch::Sender<bool>>>,
    thinking_level: RwLock<String>,
    auto_compaction_enabled: RwLock<bool>,
    message_count: RwLock<u64>,
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
        let session_path = sessions::create_session(&model).ok();
        AgentSession {
            provider,
            approval_fn: std::sync::RwLock::new(None),
            model: std::sync::RwLock::new(model),
            context_window,
            cwd,
            skills,
            context_files,
            messages: RwLock::new(Vec::new()),
            session_path: RwLock::new(session_path),
            is_streaming: Mutex::new(false),
            is_compacting: Mutex::new(false),
            abort_signal: Mutex::new(None),
            thinking_level: RwLock::new("off".to_string()),
            auto_compaction_enabled: RwLock::new(true),
            message_count: RwLock::new(0),
        }
    }

    pub fn from_config(config: OpenAIConfig) -> Self {
        Self::from_config_with(
            config,
            std::env::current_dir().unwrap_or_default().to_string_lossy().to_string(),
            Vec::new(),
            Vec::new(),
        )
    }

    pub fn from_config_with(
        config: OpenAIConfig,
        cwd: String,
        skills: Vec<Skill>,
        context_files: Vec<ContextFile>,
    ) -> Self {
        let context_window = config.context_window;
        let model = config.model.clone();
        let provider = Arc::new(OpenAIProvider::new(config));
        Self::new(provider as Arc<dyn ChatProvider>, model, context_window, cwd, skills, context_files)
    }

    /// Get the session file path, if any.
    pub async fn session_path(&self) -> Option<std::path::PathBuf> {
        self.session_path.read().await.clone()
    }

    pub fn set_model(&self, new_model: String) {
        *self.model.write().unwrap() = new_model;
    }

    pub fn model(&self) -> String {
        self.model.read().unwrap().clone()
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

    pub async fn message_count(&self) -> u64 {
        *self.message_count.read().await
    }

    /// Persist a message to the session file.
    async fn persist_message(&self, msg: &Message) {
        if let Some(ref path) = *self.session_path.read().await {
            let _ = sessions::append_message(path, msg);
        }
    }

    /// Persist a compaction record to the session file.
    async fn persist_compaction(&self, summary: &str, tokens_before: u64) {
        if let Some(ref path) = *self.session_path.read().await {
            let _ = sessions::append_compaction(path, summary, tokens_before);
        }
    }

    /// Reset the session (clear messages, create new session file).
    pub async fn reset(&self) {
        self.messages.write().await.clear();
        *self.message_count.write().await = 0;
        let new_path = sessions::create_session(&self.model.read().unwrap()).ok();
        *self.session_path.write().await = new_path;
    }

    /// Stream a prompt to the model. Events are sent to the event_tx channel.
    /// Handles multi-turn tool execution (bash, etc.) internally.
    pub async fn prompt(
        &self,
        message: &str,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        // Check if already streaming
        {
            let mut streaming = self.is_streaming.lock().await;
            if *streaming {
                return Err(AgentError::Config("Already streaming".into()));
            }
            *streaming = true;
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

        // Run the multi-turn tool loop
        let result = self.run_tool_loop(event_tx.clone()).await;

        {
            let mut streaming = self.is_streaming.lock().await;
            *streaming = false;
        }

        result
    }

    /// Internal tool loop: keeps sending messages + executing tools until final response.
    async fn run_tool_loop(
        &self,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let max_turns = 10;
        let start_time = std::time::Instant::now();
        let max_duration = std::time::Duration::from_secs(120);

        for _turn in 0..max_turns {
            if start_time.elapsed() > max_duration {
                let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                    role: "assistant".to_string(),
                    content: vec![MessageContent {
                        content_type: "text".to_string(),
                        text: Some("Request timed out".to_string()),
                    }],
                    model: Some(self.model.read().unwrap().clone()),
                    usage: None,
                    stop_reason: Some("timeout".to_string()),
                }));
                let _ = event_tx.send(AgentEvent::turn_end());
                let _ = event_tx.send(AgentEvent::agent_end());
                return Err(AgentError::Timeout);
            }
            // Check abort signal
            {
                let signal = self.abort_signal.lock().await;
                if let Some(ref tx) = *signal {
                    if *tx.borrow() {
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
                }
            }

            // Create abort signal for this round
            let (abort_tx, abort_rx) = watch::channel(false);
            {
                let mut signal = self.abort_signal.lock().await;
                *signal = Some(abort_tx);
            }

            // Build messages: system prompt with skills + context files, then history
            let prompt_text = build_system_prompt(&self.skills, &self.context_files);
            let system_msg = Message::new("system", &prompt_text);
            let mut messages_for_api = vec![system_msg];
            messages_for_api.extend(self.messages.read().await.clone());

            let current_model = self.model.read().unwrap().clone();

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
                        let retryable = matches!(&e, AgentError::Api { status_code, .. }
                            if *status_code == 429 || *status_code >= 500
                        );
                        if !retryable || attempt == 2 {
                            last_error = Some(e);
                            break;
                        }
                        let delay = std::time::Duration::from_secs(1 << attempt);
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
            let mut input_tokens = 0;
            let mut output_tokens = 0;
            let mut prompt_cost: Option<f64> = None;
            let mut completion_cost: Option<f64> = None;
            let mut total_cost: Option<f64> = None;

            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut _finish_reason: Option<String> = None;
            loop {
                let event = match tokio::time::timeout(
                    std::time::Duration::from_secs(120),
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
                        let _ = event_tx.send(AgentEvent::generation_id(id));
                    }
                    StreamEvent::Delta(delta) => {
                        full_content.push_str(&delta);
                        let _ = event_tx.send(AgentEvent::message_update(delta));
                    }
                    StreamEvent::Done(result) => {
                        full_content = result.content;
                        input_tokens = result.input_tokens;
                        output_tokens = result.output_tokens;
                        if let Some(c) = result.cost {
                            prompt_cost = Some(c.prompt_cost);
                            completion_cost = Some(c.completion_cost);
                            total_cost = Some(c.total_cost);
                        }
                        _finish_reason = Some("stop".to_string());
                    }
                    StreamEvent::ToolCalls {
                        calls,
                        content,
                        input_tokens: it,
                        output_tokens: ot,
                        cost,
                        finish_reason: fr,
                    } => {
                        full_content = content;
                        input_tokens = it;
                        output_tokens = ot;
                        if let Some(c) = cost {
                            prompt_cost = Some(c.prompt_cost);
                            completion_cost = Some(c.completion_cost);
                            total_cost = Some(c.total_cost);
                        }
                        _finish_reason = fr;
                        tool_calls = calls;
                    }
                    StreamEvent::Error(err) => {
                        let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
                            role: "assistant".to_string(),
                            content: if full_content.is_empty() {
                                vec![]
                            } else {
                                vec![MessageContent {
                                    content_type: "text".to_string(),
                                    text: Some(full_content.clone()),
                                }]
                            },
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

            // If tool calls were made, execute them and continue
                if !tool_calls.is_empty() {
                // Add assistant message with tool calls to history
                let assistant_msg = Message::tool_call(&full_content, tool_calls.clone());
                self.persist_message(&assistant_msg).await;
                self.messages.write().await.push(assistant_msg);

                // Execute each tool, emit events, and add results to history
                for tc in &tool_calls {
                    let _ = event_tx.send(AgentEvent::tool_execution_start(
                        tc.name.clone(),
                        tc.arguments.clone(),
                    ));

                    // Check approval callback
                    let allowed = {
                        let guard = self.approval_fn.read().unwrap();
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
                    let result_msg = Message::tool_result(&tc.id, &result);
                    self.persist_message(&result_msg).await;
                    self.messages.write().await.push(result_msg);
                }

                // Continue the loop for the next turn
                *self.message_count.write().await += 1;
                continue;
            }

            // No tool calls — this is the final response.
            // Run auto-compaction check in background (fire-and-forget)
            let auto_enabled = *self.auto_compaction_enabled.read().await;
            if auto_enabled {
                let msgs = self.messages.read().await.clone();
                let total = compaction::estimate_total_tokens(&msgs);
                if compaction::should_compact(total, self.context_window) {
                    // Schedule compaction in background
                    let _ = self.compact().await;
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
            let assistant_msg = Message::new("assistant", &full_content);
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
                stop_reason: Some("stop".to_string()),
            }));
            let _ = event_tx.send(AgentEvent::turn_end());
            let _ = event_tx.send(AgentEvent::agent_end());

            *self.message_count.write().await += 2;

            return Ok(());
        }

        // Max turns exceeded
        let _ = event_tx.send(AgentEvent::message_end(AgentMessage {
            role: "assistant".to_string(),
            content: vec![MessageContent {
                content_type: "text".to_string(),
                text: Some("Max iteration depth reached. Try breaking your request into smaller steps.".to_string()),
            }],
            model: Some(self.model.read().unwrap().clone()),
            usage: None,
            stop_reason: Some("max_turns".to_string()),
        }));
        let _ = event_tx.send(AgentEvent::turn_end());
        let _ = event_tx.send(AgentEvent::agent_end());

        Err(AgentError::Config("Max iteration depth reached".into()))
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
        let compact_model = self.model.read().unwrap().clone();
        let summary = compaction::generate_summary(
            &self.provider,
            &compact_model,
            messages_to_summarize,
            None,
        )
        .await?;

        // Replace summarized messages with a compaction summary message
        let mut all_messages = self.messages.write().await;
        let keep: Vec<Message> = all_messages[cut_index..].to_vec();
        *all_messages = keep;
        all_messages.insert(
            0,
            Message::new("system", &format!("[Compacted conversation history]\n{}", summary)),
        );

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
    pub async fn abort(&self) {
        let mut signal = self.abort_signal.lock().await;
        if let Some(tx) = signal.take() {
            let _ = tx.send(true);
        }
        let mut streaming = self.is_streaming.lock().await;
        *streaming = false;
    }

    /// Set an approval callback for tool execution.
    /// Called with (tool_name, args_json) before execution. Return true to allow.
    pub fn set_approval_fn(&self, f: Option<ApprovalFn>) {
        *self.approval_fn.write().unwrap() = f;
    }

    pub fn provider_model_info(&self) -> ModelInfo {
        self.provider.model_info()
    }

    pub async fn get_state(&self) -> SessionState {
        SessionState {
            model: Some(self.provider_model_info()),
            thinking_level: self.thinking_level.read().await.clone(),
            is_streaming: *self.is_streaming.lock().await,
            is_compacting: *self.is_compacting.lock().await,
            steering_mode: "all".to_string(),
            follow_up_mode: "all".to_string(),
            auto_compaction_enabled: *self.auto_compaction_enabled.read().await,
            message_count: self.messages.read().await.len(),
            pending_message_count: 0,
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
        *session.message_count.write().await = 1;

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
}
