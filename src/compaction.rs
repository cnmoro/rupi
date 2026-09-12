use std::sync::Arc;

use crate::agent::session::Message;
use crate::error::AgentError;
use crate::provider::ChatProvider;

/// Default number of recent turns to preserve when snipping old tool results.
/// Matches little-coder's `preserve_last_n_turns = 6`.
const SNIP_PRESERVE_TURNS: usize = 6;

/// Snip old tool results: truncates tool-role messages older than the last N turns.
/// Runs before LLM-based compaction (auto-compact) to reduce token count without API cost.
/// Keeps first ~40% + last ~30% of each old tool result (preserving beginning and end).
/// Returns the number of characters removed.
pub fn snip_old_tool_results(messages: &mut [Message], preserve_turns: usize) -> u64 {
    if messages.len() < 4 {
        return 0;
    }

    let preserve_turns = if preserve_turns == 0 {
        SNIP_PRESERVE_TURNS
    } else {
        preserve_turns
    };

    // Count backwards to find the preserve boundary.
    // When we find user #(preserve_turns+1), everything up to the end of
    // that exchange (user + assistant + tool results) is eligible for snip.
    let mut turn_count = 0;
    let mut boundary = 0usize;
    let mut found = false;
    for i in (0..messages.len()).rev() {
        if messages[i].role == "user" {
            turn_count += 1;
            if turn_count > preserve_turns {
                boundary = i;
                // Advance boundary past this user exchange (assistant + tool messages)
                let mut next = boundary + 1;
                while next < messages.len() && messages[next].role != "user" {
                    next += 1;
                }
                boundary = next;
                found = true;
                break;
            }
        }
    }

    if !found {
        return 0;
    }

    let mut removed: u64 = 0;
    for msg in messages.iter_mut().take(boundary) {
        if msg.role == "tool" && msg.content.len() > 500 {
            let len = msg.content.len();
            let first = floor_boundary(&msg.content, len / 5 * 2); // ~40% from start
            let last_start = ceil_boundary(&msg.content, len - len / 10 * 3); // ~30% from end
            if last_start <= first {
                continue;
            }
            let first_part = &msg.content[..first];
            let last_part = &msg.content[last_start..];
            let truncated = format!(
                "{}... [snip: {} chars truncated]\n...{}",
                first_part,
                last_start - first,
                last_part
            );
            removed += (len - truncated.len()) as u64;
            msg.content = truncated;
        }
    }

    removed
}

/// Largest character boundary at or below `index`.
fn floor_boundary(text: &str, index: usize) -> usize {
    let mut i = index.min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest character boundary at or above `index`.
fn ceil_boundary(text: &str, index: usize) -> usize {
    let mut i = index.min(text.len());
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Default number of tokens to reserve for the prompt + LLM response.
const RESERVE_TOKENS: u64 = 16384;

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary: String,
    pub tokens_before: u64,
}

/// Estimate tokens for a text using a simple heuristic (chars / 4).
pub fn estimate_tokens(text: &str) -> u64 {
    (text.len() as u64).max(1).div_ceil(4)
}

/// Estimate tokens for a single message.
pub fn estimate_message_tokens(msg: &Message) -> u64 {
    let mut total = estimate_tokens(&msg.content);
    // Add overhead for message role and structure (~4 tokens)
    total += 4;
    // Tool call payloads add overhead
    if let Some(ref calls) = msg.tool_calls {
        total += (calls.len() as u64) * 12;
        for tc in calls {
            total += estimate_tokens(&tc.name);
            total += estimate_tokens(&serde_json::to_string(&tc.arguments).unwrap_or_default());
        }
    }
    total
}

/// Estimate total tokens for a slice of messages.
pub fn estimate_total_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Whether compaction should trigger given current token count and context window.
pub fn should_compact(context_tokens: u64, context_window: u64) -> bool {
    context_tokens > context_window.saturating_sub(RESERVE_TOKENS)
}

// ---------------------------------------------------------------------------
// Tool pairing
// ---------------------------------------------------------------------------
//
// Every OpenAI-compatible endpoint rejects a request where a `tool` message has
// no preceding assistant `tool_calls` entry, or an assistant `tool_calls` entry
// has no following results. Compaction splits the history in two, and BOTH
// halves become their own request: the tail goes to the next turn, and the head
// is replayed to the summarizer. A cut in the middle of a call/result pair is a
// hard 400 on one side or the other, so the boundary is checked explicitly here
// instead of being inferred at the call site.

/// Whether `messages[..cut]` is a well-formed request prefix.
///
/// It is not when the last message opens tool calls whose results fall on the
/// other side of the cut. The summarizer replays this half verbatim, so an
/// unanswered call here is rejected by the provider.
/// Note: this checks only the message immediately before the cut. It is not
/// self-sufficient — a cut in the middle of one assistant's multi-result run would
/// pass it, because the message before the cut is then a `tool` with no calls of its
/// own. `tool_pairing_balanced_after` rejects exactly that case, and the two are
/// always used together in `snap_cut_to_boundary`. Keep them together.
pub fn tool_pairing_balanced_before(messages: &[Message], cut: usize) -> bool {
    let cut = cut.min(messages.len());
    if cut == 0 {
        return true;
    }
    !matches!(messages[cut - 1].tool_calls, Some(ref calls) if !calls.is_empty())
}

/// Whether `messages[cut..]` is a well-formed request suffix.
///
/// It is not when it starts with tool results whose assistant call was left
/// behind. The next turn sends this half after the summary message, so an
/// orphan result here is rejected by the provider.
pub fn tool_pairing_balanced_after(messages: &[Message], cut: usize) -> bool {
    let cut = cut.min(messages.len());
    if cut >= messages.len() {
        return true;
    }
    messages[cut].role != "tool"
}

/// Move a candidate cut earlier until both halves are well formed.
///
/// Walking backwards is the only safe direction: it can only grow the summarized
/// region, never leave more context than the caller budgeted for. Returns `None`
/// when no valid boundary above zero exists, which means nothing can be cut.
pub fn snap_cut_to_boundary(messages: &[Message], candidate: usize) -> Option<usize> {
    let mut cut = candidate.min(messages.len());
    while cut > 0 {
        if tool_pairing_balanced_before(messages, cut) && tool_pairing_balanced_after(messages, cut)
        {
            return Some(cut);
        }
        cut -= 1;
    }
    None
}

/// Find the cut point index: the first message to KEEP.
/// Everything before this index is summarized.
/// We keep approximately `keep_recent` tokens worth of recent messages at the end.
///
/// The returned index is always a boundary where neither half splits a tool
/// call from its results.
pub fn find_cut_point(messages: &[Message], keep_recent: u64) -> Option<usize> {
    if messages.len() < 4 {
        return None; // Not enough messages to compact
    }

    let mut accumulated: u64 = 0;
    let mut candidate: Option<usize> = None;

    // Walk backwards from the end, accumulating tokens.
    for i in (0..messages.len()).rev() {
        accumulated = accumulated.saturating_add(estimate_message_tokens(&messages[i]));
        if accumulated >= keep_recent {
            candidate = Some(i);
            break;
        }
    }

    // All messages fit in keep_recent — nothing to compact.
    let candidate = candidate?;
    let cut = snap_cut_to_boundary(messages, candidate)?;
    if cut == 0 {
        return None;
    }
    Some(cut)
}

// ---------------------------------------------------------------------------
// Summarization
// ---------------------------------------------------------------------------

/// Tags wrapping the structured summary inside a landed checkpoint message.
pub const SUMMARY_OPEN_TAG: &str = "<compacted-summary>";
/// Closing tag of a landed checkpoint summary.
pub const SUMMARY_CLOSE_TAG: &str = "</compacted-summary>";

/// Framing that makes the replacement message read as established background.
///
/// Without it the model opens the next turn by summarizing the summary, or treats
/// the checkpoint as a fresh instruction and re-plans work that is already done.
pub const CHECKPOINT_PREAMBLE: &str =
    "This is an automatically generated checkpoint that condenses an earlier span of this \
conversation to free context. Treat the captured context as established background and build on \
it without restating it. Continue the task directly from the messages that follow. Do not \
acknowledge this checkpoint.";

/// The summarization directive.
///
/// It is delivered as the FINAL user message after the replayed conversation, not
/// as a separate summarizer system prompt. Keeping the conversation's own system
/// prompt, tool schemas, and message prefix in front of it makes the auxiliary call
/// a genuine prefix of the last routed request, so the provider's KV cache is reused
/// instead of invalidated. The previous design serialized the history into one text
/// blob under a different system prompt, which was a guaranteed cache miss that
/// re-billed the whole conversation as fresh input on every compaction.
pub const COMPACTION_INSTRUCTION: &str = concat!(
    "You are now acting as a compaction engine for this coding agent. Condense the conversation ",
    "ABOVE into a structured checkpoint that lets another model resume the work with no loss of ",
    "essential context.\n\n",
    "Output EXACTLY the Markdown structure below. Keep every section, in order. Use terse bullets, ",
    "not prose paragraphs. Write \"(none)\" for an empty section. Never drop a section.\n\n",
    "## Primary Request and Intent\n",
    "- [the user's original and evolving goals; quote verbatim where the exact wording matters]\n\n",
    "## Key Technical Concepts\n",
    "- [technologies, frameworks, patterns, and conventions in play]\n\n",
    "## Files and Code\n",
    "- [exact path: why it matters, key changes or snippets]\n\n",
    "## Errors and Fixes\n",
    "- [error: how it was resolved, plus any related user feedback]\n\n",
    "## Pending Jobs\n",
    "- [explicitly requested work that is not yet complete]\n\n",
    "## Current Work\n",
    "- [precisely what was in progress at this checkpoint]\n\n",
    "## Next Step\n",
    "- [the single next action, directly in line with the most recent request, or \"(none)\"]\n\n",
    "## Critical Context\n",
    "- [decisions and their rationale, constraints, user preferences, open questions, data needed ",
    "to continue]\n\n",
    "Rules:\n",
    "- Write concise English engineering prose. Preserve exact file paths, commands, error strings, ",
    "identifiers, numeric values, function signatures, and syntax fragments.\n",
    "- Capture user feedback and explicit instructions faithfully, especially corrections.\n",
    "- Do NOT mention this summarization request or the fact that the context was compacted.\n",
    "- Output only the checkpoint text. Do not call any tool and do not take any other action.\n",
    "- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. ",
    "Do not copy it forward verbatim: preserve the facts that are still true, drop the stale ones, ",
    "and merge newer information into a single consolidated summary under the same structure."
);

/// Whether a replayed region already carries a landed checkpoint.
///
/// Used only for logging and for the merge assertion in tests. The instruction
/// itself always states the merge rule, so the summarizer needs no extra prompt.
pub fn region_contains_checkpoint(messages: &[Message]) -> bool {
    messages
        .iter()
        .any(|m| m.content.contains(SUMMARY_OPEN_TAG))
}

/// Build the exact message list sent to the summarizer.
///
/// The order is deliberate: the conversation's own system prompt, then the region
/// verbatim, then the instruction. Anything that reorders this breaks the prefix
/// match and turns a cache hit into a full re-prefill.
pub fn build_summarization_request(
    system_prompt: Option<&str>,
    region: &[Message],
) -> Vec<Message> {
    // The region must already be a whole request. Every caller derives it from
    // `find_cut_point`, which guarantees that; this catches a future caller that
    // does not, in debug builds, instead of sending the provider an invalid request.
    debug_assert!(
        region.first().map(|m| m.role.as_str()) != Some("tool"),
        "a summarization region must not start with an orphan tool result"
    );
    debug_assert!(
        tool_pairing_balanced_before(region, region.len()),
        "a summarization region must not end with an unanswered tool call"
    );
    let mut request = Vec::with_capacity(region.len() + 2);
    if let Some(system) = system_prompt {
        request.push(Message::new("system", system));
    }
    request.extend(region.iter().cloned());
    request.push(Message::new("user", COMPACTION_INSTRUCTION));
    request
}

/// Wrap a raw summary in the tags that let a later compaction recognize it.
pub fn frame_summary(summary: &str) -> String {
    format!(
        "{}\n{}\n{}",
        SUMMARY_OPEN_TAG,
        summary.trim(),
        SUMMARY_CLOSE_TAG
    )
}

/// Build the full checkpoint message body that replaces the summarized region.
pub fn build_checkpoint_body(summary: &str) -> String {
    format!(
        "{}\n{}\n\n{}",
        crate::sessions::COMPACTION_PREFIX,
        CHECKPOINT_PREAMBLE,
        frame_summary(summary)
    )
}

// ---------------------------------------------------------------------------
// Checkpoint validation
// ---------------------------------------------------------------------------
//
// Compaction replaces a large span of the conversation with whatever the
// summarizer returned. Nothing checked that the return value was a summary at
// all, so a refusal, a content-filter stub, or a response cut off at the token
// limit would be stored as the checkpoint and the whole region deleted behind
// it. The loss is silent: the agent continues from a checkpoint that says
// nothing, with no signal that its context was thrown away.
//
// The idea is borrowed from the deepseek harness and from SoL-Pi's reducer,
// which refuses a receipt whose claims it cannot verify against the archived
// source. A markdown checkpoint carries no structured evidence to verify quote
// by quote, so the check here is structural: a checkpoint has to look like one.

/// Why a summarizer response was refused as a checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointProblem {
    /// Nothing but whitespace came back.
    Empty,
    /// Far too short to be a summary of a full context window.
    TooShort(usize),
    /// The instruction demands these sections and they are absent.
    MissingSections(Vec<&'static str>),
    /// A section is present but says nothing.
    EmptySection(&'static str),
    /// The model echoed the instruction back instead of following it.
    InstructionEcho,
}

impl std::fmt::Display for CheckpointProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointProblem::Empty => write!(f, "an empty response"),
            CheckpointProblem::TooShort(chars) => {
                write!(f, "only {} characters, too short to be a checkpoint", chars)
            }
            CheckpointProblem::MissingSections(missing) => {
                write!(f, "a response missing the sections {}", missing.join(", "))
            }
            CheckpointProblem::EmptySection(section) => {
                write!(f, "a response whose {} section is empty", section)
            }
            CheckpointProblem::InstructionEcho => {
                write!(f, "the instruction echoed back instead of a summary")
            }
        }
    }
}

/// Sections the compaction instruction requires, by their heading text.
///
/// Matched after normalization, not literally. A model that writes
/// `**Primary Request and Intent**`, `## 1. Primary Request and Intent`, or
/// `##PRIMARY REQUEST AND INTENT` has followed the instruction; rejecting it
/// would throw away a good summary and fall back to a far worse checkpoint.
const REQUIRED_SECTIONS: [&str; 3] = ["primary request and intent", "current work", "next step"];

/// Sections that must carry substance, not just a heading.
///
/// `Next Step` is deliberately excluded: the instruction itself allows `(none)`
/// there, so an empty one is a valid checkpoint.
const SECTIONS_NEEDING_BODY: [&str; 2] = ["primary request and intent", "current work"];

/// Shortest response that can plausibly be a checkpoint.
const MIN_CHECKPOINT_CHARS: usize = 200;

/// Shortest body a required section must carry.
const MIN_SECTION_BODY_CHARS: usize = 12;

/// A sentence unique to the compaction instruction.
///
/// Parroting the prompt back is the most common weak-model failure, and the
/// instruction itself contains every required heading — so an echo used to sail
/// through the gate written to catch exactly that.
const INSTRUCTION_FINGERPRINT: &str = "acting as a compaction engine";

/// Reduce a line to its heading text, or `None` when it is not a heading.
///
/// Strips the markdown and numbering models decorate headings with, so matching
/// is on what the heading says rather than how it was typeset.
fn heading_text(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let is_hash = trimmed.starts_with('#');
    let is_bold = trimmed.starts_with("**");
    if !is_hash && !is_bold {
        return None;
    }
    let stripped: String = trimmed
        .trim_matches(|c: char| c == '#' || c == '*' || c == ':' || c == '.' || c.is_whitespace())
        .to_string();
    // Drop a leading section number such as `1.` or `2)`.
    let without_number = stripped
        .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == ' ');
    let text = without_number.trim().to_lowercase();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Locate a required section and return the text under it.
fn section_body(lines: &[&str], section: &str) -> Option<String> {
    // Exact match after normalization. A prefix match let `## Current Workflow`,
    // about something else entirely, satisfy the `Current Work` requirement.
    let start = lines
        .iter()
        .position(|line| heading_text(line).as_deref() == Some(section))?;
    // Stop at the next heading, ignoring anything inside a fenced code block. A
    // `#` comment in a snippet is not a heading, and treating it as one truncated
    // the section and rejected a summary that had followed the instruction to
    // preserve syntax fragments.
    let mut body: Vec<&str> = Vec::new();
    let mut in_fence = false;
    for line in &lines[start + 1..] {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            body.push(line);
            continue;
        }
        if !in_fence && heading_text(line).is_some() {
            break;
        }
        body.push(line);
    }
    Some(body.join("\n"))
}

/// Whether a section body says anything.
///
/// Bullet markers and dashes are decoration, so a body made only of them is
/// empty however long it is.
fn body_is_substantive(body: &str) -> bool {
    let meat: String = body
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '*' && *c != '_' && *c != '#')
        .collect();
    meat.chars().count() >= MIN_SECTION_BODY_CHARS
}

/// Check that a summarizer response is actually a checkpoint.
pub fn validate_summary(summary: &str) -> Result<(), CheckpointProblem> {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return Err(CheckpointProblem::Empty);
    }
    if trimmed.len() < MIN_CHECKPOINT_CHARS {
        return Err(CheckpointProblem::TooShort(trimmed.len()));
    }
    if trimmed.contains(INSTRUCTION_FINGERPRINT) {
        return Err(CheckpointProblem::InstructionEcho);
    }

    let lines: Vec<&str> = trimmed.lines().collect();
    let mut missing: Vec<&'static str> = Vec::new();
    for section in REQUIRED_SECTIONS {
        if section_body(&lines, section).is_none() {
            missing.push(section);
        }
    }
    if !missing.is_empty() {
        return Err(CheckpointProblem::MissingSections(missing));
    }
    for section in SECTIONS_NEEDING_BODY {
        let body = section_body(&lines, section).unwrap_or_default();
        if !body_is_substantive(&body) {
            return Err(CheckpointProblem::EmptySection(section));
        }
    }
    Ok(())
}

/// Longest a single recorded path or command may be in a mechanical checkpoint.
const MECHANICAL_ITEM_CHARS: usize = 120;
/// Most paths a mechanical checkpoint lists.
const MECHANICAL_MAX_FILES: usize = 40;
/// Most commands a mechanical checkpoint lists.
const MECHANICAL_MAX_COMMANDS: usize = 30;
/// Longest narration excerpt kept per message.
const MECHANICAL_NARRATION_CHARS: usize = 400;

/// Patterns whose following value is replaced before a command is recorded.
///
/// Redaction is best-effort and cannot be complete. It exists because compaction
/// is the one operation that would otherwise have dropped these strings from the
/// live context, and a mechanical checkpoint would instead promote them into
/// every later request.
const SECRET_MARKERS: [&str; 10] = [
    "authorization:",
    "bearer ",
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "--header",
    "-u ",
];

/// Shorten one model-authored string to a bounded, redacted single line.
///
/// The length cap is the load-bearing part. `file_path` and `command` come
/// straight from tool arguments, so a region of long one-liners used to produce a
/// checkpoint far larger than the span it replaced — which left the context over
/// budget with a checkpoint at index zero that `find_cut_point` could never cut
/// again, wedging the session permanently.
fn mechanical_item(value: &str) -> String {
    let line = value.lines().next().unwrap_or("").trim();
    let lowered = line.to_lowercase();
    if SECRET_MARKERS.iter().any(|marker| lowered.contains(marker)) {
        let head: String = line.chars().take(24).collect();
        return format!("{}... [redacted: may contain a credential]", head);
    }
    if line.chars().count() <= MECHANICAL_ITEM_CHARS {
        return line.to_string();
    }
    let head: String = line.chars().take(MECHANICAL_ITEM_CHARS).collect();
    format!("{}... [{} chars]", head, line.chars().count())
}

/// Build a checkpoint from the region with no model call.
///
/// Used when the summarizer will not produce a usable one. The two obvious
/// responses to that are both bad: storing the bad summary destroys the region
/// silently, and refusing to compact leaves the next request over the provider's
/// limit. This is the third option — a summary that states only what can be read
/// straight off the messages, so it invents nothing and always succeeds.
///
/// Every part of it is bounded. It runs precisely when the context is already
/// over budget, so it is the one summary that must never be large.
pub fn mechanical_checkpoint(region: &[Message]) -> String {
    let mut files: Vec<String> = Vec::new();
    let mut commands: Vec<String> = Vec::new();
    let mut files_seen = 0usize;
    let mut commands_seen = 0usize;
    for msg in region {
        let Some(calls) = msg.tool_calls.as_ref() else {
            continue;
        };
        for call in calls {
            // `or_else` only fires when the key is absent, so a `file_path` of the
            // wrong type still falls through to `path`.
            if let Some(path) = call
                .arguments
                .get("file_path")
                .and_then(|p| p.as_str())
                .or_else(|| call.arguments.get("path").and_then(|p| p.as_str()))
            {
                let item = mechanical_item(path);
                if !item.is_empty() && !files.contains(&item) {
                    files_seen += 1;
                    if files.len() < MECHANICAL_MAX_FILES {
                        files.push(item);
                    }
                }
            }
            if let Some(command) = call.arguments.get("command").and_then(|c| c.as_str()) {
                let item = mechanical_item(command);
                if !item.is_empty() && !commands.contains(&item) {
                    commands_seen += 1;
                    if commands.len() < MECHANICAL_MAX_COMMANDS {
                        commands.push(item);
                    }
                }
            }
        }
    }

    let mut out = String::new();
    out.push_str("## Primary Request and Intent\n");
    out.push_str(
        "- The summarizer did not return a usable checkpoint, so this one was built mechanically \
from the messages. It records only what the conversation shows. Read the files below before you \
rely on any of it.\n\n",
    );

    out.push_str("## Files and Code\n");
    if files.is_empty() {
        out.push_str("- (none recorded)\n");
    } else {
        for path in &files {
            out.push_str(&format!("- {}\n", path));
        }
        // Say when the list is partial. Without this, whoever resumes has no signal
        // that the region touched more files than are recorded here.
        if files_seen > files.len() {
            out.push_str(&format!(
                "- ... and {} more not listed\n",
                files_seen - files.len()
            ));
        }
    }

    out.push_str("\n## Commands Run\n");
    if commands.is_empty() {
        out.push_str("- (none recorded)\n");
    } else {
        for command in &commands {
            out.push_str(&format!("- {}\n", command));
        }
        if commands_seen > commands.len() {
            out.push_str(&format!(
                "- ... and {} more not listed\n",
                commands_seen - commands.len()
            ));
        }
    }

    out.push_str("\n## Current Work\n");
    let tail: Vec<&Message> = region
        .iter()
        .rev()
        .filter(|m| (m.role == "assistant" || m.role == "user") && !m.content.trim().is_empty())
        .take(3)
        .collect();
    if tail.is_empty() {
        out.push_str("- no narration was recorded in the replaced messages\n");
    } else {
        for msg in tail.iter().rev() {
            let text: String = msg
                .content
                .chars()
                .take(MECHANICAL_NARRATION_CHARS)
                .collect::<String>()
                .replace('\n', " ");
            out.push_str(&format!("- {}: {}\n", msg.role, text));
        }
    }

    out.push_str(&format!(
        "\n## Next Step\n- Re-read the files above and continue the task stated in the \
active-task block. {} messages were replaced by this checkpoint.\n",
        region.len()
    ));
    out
}

/// Generate a summary of `region` by replaying it to the model.
///
/// `system_prompt` must be the conversation's own system prompt. Passing `None`
/// still works but gives up the cache alignment that is the point of this call.
pub async fn generate_summary(
    provider: &Arc<dyn ChatProvider>,
    model: &str,
    system_prompt: Option<&str>,
    region: &[Message],
) -> Result<String, AgentError> {
    let request = build_summarization_request(system_prompt, region);
    let response = provider.complete_aligned(model, &request).await?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCall;

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
            raw_arguments: None,
        }
    }

    fn assistant_with_calls(ids: &[&str]) -> Message {
        Message::tool_call("running", ids.iter().map(|id| call(id)).collect())
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello"), 2); // 5/4 = 1.25 -> 2
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("a"), 1);
    }

    #[test]
    fn test_should_compact() {
        // 128K window, compact triggers at > 128K - 16K = 112K
        assert!(!should_compact(100_000, 128_000));
        assert!(should_compact(120_000, 128_000));
        assert!(!should_compact(50_000, 128_000));
    }

    #[test]
    fn test_estimate_message_tokens() {
        let msg = Message::new("user", "hello world");
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens >= 3); // content + role overhead
    }

    #[test]
    fn test_estimate_total_tokens() {
        let msgs = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "world"),
        ];
        let total = estimate_total_tokens(&msgs);
        assert!(total > 0);
    }

    #[test]
    fn test_find_cut_point_too_few_messages() {
        let msgs = vec![
            Message::new("user", "hi"),
            Message::new("assistant", "hello"),
        ];
        assert!(find_cut_point(&msgs, 1000).is_none());
    }

    #[test]
    fn test_find_cut_point_large_conversation() {
        let mut msgs = Vec::new();
        for i in 0..20 {
            msgs.push(Message::new("user", &format!("message {}", i)));
            msgs.push(Message::new("assistant", &format!("response {}", i)));
        }
        // With a small keep_recent, it should find a cut point
        let cut = find_cut_point(&msgs, 10);
        assert!(cut.is_some());
        let cut = cut.unwrap();
        assert!(cut > 0);
        assert!(cut < msgs.len());
    }

    // ---- tool pairing ----

    #[test]
    fn balanced_before_rejects_an_unanswered_call() {
        let msgs = vec![
            Message::new("user", "go"),
            assistant_with_calls(&["c1"]),
            Message::tool_result("c1", "done"),
        ];
        // Cutting at 2 leaves the assistant call in the head with no result.
        assert!(!tool_pairing_balanced_before(&msgs, 2));
        // Cutting at 1 or 3 keeps both halves whole.
        assert!(tool_pairing_balanced_before(&msgs, 1));
        assert!(tool_pairing_balanced_before(&msgs, 3));
    }

    #[test]
    fn balanced_after_rejects_an_orphan_result() {
        let msgs = vec![
            Message::new("user", "go"),
            assistant_with_calls(&["c1"]),
            Message::tool_result("c1", "done"),
            Message::new("assistant", "finished"),
        ];
        assert!(!tool_pairing_balanced_after(&msgs, 2));
        assert!(tool_pairing_balanced_after(&msgs, 1));
        assert!(tool_pairing_balanced_after(&msgs, 3));
    }

    #[test]
    fn balanced_checks_handle_the_edges() {
        let msgs = vec![Message::new("user", "go")];
        assert!(tool_pairing_balanced_before(&msgs, 0));
        assert!(tool_pairing_balanced_after(&msgs, msgs.len()));
        // An out-of-range index is clamped, never a panic.
        assert!(tool_pairing_balanced_before(&msgs, 99));
        assert!(tool_pairing_balanced_after(&msgs, 99));
    }

    #[test]
    fn snap_walks_back_to_a_whole_boundary() {
        let msgs = vec![
            Message::new("user", "go"),
            Message::new("assistant", "ok"),
            assistant_with_calls(&["c1", "c2"]),
            Message::tool_result("c1", "a"),
            Message::tool_result("c2", "b"),
            Message::new("assistant", "done"),
        ];
        // Index 3 splits the pair; index 4 does too. Both snap back to 2.
        assert_eq!(snap_cut_to_boundary(&msgs, 3), Some(2));
        assert_eq!(snap_cut_to_boundary(&msgs, 4), Some(2));
        // Index 5 is already whole.
        assert_eq!(snap_cut_to_boundary(&msgs, 5), Some(5));
    }

    #[test]
    fn snap_returns_none_when_no_boundary_exists() {
        // Every candidate above zero splits the one and only pair.
        let msgs = vec![
            assistant_with_calls(&["c1"]),
            Message::tool_result("c1", "a"),
        ];
        assert_eq!(snap_cut_to_boundary(&msgs, 1), None);
    }

    #[test]
    fn find_cut_point_never_splits_a_tool_pair() {
        // Build a long conversation where tool pairs land on many boundaries.
        let mut msgs = Vec::new();
        for i in 0..40 {
            msgs.push(Message::new("user", &format!("turn {}", i)));
            msgs.push(assistant_with_calls(&["c"]));
            msgs.push(Message::tool_result("c", &"x".repeat(200)));
            msgs.push(Message::new("assistant", "summary of the step"));
        }
        for keep in [50u64, 200, 800, 3000, 9000] {
            if let Some(cut) = find_cut_point(&msgs, keep) {
                assert!(
                    tool_pairing_balanced_before(&msgs, cut),
                    "keep={} cut={} split the head",
                    keep,
                    cut
                );
                assert!(
                    tool_pairing_balanced_after(&msgs, cut),
                    "keep={} cut={} split the tail",
                    keep,
                    cut
                );
            }
        }
    }

    // ---- summarization request shape ----

    #[test]
    fn summarization_request_is_a_prefix_of_the_conversation() {
        let region = vec![
            Message::new("user", "fix the parser"),
            Message::new("assistant", "reading it now"),
        ];
        let request = build_summarization_request(Some("SYSTEM PROMPT"), &region);

        // system + region verbatim + instruction, in that exact order.
        assert_eq!(request.len(), 4);
        assert_eq!(request[0].role, "system");
        assert_eq!(request[0].content, "SYSTEM PROMPT");
        assert_eq!(request[1].content, "fix the parser");
        assert_eq!(request[2].content, "reading it now");
        assert_eq!(request[3].role, "user");
        assert_eq!(request[3].content, COMPACTION_INSTRUCTION);
    }

    #[test]
    fn summarization_request_works_without_a_system_prompt() {
        let region = vec![Message::new("user", "hi")];
        let request = build_summarization_request(None, &region);
        assert_eq!(request.len(), 2);
        assert_eq!(request[0].content, "hi");
        assert_eq!(request[1].content, COMPACTION_INSTRUCTION);
    }

    #[test]
    fn instruction_states_the_prior_checkpoint_merge_rule() {
        assert!(COMPACTION_INSTRUCTION.contains("PRIOR checkpoint"));
        assert!(COMPACTION_INSTRUCTION.contains("merge newer information"));
        assert!(COMPACTION_INSTRUCTION.contains(SUMMARY_OPEN_TAG));
    }

    #[test]
    fn region_checkpoint_detection() {
        let plain = vec![Message::new("user", "hello")];
        assert!(!region_contains_checkpoint(&plain));

        let carried = vec![Message::new(
            "user",
            &build_checkpoint_body("## Summary\n- did things"),
        )];
        assert!(region_contains_checkpoint(&carried));
    }

    #[test]
    fn checkpoint_body_carries_prefix_preamble_and_tags() {
        let body = build_checkpoint_body("## Summary\n- did things");
        assert!(body.starts_with(crate::sessions::COMPACTION_PREFIX));
        assert!(body.contains(CHECKPOINT_PREAMBLE));
        assert!(body.contains(SUMMARY_OPEN_TAG));
        assert!(body.contains(SUMMARY_CLOSE_TAG));
        assert!(body.contains("did things"));
    }

    // ---- checkpoint validation ----

    fn good_summary() -> String {
        format!(
            "## Primary Request and Intent\n- refactor the parser\n\n\
             ## Key Technical Concepts\n- rust, nom\n\n\
             ## Files and Code\n- src/parse.rs: the tokenizer\n\n\
             ## Errors and Fixes\n- (none)\n\n\
             ## Pending Jobs\n- write the tests\n\n\
             ## Current Work\n- rewriting the lexer\n\n\
             ## Next Step\n- run the suite\n\n\
             ## Critical Context\n- keep the public API stable{}\n",
            " ".repeat(50)
        )
    }

    #[test]
    fn a_real_checkpoint_validates() {
        assert_eq!(validate_summary(&good_summary()), Ok(()));
    }

    #[test]
    fn an_empty_response_is_refused() {
        assert_eq!(validate_summary(""), Err(CheckpointProblem::Empty));
        assert_eq!(validate_summary("   \n\t "), Err(CheckpointProblem::Empty));
    }

    #[test]
    fn a_refusal_is_refused() {
        // The exact shape a filtered or declining response takes. Storing this
        // would delete the whole region and leave the agent with nothing.
        for refusal in [
            "I'm sorry, I can't help with that.",
            "I cannot summarize this conversation.",
            "```\n```",
        ] {
            assert!(
                validate_summary(refusal).is_err(),
                "accepted a refusal: {}",
                refusal
            );
        }
    }

    #[test]
    fn a_truncated_response_is_refused() {
        // A stream cut off at the token limit keeps the early sections and loses
        // the rest.
        let truncated = format!(
            "## Primary Request and Intent\n- refactor the parser{}\n\n## Key Technical Concepts\n- rust",
            " ".repeat(300)
        );
        match validate_summary(&truncated) {
            Err(CheckpointProblem::MissingSections(missing)) => {
                assert!(missing.contains(&"current work"), "{:?}", missing);
                assert!(missing.contains(&"next step"), "{:?}", missing);
            }
            other => panic!("expected missing sections, got {:?}", other),
        }
    }

    #[test]
    fn a_similar_heading_does_not_satisfy_a_required_one() {
        // `## Current Workflow` is about something else. A prefix match accepted it
        // as the `Current Work` section, so the real one could be absent entirely.
        let text = format!(
            "## Primary Request and Intent\n- refactor the parser thoroughly\n\n\
             ## Current Workflow\n- the CI pipeline runs on every push{}\n\n\
             ## Next Step\n- run the suite\n",
            " ".repeat(120)
        );
        match validate_summary(&text) {
            Err(CheckpointProblem::MissingSections(missing)) => {
                assert!(missing.contains(&"current work"), "{:?}", missing);
            }
            other => panic!("expected the section to be missing, got {:?}", other),
        }
    }

    #[test]
    fn heading_styles_models_actually_emit_are_accepted() {
        for style in [
            "## Primary Request and Intent",
            "**Primary Request and Intent**",
            "##Primary Request and Intent",
            "## primary request and intent",
            "### 1. Primary Request and Intent",
        ] {
            let text = format!(
                "{}\n- refactor the parser thoroughly and keep the API stable\n\n\
                 ## Current Work\n- rewriting the lexer right now{}\n\n\
                 ## Next Step\n- run the suite\n",
                style,
                " ".repeat(100)
            );
            assert_eq!(
                validate_summary(&text),
                Ok(()),
                "rejected heading style {:?}",
                style
            );
        }
    }

    #[test]
    fn a_code_fence_inside_a_section_does_not_end_it() {
        // The instruction asks the model to preserve syntax fragments. A `#`
        // comment inside a fence was read as the next heading, truncating the
        // section and rejecting a summary that had done as it was told.
        let text = format!(
            "## Primary Request and Intent\n- refactor the parser thoroughly\n\n\
             ## Current Work\n```python\n# initialize the parser\ndef foo(): pass\n```\n\
             - rewrote the lexer and it now passes{}\n\n\
             ## Next Step\n- run the suite\n",
            " ".repeat(100)
        );
        assert_eq!(validate_summary(&text), Ok(()));
    }

    #[test]
    fn the_instruction_echoed_back_is_refused() {
        // The instruction contains every required heading, so parroting it used to
        // pass the gate written to catch exactly that.
        assert_eq!(
            validate_summary(COMPACTION_INSTRUCTION),
            Err(CheckpointProblem::InstructionEcho)
        );
    }

    #[test]
    fn headings_with_no_body_are_refused() {
        let bare = format!(
            "## Primary Request and Intent\n\n## Current Work\n\n## Next Step\n{}",
            "-".repeat(220)
        );
        assert!(matches!(
            validate_summary(&bare),
            Err(CheckpointProblem::EmptySection(_))
        ));
    }

    #[test]
    fn the_fallback_says_when_its_lists_are_partial() {
        let region: Vec<Message> = (0..60)
            .map(|i| {
                Message::tool_call(
                    "",
                    vec![ToolCall {
                        id: format!("c{}", i),
                        name: "edit".into(),
                        arguments: serde_json::json!({"file_path": format!("src/f{}.rs", i)}),
                        raw_arguments: None,
                    }],
                )
            })
            .collect();
        let checkpoint = mechanical_checkpoint(&region);
        assert!(
            checkpoint.contains("and 20 more not listed"),
            "{}",
            checkpoint
        );
    }

    #[test]
    fn the_fallback_bounds_a_region_of_enormous_commands() {
        // The defect this closes: item COUNT was capped but item LENGTH was not, so
        // a region of long one-liners produced a checkpoint larger than the span it
        // replaced. The context then stayed over budget with an uncuttable
        // checkpoint at index zero, and the session could never compact again.
        let region: Vec<Message> = (0..30)
            .map(|i| {
                Message::tool_call(
                    "",
                    vec![ToolCall {
                        id: format!("c{}", i),
                        name: "bash".into(),
                        arguments: serde_json::json!({
                            "command": format!("python3 -c \"d={}\"", "0".repeat(100_000))
                        }),
                        raw_arguments: None,
                    }],
                )
            })
            .collect();
        let checkpoint = mechanical_checkpoint(&region);
        assert!(
            checkpoint.len() < 16_000,
            "fallback was {} bytes",
            checkpoint.len()
        );
        assert_eq!(validate_summary(&checkpoint), Ok(()));
    }

    #[test]
    fn the_fallback_redacts_a_credential_bearing_command() {
        let region = vec![Message::tool_call(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "curl -H 'Authorization: Bearer sk-live-DEADBEEF' https://api.example"
                }),
                raw_arguments: None,
            }],
        )];
        let checkpoint = mechanical_checkpoint(&region);
        assert!(!checkpoint.contains("sk-live-DEADBEEF"), "{}", checkpoint);
        assert!(checkpoint.contains("redacted"), "{}", checkpoint);
    }

    #[test]
    fn a_short_response_is_refused_before_the_section_check() {
        let short = "## Primary Request and Intent\n## Current Work\n## Next Step";
        assert!(matches!(
            validate_summary(short),
            Err(CheckpointProblem::TooShort(_))
        ));
    }

    #[test]
    fn the_problem_reads_as_a_sentence() {
        assert!(format!("{}", CheckpointProblem::Empty).contains("empty"));
        assert!(format!("{}", CheckpointProblem::TooShort(12)).contains("12"));
        assert!(
            format!("{}", CheckpointProblem::MissingSections(vec!["next step"]))
                .contains("next step")
        );
        assert!(format!("{}", CheckpointProblem::InstructionEcho).contains("echoed back"));
        assert!(format!("{}", CheckpointProblem::EmptySection("current work")).contains("empty"));
    }

    // ---- mechanical fallback ----

    #[test]
    fn the_mechanical_checkpoint_validates_as_one() {
        let region = vec![
            Message::new("user", "fix the parser"),
            Message::tool_call(
                "reading",
                vec![ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"file_path": "src/parse.rs"}),
                    raw_arguments: None,
                }],
            ),
            Message::tool_result("c1", "file body"),
            Message::new("assistant", "I rewrote the lexer"),
        ];
        let checkpoint = mechanical_checkpoint(&region);
        // The fallback must itself pass the gate, or compaction has no way out.
        assert_eq!(validate_summary(&checkpoint), Ok(()));
    }

    #[test]
    fn the_mechanical_checkpoint_reports_only_what_it_can_read() {
        let region = vec![
            Message::new("user", "fix the parser"),
            Message::tool_call(
                "working",
                vec![
                    ToolCall {
                        id: "c1".into(),
                        name: "edit".into(),
                        arguments: serde_json::json!({"file_path": "src/parse.rs"}),
                        raw_arguments: None,
                    },
                    ToolCall {
                        id: "c2".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "cargo test\n--lib"}),
                        raw_arguments: None,
                    },
                ],
            ),
            Message::tool_result("c1", "ok"),
            Message::new("assistant", "the lexer is rewritten"),
        ];
        let checkpoint = mechanical_checkpoint(&region);
        assert!(checkpoint.contains("src/parse.rs"));
        assert!(checkpoint.contains("cargo test"), "{}", checkpoint);
        assert!(checkpoint.contains("the lexer is rewritten"));
        assert!(checkpoint.contains("4 messages were replaced"));
        // It must say plainly that it is not a model summary.
        assert!(checkpoint.contains("mechanically"));
    }

    #[test]
    fn the_mechanical_checkpoint_survives_an_empty_region() {
        let checkpoint = mechanical_checkpoint(&[]);
        assert_eq!(validate_summary(&checkpoint), Ok(()));
        assert!(checkpoint.contains("(none recorded)"));
    }

    #[test]
    fn the_mechanical_checkpoint_bounds_what_it_lists() {
        let mut region = Vec::new();
        for i in 0..200 {
            region.push(Message::tool_call(
                "",
                vec![ToolCall {
                    id: format!("c{}", i),
                    name: "edit".into(),
                    arguments: serde_json::json!({"file_path": format!("src/file{}.rs", i)}),
                    raw_arguments: None,
                }],
            ));
        }
        let checkpoint = mechanical_checkpoint(&region);
        assert_eq!(
            checkpoint.matches("- src/file").count(),
            40,
            "file list is not capped"
        );
        // A fallback that grew with the region would defeat the compaction.
        assert!(
            checkpoint.len() < 8000,
            "fallback was {} chars",
            checkpoint.len()
        );
    }

    #[test]
    fn the_mechanical_checkpoint_deduplicates() {
        let region: Vec<Message> = (0..10)
            .map(|i| {
                Message::tool_call(
                    "",
                    vec![ToolCall {
                        id: format!("c{}", i),
                        name: "bash".into(),
                        arguments: serde_json::json!({"command": "cargo test"}),
                        raw_arguments: None,
                    }],
                )
            })
            .collect();
        let checkpoint = mechanical_checkpoint(&region);
        assert_eq!(checkpoint.matches("- cargo test").count(), 1);
    }

    // ---- snip ----

    #[test]
    fn test_snip_old_tool_results_short_conversation() {
        let mut msgs = vec![
            Message::new("user", "hi"),
            Message::new("assistant", "hello"),
        ];
        let removed = snip_old_tool_results(&mut msgs, 6);
        assert_eq!(removed, 0);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_snip_truncates_long_tool_output() {
        let mut msgs = vec![
            Message::new("user", "first msg"),
            Message::new("assistant", "let me check"),
            Message::tool_result("call_1", &"x".repeat(2000)),
            Message::new("user", "second msg"),
            Message::new("assistant", "a2"),
            Message::new("user", "third msg"),
            Message::new("assistant", "a3"),
        ];
        let removed = snip_old_tool_results(&mut msgs, 2);
        assert!(removed > 0);
        // The first exchange's tool result should have been truncated
        assert!(msgs[2].content.len() < 1500);
        assert!(msgs[2].content.contains("[snip:"));
    }

    #[test]
    fn test_snip_preserves_recent_tools() {
        let mut msgs = vec![
            Message::new("user", "first"),
            Message::new("assistant", "a1"),
            Message::tool_result("c1", &"x".repeat(1000)),
            Message::new("user", "second"),
            Message::new("assistant", "a2"),
            Message::tool_result("c2", &"y".repeat(1000)),
            Message::new("user", "third"),
            Message::new("assistant", "a3"),
            Message::tool_result("c3", &"z".repeat(1000)),
        ];
        let removed = snip_old_tool_results(&mut msgs, 2);
        assert!(removed > 0);
        // First tool result (turn 1 of 3) should be snipped (exceeds preserve_turns=2)
        assert!(msgs[2].content.contains("[snip:"));
        // Second and third tool results should NOT be snipped (within last 2 turns)
        assert!(
            !msgs[5].content.contains("[snip:"),
            "second tool should be preserved: {}",
            msgs[5].content
        );
        assert!(
            !msgs[8].content.contains("[snip:"),
            "third tool should be preserved: {}",
            msgs[8].content
        );
    }

    #[test]
    fn test_snip_short_tool_output_not_truncated() {
        let mut msgs = vec![
            Message::new("user", "first"),
            Message::new("assistant", "a1"),
            Message::tool_result("c1", "short"),
        ];
        let removed = snip_old_tool_results(&mut msgs, 6);
        assert_eq!(removed, 0);
        assert_eq!(msgs[2].content, "short");
    }

    #[test]
    fn snip_does_not_split_multibyte_tool_output() {
        // 3 bytes per character, so naive byte offsets land mid-character.
        let body = "の".repeat(400);
        let mut msgs = vec![
            Message::new("user", "first"),
            Message::new("assistant", "a1"),
            Message::tool_result("c1", &body),
            Message::new("user", "second"),
            Message::new("assistant", "a2"),
            Message::new("user", "third"),
            Message::new("assistant", "a3"),
        ];
        let removed = snip_old_tool_results(&mut msgs, 2);
        assert!(removed > 0);
        assert!(msgs[2].content.contains("[snip:"));
        assert!(msgs[2].content.contains("の"));
    }
}
