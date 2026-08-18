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
pub fn snip_old_tool_results(messages: &mut Vec<Message>, preserve_turns: usize) -> u64 {
    if messages.len() < 4 {
        return 0;
    }

    let preserve_turns = if preserve_turns == 0 { SNIP_PRESERVE_TURNS } else { preserve_turns };

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

/// Default number of recent tokens to keep after compaction cuts.
const KEEP_RECENT_TOKENS: u64 = 20000;

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
    messages.iter().map(|m| estimate_message_tokens(m)).sum()
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
pub fn tool_pairing_balanced_before(messages: &[Message], cut: usize) -> bool {
    let cut = cut.min(messages.len());
    if cut == 0 {
        return true;
    }
    match messages[cut - 1].tool_calls {
        Some(ref calls) if !calls.is_empty() => false,
        _ => true,
    }
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
    messages.iter().any(|m| m.content.contains(SUMMARY_OPEN_TAG))
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
    format!("{}\n{}\n{}", SUMMARY_OPEN_TAG, summary.trim(), SUMMARY_CLOSE_TAG)
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

/// Run compaction: find cut point, generate summary, return result.
pub async fn run_compaction(
    provider: &Arc<dyn ChatProvider>,
    model: &str,
    system_prompt: Option<&str>,
    messages: &[Message],
    auto_compaction_enabled: bool,
    context_window: u64,
) -> Result<Option<CompactionResult>, AgentError> {
    if !auto_compaction_enabled {
        return Ok(None);
    }

    let total_tokens = estimate_total_tokens(messages);
    if !should_compact(total_tokens, context_window) {
        return Ok(None);
    }

    let cut_index = match find_cut_point(messages, KEEP_RECENT_TOKENS) {
        Some(i) => i,
        None => return Ok(None),
    };

    if cut_index == 0 {
        return Ok(None);
    }

    let region = &messages[..cut_index];
    let summary = generate_summary(provider, model, system_prompt, region).await?;

    Ok(Some(CompactionResult {
        summary,
        tokens_before: total_tokens,
    }))
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

        let carried = vec![Message::new("user", &build_checkpoint_body("## Summary\n- did things"))];
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
        assert!(!msgs[5].content.contains("[snip:"), "second tool should be preserved: {}", msgs[5].content);
        assert!(!msgs[8].content.contains("[snip:"), "third tool should be preserved: {}", msgs[8].content);
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

    #[test]
    fn test_run_compaction_disabled() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = crate::provider::openai::OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test".into(),
            model: "gpt-4".into(),
            context_window: 128000,
            timeout_secs: 0,
            reasoning: false,
        };
        let provider: Arc<dyn ChatProvider> = Arc::new(crate::provider::openai::OpenAIProvider::new(config));
        let msgs = vec![Message::new("user", "hi"), Message::new("assistant", "hello")];

        let result = rt.block_on(run_compaction(&provider, "gpt-4", None, &msgs, false, 128000));
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_run_compaction_below_threshold() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = crate::provider::openai::OpenAIConfig {
            base_url: "https://api.example.com".into(),
            api_key: "test".into(),
            model: "gpt-4".into(),
            context_window: 128000,
            timeout_secs: 0,
            reasoning: false,
        };
        let provider: Arc<dyn ChatProvider> = Arc::new(crate::provider::openai::OpenAIProvider::new(config));
        let msgs = vec![Message::new("user", "hi"), Message::new("assistant", "hello")];

        let result = rt.block_on(run_compaction(&provider, "gpt-4", None, &msgs, true, 128000));
        assert!(result.is_ok());
        // Below threshold, should return None
        assert!(result.unwrap().is_none());
    }
}
