use std::sync::Arc;

use crate::agent::session::Message;
use crate::error::AgentError;
use crate::provider::ChatProvider;

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

/// Find the cut point index: the first message to KEEP.
/// Everything before this index is summarized.
/// We keep approximately `keep_recent` tokens worth of recent messages at the end.
pub fn find_cut_point(messages: &[Message], keep_recent: u64) -> Option<usize> {
    if messages.len() < 4 {
        return None; // Not enough messages to compact
    }

    let mut accumulated: u64 = 0;

    // Walk backwards from the end, accumulating tokens
    for i in (0..messages.len()).rev() {
        accumulated = accumulated.saturating_add(estimate_message_tokens(&messages[i]));
        if accumulated >= keep_recent {
            // We've accumulated enough tokens to keep. Cut here.
            // But don't cut at tool messages — they need their preceding assistant call.
            if i > 0 && messages[i].role == "tool" {
                // Include the tool message, cut before its corresponding assistant
                let mut cut = i.saturating_sub(1);
                while cut > 0 && messages[cut].role == "tool" {
                    cut = cut.saturating_sub(1);
                }
                return Some(cut);
            }
            if i > 0 && messages[i].role == "assistant" {
                // Check if next message is a tool — if so, include both
                return Some(i);
            }
            return Some(i);
        }
    }

    // All messages fit in keep_recent, no compaction needed
    None
}

/// Serialize messages into plain text for the summarization prompt.
fn serialize_conversation(messages: &[Message]) -> String {
    let mut text = String::new();
    for msg in messages {
        let role_upper = msg.role.to_uppercase();
        if msg.role == "tool" {
            // Truncate long tool results
            let content = if msg.content.len() > 2000 {
                format!("{}... [truncated]", &msg.content[..2000])
            } else {
                msg.content.clone()
            };
            text.push_str(&format!("<{}>\n{}\n</{}>\n", role_upper, content, role_upper));
        } else {
            text.push_str(&format!("<{}>\n{}\n</{}>\n", role_upper, msg.content, role_upper));
        }
    }
    text
}

const SUMMARIZATION_SYSTEM_PROMPT: &str = r#"You are a context summarization assistant for a coding agent.
Your task is to produce a concise structured summary of the conversation history provided.
Do NOT continue the conversation. Do NOT add new information.
Output only the summary in the following format:

## Summary
<concise summary of what was accomplished>

## Key Files
<list of files that were read or modified>

## Decisions Made
<important decisions that affect future work>

## Next Steps
<pending or planned next steps>"#;

/// Generate a summary of messages by calling the LLM.
pub async fn generate_summary(
    provider: &Arc<dyn ChatProvider>,
    model: &str,
    messages: &[Message],
    previous_summary: Option<&str>,
) -> Result<String, AgentError> {
    let conversation_text = serialize_conversation(messages);

    let user_prompt = match previous_summary {
        Some(prev) => format!(
            "Previous summary:\n{}\n\nNew conversation to merge:\n<conversation>\n{}\n</conversation>",
            prev, conversation_text
        ),
        None => format!(
            "Summarize the following conversation:\n<conversation>\n{}\n</conversation>",
            conversation_text
        ),
    };

    let system_msg = Message::new("system", SUMMARIZATION_SYSTEM_PROMPT);
    let user_msg = Message::new("user", &user_prompt);

    let response = provider.complete(model, &[system_msg, user_msg]).await?;
    Ok(response)
}

/// Run compaction: find cut point, generate summary, return result.
pub async fn run_compaction(
    provider: &Arc<dyn ChatProvider>,
    model: &str,
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

    let messages_to_summarize = &messages[..cut_index];
    let summary = generate_summary(provider, model, messages_to_summarize, None).await?;

    Ok(Some(CompactionResult {
        summary,
        tokens_before: total_tokens,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_serialize_conversation() {
        let msgs = vec![
            Message::new("user", "hello"),
            Message::new("assistant", "world"),
        ];
        let text = serialize_conversation(&msgs);
        assert!(text.contains("<USER>"));
        assert!(text.contains("hello"));
        assert!(text.contains("<ASSISTANT>"));
        assert!(text.contains("world"));
    }

    #[test]
    fn test_truncate_long_tool_result() {
        let long = "x".repeat(3000);
        let msgs = vec![Message::tool_result("call_1", &long)];
        let text = serialize_conversation(&msgs);
        assert!(text.contains("[truncated]"));
        assert!(text.len() < 2500); // truncated
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

        let result = rt.block_on(run_compaction(&provider, "gpt-4", &msgs, false, 128000));
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

        let result = rt.block_on(run_compaction(&provider, "gpt-4", &msgs, true, 128000));
        assert!(result.is_ok());
        // Below threshold, should return None
        assert!(result.unwrap().is_none());
    }
}
