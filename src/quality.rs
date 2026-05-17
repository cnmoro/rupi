use crate::tools::ToolCall;

/// A quality issue detected in the assistant's response.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityIssue {
    /// Empty response — no text content and no tool calls.
    Empty,
    /// Tool call to a name not in the known tools list.
    Hallucinated(String),
    /// Same tool call (name + arguments) repeated N times in recent history.
    Loop(String),
    /// Other issue.
    Other(String),
}

/// Result of assessing response quality.
#[derive(Debug)]
pub struct QualityVerdict {
    pub ok: bool,
    pub reason: Option<QualityIssue>,
}

/// Assess the quality of an assistant response.
///
/// Checks:
/// 1. Empty response (no text, no tool calls)
/// 2. Hallucinated tool names (calling tools that don't exist)
/// 3. Loop detection (same tool call repeated in consecutive turns)
///
/// `text`: the assistant's text response.
/// `current_calls`: tool calls from this turn.
/// `recent_calls: &[Vec<ToolCall>]`: tool calls from recent turns (most recent first).
/// `known_tools`: set of valid tool names.
pub fn assess_response(
    text: &str,
    current_calls: &[ToolCall],
    recent_calls: &[Vec<ToolCall>],
    known_tools: &[&str],
) -> QualityVerdict {
    // 1. Empty response
    let has_text = !text.trim().is_empty();
    if !has_text && current_calls.is_empty() {
        return QualityVerdict {
            ok: false,
            reason: Some(QualityIssue::Empty),
        };
    }

    // 2. Hallucinated tool names
    for call in current_calls {
        if !known_tools.contains(&call.name.as_str()) {
            return QualityVerdict {
                ok: false,
                reason: Some(QualityIssue::Hallucinated(call.name.clone())),
            };
        }
    }

    // 3. Loop detection: same tool name + same arguments in last N calls
    if !current_calls.is_empty() {
        let mut loop_count = 1;
        for prev_calls in recent_calls.iter().take(4) {
            if calls_match(current_calls, prev_calls) {
                loop_count += 1;
            } else {
                break;
            }
        }
        if loop_count >= 3 {
            let names: Vec<&str> = current_calls.iter().map(|c| c.name.as_str()).collect();
            return QualityVerdict {
                ok: false,
                reason: Some(QualityIssue::Loop(names.join(", "))),
            };
        }
    }

    QualityVerdict { ok: true, reason: None }
}

/// Check if two sets of tool calls are structurally identical (same names + same args).
fn calls_match(a: &[ToolCall], b: &[ToolCall]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (ca, cb) in a.iter().zip(b.iter()) {
        if ca.name != cb.name {
            return false;
        }
        if ca.arguments != cb.arguments {
            return false;
        }
    }
    true
}

/// Build a correction message to nudge the model back on track.
pub fn build_correction_message(issue: &QualityIssue) -> String {
    match issue {
        QualityIssue::Empty => {
            "Your previous response was empty. Please use your available tools to make progress on the task. \
             If you need to read a file, run a command, or write/edit code, do so now."
                .to_string()
        }
        QualityIssue::Hallucinated(name) => {
            format!(
                 "Your previous response included a call to tool '{}', which is not a valid tool. \
                  Available tools are: bash, read, write, edit, grep, find, ls, search_code. \
                  Please re-issue your response using only these tools.",
                name
            )
        }
        QualityIssue::Loop(names) => {
            format!(
                "You have been repeating the same tool call(s) [{}] multiple times without making progress. \
                 Try a different approach — read the error output, examine the file structure, or reconsider \
                 your strategy before making the same call again.",
                names
            )
        }
        QualityIssue::Other(msg) => msg.clone(),
    }
}

pub fn known_tool_names() -> Vec<&'static str> {
    vec!["bash", "read", "write", "edit", "grep", "find", "ls", "search_code"]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tc(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "test".into(),
            name: name.into(),
            arguments: args,
        }
    }

    #[test]
    fn test_empty_response_detected() {
        let v = assess_response("", &[], &[], &["bash", "read"]);
        assert!(!v.ok);
        assert_eq!(v.reason, Some(QualityIssue::Empty));
    }

    #[test]
    fn test_hallucinated_tool_detected() {
        let calls = vec![tc("nonexistent_tool", json!({}))];
        let v = assess_response("hello", &calls, &[], &["bash", "read"]);
        assert!(!v.ok);
        assert_eq!(v.reason, Some(QualityIssue::Hallucinated("nonexistent_tool".into())));
    }

    #[test]
    fn test_valid_response_passes() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let v = assess_response("Running...", &calls, &[], &["bash", "read", "write", "edit"]);
        assert!(v.ok);
    }

    #[test]
    fn test_no_tool_calls_but_has_text_passes() {
        let v = assess_response("hello world", &[], &[], &["bash"]);
        assert!(v.ok);
    }

    #[test]
    fn test_loop_detected() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let recent = vec![
            vec![tc("bash", json!({"command": "ls"}))],
            vec![tc("bash", json!({"command": "ls"}))],
        ];
        // Third time in a row
        let v = assess_response("", &calls, &recent, &["bash", "read"]);
        assert!(!v.ok);
        assert!(matches!(v.reason, Some(QualityIssue::Loop(_))));
    }

    #[test]
    fn test_two_identical_is_not_loop() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let recent = vec![
            vec![tc("bash", json!({"command": "ls"}))],
        ];
        // Only second time — not a loop yet
        let v = assess_response("", &calls, &recent, &["bash", "read"]);
        assert!(v.ok);
    }

    #[test]
    fn test_different_args_breaks_loop() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let recent = vec![
            vec![tc("bash", json!({"command": "ls"}))],
            vec![tc("bash", json!({"command": "pwd"}))],
        ];
        let v = assess_response("", &calls, &recent, &["bash", "read"]);
        assert!(v.ok);
    }

    #[test]
    fn test_build_correction_message() {
        let msg = build_correction_message(&QualityIssue::Empty);
        assert!(!msg.is_empty());
        assert!(msg.contains("tool"));

        let msg2 = build_correction_message(&QualityIssue::Hallucinated("foo".into()));
        assert!(msg2.contains("foo"));

        let msg3 = build_correction_message(&QualityIssue::Loop("bash".into()));
        assert!(msg3.contains("repeating"));
    }

    #[test]
    fn test_known_tool_names() {
        let names = known_tool_names();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"search_code"));
        assert_eq!(names.len(), 8);
    }
}
