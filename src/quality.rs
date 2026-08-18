use crate::tools::ToolCall;
use serde_json::Value;

/// A quality issue detected in the assistant's response.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityIssue {
    /// Empty response — no text content and no tool calls.
    Empty,
    /// Tool call to a name not in the known tools list.
    Hallucinated(String),
    /// The same tool call repeated without progress, at a reminder threshold.
    Repeat {
        tool: String,
        count: u32,
        arguments: String,
    },
    /// Response was truncated (finish_reason: "length").
    Truncated,
    /// Other issue.
    Other(String),
}

/// Result of assessing response quality.
#[derive(Debug)]
pub struct QualityVerdict {
    pub ok: bool,
    pub reason: Option<QualityIssue>,
}

/// Consecutive-repeat counts that earn a reminder.
///
/// A single cut-off fires once and then goes quiet, which leaves a model that
/// ignored the first nudge free to spin. The ladder escalates instead: a gentle
/// note, then two increasingly specific ones. The ladder is also its own loop
/// protection, because it fires at most three times for one run of repeats.
pub const REPEAT_THRESHOLDS: &[u32] = &[3, 5, 8];

/// Maximum characters of canonical arguments quoted in a detailed reminder.
///
/// The payload that is being repeated can be a whole file body or a long command.
/// Quoting it unbounded would carry it into the next request — precisely in the
/// scenario where context is already being wasted. The cap bounds the reminder and
/// never the detection, which always compares the full canonical string.
pub const ARGUMENTS_PREVIEW_CHARS: usize = 500;

/// Serialize a tool call's arguments with object keys in sorted order.
///
/// Two calls that differ only in JSON key order are the same call. Comparing the
/// raw serialization missed that and let a model loop forever by shuffling keys.
pub fn canonical_arguments(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_canonical(&map[*key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

/// A stable key for one turn's set of tool calls.
fn calls_key(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|c| format!("{}({})", c.name, canonical_arguments(&c.arguments)))
        .collect::<Vec<_>>()
        .join("|")
}

/// How many consecutive turns end with this exact set of tool calls.
///
/// Counts the current turn, so an unrepeated call returns 1.
pub fn consecutive_repeat_count(current: &[ToolCall], recent: &[Vec<ToolCall>]) -> u32 {
    if current.is_empty() {
        return 0;
    }
    let key = calls_key(current);
    let mut count = 1;
    for previous in recent {
        if calls_key(previous) == key {
            count += 1;
        } else {
            break;
        }
    }
    count
}

/// Assess the quality of an assistant response.
///
/// Checks:
/// 1. Empty response (no text, no tool calls)
/// 2. Hallucinated tool names (calling tools that don't exist)
/// 3. Repeated identical tool calls, reported at the ladder thresholds
///
/// `text`: the assistant's text response.
/// `current_calls`: tool calls from this turn.
/// `recent_calls`: tool calls from recent turns (most recent first).
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

    // 3. Repeat detection. Reporting only ON a threshold keeps the reminder out of
    //    the request on every other repeat, so the escalation stays legible.
    let count = consecutive_repeat_count(current_calls, recent_calls);
    if REPEAT_THRESHOLDS.contains(&count) {
        let names: Vec<&str> = current_calls.iter().map(|c| c.name.as_str()).collect();
        let arguments = current_calls
            .first()
            .map(|c| canonical_arguments(&c.arguments))
            .unwrap_or_default();
        return QualityVerdict {
            ok: false,
            reason: Some(QualityIssue::Repeat {
                tool: names.join(", "),
                count,
                arguments,
            }),
        };
    }

    QualityVerdict { ok: true, reason: None }
}

/// Trim a canonical argument string to the reminder preview budget.
fn preview(arguments: &str) -> String {
    if arguments.len() <= ARGUMENTS_PREVIEW_CHARS {
        return arguments.to_string();
    }
    let mut end = ARGUMENTS_PREVIEW_CHARS;
    while end > 0 && !arguments.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... [{} more chars]", &arguments[..end], arguments.len() - end)
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
                  Available tools are: {}. \
                  Please re-issue your response using only these tools.",
                name,
                known_tool_names().join(", ")
            )
        }
        QualityIssue::Repeat { tool, count, arguments } => {
            // The first rung stays gentle: a model that is one retry into a
            // transient failure does not need a lecture. Later rungs name the tool,
            // the run length, and the exact arguments, because by then the model has
            // ignored the gentle note and needs the specifics to break out.
            if Some(count) == REPEAT_THRESHOLDS.first().map(|t| t) {
                "You are repeating the exact same tool call with identical arguments. \
                 Carefully analyze the previous result before calling again: if the task is \
                 not complete, try a different approach or different arguments instead of \
                 repeating the call."
                    .to_string()
            } else {
                format!(
                    "Repeated tool call detected:\n\
                     - tool: {}\n\
                     - consecutive_calls: {}\n\
                     - arguments: {}\n\
                     The repeated calls are not making progress. Do not call this tool with \
                     these exact arguments again. Inspect the latest result and choose a \
                     different action, different arguments, or finish the task if you have \
                     gathered enough evidence.",
                    tool,
                    count,
                    preview(arguments)
                )
            }
        }
        QualityIssue::Truncated => {
            "Your response was truncated (hit the output token limit). \
             If you were writing a file, it may have been written partially with a RUPI_TRUNCATED marker. \
             Read the file to see where it was cut off, then use Edit to continue from that point. \
             If the file was not written, try again with a shorter response or break the task into smaller steps."
                .to_string()
        }
        QualityIssue::Other(msg) => msg.clone(),
    }
}

/// Whether an issue is exempt from the per-session correction cap.
///
/// The cap exists to stop a correction loop. The repeat ladder cannot loop: it
/// fires at most once per threshold for one run of identical calls, so capping it
/// only silences the escalation that was doing the work.
pub fn is_self_limiting(issue: &QualityIssue) -> bool {
    matches!(issue, QualityIssue::Repeat { .. })
}

pub fn known_tool_names() -> &'static [&'static str] {
    // Derived from the registry so a new tool cannot go stale here, but computed
    // once: this runs on every assistant response, and `all_tools` rebuilds ten
    // JSON schemas each call.
    static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| crate::tools::all_tools().into_iter().map(|t| t.name).collect())
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
            raw_arguments: None,
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

    // ---- repeat ladder ----

    #[test]
    fn repeat_fires_at_the_first_threshold() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let recent = vec![
            vec![tc("bash", json!({"command": "ls"}))],
            vec![tc("bash", json!({"command": "ls"}))],
        ];
        let v = assess_response("", &calls, &recent, &["bash", "read"]);
        assert!(!v.ok);
        match v.reason {
            Some(QualityIssue::Repeat { count, ref tool, .. }) => {
                assert_eq!(count, 3);
                assert_eq!(tool, "bash");
            }
            other => panic!("expected a repeat issue, got {:?}", other),
        }
    }

    #[test]
    fn repeat_stays_quiet_between_thresholds() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        // Four in a row: past the first rung, not yet at the second.
        let recent = vec![vec![tc("bash", json!({"command": "ls"}))]; 3];
        let v = assess_response("", &calls, &recent, &["bash"]);
        assert!(v.ok, "count 4 is between rungs and must not fire");
    }

    #[test]
    fn repeat_fires_again_at_the_later_rungs() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        for rung in [5usize, 8] {
            let recent = vec![vec![tc("bash", json!({"command": "ls"}))]; rung - 1];
            let v = assess_response("", &calls, &recent, &["bash"]);
            assert!(!v.ok, "rung {} must fire", rung);
            match v.reason {
                Some(QualityIssue::Repeat { count, .. }) => assert_eq!(count as usize, rung),
                other => panic!("expected a repeat issue, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_two_identical_is_not_a_repeat_yet() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let recent = vec![vec![tc("bash", json!({"command": "ls"}))]];
        let v = assess_response("", &calls, &recent, &["bash", "read"]);
        assert!(v.ok);
    }

    #[test]
    fn test_different_args_breaks_the_run() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        let recent = vec![
            vec![tc("bash", json!({"command": "ls"}))],
            vec![tc("bash", json!({"command": "pwd"}))],
        ];
        let v = assess_response("", &calls, &recent, &["bash", "read"]);
        assert!(v.ok);
    }

    #[test]
    fn reordered_keys_are_the_same_call() {
        let calls = vec![tc("bash", json!({"command": "ls", "timeout": 5}))];
        let recent = vec![
            vec![tc("bash", json!({"timeout": 5, "command": "ls"}))],
            vec![tc("bash", json!({"command": "ls", "timeout": 5}))],
        ];
        let v = assess_response("", &calls, &recent, &["bash"]);
        assert!(!v.ok, "key order must not hide a repeat");
    }

    #[test]
    fn canonical_arguments_sorts_nested_objects() {
        let a = canonical_arguments(&json!({"b": {"y": 1, "x": 2}, "a": [3, {"n": 1, "m": 2}]}));
        let b = canonical_arguments(&json!({"a": [3, {"m": 2, "n": 1}], "b": {"x": 2, "y": 1}}));
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_arguments_keeps_array_order() {
        let a = canonical_arguments(&json!([1, 2]));
        let b = canonical_arguments(&json!([2, 1]));
        assert_ne!(a, b);
    }

    #[test]
    fn consecutive_repeat_count_counts_the_current_turn() {
        let calls = vec![tc("bash", json!({"command": "ls"}))];
        assert_eq!(consecutive_repeat_count(&calls, &[]), 1);
        assert_eq!(consecutive_repeat_count(&[], &[]), 0);
    }

    #[test]
    fn gentle_reminder_first_then_detailed() {
        let gentle = build_correction_message(&QualityIssue::Repeat {
            tool: "bash".into(),
            count: 3,
            arguments: "{\"command\":\"ls\"}".into(),
        });
        assert!(gentle.contains("Carefully analyze the previous result"));
        assert!(!gentle.contains("consecutive_calls"));

        let detailed = build_correction_message(&QualityIssue::Repeat {
            tool: "bash".into(),
            count: 5,
            arguments: "{\"command\":\"ls\"}".into(),
        });
        assert!(detailed.contains("consecutive_calls: 5"));
        assert!(detailed.contains("tool: bash"));
        assert!(detailed.contains("{\"command\":\"ls\"}"));
    }

    #[test]
    fn detailed_reminder_caps_the_argument_preview() {
        let huge = "y".repeat(4000);
        let message = build_correction_message(&QualityIssue::Repeat {
            tool: "write".into(),
            count: 8,
            arguments: huge,
        });
        assert!(message.contains("more chars]"));
        assert!(message.len() < ARGUMENTS_PREVIEW_CHARS + 600);
    }

    #[test]
    fn preview_does_not_split_multibyte_characters() {
        let text = "の".repeat(ARGUMENTS_PREVIEW_CHARS);
        let shown = preview(&text);
        assert!(shown.contains("more chars]"));
        assert!(shown.contains("の"));
    }

    #[test]
    fn repeat_issues_bypass_the_correction_cap() {
        assert!(is_self_limiting(&QualityIssue::Repeat {
            tool: "bash".into(),
            count: 3,
            arguments: "{}".into()
        }));
        assert!(!is_self_limiting(&QualityIssue::Empty));
    }

    #[test]
    fn test_build_correction_message() {
        let msg = build_correction_message(&QualityIssue::Empty);
        assert!(!msg.is_empty());
        assert!(msg.contains("tool"));

        let msg2 = build_correction_message(&QualityIssue::Hallucinated("foo".into()));
        assert!(msg2.contains("foo"));
        // The list of valid tools is derived, so a new tool cannot go stale here.
        assert!(msg2.contains("todo_write"));

        let msg4 = build_correction_message(&QualityIssue::Truncated);
        assert!(msg4.contains("truncated"));
        assert!(msg4.contains("RUPI_TRUNCATED"));
    }

    #[test]
    fn test_known_tool_names_tracks_the_registry() {
        let names = known_tool_names();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"todo_write"));
        assert!(names.contains(&"goal"));
        assert_eq!(names.len(), crate::tools::all_tools().len());
    }
}
