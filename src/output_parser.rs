use serde_json::Value;

use crate::tools::ToolCall;

/// Extract tool calls from assistant text when the model emits them
/// in non-native formats (fenced blocks, tags, bare JSON arrays).
///
/// Handles these patterns:
///   ```tool\n{"name":"tool","input":{...}}\n```
///   <tool_call>\n{"name":"tool","input":{...}}\n</tool_call>
///   single JSON object with name + input
///
/// Returns extracted tool calls with synthetic IDs.
pub fn extract_tool_calls_from_text(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Pattern 1: fenced ```tool blocks
    calls.extend(extract_fenced_blocks(text));

    // Pattern 2: <tool_call>...</tool_call> tags
    calls.extend(extract_tool_call_tags(text));

    // Pattern 3: single JSON object with name+input (bare JSON)
    if calls.is_empty() {
        calls.extend(extract_bare_json(text));
    }

    calls
}

fn extract_fenced_blocks(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut pos = 0;
    let start_marker = "```tool";
    let end_marker = "```";

    while let Some(start) = text[pos..].find(start_marker) {
        let content_start = pos + start + start_marker.len();
        // Find the opening of the JSON (first { after the marker)
        let json_start = match text[content_start..].find('{') {
            Some(i) => content_start + i,
            None => break,
        };

        // Find the closing of the JSON (matching })
        let after_json = match find_matching_brace(&text[json_start..]) {
            Some(end) => json_start + end + 1,
            None => break,
        };

        let json_str = &text[json_start..after_json];
        if let Some(call) = parse_single_tool_call(json_str) {
            calls.push(call);
        }

        // Find end marker after the JSON
        let remaining = &text[after_json..];
        if let Some(end) = remaining.find(end_marker) {
            pos = after_json + end + end_marker.len();
        } else {
            break;
        }
    }

    calls
}

fn extract_tool_call_tags(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut pos = 0;
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    while let Some(start) = text[pos..].find(start_tag) {
        let content_start = pos + start + start_tag.len();
        // Find opening brace
        let json_start = match text[content_start..].find('{') {
            Some(i) => content_start + i,
            None => break,
        };

        let after_json = match find_matching_brace(&text[json_start..]) {
            Some(end) => json_start + end + 1,
            None => break,
        };

        let json_str = &text[json_start..after_json];
        if let Some(call) = parse_single_tool_call(json_str) {
            calls.push(call);
        }

        // Find closing tag after the JSON
        let remaining = &text[after_json..];
        if let Some(end) = remaining.find(end_tag) {
            pos = after_json + end + end_tag.len();
        } else {
            break;
        }
    }

    calls
}

fn extract_bare_json(text: &str) -> Vec<ToolCall> {
    let trimmed = text.trim();
    if !trimmed.starts_with('{') {
        return Vec::new();
    }

    if !trimmed.contains("\"name\"") {
        return Vec::new();
    }

    if let Some(end) = find_matching_brace(trimmed) {
        let json_str = &trimmed[..=end];
        if let Some(call) = parse_single_tool_call(json_str) {
            return vec![call];
        }
    }

    Vec::new()
}

fn parse_single_tool_call(json_str: &str) -> Option<ToolCall> {
    let repaired = repair_json(json_str);
    let v: Value = serde_json::from_str(&repaired).ok()?;
    tool_call_from_value(&v)
}

fn tool_call_from_value(v: &Value) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?.to_string();
    let input = v
        .get("input")
        .or_else(|| v.get("arguments"))
        .or_else(|| v.get("args"))?;
    let input_obj = if let Some(s) = input.as_str() {
        serde_json::from_str(s).unwrap_or_else(|_| serde_json::json!({"value": s}))
    } else {
        input.clone()
    };
    Some(ToolCall {
        id: format!(
            "call_text_{}",
            uuid::Uuid::new_v4()
                .to_string()
                .split('-')
                .next()
                .unwrap_or("x")
        ),
        name,
        arguments: input_obj,
    })
}

fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Repair common JSON issues from small models:
///   - trailing commas before `]` or `}`
pub fn repair_json(s: &str) -> String {
    let mut r = s.to_string();
    r = r.replace(",}", "}");
    r = r.replace(",]", "]");
    r.trim().to_string()
}

/// Check if assistant text contains embedded tool calls that should be extracted.
pub fn contains_embedded_tool_calls(text: &str) -> bool {
    text.contains("```tool") || text.contains("<tool_call>") || (text.contains("\"name\"") && text.contains("\"input\""))
}

pub fn has_native_tool_calls(message: &crate::agent::session::Message) -> bool {
    message.tool_calls.as_ref().map_or(false, |c| !c.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_fenced_tool_call() {
        let text = "Let me check that file.\n```tool\n{\"name\":\"read\",\"input\":{\"file_path\":\"/tmp/test.txt\"}}\n```\nDone.";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["file_path"], "/tmp/test.txt");
    }

    #[test]
    fn test_extract_tool_call_tags() {
        let text = "I need to run:\n<tool_call>\n{\"name\":\"bash\",\"input\":{\"command\":\"ls -la\"}}\n</tool_call>\nThat's it.";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls -la");
    }

    #[test]
    fn test_extract_multiple_fenced_calls() {
        let text = "Let me do two things:\n```tool\n{\"name\":\"read\",\"input\":{\"file_path\":\"a.txt\"}}\n```\n```tool\n{\"name\":\"read\",\"input\":{\"file_path\":\"b.txt\"}}\n```";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[1].name, "read");
    }

    #[test]
    fn test_no_false_positive() {
        let text = "This is just a normal response without any tool calls.";
        let calls = extract_tool_calls_from_text(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_repair_trailing_commas() {
        let result = repair_json("{\"name\":\"read\",\"input\":{\"file_path\":\"test.txt\",}}");
        assert!(!result.contains(",}"));
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["name"], "read");
    }

    #[test]
    fn test_extract_single_object() {
        let text = "{\"name\":\"read\",\"input\":{\"file_path\":\"test.txt\"}}";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
    }

    #[test]
    fn test_contains_embedded_tool_calls() {
        assert!(contains_embedded_tool_calls("```tool\n..."));
        assert!(contains_embedded_tool_calls("<tool_call>..."));
        assert!(!contains_embedded_tool_calls("Just text"));
    }

    #[test]
    fn test_extract_with_trailing_comma_fix() {
        let text = "```tool\n{\"name\":\"write\",\"input\":{\"file_path\":\"/tmp/test.txt\",\"content\":\"hello\",}}\n```";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write");
    }

    #[test]
    fn test_no_tool_calls_in_empty() {
        let calls = extract_tool_calls_from_text("");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_has_native_tool_calls() {
        use crate::agent::session::Message;
        let m = Message::tool_call(
            "",
            vec![ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: serde_json::json!({}),
            }],
        );
        assert!(has_native_tool_calls(&m));
        let m2 = Message::new("assistant", "hello");
        assert!(!has_native_tool_calls(&m2));
    }

    #[test]
    fn test_brace_depth_matching() {
        let s = "{\"a\":{\"b\":\"c\"}}";
        assert_eq!(find_matching_brace(s), Some(s.len() - 1));
    }

    #[test]
    fn test_nested_brace_matching() {
        let s = "{\"a\":{\"b\":[1,2,{\"x\":\"y\"}]}}extra";
        let idx = find_matching_brace(s);
        assert!(idx.is_some());
        // The match should include all nested braces
        assert_eq!(&s[..=idx.unwrap()], "{\"a\":{\"b\":[1,2,{\"x\":\"y\"}]}}");
    }

    #[test]
    fn test_repair_multiple_trailing_commas() {
        let result = repair_json("{\"a\":1,,\"b\":2,}");
        // repair_json only handles ",}" and ",]" patterns
        assert!(!result.contains(",}"));
    }

    #[test]
    fn test_extract_bare_json_with_arguments_key() {
        let text = "{\"name\":\"edit\",\"arguments\":{\"file_path\":\"/tmp/x.txt\",\"old_text\":\"old\",\"new_text\":\"new\"}}";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "edit");
    }

    #[test]
    fn test_tool_call_tags_multiline_json() {
        let text = "<tool_call>\n{\n\"name\":\"bash\",\n\"input\":{\"command\":\"echo hi\"}\n}\n</tool_call>";
        let calls = extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }

    #[test]
    fn test_not_confused_by_regular_text() {
        let text = "The input parameter should be a string";
        let calls = extract_tool_calls_from_text(text);
        assert!(calls.is_empty());
    }
}
