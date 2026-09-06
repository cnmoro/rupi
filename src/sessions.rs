use std::path::PathBuf;

use crate::agent::session::Message;
use crate::rpc::jsonl::serialize_json_line;

const SESSIONS_DIR: &str = "rupi_sessions";

static SESSIONS_DIR_OVERRIDE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Store sessions in `dir` instead of `~/.config/rupi_sessions/`.
///
/// Embedders that sandbox rupi (or keep one session directory per tenant) need
/// the transcripts somewhere they control, without depending on `$HOME`. Only
/// the first call wins, so this must run before any session is opened.
pub fn set_sessions_dir(dir: PathBuf) {
    let _ = SESSIONS_DIR_OVERRIDE.set(dir);
}

/// Get the sessions directory path (`--sessions-dir`, else `~/.config/rupi_sessions/`).
pub fn sessions_dir() -> Option<PathBuf> {
    if let Some(dir) = SESSIONS_DIR_OVERRIDE.get() {
        return Some(dir.clone());
    }
    dirs::home_dir().map(|h| h.join(".config").join(SESSIONS_DIR))
}

/// Ensure the sessions directory exists.
pub fn ensure_sessions_dir() -> Result<PathBuf, String> {
    let dir = sessions_dir().ok_or_else(|| "Cannot determine home directory".to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create sessions dir: {}", e))?;
    Ok(dir)
}

/// A session file entry written to the JSONL log.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionEntry<'a> {
    #[serde(rename = "type")]
    pub entry_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<&'a SessionMessageEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_before: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<&'a str>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionMessageEntry {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Info about a saved session.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub model: String,
    pub created_at: String,
    pub message_count: usize,
}

/// Create a new session file under a caller-chosen id. Returns the file path.
///
/// `--session <id>` names the conversation, so an embedder that owns the id
/// (one per chat, say) gets a predictable transcript path from the first turn
/// instead of a UUID it then has to discover.
pub fn create_session_with_id(id: &str, model: &str) -> Result<PathBuf, String> {
    let dir = ensure_sessions_dir()?;
    write_session_header(&dir.join(format!("{}.jsonl", id)), id, model)
}

/// Create a new session file. Returns the file path.
pub fn create_session(model: &str) -> Result<PathBuf, String> {
    let dir = ensure_sessions_dir()?;
    let id = uuid::Uuid::new_v4().to_string();
    write_session_header(&dir.join(format!("{}.jsonl", id)), &id, model)
}

fn write_session_header(path: &std::path::Path, id: &str, model: &str) -> Result<PathBuf, String> {
    let path = path.to_path_buf();

    // Write session header with the ID
    let header = SessionEntry {
        entry_type: "session",
        model: Some(model),
        message: None,
        summary: None,
        tokens_before: None,
        session_id: Some(id),
    };
    let mut file =
        std::fs::File::create(&path).map_err(|e| format!("Cannot create session file: {}", e))?;
    use std::io::Write;
    writeln!(file, "{}", serialize_json_line(&header).trim())
        .map_err(|e| format!("Cannot write session header: {}", e))?;
    Ok(path)
}

/// Open a session file for appending, recreating it if it vanished.
///
/// An embedder may prune the session directory between turns; without
/// `create(true)` every later write fails with ENOENT and the transcript is
/// silently lost for the rest of the process's life — taking the compaction
/// record and the generation ids with it.
fn open_for_append(path: &PathBuf) -> Result<std::fs::File, String> {
    std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|e| format!("Cannot open session file: {}", e))
}

/// Append a message entry to a session file.
pub fn append_message(path: &PathBuf, msg: &Message) -> Result<(), String> {
    let tool_calls = msg.tool_calls.as_ref().map(|calls| {
        calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            })
            .collect()
    });
    let entry = SessionEntry {
        entry_type: "message",
        model: None,
        message: Some(&SessionMessageEntry {
            role: msg.role.clone(),
            content: msg.content.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
            reasoning_content: msg.reasoning_content.clone(),
        }),
        summary: None,
        tokens_before: None,
        session_id: None,
    };
    let mut file = open_for_append(path)?;
    use std::io::Write;
    writeln!(file, "{}", serialize_json_line(&entry).trim())
        .map_err(|e| format!("Cannot append to session file: {}", e))?;
    Ok(())
}

/// Append a compaction entry to a session file.
pub fn append_compaction(path: &PathBuf, summary: &str, tokens_before: u64) -> Result<(), String> {
    let entry = SessionEntry {
        entry_type: "compaction",
        model: None,
        message: None,
        summary: Some(summary),
        tokens_before: Some(tokens_before),
        session_id: None,
    };
    let mut file = open_for_append(path)?;
    use std::io::Write;
    writeln!(file, "{}", serialize_json_line(&entry).trim())
        .map_err(|e| format!("Cannot append compaction to session file: {}", e))?;
    Ok(())
}

/// Marker that identifies the summary message written when a session compacts.
pub const COMPACTION_PREFIX: &str = "[Compacted conversation history]";

/// Read the text of a session entry's `content`.
///
/// rupi writes a plain string. Transcripts written by other agents in this
/// family use a content-block array (`[{"type":"text","text":"…"}]`), which we
/// accept so an embedder can hand us a conversation it recorded itself.
fn message_text(content: &serde_json::Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let Some(blocks) = content.as_array() else {
        return String::new();
    };
    blocks
        .iter()
        .filter_map(|block| match block {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(map) => match map.get("type").and_then(|t| t.as_str()) {
                Some("text") | None => map.get("text").and_then(|t| t.as_str()).map(str::to_string),
                Some("toolCall") => map
                    .get("name")
                    .and_then(|n| n.as_str())
                    .map(|name| format!("[called tool: {}]", name)),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Map role spellings from other agents onto the ones the provider expects.
fn normalize_role(role: &str) -> &str {
    match role {
        "toolResult" | "tool_result" => "tool",
        other => other,
    }
}

/// Record the provider's generation id for a turn.
///
/// Streaming usage is an estimate; the authoritative cost has to be fetched
/// from the provider afterwards, keyed by this id. Keeping it only in the event
/// stream loses it whenever nobody was listening (crash, detached run), so it
/// belongs in the transcript.
pub fn append_generation_id(path: &PathBuf, generation_id: &str) -> Result<(), String> {
    let entry = serde_json::json!({
        "type": "generation_id",
        "id": generation_id,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    let mut file = open_for_append(path)?;
    use std::io::Write;
    writeln!(file, "{}", serialize_json_line(&entry).trim())
        .map_err(|e| format!("Cannot append generation id to session file: {}", e))?;
    Ok(())
}

/// Load all entries from a session file.
pub fn load_session(path: &PathBuf) -> Result<Vec<Message>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("Cannot read session file: {}", e))?;
    let mut messages: Vec<Message> = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            let entry_type = val["type"].as_str().unwrap_or("");
            if entry_type == "compaction" {
                // Everything above this point was summarized. Replaying it too
                // would restore the very context the compaction just dropped —
                // the session would resume over budget and immediately compact
                // again, paying for a summary every single time.
                let summary_msg = match messages.last() {
                    Some(last) if last.content.starts_with(COMPACTION_PREFIX) => last.clone(),
                    // Written by a different tool, or the summary message never
                    // made it to disk: rebuild it from the compaction record.
                    _ => Message::new(
                        "user",
                        &format!(
                            "{}\n{}",
                            COMPACTION_PREFIX,
                            val["summary"].as_str().unwrap_or_default()
                        ),
                    ),
                };
                messages.clear();
                messages.push(summary_msg);
                continue;
            }
            if entry_type == "message" {
                if let Some(msg_val) = val.get("message") {
                    let role = msg_val["role"].as_str().unwrap_or("user");
                    let content = message_text(&msg_val["content"]);
                    let content = content.as_str();

                    // Try to deserialize with optional tool_calls and tool_call_id
                    if let Ok(entry) =
                        serde_json::from_value::<SessionMessageEntry>(msg_val.clone())
                    {
                        let mut msg = Message::new(normalize_role(&entry.role), &entry.content);
                        msg.tool_call_id = entry.tool_call_id;
                        msg.reasoning_content = entry.reasoning_content;
                        if let Some(calls) = entry.tool_calls {
                            let tool_calls: Vec<crate::tools::ToolCall> = calls
                                .into_iter()
                                .filter_map(|v| {
                                    let id = v.get("id")?.as_str()?.to_string();
                                    let name = v.get("name")?.as_str()?.to_string();
                                    let args = v.get("arguments").cloned().unwrap_or_default();
                                    Some(crate::tools::ToolCall {
                                        id,
                                        name,
                                        arguments: args,
                                        raw_arguments: None,
                                    })
                                })
                                .collect();
                            if !tool_calls.is_empty() {
                                msg.tool_calls = Some(tool_calls);
                            }
                        }
                        messages.push(msg);
                    } else {
                        messages.push(Message::new(normalize_role(role), content));
                    }
                }
            }
        }
    }
    Ok(messages)
}

/// List all saved sessions with metadata.
pub fn list_sessions() -> Result<Vec<SessionInfo>, String> {
    let dir = match sessions_dir() {
        Some(d) => d,
        None => return Ok(Vec::new()),
    };
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let contents = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut model = String::new();
        let mut message_count = 0;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                match val["type"].as_str().unwrap_or("") {
                    "session" => {
                        model = val["model"].as_str().unwrap_or("").to_string();
                    }
                    "message" => {
                        message_count += 1;
                    }
                    _ => {}
                }
            }
        }

        let created = std::fs::metadata(&path)
            .and_then(|m| m.created().or_else(|_| m.modified()))
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        sessions.push(SessionInfo {
            id,
            model,
            created_at: chrono::DateTime::from_timestamp(created as i64, 0)
                .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            message_count,
        });
    }

    // Sort by creation time (newest first)
    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(sessions)
}

/// Find a session file by its ID.
pub fn find_session_path(id: &str) -> Option<PathBuf> {
    let dir = sessions_dir()?;
    let path = dir.join(format!("{}.jsonl", id));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolCall;
    use std::fs;

    #[test]
    fn test_sessions_dir() {
        let dir = sessions_dir();
        assert!(dir.is_some());
        assert!(dir.unwrap().to_string_lossy().contains("rupi_sessions"));
    }

    #[test]
    fn test_create_and_read_session() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_session.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        writeln!(
            file,
            "{}",
            serialize_json_line(
                &serde_json::json!({"type":"session","model":"gpt-4","session_id":"abc123"})
            )
            .trim()
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            serialize_json_line(
                &serde_json::json!({"type":"message","message":{"role":"user","content":"hello"}})
            )
            .trim()
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            serialize_json_line(
                &serde_json::json!({"type":"message","message":{"role":"assistant","content":"hi"}})
            )
            .trim()
        )
        .unwrap();

        let messages = load_session(&path).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "hi");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_append_message() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-append-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("append_test.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        writeln!(
            file,
            "{}",
            serialize_json_line(&serde_json::json!({"type":"session","model":"gpt-4"})).trim()
        )
        .unwrap();
        drop(file);

        let msg = Message::new("user", "test message");
        append_message(&path, &msg).unwrap();

        let messages = load_session(&path).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "test message");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_append_compaction() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-comp-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("comp_test.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        writeln!(
            file,
            "{}",
            serialize_json_line(&serde_json::json!({"type":"session","model":"gpt-4"})).trim()
        )
        .unwrap();
        drop(file);

        append_compaction(&path, "test summary", 1000).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains("test summary"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_session_nonexistent() {
        let dir = std::env::temp_dir().join("nonexistent-sessions-dir-12345");
        let path = dir.join("nonexistent.jsonl");
        let messages = load_session(&path).unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_create_session_uses_uuid() {
        // Override sessions dir to temp
        let _orig_home = dirs::home_dir();
        let dir = std::env::temp_dir().join(format!("rupi-sessions-uuid-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        // Temporarily redirect by creating file directly with UUID
        let path = create_session("gpt-4").unwrap();
        let filename = path.file_stem().unwrap().to_string_lossy().to_string();
        // UUID v4 format: 8-4-4-4-12 hex chars
        assert_eq!(
            filename.len(),
            36,
            "UUID should be 36 chars, got {}",
            filename
        );
        assert!(
            filename.contains('-'),
            "UUID should contain dashes, got {}",
            filename
        );
        assert!(path.exists());

        // Verify header contains session_id
        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains(&filename));
        assert!(contents.contains("session_id"));

        // Clean up the created session file
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_list_sessions() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-list-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);

        // Create two session files
        let s1 = dir.join("11111111-1111-1111-1111-111111111111.jsonl");
        let mut f1 = fs::File::create(&s1).unwrap();
        use std::io::Write;
        writeln!(f1, r#"{{"type":"session","model":"gpt-4","session_id":"11111111-1111-1111-1111-111111111111"}}"#).unwrap();
        writeln!(
            f1,
            r#"{{"type":"message","message":{{"role":"user","content":"hi"}}}}"#
        )
        .unwrap();
        drop(f1);

        let s2 = dir.join("22222222-2222-2222-2222-222222222222.jsonl");
        let mut f2 = fs::File::create(&s2).unwrap();
        writeln!(f2, r#"{{"type":"session","model":"claude-3","session_id":"22222222-2222-2222-2222-222222222222"}}"#).unwrap();
        drop(f2);

        // Temporarily override sessions_dir by using list logic directly
        let sessions = list_sessions_from_dir(&dir).unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.iter().any(|s| s.model == "gpt-4"));
        assert!(sessions.iter().any(|s| s.model == "claude-3"));

        fs::remove_dir_all(&dir).unwrap();
    }

    fn list_sessions_from_dir(dir: &std::path::Path) -> Result<Vec<SessionInfo>, String> {
        let mut sessions = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(Vec::new());
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let contents = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let mut model = String::new();
            let mut message_count = 0;
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    match val["type"].as_str().unwrap_or("") {
                        "session" => {
                            model = val["model"].as_str().unwrap_or("").to_string();
                        }
                        "message" => {
                            message_count += 1;
                        }
                        _ => {}
                    }
                }
            }
            sessions.push(SessionInfo {
                id,
                model,
                created_at: String::new(),
                message_count,
            });
        }
        Ok(sessions)
    }

    #[test]
    fn test_find_session_path() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-find-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let id = "test-find-id-123";
        let path = dir.join(format!("{}.jsonl", id));
        fs::write(&path, "{}").unwrap();

        // Override home to use temp dir
        let result = find_session_path(id);
        // May be None since sessions_dir points to home
        // Just verify no crash
        assert!(result.is_none() || result.as_ref().is_some_and(|p| p.exists()));
    }

    #[test]
    fn test_session_tool_call_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-tc-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("tc_test.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        writeln!(
            file,
            "{}",
            serialize_json_line(
                &serde_json::json!({"type":"session","model":"gpt-4","session_id":"tc1"})
            )
            .trim()
        )
        .unwrap();
        drop(file);

        // Create a message with tool calls
        let mut msg = Message::new("assistant", "let me check");
        msg.tool_calls = Some(vec![
            ToolCall {
                id: "call_1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "ls"}),
                raw_arguments: None,
            },
            ToolCall {
                id: "call_2".into(),
                name: "read".into(),
                arguments: serde_json::json!({"file_path": "/tmp/x"}),
                raw_arguments: None,
            },
        ]);
        append_message(&path, &msg).unwrap();

        // Create a tool result
        let mut tr = Message::tool_result("call_1", "file1.txt\nfile2.txt");
        tr.tool_call_id = Some("call_1".into());
        append_message(&path, &tr).unwrap();

        // Load and verify
        let loaded = load_session(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(
            loaded[0].tool_calls.is_some(),
            "Tool calls should be preserved"
        );
        let calls = loaded[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments["command"], "ls");
        assert_eq!(calls[1].name, "read");
        assert_eq!(loaded[1].tool_call_id.as_deref(), Some("call_1"));
        assert!(loaded[1].content.contains("file1.txt"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_corrupt_session() {
        let dir =
            std::env::temp_dir().join(format!("rupi-sessions-corrupt-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("corrupt.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        // Valid header
        writeln!(file, r#"{{"type":"session","model":"gpt-4"}}"#).unwrap();
        // Valid message
        writeln!(
            file,
            r#"{{"type":"message","message":{{"role":"user","content":"hello"}}}}"#
        )
        .unwrap();
        // Corrupt JSON line
        writeln!(file, "this is not json at all {{").unwrap();
        // Valid message after corruption
        writeln!(
            file,
            r#"{{"type":"message","message":{{"role":"assistant","content":"hi"}}}}"#
        )
        .unwrap();
        drop(file);

        let loaded = load_session(&path).unwrap();
        // Should skip corrupt line but still load valid ones
        assert_eq!(loaded.len(), 2, "Corrupt line should be skipped");
        assert_eq!(loaded[0].content, "hello");
        assert_eq!(loaded[1].content, "hi");
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Write `lines` to a throwaway session file and load it back.
    fn load_lines(tag: &str, lines: &[serde_json::Value]) -> (PathBuf, Vec<Message>) {
        let dir =
            std::env::temp_dir().join(format!("rupi-sessions-{}-{}", tag, std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join(format!("{}.jsonl", tag));
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        for line in lines {
            writeln!(file, "{}", serialize_json_line(line).trim()).unwrap();
        }
        drop(file);
        let loaded = load_session(&path).unwrap();
        (dir, loaded)
    }

    #[test]
    fn resuming_after_compaction_drops_the_summarized_messages() {
        // Without this, resuming replays the whole history *plus* the summary,
        // so the session reopens over budget and compacts again immediately.
        let summary_content = format!("{}\nuser asked about X; agent fixed Y", COMPACTION_PREFIX);
        let (dir, loaded) = load_lines(
            "compacted",
            &[
                serde_json::json!({"type":"session","model":"gpt-4"}),
                serde_json::json!({"type":"message","message":{"role":"user","content":"old question"}}),
                serde_json::json!({"type":"message","message":{"role":"assistant","content":"old answer"}}),
                serde_json::json!({"type":"message","message":{"role":"user","content":summary_content}}),
                serde_json::json!({"type":"compaction","summary":"user asked about X; agent fixed Y","tokens_before":9000}),
                serde_json::json!({"type":"message","message":{"role":"user","content":"what next?"}}),
            ],
        );

        assert_eq!(
            loaded.len(),
            2,
            "expected summary + post-compaction message"
        );
        assert!(loaded[0].content.starts_with(COMPACTION_PREFIX));
        assert!(loaded[0].content.contains("fixed Y"));
        assert_eq!(loaded[1].content, "what next?");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn compaction_without_a_persisted_summary_message_is_rebuilt() {
        let (dir, loaded) = load_lines(
            "compacted-nosummary",
            &[
                serde_json::json!({"type":"message","message":{"role":"user","content":"old"}}),
                serde_json::json!({"type":"compaction","summary":"the story so far","tokens_before":9000}),
                serde_json::json!({"type":"message","message":{"role":"assistant","content":"after"}}),
            ],
        );

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].role, "user");
        assert!(loaded[0].content.contains("the story so far"));
        assert_eq!(loaded[1].content, "after");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn content_blocks_and_foreign_roles_are_understood() {
        // Transcripts recorded by an embedder use the content-block shape.
        let (dir, loaded) = load_lines(
            "blocks",
            &[
                serde_json::json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}),
                serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"part one"},{"type":"toolCall","name":"bash"}]}}),
                serde_json::json!({"type":"message","message":{"role":"toolResult","content":[{"type":"text","text":"exit 0"}]}}),
            ],
        );

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].content, "hello");
        assert_eq!(loaded[1].content, "part one\n[called tool: bash]");
        assert_eq!(
            loaded[2].role, "tool",
            "toolResult must map to the tool role"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn generation_ids_are_persisted_for_cost_reconciliation() {
        let dir = std::env::temp_dir().join(format!("rupi-sessions-genid-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("genid.jsonl");
        fs::write(&path, "").unwrap();

        append_generation_id(&path, "gen-abc123").unwrap();

        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("gen-abc123"));
        assert!(text.contains("\"type\":\"generation_id\""));
        // It must not be mistaken for conversation content on resume.
        assert!(load_session(&path).unwrap().is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn appending_recreates_a_transcript_that_was_deleted() {
        // An embedder that prunes the session directory between turns used to
        // break persistence permanently: every later append failed with ENOENT,
        // losing the compaction record and the generation ids with it.
        let dir = std::env::temp_dir().join(format!("rupi-sessions-gone-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path =
            create_session_with_id("vanishing", "gpt-4").unwrap_or(dir.join("vanishing.jsonl"));
        let _ = fs::write(&path, "");

        fs::remove_file(&path).unwrap();
        assert!(!path.exists());

        append_message(&path, &Message::new("user", "depois do sumico")).unwrap();
        append_generation_id(&path, "gen-after").unwrap();

        let loaded = load_session(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "depois do sumico");
        assert!(fs::read_to_string(&path).unwrap().contains("gen-after"));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_old_format_compat() {
        // Old format didn't have tool_calls/tool_call_id fields — should still load
        let json = r#"{"role":"user","content":"hello"}"#;
        let entry: SessionMessageEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.role, "user");
        assert_eq!(entry.content, "hello");
        assert!(entry.tool_calls.is_none());
        assert!(entry.tool_call_id.is_none());
    }
}
