use std::path::PathBuf;

use crate::agent::session::Message;
use crate::rpc::jsonl::serialize_json_line;

const SESSIONS_DIR: &str = "rupi_sessions";

/// Get the sessions directory path (~/.config/rupi_sessions/).
pub fn sessions_dir() -> Option<PathBuf> {
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
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionMessageEntry {
    pub role: String,
    pub content: String,
}

/// Create a new session file. Returns the file path.
pub fn create_session(model: &str) -> Result<PathBuf, String> {
    let dir = ensure_sessions_dir()?;
    let id = format!("{}", chrono::Utc::now().format("%Y%m%d_%H%M%S_%6f"));
    let path = dir.join(format!("{}.jsonl", id));

    // Write session header
    let header = SessionEntry {
        entry_type: "session",
        model: Some(model),
        message: None,
        summary: None,
        tokens_before: None,
    };
    let mut file = std::fs::File::create(&path).map_err(|e| format!("Cannot create session file: {}", e))?;
    use std::io::Write;
    writeln!(file, "{}", serialize_json_line(&header).trim())
        .map_err(|e| format!("Cannot write session header: {}", e))?;
    Ok(path)
}

/// Append a message entry to a session file.
pub fn append_message(path: &PathBuf, msg: &Message) -> Result<(), String> {
    let entry = SessionEntry {
        entry_type: "message",
        model: None,
        message: Some(&SessionMessageEntry {
            role: msg.role.clone(),
            content: msg.content.clone(),
        }),
        summary: None,
        tokens_before: None,
    };
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|e| format!("Cannot open session file: {}", e))?;
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
    };
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|e| format!("Cannot open session file: {}", e))?;
    use std::io::Write;
    writeln!(file, "{}", serialize_json_line(&entry).trim())
        .map_err(|e| format!("Cannot append compaction to session file: {}", e))?;
    Ok(())
}

/// Load all entries from a session file.
pub fn load_session(path: &PathBuf) -> Result<Vec<Message>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read session file: {}", e))?;
    let mut messages = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            let entry_type = val["type"].as_str().unwrap_or("");
            if entry_type == "message" {
                if let Some(msg_val) = val.get("message") {
                    let role = msg_val["role"].as_str().unwrap_or("user");
                    let content = msg_val["content"].as_str().unwrap_or("");
                    messages.push(Message::new(role, content));
                }
            }
            // Skip session headers and compaction entries for message loading
        }
    }
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // Temporarily override sessions_dir by creating file directly
        let path = dir.join("test_session.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        writeln!(file, "{}", serialize_json_line(&serde_json::json!({"type":"session","model":"gpt-4"})).trim()).unwrap();
        writeln!(file, "{}", serialize_json_line(&serde_json::json!({"type":"message","message":{"role":"user","content":"hello"}})).trim()).unwrap();
        writeln!(file, "{}", serialize_json_line(&serde_json::json!({"type":"message","message":{"role":"assistant","content":"hi"}})).trim()).unwrap();

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

        // Create session header
        let mut file = fs::File::create(&path).unwrap();
        use std::io::Write;
        writeln!(file, "{}", serialize_json_line(&serde_json::json!({"type":"session","model":"gpt-4"})).trim()).unwrap();
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
        writeln!(file, "{}", serialize_json_line(&serde_json::json!({"type":"session","model":"gpt-4"})).trim()).unwrap();
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
}
