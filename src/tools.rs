use std::path::PathBuf;
use serde_json::Value;

/// A tool definition sent to the API.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
}

/// A tool call from the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Get all available tool definitions.
pub fn all_tools() -> Vec<ToolDef> {
    vec![
        bash_tool(),
        read_tool(),
        write_tool(),
        edit_tool(),
        grep_tool(),
        find_tool(),
        ls_tool(),
    ]
}

fn bash_tool() -> ToolDef {
    ToolDef {
        name: "bash",
        description: "Execute a bash command. Use it to run shell commands, scripts, compilers, curl, git, or anything else in the terminal. The command runs in the current working directory.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The bash command to execute" },
                "description": { "type": "string", "description": "A short description (for logging)" },
                "timeout": { "type": "number", "description": "Timeout in seconds (max 120)", "default": 30 }
            },
            "required": ["command"]
        }),
    }
}

fn read_tool() -> ToolDef {
    ToolDef {
        name: "read",
        description: "Read the contents of a file. Supports optional offset and limit to read specific portions. Use this to examine source code, configuration files, logs, and other text files.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The absolute or relative path to the file to read" },
                "offset": { "type": "integer", "description": "Line number to start reading from (1-indexed)", "default": 1 },
                "limit": { "type": "integer", "description": "Maximum number of lines to read", "default": 2000 }
            },
            "required": ["file_path"]
        }),
    }
}

fn write_tool() -> ToolDef {
    ToolDef {
        name: "write",
        description: "Create a NEW file with the given content. REFUSES if the file already exists — use edit to modify existing files. Creates parent directories automatically.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The absolute file path for the new file" },
                "content": { "type": "string", "description": "The full content to write" }
            },
            "required": ["file_path", "content"]
        }),
    }
}

fn edit_tool() -> ToolDef {
    ToolDef {
        name: "edit",
        description: "Replace exact text in a file. Uses exact string matching (not regex). Each old_text must be found exactly once in the file. Supports batch edits via the edits array. Prefer this over write for any change to an existing file.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The absolute or relative path to the file to edit" },
                "old_text": { "type": "string", "description": "(Deprecated) Use edits array instead. The exact text to find and replace (must match exactly once)" },
                "new_text": { "type": "string", "description": "(Deprecated) Use edits array instead. The replacement text" },
                "edits": {
                    "type": "array",
                    "description": "Array of edits to apply. Each edit's old_text is matched against the ORIGINAL file content (not after other edits). Edits must not overlap.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": { "type": "string", "description": "Exact text to find (must match exactly once in the file)" },
                            "new_text": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["old_text", "new_text"]
                    }
                }
            }
        }),
    }
}

fn grep_tool() -> ToolDef {
    ToolDef {
        name: "grep",
        description: "Search file contents using a regular expression. Supports case-insensitive search and glob file patterns. Use this to find where functions are defined, search for specific patterns, or locate code references.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "The regex pattern to search for" },
                "include": { "type": "string", "description": "Glob pattern for files to include (e.g. '*.rs', '*.{ts,js}')", "default": "" },
                "path": { "type": "string", "description": "Directory to search in (default: current directory)", "default": "." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive search", "default": false }
            },
            "required": ["pattern"]
        }),
    }
}

fn find_tool() -> ToolDef {
    ToolDef {
        name: "find",
        description: "Find files matching a glob pattern. Use this to locate files by name pattern, discover project structure, or find configuration files.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern to match (e.g. '**/*.rs', 'src/**/*.ts', '*.json')" },
                "path": { "type": "string", "description": "Directory to search in (default: current directory)", "default": "." }
            },
            "required": ["pattern"]
        }),
    }
}

fn ls_tool() -> ToolDef {
    ToolDef {
        name: "ls",
        description: "List files and directories in a given path. Use this to explore project structure, see what files are in a directory, or verify file existence.",
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path to list (default: current directory)", "default": "." }
            },
            "required": []
        }),
    }
}

/// Serialize tool definitions to the OpenAI API format.
pub fn serialize_tools(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            })
        })
        .collect()
}

/// Execute a tool call and return the result.
pub fn execute_tool(tool_call: &ToolCall) -> String {
    match tool_call.name.as_str() {
        "bash" => execute_bash(&tool_call.arguments),
        "read" => execute_read(&tool_call.arguments),
        "write" => execute_write(&tool_call.arguments),
        "edit" => execute_edit(&tool_call.arguments),
        "grep" => execute_grep(&tool_call.arguments),
        "find" => execute_find(&tool_call.arguments),
        "ls" => execute_ls(&tool_call.arguments),
        _ => format!("Unknown tool: {}", tool_call.name),
    }
}

fn get_arg<'a>(args: &'a Value, name: &str) -> Option<&'a str> {
    args.get(name).and_then(|v| v.as_str())
}

fn get_arg_i64(args: &Value, name: &str) -> Option<i64> {
    args.get(name).and_then(|v| v.as_i64())
}

fn get_arg_bool(args: &Value, name: &str, default: bool) -> bool {
    args.get(name).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn resolve_path(path_str: &str) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    }
}

// ---- bash ----
fn execute_bash(args: &Value) -> String {
    let command = match get_arg(args, "command") {
        Some(cmd) => cmd,
        None => return "Error: missing 'command' argument".to_string(),
    };
    let timeout_secs: u64 = args.get("timeout").and_then(|t| t.as_u64()).unwrap_or(30).min(120);

    match std::panic::catch_unwind(|| {
        let mut child = match std::process::Command::new("bash")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return format!("Failed to spawn bash: {}", e),
        };

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(timeout_secs);

        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let output = child.wait_with_output().unwrap();
                    let mut result = String::new();
                    if !output.stdout.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&output.stdout));
                    }
                    if !output.stderr.is_empty() {
                        if !result.is_empty() { result.push('\n'); }
                        result.push_str(&String::from_utf8_lossy(&output.stderr));
                    }
                    if !status.success() {
                        let ec = status.code().unwrap_or(-1);
                        result.push_str(&format!("\n[exit code: {}]", ec));
                    }
                    if result.trim().is_empty() {
                        result = format!("[command completed with exit code {}]", status.code().unwrap_or(-1));
                    }
                    return result;
                }
                Ok(None) => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        return format!("[timed out after {}s]", timeout_secs);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => return format!("Error waiting for bash: {}", e),
            }
        }
    }) {
        Ok(r) => r,
        Err(_) => "Error: bash execution panicked".to_string(),
    }
}

// ---- read ----
fn execute_read(args: &Value) -> String {
    let file_path = match get_arg(args, "file_path") {
        Some(p) => p,
        None => return "Error: missing 'file_path' argument".to_string(),
    };
    let offset = get_arg_i64(args, "offset").unwrap_or(1).max(1) as usize;
    let limit = get_arg_i64(args, "limit").unwrap_or(2000).max(1) as usize;

    let path = resolve_path(file_path);
    if !path.exists() {
        return format!("Error: file not found: {}", path.display());
    }
    if !path.is_file() {
        return format!("Error: not a file: {}", path.display());
    }

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let start = (offset - 1).min(lines.len());
            let end = (start + limit).min(lines.len());
            let selected = &lines[start..end];
            let mut result = String::new();
            for (i, line) in selected.iter().enumerate() {
                result.push_str(&format!("{:6}  {}\n", start + i + 1, line));
            }
            if end < lines.len() {
                result.push_str(&format!("... {} more lines\n", lines.len() - end));
            }
            if result.is_empty() {
                result = "[empty file]".to_string();
            }
            result
        }
        Err(e) => format!("Error reading file: {}", e),
    }
}

// ---- write (with guard) ----
fn execute_write(args: &Value) -> String {
    let file_path = match get_arg(args, "file_path") {
        Some(p) => p,
        None => return "Error: missing 'file_path' argument".to_string(),
    };
    let content = match get_arg(args, "content") {
        Some(c) => c,
        None => return "Error: missing 'content' argument".to_string(),
    };

    let path = resolve_path(file_path);

    // Write guard: refuse if file already exists
    if path.exists() {
        return format!(
            "Error: Write refused — {} already exists.\n\
             \n\
             Write is only for creating NEW files. To change an existing file, use Edit:\n\
             {{\"name\":\"edit\",\"input\":{{\"file_path\":\"{}\",\"edits\":[{{\"old_text\":\"<exact text>\",\"new_text\":\"<replacement>\"}}]}}}}\n\
             \n\
             If you do not already know the file's current content, Read it first to get the exact text. \
             Do NOT retry Write — it will be refused again.",
            path.display(),
            path.display(),
        );
    }

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return format!("Error creating parent directory: {}", e);
            }
        }
    }

    match std::fs::write(&path, content) {
        Ok(_) => {
            let line_count = content.lines().count();
            format!("Successfully wrote {} lines to {}", line_count, path.display())
        }
        Err(e) => format!("Error writing file: {}", e),
    }
}

// ---- edit (multi-edit support) ----
fn execute_edit(args: &Value) -> String {
    let file_path = match get_arg(args, "file_path") {
        Some(p) => p,
        None => return "Error: missing 'file_path' argument".to_string(),
    };

    let path = resolve_path(file_path);
    if !path.exists() {
        return format!("Error: file not found: {}", path.display());
    }

    // Collect edits from either the singular old_text/new_text or the edits array
    let edits: Vec<(String, String)> = if let Some(edits_val) = args.get("edits").and_then(|v| v.as_array()) {
        if edits_val.is_empty() {
            return "Error: edits array is empty".to_string();
        }
        edits_val.iter().filter_map(|e| {
            let old = e.get("old_text")?.as_str()?.to_string();
            let new = e.get("new_text")?.as_str()?.to_string();
            Some((old, new))
        }).collect()
    } else {
        let old_text = match get_arg(args, "old_text") {
            Some(t) => t,
            None => return "Error: missing 'old_text' argument (or provide 'edits' array)".to_string(),
        };
        let new_text = match get_arg(args, "new_text") {
            Some(t) => t,
            None => return "Error: missing 'new_text' argument (or provide 'edits' array)".to_string(),
        };
        vec![(old_text.to_string(), new_text.to_string())]
    };

    if edits.is_empty() {
        return "Error: no valid edits provided".to_string();
    }

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            // Check all old_text exist exactly once in the ORIGINAL content
            let mut results: Vec<String> = Vec::new();
            let mut has_error = false;

            for (old_text, _new_text) in &edits {
                let count = content.matches(old_text).count();
                if count == 0 {
                    results.push(format!("old_text not found: {:?}", old_text));
                    has_error = true;
                } else if count > 1 {
                    results.push(format!("old_text found {} times: {:?}", count, old_text));
                    has_error = true;
                }
            }

            if has_error {
                let mut msg = "Edit failed:\n".to_string();
                for r in &results {
                    msg.push_str(&format!("  - {}\n", r));
                }
                msg.push_str("\nRecovery: Read the file to get the exact current content, then retry with the exact text.");
                return msg;
            }

            // Check for overlapping edits by finding positions in original content
            let mut positions: Vec<(usize, usize, &str, &str)> = Vec::new(); // (start, end, old_text, new_text)
            for (old_text, new_text) in &edits {
                if let Some(pos) = content.find(old_text) {
                    positions.push((pos, pos + old_text.len(), old_text, new_text));
                }
            }

            // Sort by position (ascending)
            positions.sort_by_key(|&(start, _, _, _)| start);

            // Check for overlaps
            for i in 1..positions.len() {
                if positions[i].0 < positions[i - 1].1 {
                    return format!(
                        "Error: edits overlap. First edit ends at position {} but next edit starts at {}. \
                         Edits must not overlap — each old_text is matched against the original file content.",
                        positions[i - 1].1, positions[i].0
                    );
                }
            }

            // Apply edits in reverse position order to preserve positions
            let mut new_content = content.clone();
            for &(start, end, _old, new) in positions.iter().rev() {
                new_content.replace_range(start..end, new);
            }

            match std::fs::write(&path, &new_content) {
                Ok(_) => {
                    if positions.len() == 1 {
                        format!("Successfully applied edit to {}", path.display())
                    } else {
                        format!("Successfully applied {} edits to {}", positions.len(), path.display())
                    }
                }
                Err(e) => format!("Error writing file: {}", e),
            }
        }
        Err(e) => format!("Error reading file: {}", e),
    }
}

// ---- grep ----
fn execute_grep(args: &Value) -> String {
    let pattern = match get_arg(args, "pattern") {
        Some(p) => p,
        None => return "Error: missing 'pattern' argument".to_string(),
    };
    let include = get_arg(args, "include").unwrap_or("");
    let search_path = get_arg(args, "path").unwrap_or(".");
    let ignore_case = get_arg_bool(args, "ignore_case", false);

    // Build rg (ripgrep) command
    let mut cmd = std::process::Command::new("rg");
    cmd.arg("--line-number").arg("--color").arg("never");
    if !include.is_empty() {
        cmd.arg("--glob").arg(include);
    }
    if ignore_case {
        cmd.arg("-i");
    }
    cmd.arg(pattern);
    cmd.arg(search_path);

    match cmd.output() {
        Ok(output) => {
            let mut result = String::new();
            if !output.stdout.is_empty() {
                result.push_str(&String::from_utf8_lossy(&output.stdout));
            }
            if !output.stderr.is_empty() && result.is_empty() {
                result.push_str(&String::from_utf8_lossy(&output.stderr));
            }
            if result.is_empty() {
                result = "No matches found.".to_string();
            }
            // Truncate long output
            if result.len() > 10000 {
                result.truncate(10000);
                result.push_str("\n... [output truncated]");
            }
            result
        }
        Err(e) => {
            // Fallback to grep if rg is not available
            let mut cmd = std::process::Command::new("grep");
            cmd.arg("-rn");
            if ignore_case { cmd.arg("-i"); }
            if !include.is_empty() {
                cmd.arg("--include").arg(include);
            }
            cmd.arg(pattern);
            cmd.arg(search_path);

            match cmd.output() {
                Ok(o) => {
                    let mut r = String::new();
                    if !o.stdout.is_empty() { r.push_str(&String::from_utf8_lossy(&o.stdout)); }
                    if r.is_empty() { r = "No matches found.".to_string(); }
                    if r.len() > 10000 { r.truncate(10000); r.push_str("\n... [output truncated]"); }
                    r
                }
                Err(e2) => format!("Error running grep: {} (rg also unavailable: {})", e2, e),
            }
        }
    }
}

// ---- find ----
fn execute_find(args: &Value) -> String {
    let pattern = match get_arg(args, "pattern") {
        Some(p) => p,
        None => return "Error: missing 'pattern' argument".to_string(),
    };
    let search_path = get_arg(args, "path").unwrap_or(".");

    // Try fd first, fallback to find
    let mut cmd = std::process::Command::new("fd");
    cmd.arg("--glob").arg(pattern).arg(search_path);

    match cmd.output() {
        Ok(output) => {
            let mut result = String::new();
            if !output.stdout.is_empty() {
                result.push_str(&String::from_utf8_lossy(&output.stdout));
            }
            if result.is_empty() {
                result = "No files found.".to_string();
            }
            if result.len() > 5000 {
                result.truncate(5000);
                result.push_str(&format!("\n... [{} results total, truncated]", result.matches('\n').count()));
            }
            result
        }
        Err(_) => {
            // Fallback to find
            let mut cmd = std::process::Command::new("find");
            cmd.arg(search_path).arg("-name").arg(pattern);

            match cmd.output() {
                Ok(o) => {
                    let mut r = String::new();
                    if !o.stdout.is_empty() { r.push_str(&String::from_utf8_lossy(&o.stdout)); }
                    if r.is_empty() { r = "No files found.".to_string(); }
                    if r.len() > 5000 { r.truncate(5000); r.push_str("\n... [truncated]"); }
                    r
                }
                Err(e) => format!("Error running find: {}", e),
            }
        }
    }
}

// ---- ls ----
fn execute_ls(args: &Value) -> String {
    let dir_path = get_arg(args, "path").unwrap_or(".");
    let path = resolve_path(dir_path);

    if !path.exists() {
        return format!("Error: path not found: {}", path.display());
    }
    if !path.is_dir() {
        return format!("Error: not a directory: {}", path.display());
    }

    match std::fs::read_dir(&path) {
        Ok(entries) => {
            let mut files = Vec::new();
            let mut dirs = Vec::new();
            let mut total = 0;

            for entry in entries.flatten() {
                total += 1;
                if total > 200 {
                    break;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.') {
                    continue;
                }
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    dirs.push(format!("{}/", name));
                } else {
                    files.push(name);
                }
            }

            files.sort();
            dirs.sort();

            let mut result = String::new();
            for d in &dirs {
                result.push_str(&format!("  {}\n", d));
            }
            for f in &files {
                result.push_str(&format!("  {}\n", f));
            }
            if result.is_empty() {
                result = "(empty directory)".to_string();
            }
            result
        }
        Err(e) => format!("Error reading directory: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_bash_echo() {
        let args = serde_json::json!({"command": "echo hello"});
        let result = execute_bash(&args);
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn test_read_file() {
        let dir = std::env::temp_dir().join("rupi-tools-test-read-file".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        fs::write(&path, "line1\nline2\nline3\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy()});
        let result = execute_read(&args);
        assert!(result.contains("line1"));
        assert!(result.contains("line2"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_file_with_offset() {
        let dir = std::env::temp_dir().join("rupi-tools-test-read-offset".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        fs::write(&path, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "offset": 3, "limit": 2});
        let result = execute_read(&args);
        assert!(result.contains("line3"));
        assert!(result.contains("line4"));
        assert!(!result.contains("line1"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_creates_new_file() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-file".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("new_file.txt");

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "content": "hello\nworld"});
        let result = execute_write(&args);
        assert!(result.contains("Successfully wrote"));
        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello\nworld");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_refuses_on_existing_file() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-refuse".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("existing.txt");
        fs::write(&path, "original content").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "content": "new content"});
        let result = execute_write(&args);
        assert!(result.contains("Write refused"));
        // Content should be unchanged
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "original content");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_guard_returns_edit_recipe() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-recipe".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("existing.txt");
        fs::write(&path, "content").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "content": "new"});
        let result = execute_write(&args);
        assert!(result.contains("use Edit"));
        assert!(result.contains("old_text"));
        assert!(result.contains("new_text"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_singular() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-file".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "old_text": "hello", "new_text": "goodbye"});
        let result = execute_edit(&args);
        assert!(result.contains("Successfully applied edit"));
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "goodbye world\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multi_edits() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-multi".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi.txt");
        fs::write(&path, "AAA line\nBBB line\nCCC line\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "edits": [
                {"old_text": "AAA", "new_text": "XXX"},
                {"old_text": "CCC", "new_text": "ZZZ"}
            ]
        });
        let result = execute_edit(&args);
        assert!(result.contains("2 edits"));
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "XXX line\nBBB line\nZZZ line\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_not_found() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-not-found".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "old_text": "nonexistent", "new_text": "foo"});
        let result = execute_edit(&args);
        assert!(result.contains("not found"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multiple_matches_singular() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-multi-singular".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edit.txt");
        fs::write(&path, "hello hello\n").unwrap();

        let args = serde_json::json!({"file_path": path.to_string_lossy(), "old_text": "hello", "new_text": "foo"});
        let result = execute_edit(&args);
        assert!(result.contains("2 times"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multi_overlap_detected() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-overlap".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overlap.txt");
        fs::write(&path, "hello world foo\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "edits": [
                {"old_text": "hello world", "new_text": "goodbye"},
                {"old_text": "world foo", "new_text": "moon"}
            ]
        });
        let result = execute_edit(&args);
        assert!(result.contains("overlap"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_edit_multi_one_fails_all_fail() {
        let dir = std::env::temp_dir().join("rupi-tools-test-edit-partial".to_string());
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial.txt");
        fs::write(&path, "AAA line\nBBB line\n").unwrap();

        let args = serde_json::json!({
            "file_path": path.to_string_lossy(),
            "edits": [
                {"old_text": "AAA", "new_text": "XXX"},
                {"old_text": "NONEXISTENT", "new_text": "YYY"}
            ]
        });
        let result = execute_edit(&args);
        assert!(result.contains("Edit failed"));
        // File should be unchanged
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "AAA line\nBBB line\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_grep() {
        let dir = std::env::temp_dir().join("rupi-tools-test-grep".to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("test.txt"), "hello world\nfoo bar\n").unwrap();

        let args = serde_json::json!({"pattern": "hello", "path": dir.to_string_lossy()});
        let result = execute_grep(&args);
        assert!(result.contains("hello"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_ls() {
        let dir = std::env::temp_dir().join("rupi-tools-test-ls".to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(dir.join("b.txt"), "b").unwrap();

        let args = serde_json::json!({"path": dir.to_string_lossy()});
        let result = execute_ls(&args);
        assert!(result.contains("a.txt"));
        assert!(result.contains("b.txt"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_serialize_tools() {
        let tools = all_tools();
        let serialized = serialize_tools(&tools);
        assert_eq!(serialized.len(), 7);
        assert_eq!(serialized[0]["function"]["name"], "bash");
        assert_eq!(serialized[1]["function"]["name"], "read");
        assert_eq!(serialized[2]["function"]["name"], "write");
        assert_eq!(serialized[3]["function"]["name"], "edit");
        assert_eq!(serialized[4]["function"]["name"], "grep");
        assert_eq!(serialized[5]["function"]["name"], "find");
        assert_eq!(serialized[6]["function"]["name"], "ls");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let tc = ToolCall {
            id: "call_1".into(), name: "nonexistent".into(),
            arguments: serde_json::json!({}),
        };
        let result = execute_tool(&tc);
        assert!(result.contains("Unknown tool"));
    }

    #[test]
    fn test_read_file_not_found() {
        let args = serde_json::json!({"file_path": "/nonexistent/path/file.txt"});
        let result = execute_read(&args);
        assert!(result.contains("not found"));
    }

    #[test]
    fn test_read_missing_path() {
        let args = serde_json::json!({});
        let result = execute_read(&args);
        assert!(result.contains("missing"));
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-parent".to_string());
        let nested = dir.join("nested").join("deep").join("file.txt");
        let args = serde_json::json!({"file_path": nested.to_string_lossy(), "content": "test"});
        let result = execute_write(&args);
        assert!(result.contains("Successfully wrote"));
        assert!(nested.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_creates_new_even_if_parent_doesnt_exist() {
        let dir = std::env::temp_dir().join("rupi-tools-test-write-deep".to_string());
        let nested = dir.join("a").join("b").join("c").join("f.txt");
        let args = serde_json::json!({"file_path": nested.to_string_lossy(), "content": "hi"});
        let result = execute_write(&args);
        assert!(result.contains("Successfully wrote"));
        assert!(nested.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_tool_def_names() {
        for tool in all_tools() {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty());
            assert!(tool.parameters.get("properties").is_some() || tool.parameters.get("type").is_some());
        }
    }
}
