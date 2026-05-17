use std::collections::HashMap;

/// Apply RTK-style output compression to a bash command's output.
/// Detects the command type from the command string and applies
/// the appropriate post-hoc filter. If no filter matches, returns
/// the output with generic compression applied.
pub fn filter_output(command: &str, output: &str) -> String {
    let trimmed = command.trim();

    // Try command-specific filters first (most specific wins)
    let filtered = if trimmed.starts_with("git ") {
        filter_git(trimmed, output)
    } else if trimmed.starts_with("cargo ") {
        filter_cargo(trimmed, output)
    } else if trimmed.starts_with("ls ") || trimmed == "ls" {
        filter_ls(output)
    } else if trimmed.starts_with("find ") || trimmed.starts_with("fd ") {
        filter_find(output)
    } else {
        None
    };

    match filtered {
        Some(f) => f,
        None => filter_generic(output),
    }
}

// ── Git filters ──────────────────────────────────────────────────────────

fn filter_git(cmd: &str, output: &str) -> Option<String> {
    let rest = cmd.trim_start_matches("git ").trim();
    if rest.is_empty() {
        return None;
    }

    // git status
    if rest == "status" || rest.starts_with("status ") {
        return Some(filter_git_status(output));
    }

    // git diff
    if rest == "diff" || rest.starts_with("diff ") {
        return Some(filter_git_diff(output));
    }

    // git log
    if rest == "log" || rest.starts_with("log ") {
        return Some(filter_git_log(output));
    }

    // git add
    if rest == "add" || rest.starts_with("add ") {
        return Some(filter_git_add(output));
    }

    // git commit
    if rest == "commit" || rest.starts_with("commit ") {
        return Some(filter_git_commit(output));
    }

    // git push / pull
    if rest.starts_with("push") || rest.starts_with("pull") {
        return Some(filter_git_push_pull(output));
    }

    // git branch
    if rest == "branch" || rest.starts_with("branch ") {
        return Some(filter_git_branch(output));
    }

    None
}

fn filter_git_status(output: &str) -> String {
    let mut result = String::new();
    let lines: Vec<&str> = output.lines().collect();

    // Track if we're in a rebase/merge state
    let mut state_line = String::new();
    let mut changes: Vec<String> = Vec::new();

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip hints and suggestions
        if trimmed.starts_with("(use \"")
            || trimmed.starts_with("(use `")
            || trimmed.starts_with("(fix conflicts")
            || trimmed.contains("no changes added to commit")
            || trimmed == "nothing added to commit but untracked files present (use \"git add\" to track)"
            || trimmed == "nothing to commit, working tree clean"
            || trimmed.starts_with("Changes not staged for commit:")
            || trimmed.starts_with("Changes to be committed:")
            || trimmed.starts_with("Untracked files:")
        {
            continue;
        }
        // Rebase/merge state
        if trimmed.contains("You have unmerged paths")
            || trimmed.contains("All conflicts fixed")
            || trimmed.starts_with("You are currently rebasing")
            || trimmed.starts_with("You are in a rebase")
            || trimmed.starts_with("You have unstaged changes")
        {
            state_line = trimmed.to_string();
            continue;
        }
        // File status lines (look for "modified:" / "new file:" / "deleted:" patterns)
        if trimmed.contains("modified:") || trimmed.contains("new file:") || trimmed.contains("deleted:") || trimmed.contains("renamed:") {
            // Extract file path after the colon+space
            if let Some(pos) = trimmed.rfind(':') {
                let file = trimmed[pos + 1..].trim().to_string();
                if !file.is_empty() {
                    changes.push(if trimmed.starts_with("modified:") || line.trim().starts_with("\tmodified") {
                        format!(" M {}", file)
                    } else if trimmed.starts_with("new file:") || line.trim().starts_with("\tnew file") {
                        format!(" A {}", file)
                    } else if trimmed.starts_with("deleted:") || line.trim().starts_with("\tdeleted") {
                        format!(" D {}", file)
                    } else {
                        format!("  {}", file)
                    });
                }
            }
            continue;
        }
        // Untracked files
        if trimmed.starts_with("\t") || trimmed.starts_with("        ") {
            let file = trimmed.trim().to_string();
            if !file.is_empty() && !changes.iter().any(|c| c.contains(&file)) {
                changes.push(format!("?? {}", file));
            }
            continue;
        }
        // Branch line
        if trimmed.starts_with("On branch ") || trimmed.starts_with("HEAD detached") {
            result.push_str(trimmed);
            result.push('\n');
            continue;
        }
        // Modified/Created/Deleted header counts
        if trimmed.contains("file changed") || trimmed.contains("files changed")
            || trimmed.contains("insertion") || trimmed.contains("deletion")
        {
            continue;
        }
    }

    if !state_line.is_empty() {
        result.push_str(&format!("[!] {}\n", state_line));
    }

    // If clean, say so concisely
    if changes.is_empty() {
        if result.is_empty() || result.trim() == "" {
            result = "clean — nothing to commit, working tree clean\n".to_string();
        } else if result.ends_with('\n') {
            result.push_str("clean — nothing to commit\n");
        }
        return result;
    }

    // Group changes by type
    #[derive(Clone)]
    struct Change {
        status: char,
        file: String,
    }
    let mut staged: Vec<Change> = Vec::new();
    let mut unstaged: Vec<Change> = Vec::new();
    let mut untracked: Vec<String> = Vec::new();

    for ch in &changes {
        let bytes = ch.as_bytes();
        if bytes.len() < 2 {
            continue;
        }
        let idx = if bytes[0] == b' ' { 1 } else { 0 };
        if bytes.len() <= idx + 1 {
            continue;
        }
        let status = bytes[idx] as char;
        let file = ch[idx + 1..].trim().to_string();
        match status {
            '?' => untracked.push(file),
            'M' | 'A' | 'D' | 'R' => staged.push(Change { status, file }),
            ' ' | 'm' | 'a' | 'd' => unstaged.push(Change { status: bytes[0] as char, file }),
            _ => {}
        }
    }

    if !staged.is_empty() {
        result.push_str(&format!("staged ({}):\n", staged.len()));
        for c in staged.iter().take(10) {
            result.push_str(&format!("  {} {}\n", c.status, c.file));
        }
        if staged.len() > 10 {
            result.push_str(&format!("  ... +{} more\n", staged.len() - 10));
        }
    }
    if !unstaged.is_empty() {
        result.push_str(&format!("unstaged ({}):\n", unstaged.len()));
        for c in unstaged.iter().take(10) {
            result.push_str(&format!("  {} {}\n", c.status, c.file));
        }
        if unstaged.len() > 10 {
            result.push_str(&format!("  ... +{} more\n", unstaged.len() - 10));
        }
    }
    if !untracked.is_empty() {
        result.push_str(&format!("untracked ({}):\n", untracked.len()));
        for f in untracked.iter().take(10) {
            result.push_str(&format!("  {} \n", f));
        }
        if untracked.len() > 10 {
            result.push_str(&format!("  ... +{} more\n", untracked.len() - 10));
        }
    }

    result
}

fn filter_git_diff(output: &str) -> String {
    let mut result = String::new();
    for line in output.lines() {
        // Skip diff metadata headers
        if line.starts_with("diff --git")
            || line.starts_with("index ")
            || line.starts_with("--- ")
            || line.starts_with("+++ ")
            || line.starts_with("new file mode")
            || line.starts_with("deleted file mode")
            || line.starts_with("old mode")
            || line.starts_with("new mode")
            || line.starts_with("similarity index")
            || line.starts_with("rename from")
            || line.starts_with("rename to")
            || line.starts_with("copy from")
            || line.starts_with("copy to")
        {
            continue;
        }
        // Keep hunk headers and +/- lines (strip context lines)
        if line.starts_with("@@ ") {
            result.push_str(line);
        } else if line.starts_with('+') && !line.starts_with("+++") {
            result.push_str(line);
        } else if line.starts_with('-') && !line.starts_with("---") {
            result.push_str(line);
        }
        // Context lines (starting with ' ') are stripped
        result.push('\n');
    }
    // Append recovery hint if output was large
    if result.len() < output.len() {
        result.push_str("[diff filtered: context lines stripped]");
    }
    result
}

fn filter_git_log(output: &str) -> String {
    let mut result = String::new();
    for line in output.lines() {
        // Keep lines that look like oneline format (hash + message)
        if line.len() > 8 && line.as_bytes().iter().take(7).all(|b| b.is_ascii_hexdigit()) {
            result.push_str(line);
            result.push('\n');
        }
    }
    if result.is_empty() {
        return output.to_string();
    }
    result
}

fn filter_git_add(output: &str) -> String {
    // git add outputs nothing on success — compress to "ok"
    if output.trim().is_empty() || output.contains("nothing to add") {
        return "ok".to_string();
    }
    // If there was an error, show it
    if output.contains("fatal:") || output.contains("error:") {
        return output.to_string();
    }
    "ok".to_string()
}

fn filter_git_commit(output: &str) -> String {
    // Extract short hash: "[main abc1234] message"
    for line in output.lines() {
        if line.starts_with('[') && line.contains("] ") {
            if let Some(end) = line.find(']') {
                let bracket = &line[1..end];
                if let Some(hash_start) = bracket.rfind(' ') {
                    let hash = bracket[hash_start + 1..].trim();
                    if hash.len() >= 7 {
                        return format!("ok {} [savings: ~95%]",
                            &hash[..7.min(hash.len())]);
                    }
                }
            }
        }
    }
    if output.contains("nothing to commit") {
        return "ok (nothing to commit)".to_string();
    }
    if output.contains("fatal:") || output.contains("error:") {
        return output.to_string();
    }
    output.to_string()
}

fn filter_git_push_pull(output: &str) -> String {
    // Extract "->" ref from push output
    for line in output.lines() {
        if line.contains("->") && line.contains("refs/") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(last) = parts.last() {
                return format!("ok {} [savings: ~90%]", last);
            }
        }
        if line.contains("Already up to date") || line.contains("Everything up-to-date") {
            return "ok (up-to-date)".to_string();
        }
        if line.contains("Already up-to-date") {
            return "ok (up-to-date)".to_string();
        }
    }
    if output.contains("fatal:") || output.contains("error:") {
        return output.to_string();
    }
    output.to_string()
}

fn filter_git_branch(output: &str) -> String {
    let mut result = String::new();
    for line in output.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with("* ") || t.starts_with("  ") || t.starts_with('+') {
            result.push_str(t);
            result.push('\n');
        } else {
            result.push_str(t);
            result.push('\n');
        }
    }
    result
}

// ── Cargo filters ────────────────────────────────────────────────────────

fn filter_cargo(cmd: &str, output: &str) -> Option<String> {
    let rest = cmd.trim_start_matches("cargo ").trim();
    if rest.starts_with("test") || rest.starts_with("nextest") {
        return Some(filter_cargo_test(output));
    }
    if rest.starts_with("build")
        || rest.starts_with("check")
        || rest.starts_with("clippy")
        || rest.starts_with("rustc")
    {
        return Some(filter_cargo_build(output));
    }
    None
}

fn filter_cargo_test(output: &str) -> String {
    let mut result = String::new();
    let mut in_failure = false;
    let mut failure_block = String::new();
    let mut failure_count = 0;
    let mut test_count = 0u32;
    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut has_summary = false;

    for line in output.lines() {
        let trimmed = line.trim();

        // Skip noise lines
        if trimmed.starts_with("Compiling ")
            || trimmed.starts_with("Finished ")
            || trimmed.starts_with("Downloading ")
            || trimmed.starts_with("   Compiling")
            || trimmed.starts_with("    Checking")
        {
            continue;
        }

        // Track individual test results
        if trimmed.contains("test ") && (trimmed.contains(" ... ok") || trimmed.contains("...ok")) {
            test_count += 1;
            passed += 1;
            continue; // skip passing tests
        }
        if trimmed.contains("test ") && (trimmed.contains(" ... FAILED") || trimmed.contains("...FAILED")) {
            test_count += 1;
            failed += 1;
            continue; // will be shown in failure block
        }

        // Failure block start
        if trimmed.starts_with("----") || trimmed.starts_with("failures:") {
            in_failure = true;
            failure_block.clear();
            failure_block.push_str(line);
            failure_block.push('\n');
            continue;
        }

        if in_failure {
            if trimmed.starts_with("----") || trimmed.is_empty() && failure_block.lines().count() > 1 {
                // End of failure block
                in_failure = false;
                if failure_count < 10 {
                    result.push_str(&failure_block);
                    result.push('\n');
                }
                failure_count += 1;
                if !trimmed.is_empty() {
                    failure_block = format!("{}\n", line);
                    in_failure = trimmed.starts_with("----") && trimmed.contains("FAILED");
                }
                continue;
            }
            failure_block.push_str(line);
            failure_block.push('\n');
            continue;
        }

        // Summary line
        if trimmed.starts_with("test result:") {
            has_summary = true;
            // Keep summary but strip ANSI
            let clean = strip_ansi(trimmed);
            result.push_str(&clean);
            result.push('\n');
            // Parse counts from summary (format: "ok. N passed; M failed" or "FAILED. N passed; M failed")
            if let Some(rest) = clean.strip_prefix("test result: ") {
                for part in rest.split(';').map(|s| s.trim()) {
                    let part_trimmed = part.trim();
                    if let Some(num_str) = part_trimmed.split(' ').find(|p| p.chars().all(|c| c.is_ascii_digit())) {
                        if let Ok(n) = num_str.parse::<u32>() {
                            if part_trimmed.contains("passed") {
                                passed = n;
                            } else if part_trimmed.contains("failed") {
                                failed = n;
                            }
                        }
                    }
                }
            }
            continue;
        }

        // Pass through errors
        if trimmed.starts_with("error[") || trimmed.starts_with("error:") || trimmed.starts_with("error:") {
            result.push_str(line);
            result.push('\n');
        }
    }

    // Flush remaining failure block
    if in_failure && failure_count < 10 {
        result.push_str(&failure_block);
        result.push('\n');
    }

    // If no failures and no summary, try to produce one
    if !has_summary && test_count > 0 {
        result.push_str(&format!("test result: {} passed, {} failed ({} tests total)\n", passed, failed, test_count));
    }

    // If all passed, condense to single line
    if failed == 0 && has_summary {
        // Already have summary line, but strip any passing test noise
        let lines: Vec<&str> = result.lines().filter(|l| {
            !l.contains("test ... ok")
        }).collect();
        result = lines.join("\n");
        if !result.ends_with('\n') {
            result.push('\n');
        }
    }

    // If all passed and result is just the summary (no failures), add savings note
    if has_summary && failure_count == 0 && passed > 0 && failed == 0 {
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str("[savings: ~90%]\n");
    }

    if result.is_empty() {
        return output.to_string();
    }
    result
}

fn filter_cargo_build(output: &str) -> String {
    let mut result = String::new();
    let mut crate_count = 0u32;
    let mut error_count = 0u32;
    let mut warning_count = 0u32;
    let mut shown_warnings = 0u32;
    let mut block = String::new();
    let mut block_type: &str = "";

    for line in output.lines() {
        let trimmed = line.trim();

        // Count compiled crates
        if trimmed.starts_with("Compiling ") || trimmed.starts_with("   Compiling ") {
            crate_count += 1;
            continue;
        }
        if trimmed.starts_with("Checking ") || trimmed.starts_with("   Checking ") {
            crate_count += 1;
            continue;
        }

        if trimmed.starts_with("Finished ") || trimmed.starts_with("Downloading ") || trimmed.starts_with("   Downloading") {
            continue;
        }

        if trimmed.starts_with("error[") || trimmed.starts_with("error:") {
            if !block.is_empty() && block_type == "error" && error_count <= 10 {
                result.push_str(&block);
                result.push('\n');
            }
            block = format!("{}\n", line);
            block_type = "error";
            error_count += 1;
            continue;
        }

        if !block.is_empty() {
            let ends = trimmed.starts_with("error[") || trimmed.starts_with("error:") || trimmed.starts_with("warning[") || trimmed.starts_with("warning:") || (trimmed.is_empty() && block.lines().count() > 1);
            if ends {
                if block_type == "error" && error_count <= 10 {
                    result.push_str(&block);
                    result.push('\n');
                } else if block_type == "warning" && shown_warnings < 5 {
                    result.push_str(&block);
                    result.push('\n');
                    shown_warnings += 1;
                }
                if trimmed.starts_with("error[") || trimmed.starts_with("error:") {
                    block = format!("{}\n", line);
                    block_type = "error";
                    error_count += 1;
                } else if trimmed.starts_with("warning[") || trimmed.starts_with("warning:") {
                    block = format!("{}\n", line);
                    block_type = "warning";
                    warning_count += 1;
                } else {
                    block.clear();
                    block_type = "";
                }
                continue;
            }
            block.push_str(line);
            block.push('\n');
            continue;
        }

        if trimmed.starts_with("warning[") || trimmed.starts_with("warning:") {
            block = format!("{}\n", line);
            block_type = "warning";
            warning_count += 1;
            continue;
        }
    }

    if !block.is_empty() {
        if block_type == "error" && error_count <= 10 {
            result.push_str(&block);
            result.push('\n');
        } else if block_type == "warning" && shown_warnings < 5 {
            result.push_str(&block);
            result.push('\n');
            shown_warnings += 1;
        }
    }

    let mut summary = format!("cargo build ({} crates)\n", crate_count);
    if error_count > 0 {
        summary.push_str(&format!("{} error(s)", error_count));
    }
    if warning_count > 0 {
        if error_count > 0 {
            summary.push_str(", ");
        }
        summary.push_str(&format!("{} warning(s)\n", warning_count));
    }
    if error_count == 0 && warning_count == 0 {
        summary.push_str("no errors, no warnings\n");
    }

    if error_count > 10 {
        summary.push_str(&format!("[+ {} more errors not shown]\n", error_count - 10));
    }
    if warning_count > shown_warnings {
        summary.push_str(&format!("[+ {} more warnings not shown]\n", warning_count - shown_warnings));
    }

    result = summary + &result;
    if result.trim().is_empty() {
        return output.to_string();
    }
    result
}

// ── System filters ───────────────────────────────────────────────────────

fn filter_ls(output: &str) -> Option<String> {
    let mut result = String::new();
    let mut files: Vec<String> = Vec::new();
    let mut dirs: Vec<String> = Vec::new();
    let mut total = 0usize;
    let max_items = 100;

    let lines: Vec<&str> = output.lines().collect();
    if lines.len() < 2 {
        return None; // Not ls output
    }

    // Check if first like looks like "total N" (ls -l style)
    if !lines[0].starts_with("total ") {
        return None; // Not standard ls -l output, let generic handle it
    }

    for line in &lines[1..] {
        if line.trim().is_empty() {
            continue;
        }
        if total >= max_items {
            break;
        }

        // Parse ls -l line: "permissions links owner group size date name"
        // We extract only the name (and type indicator)
        if line.len() < 50 {
            continue;
        }
        let bytes = line.as_bytes();
        let file_type = bytes[0] as char;
        let name_start = if line.rfind(|c: char| c.is_whitespace()).is_some() {
            // Find name: it's after the date field
            // Strategy: find the 5th space-separated field, everything after is name
            let mut spaces = 0;
            let mut name_pos = 0;
            for (i, &b) in bytes.iter().enumerate() {
                if b == b' ' && i > 0 && bytes[i - 1] != b' ' {
                    spaces += 1;
                    if spaces == 8 {
                        name_pos = i + 1;
                        break;
                    }
                }
            }
            name_pos
        } else {
            continue;
        };

        if name_start >= line.len() {
            continue;
        }
        let name = line[name_start..].trim().to_string();

        // Skip noise dirs
        if name == "." || name == ".." {
            continue;
        }

        if file_type == 'd' {
            dirs.push(name);
        } else {
            files.push(name);
        }
        total += 1;
    }

    dirs.sort();
    files.sort();

    for d in &dirs {
        result.push_str(&format!("{}/\n", d));
    }
    for f in &files {
        result.push_str(&format!("{}\n", f));
    }

    if result.is_empty() {
        return None;
    }

    let savings = if lines.len() > 10 {
        format!(" [savings: ~{}%]", (lines.len() - result.lines().count()) * 100 / lines.len().max(1))
    } else {
        String::new()
    };
    result.push_str(&format!("Summary: {} files, {} dirs{}", files.len(), dirs.len(), savings));

    Some(result)
}

fn filter_find(output: &str) -> Option<String> {
    if output.is_empty() || output == "No files found." {
        return None;
    }

    let lines: Vec<&str> = output.lines().collect();
    if lines.is_empty() {
        return None;
    }

    // Group by parent directory
    let mut by_dir: HashMap<String, Vec<String>> = HashMap::new();
    for line in &lines {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(parent) = std::path::Path::new(t).parent() {
            let dir = parent.to_string_lossy().to_string();
            let name = std::path::Path::new(t)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| t.to_string());
            by_dir.entry(dir).or_default().push(name);
        } else {
            by_dir.entry(".".to_string()).or_default().push(t.to_string());
        }
    }

    // Collect extensions
    let mut ext_counts: HashMap<String, u32> = HashMap::new();
    let mut total_files = 0u32;
    for names in by_dir.values() {
        for name in names {
            total_files += 1;
            if let Some(dot) = name.rfind('.') {
                let ext = name[dot..].to_string();
                *ext_counts.entry(ext).or_default() += 1;
            }
        }
    }

    let mut result = String::new();
    result.push_str(&format!("{} files found\n", total_files));

    // Show top dirs
    let mut dirs: Vec<(&String, &Vec<String>)> = by_dir.iter().collect();
    dirs.sort_by_key(|(d, _)| d.len());
    let mut shown = 0u32;
    for (dir, names) in &dirs {
        if shown >= 20 {
            break;
        }
        if names.len() == 1 {
            result.push_str(&format!("{}/{}\n", dir, names[0]));
        } else {
            result.push_str(&format!("{}/\n", dir));
            for name in names.iter().take(5) {
                result.push_str(&format!("  {}\n", name));
            }
            if names.len() > 5 {
                result.push_str(&format!("  ... +{} more\n", names.len() - 5));
            }
        }
        shown += 1;
    }
    if dirs.len() > 20 {
        result.push_str(&format!("... +{} more directories\n", dirs.len() - 20));
    }

    // Extension summary
    let mut exts: Vec<(&String, &u32)> = ext_counts.iter().collect();
    exts.sort_by(|a, b| b.1.cmp(a.1));
    if !exts.is_empty() {
        result.push_str("Extensions: ");
        let ext_parts: Vec<String> = exts.iter().take(5).map(|(e, c)| format!("{}({})", e, c)).collect();
        result.push_str(&ext_parts.join(", "));
        if exts.len() > 5 {
            result.push_str(&format!(", +{} more", exts.len() - 5));
        }
        result.push('\n');
    }

    Some(result)
}

fn filter_generic(output: &str) -> String {
    let mut result = String::new();
    let mut prev_line = String::new();
    let mut dup_count = 0u32;
    let mut skipped_lines = 0u32;
    let max_lines = 500;

    for line in output.lines() {
        let clean = strip_ansi(line);
        let trimmed = clean.trim().to_string();

        // Strip trailing whitespace
        let processed = trimmed.trim_end().to_string();

        // Deduplicate consecutive identical lines
        if processed == prev_line {
            dup_count += 1;
            skipped_lines += 1;
            continue;
        }

        // Flush dedup count
        if dup_count > 0 {
            result.push_str(&format!("  [previous line repeated {} times]\n", dup_count));
            dup_count = 0;
        }

        // Compress multiple blank lines
        if processed.is_empty() && result.ends_with("\n\n") {
            skipped_lines += 1;
            continue;
        }

        prev_line = processed.clone();

        // Cap output lines
        if result.lines().count() >= max_lines {
            skipped_lines += 1;
            continue;
        }

        if clean != line {
            result.push_str(&processed);
        } else {
            result.push_str(line.trim_end());
        }
        result.push('\n');
    }

    // Flush final dedup count
    if dup_count > 0 {
        result.push_str(&format!("  [previous line repeated {} times]\n", dup_count));
    }

    // If nothing changed, return original
    if skipped_lines == 0 && result == output {
        return output.to_string();
    }

    let original_lines = output.lines().count();
    let new_lines = result.lines().count();
    if new_lines > 0 && original_lines > new_lines {
        let pct = (original_lines - new_lines) * 100 / original_lines.max(1);
        result.push_str(&format!("[filtered: {} lines -> {} ({}% savings)]", original_lines, new_lines, pct));
    }

    result
}

/// Strip ANSI escape codes from text.
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;
    for c in s.chars() {
        if in_escape {
            if c == 'm' || c == 'H' || c == 'J' || c == 'K' || (c as u8) < 0x20 {
                in_escape = false;
                if c == 'm' { continue; }
            }
            continue;
        }
        if c == '\x1b' {
            in_escape = true;
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_status_clean() {
        let out = "On branch main\nnothing to commit, working tree clean\n";
        let filtered = filter_git_status(out);
        assert!(filtered.contains("clean"));
        assert!(!filtered.contains("(use \"git add\""));
    }

    #[test]
    fn test_git_status_with_changes() {
        let out = "On branch main\nChanges not staged for commit:\n  (use \"git add <file>...\" to update)\n  (use \"git restore <file>...\" to discard)\n\tmodified:   src/main.rs\n\tmodified:   src/lib.rs\n\nUntracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n\tnew.txt\n\nno changes added to commit\n";
        let filtered = filter_git_status(out);
        assert!(!filtered.contains("(use \""));
        assert!(filtered.contains("src/main.rs") || filtered.contains("main.rs"));
        assert!(filtered.contains("staged") || filtered.contains("unstaged") || filtered.contains("untracked"));
    }

    #[test]
    fn test_git_diff_strips_context() {
        let out = "diff --git a/src/main.rs b/src/main.rs\nindex abc..def 100644\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,5 +1,6 @@\n fn main() {\n-    println!(\"hello\");\n+    println!(\"hello world\");\n     let x = 1;\n }\n";
        let filtered = filter_git_diff(out);
        assert!(!filtered.contains("diff --git"));
        assert!(filtered.contains("@@ -1,5 +1,6 @@"));
        assert!(filtered.contains("-    println!(\"hello\");"));
        assert!(filtered.contains("+    println!(\"hello world\");"));
        assert!(!filtered.contains("let x = 1")); // context stripped
    }

    #[test]
    fn test_git_log_filters() {
        let out = "abc1234 first commit\ndef5678 second commit\n";
        let filtered = filter_git_log(out);
        assert!(filtered.contains("abc1234"));
        assert!(filtered.contains("def5678"));
    }

    #[test]
    fn test_git_add_success() {
        assert_eq!(filter_git_add(""), "ok");
        assert_eq!(filter_git_add("nothing to add"), "ok");
    }

    #[test]
    fn test_git_commit_extracts_hash() {
        let out = "[main abc1234f] my commit message\n 1 file changed, 1 insertion(+)\n";
        let filtered = filter_git_commit(out);
        assert!(filtered.contains("abc1234"));
    }

    #[test]
    fn test_cargo_test_all_pass() {
        let out = "   Compiling myapp v0.1.0\n    Finished dev [unoptimized + debuginfo]\ntest result: ok. 42 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n";
        let filtered = filter_cargo_test(out);
        assert!(filtered.contains("42 passed"));
        assert!(filtered.contains("savings"));
        assert!(!filtered.contains("Compiling"));
    }

    #[test]
    fn test_cargo_test_with_failures() {
        let out = "test test_foo ... ok\ntest test_bar ... FAILED\n\n---- test_bar stdout ----\n\tthread 'test_bar' panicked at src/lib.rs:10:\n\tassertion failed\n\nfailures:\n    test_bar\n\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n";
        let filtered = filter_cargo_test(out);
        assert!(filtered.contains("test_bar"));
        assert!(filtered.contains("assertion failed"));
        assert!(!filtered.contains("test_foo ... ok"));
    }

    #[test]
    fn test_cargo_build_success() {
        let out = "   Compiling myapp v0.1.0 (/tmp)\n   Compiling dep v0.2.0\n    Finished dev [unoptimized + debuginfo]\n";
        let filtered = filter_cargo_build(out);
        assert!(filtered.contains("2 crates"));
        assert!(filtered.contains("no errors"));
        assert!(!filtered.contains("Compiling"));
    }

    #[test]
    fn test_cargo_build_with_errors() {
        let out = "   Compiling myapp v0.1.0\nerror[E0308]: mismatched types\n --> src/main.rs:1:20\n  |\n1 | let x: i32 = \"hello\";\n  |     ^^^^^^^ expected i32, found &str\n\n   Compiling dep v0.2.0\n    Finished dev [unoptimized + debuginfo]\n";
        let filtered = filter_cargo_build(out);
        assert!(filtered.contains("E0308"));
        assert!(filtered.contains("1 error(s)"));
        assert!(!filtered.contains("Compiling"));
    }

    #[test]
    fn test_filter_generic_dedup() {
        let out = "line1\nline1\nline1\nline2\n";
        let filtered = filter_generic(out);
        assert!(filtered.contains("repeated"));
    }

    #[test]
    fn test_filter_generic_ansi() {
        let out = "\x1b[31mhello\x1b[0m\n";
        let filtered = filter_generic(out);
        assert!(filtered.contains("hello"));
        assert!(!filtered.contains("\x1b"));
    }

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("\x1b[31mhello\x1b[0m"), "hello");
        assert_eq!(strip_ansi("hello"), "hello");
        assert_eq!(strip_ansi("\x1b[1;32mok\x1b[m"), "ok");
    }

    #[test]
    fn test_filter_find() {
        let out = "src/main.rs\nsrc/lib.rs\ntests/test.rs\n";
        let filtered = filter_find(out);
        assert!(filtered.is_some());
        let f = filtered.unwrap();
        assert!(f.contains("3 files"));
        assert!(f.contains("Extensions"));
    }

    #[test]
    fn test_filter_output_routes_git() {
        let out = "On branch main\nnothing to commit, working tree clean\n";
        let filtered = filter_output("git status", out);
        assert!(filtered.contains("clean"));
    }

    #[test]
    fn test_filter_output_routes_cargo() {
        let out = "test result: ok. 10 passed; 0 failed\n";
        let filtered = filter_output("cargo test", out);
        assert!(filtered.contains("10 passed"));
    }

    #[test]
    fn test_git_branch() {
        let out = "* main\n  feature\n  bugfix\n";
        let filtered = filter_git_branch(out);
        assert!(filtered.contains("* main"));
        assert!(filtered.contains("feature"));
    }

    #[test]
    fn test_git_push_up_to_date() {
        let out = "Everything up-to-date\n";
        let filtered = filter_git_push_pull(out);
        assert!(filtered.contains("up-to-date"));
    }

    #[test]
    fn test_cargo_build_many_crates() {
        let mut out = String::new();
        for i in 0..20 {
            out.push_str(&format!("   Compiling crate{} v0.1.0\n", i));
        }
        out.push_str("    Finished dev [unoptimized + debuginfo]\n");
        let filtered = filter_cargo_build(&out);
        assert!(filtered.contains("20 crates"));
        assert!(filtered.contains("no errors"));
    }
}
