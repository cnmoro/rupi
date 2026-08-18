//! Spill: keep oversized tool output reachable instead of destroying it.
//!
//! A tool result larger than the storage cap used to be truncated in place, and the
//! discarded bytes were gone for the rest of the session. Spill writes the full
//! result to a session-scoped file first and puts the path in the truncation notice,
//! so the agent can read back exactly what was cut with the `read` tool it already
//! has. The conversation stays small and no data is lost.
//!
//! Spill is best-effort by contract. A storage failure is never fatal: the caller
//! keeps the plain truncated result and the turn continues.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic counter that makes spill file names collision-free within a process.
static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

/// A saved spill artifact.
#[derive(Debug, Clone)]
pub struct SpillRef {
    /// Absolute path the agent can hand straight to the `read` tool.
    pub path: PathBuf,
    /// Exact byte length of the saved content.
    pub bytes: usize,
}

/// Directory that holds spill artifacts for one session.
pub fn spill_dir(session_id: &str) -> Option<PathBuf> {
    Some(crate::sessions::sessions_dir()?.join("spill").join(sanitize(session_id)))
}

/// Reduce an identifier to characters that are safe in a path segment.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if cleaned.is_empty() { "session".to_string() } else { cleaned }
}

/// Persist the full text of one tool result.
///
/// Returns `None` when no sessions directory is available or the write fails.
/// Both cases are normal degraded operation, not errors the caller must handle.
pub fn save_text(session_id: &str, tool_name: &str, content: &str) -> Option<SpillRef> {
    save_text_in(&crate::sessions::sessions_dir()?, session_id, tool_name, content)
}

/// Persist the full text of one tool result under an explicit base directory.
///
/// Split out from `save_text` so the storage behaviour is testable without touching
/// the process-wide sessions directory, which is set once and never changes.
pub fn save_text_in(
    base: &std::path::Path,
    session_id: &str,
    tool_name: &str,
    content: &str,
) -> Option<SpillRef> {
    let dir = base.join("spill").join(sanitize(session_id));
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let seq = SPILL_SEQ.fetch_add(1, Ordering::SeqCst);
    let path = dir.join(format!("{}-{:06}.txt", sanitize(tool_name), seq));
    if std::fs::write(&path, content).is_err() {
        return None;
    }
    Some(SpillRef { path, bytes: content.len() })
}

/// The retrieval hint appended to a truncated result that has a spill artifact.
pub fn retrieval_hint(spill: &SpillRef) -> String {
    format!(
        "\n[The complete {} bytes of this result were saved to {}. \
Use the read tool on that path to see any part that was cut, \
or grep it to search the full output.]",
        spill.bytes,
        spill.path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rupi-spill-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sanitize_strips_path_separators() {
        // Six leading separator characters become six underscores.
        assert_eq!(sanitize("../../etc/passwd"), "______etc_passwd");
        assert_eq!(sanitize("abc-123_X"), "abc-123_X");
        assert_eq!(sanitize(""), "session");
    }

    #[test]
    fn a_traversing_session_id_stays_inside_the_base() {
        let base = scratch("traverse");
        let body = "secret";
        let spill = save_text_in(&base, "../../escape", "bash", body).unwrap();
        assert!(spill.path.starts_with(&base), "spill escaped its base: {:?}", spill.path);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn save_text_writes_the_full_content() {
        let base = scratch("full");
        let body = "x".repeat(9000);
        let spill = save_text_in(&base, "sess-1", "bash", &body).unwrap();

        assert_eq!(spill.bytes, 9000);
        assert_eq!(std::fs::read_to_string(&spill.path).unwrap(), body);

        let hint = retrieval_hint(&spill);
        assert!(hint.contains("9000 bytes"));
        assert!(hint.contains("read tool"));
        assert!(hint.contains(&spill.path.display().to_string()));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn save_text_names_are_unique() {
        let base = scratch("uniq");
        let a = save_text_in(&base, "sess-2", "bash", "one").unwrap();
        let b = save_text_in(&base, "sess-2", "bash", "two").unwrap();
        assert_ne!(a.path, b.path);
        assert_eq!(std::fs::read_to_string(&a.path).unwrap(), "one");
        assert_eq!(std::fs::read_to_string(&b.path).unwrap(), "two");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn save_text_scopes_artifacts_by_session() {
        let base = scratch("scope");
        let a = save_text_in(&base, "sess-a", "read", "one").unwrap();
        let b = save_text_in(&base, "sess-b", "read", "two").unwrap();
        assert_ne!(a.path.parent(), b.path.parent());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_unwritable_base_degrades_to_none() {
        // A regular file where the directory must go: create_dir_all fails, and the
        // caller keeps the plain truncated result instead of losing the turn.
        let base = scratch("unwritable").join("blocked");
        std::fs::write(&base, "not a directory").unwrap();
        assert!(save_text_in(&base, "sess", "bash", "body").is_none());
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }
}
