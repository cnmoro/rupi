//! Spill: keep oversized tool output reachable instead of destroying it.
//!
//! A tool result larger than the storage cap used to be truncated in place, and the
//! discarded bytes were gone for the rest of the session. Spill writes the full
//! result to a session-scoped file first and puts the path in the truncation notice,
//! so the agent can read back exactly what was cut with the `read` tool it already
//! has. The conversation stays small and no data is lost.
//!
//! Artifacts are content addressed. An agent that runs the same failing command ten
//! times produces one archive, not ten, and the name is stable, so a later identical
//! result points at a file that is already there.
//!
//! Spill is best-effort by contract. A storage failure is never fatal: the caller
//! keeps the plain truncated result and the turn continues.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// A saved spill artifact.
///
/// Non-exhaustive: this is public, and adding a field would otherwise break any
/// embedder that destructures it without a rest pattern.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpillRef {
    /// Absolute path the agent can hand straight to the `read` tool.
    pub path: PathBuf,
    /// Exact byte length of the saved content.
    pub bytes: usize,
    /// Whether an identical archive already existed and was reused.
    pub reused: bool,
}

/// Characters of the content digest used in an artifact name.
///
/// 24 hex characters is 96 bits. A collision inside one session is not credible,
/// and a collision is caught anyway: an existing file is compared byte for byte
/// before it is reused.
const DIGEST_CHARS: usize = 24;

/// Hex digest of the content, used as its address.
fn digest(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let full = format!("{:x}", hasher.finalize());
    full[..DIGEST_CHARS].to_string()
}

/// Reduce an identifier to characters that are safe in a path segment.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        // Bounded well under NAME_MAX. `tool_name` comes from the model, and an
        // over-long one made every candidate open fail with the same name-too-long
        // error, burning all sixteen attempts before giving up.
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "session".to_string()
    } else {
        cleaned
    }
}

/// Persist the full text of one tool result.
///
/// Returns `None` when no sessions directory is available or the write fails.
/// Both cases are normal degraded operation, not errors the caller must handle.
pub fn save_text(session_id: &str, tool_name: &str, content: &str) -> Option<SpillRef> {
    save_text_in(
        &crate::sessions::sessions_dir()?,
        session_id,
        tool_name,
        content,
    )
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
    let spill_root = base.join("spill");
    let dir = spill_root.join(sanitize(session_id));
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    // Restrict both levels. `create_dir_all` also creates `spill/`, and leaving it
    // at the umask default makes every session id world-listable.
    restrict_dir(&spill_root);
    restrict_dir(&dir);

    let digest = digest(content);
    let bytes = content.len();

    for attempt in 0..16u32 {
        let path = if attempt == 0 {
            dir.join(format!("{}-{}.txt", sanitize(tool_name), digest))
        } else {
            dir.join(format!(
                "{}-{}-{}.txt",
                sanitize(tool_name),
                digest,
                attempt
            ))
        };

        match create_exclusive(&path) {
            Ok(mut file) => {
                use std::io::Write;
                if file.write_all(content.as_bytes()).is_err() {
                    return None;
                }
                return Some(SpillRef {
                    path,
                    bytes,
                    reused: false,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                // Something is already at this address. Reuse it only when it holds
                // exactly this content. Anything else — a digest collision, a file a
                // concurrent writer is part way through, a symlink — is not this
                // result, and serving it would hand the agent the wrong output under
                // a name it trusts. Move to the next candidate instead.
                if archive_matches(&path, content) {
                    // An archive restored from a backup, or written by an older
                    // build, can still be world-readable. Restrict it on the way out.
                    restrict_file(&path);
                    return Some(SpillRef {
                        path,
                        bytes,
                        reused: true,
                    });
                }
                continue;
            }
            // A transient failure on one candidate must not abandon the rest.
            Err(_) => continue,
        }
    }
    // The one path that actually loses the archive. It was silent, while a
    // cosmetic permission failure logged — exactly backwards.
    eprintln!(
        "rupi: could not archive a {} byte {} result after 16 attempts in {}",
        bytes,
        tool_name,
        dir.display()
    );
    None
}

/// Create a file that must not already exist, restricted to the owner from birth.
///
/// The mode is set at creation, not afterwards. `std::fs::write` creates at
/// `0666 & ~umask`, writes the whole payload, and only then could the mode be
/// changed — so for a large result the content sits world-readable for the entire
/// duration of the write. Tool output carries environment dumps and credentials,
/// which is the threat this module exists to bound.
///
/// `O_NOFOLLOW` refuses a symlink at the path. Without it a planted dangling link
/// turns an archive write into an arbitrary file write with attacker-chosen content.
#[cfg(unix)]
fn create_exclusive(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(not(unix))]
fn create_exclusive(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// Whether the file at `path` is a regular file holding exactly `content`.
///
/// Checks the link type and the size before reading, so a symlink is never
/// followed and a large mismatch costs one metadata call rather than a full read.
fn archive_matches(path: &std::path::Path, content: &str) -> bool {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() != content.len() as u64 {
        return false;
    }
    matches!(std::fs::read_to_string(path), Ok(existing) if existing == content)
}

/// Restrict an artifact to the owner.
///
/// Tool output is whatever ran on the user's machine: environment dumps, tokens in
/// a failing request, private source. The default umask often leaves it
/// world-readable, and these files persist after the session ends.
#[cfg(unix)]
fn restrict_file(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    if std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).is_err() {
        // Say so. A silent failure leaves tool output world-readable with nothing
        // anywhere to show for it.
        eprintln!("rupi: could not restrict spill artifact {}", path.display());
    }
}

/// Restrict a spill directory to the owner.
#[cfg(unix)]
fn restrict_dir(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    if std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).is_err() {
        eprintln!(
            "rupi: could not restrict spill directory {}",
            path.display()
        );
    }
}

#[cfg(not(unix))]
fn restrict_file(_path: &std::path::Path) {}

#[cfg(not(unix))]
fn restrict_dir(_path: &std::path::Path) {}

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
        assert_eq!(sanitize("../../etc/passwd"), "______etc_passwd");
        assert_eq!(sanitize("abc-123_X"), "abc-123_X");
        assert_eq!(sanitize(""), "session");
    }

    #[test]
    fn a_traversing_session_id_stays_inside_the_base() {
        let base = scratch("traverse");
        let spill = save_text_in(&base, "../../escape", "bash", "secret").unwrap();
        assert!(
            spill.path.starts_with(&base),
            "spill escaped its base: {:?}",
            spill.path
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn save_text_writes_the_full_content() {
        let base = scratch("full");
        let body = "x".repeat(9000);
        let spill = save_text_in(&base, "sess-1", "bash", &body).unwrap();

        assert_eq!(spill.bytes, 9000);
        assert!(!spill.reused);
        assert_eq!(std::fs::read_to_string(&spill.path).unwrap(), body);

        let hint = retrieval_hint(&spill);
        assert!(hint.contains("9000 bytes"));
        assert!(hint.contains("read tool"));
        assert!(hint.contains(&spill.path.display().to_string()));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn different_content_gets_different_names() {
        let base = scratch("uniq");
        let a = save_text_in(&base, "sess-2", "bash", "one").unwrap();
        let b = save_text_in(&base, "sess-2", "bash", "two").unwrap();
        assert_ne!(a.path, b.path);
        assert_eq!(std::fs::read_to_string(&a.path).unwrap(), "one");
        assert_eq!(std::fs::read_to_string(&b.path).unwrap(), "two");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn identical_content_is_archived_once() {
        let base = scratch("dedup");
        let body = "the same failing test output".repeat(100);

        let first = save_text_in(&base, "sess-3", "bash", &body).unwrap();
        assert!(!first.reused);
        for _ in 0..5 {
            let again = save_text_in(&base, "sess-3", "bash", &body).unwrap();
            assert_eq!(
                again.path, first.path,
                "the same content must share one address"
            );
            assert!(again.reused);
            assert_eq!(again.bytes, first.bytes);
        }

        let files: Vec<_> = std::fs::read_dir(base.join("spill").join("sess-3"))
            .unwrap()
            .filter_map(|entry| entry.ok())
            .collect();
        assert_eq!(
            files.len(),
            1,
            "a repeated result must not be stored six times"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_corrupted_archive_is_never_served_as_this_result() {
        let base = scratch("corrupt");
        let body = "the real output";
        let first = save_text_in(&base, "sess-4", "bash", body).unwrap();

        // Something else rewrote the archive, or a digest collided. Either way the
        // stored bytes are not this result, so reusing them would hand the agent
        // the wrong output under a name it trusts.
        std::fs::write(&first.path, "different bytes entirely").unwrap();

        let second = save_text_in(&base, "sess-4", "bash", body).unwrap();
        assert_ne!(
            second.path, first.path,
            "a mismatched archive must not be reused"
        );
        assert!(!second.reused);
        assert_eq!(std::fs::read_to_string(&second.path).unwrap(), body);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn save_text_scopes_artifacts_by_session() {
        let base = scratch("scope");
        let a = save_text_in(&base, "sess-a", "read", "one").unwrap();
        let b = save_text_in(&base, "sess-b", "read", "one").unwrap();
        assert_ne!(a.path.parent(), b.path.parent());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn artifacts_are_readable_only_by_their_owner() {
        use std::os::unix::fs::PermissionsExt;
        let base = scratch("modes");
        let spill = save_text_in(&base, "sess-5", "bash", "API_KEY=secret").unwrap();

        // Tool output can carry credentials, and these files outlive the session.
        let file_mode = std::fs::metadata(&spill.path).unwrap().permissions().mode() & 0o777;
        assert_eq!(file_mode, 0o600, "artifact mode was {:o}", file_mode);

        let dir_mode = std::fs::metadata(spill.path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700, "directory mode was {:o}", dir_mode);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn a_planted_symlink_is_never_followed() {
        let base = scratch("symlink");
        let content = "the archive content";

        // Work out the address this content will take, then plant a dangling
        // symlink there. Following it would create the victim with content the
        // planter chose, because a tool result is often attacker-influenced.
        let dir = base.join("spill").join("sess-6");
        std::fs::create_dir_all(&dir).unwrap();
        let address = dir.join(format!("bash-{}.txt", digest(content)));
        let victim = base.join("victim.txt");
        std::os::unix::fs::symlink(&victim, &address).unwrap();

        let spill = save_text_in(&base, "sess-6", "bash", content).unwrap();
        assert!(
            !victim.exists(),
            "the write followed a symlink and created {:?}",
            victim
        );
        assert_ne!(
            spill.path, address,
            "the symlinked address must not be used"
        );
        assert_eq!(std::fs::read_to_string(&spill.path).unwrap(), content);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_corrupted_archive_does_not_multiply_copies() {
        let base = scratch("no-multiply");
        let content = "the same failing output";
        let first = save_text_in(&base, "sess-7", "bash", content).unwrap();
        std::fs::write(&first.path, "different bytes entirely").unwrap();

        // Every later save of the same content must settle on ONE alternative
        // address and reuse it, not write a fresh copy each time and then start
        // failing once the candidates run out.
        let mut paths = std::collections::HashSet::new();
        for attempt in 0..25 {
            let spill = save_text_in(&base, "sess-7", "bash", content)
                .unwrap_or_else(|| panic!("save returned None on attempt {}", attempt));
            paths.insert(spill.path);
        }
        assert_eq!(
            paths.len(),
            1,
            "the same content settled on {} addresses",
            paths.len()
        );

        let files = std::fs::read_dir(base.join("spill").join("sess-7"))
            .unwrap()
            .count();
        assert_eq!(
            files, 2,
            "expected the corrupted file plus one archive, found {}",
            files
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn the_spill_root_is_restricted_too() {
        use std::os::unix::fs::PermissionsExt;
        let base = scratch("root-mode");
        let spill = save_text_in(&base, "sess-8", "bash", "body").unwrap();

        // `create_dir_all` makes `spill/` as well. Left at the umask default it
        // lists every session id to anyone on the machine.
        let root = spill.path.parent().unwrap().parent().unwrap();
        let mode = std::fs::metadata(root).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "spill root mode was {:o}", mode);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn reuse_restricts_a_world_readable_archive() {
        use std::os::unix::fs::PermissionsExt;
        let base = scratch("reuse-mode");
        let content = "restored from a backup";
        let first = save_text_in(&base, "sess-9", "bash", content).unwrap();

        // An archive written by an older build, or restored from a tarball, can
        // arrive world-readable. Reusing it must not leave it that way.
        std::fs::set_permissions(&first.path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let again = save_text_in(&base, "sess-9", "bash", content).unwrap();
        assert!(again.reused);
        let mode = std::fs::metadata(&again.path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "reused archive mode was {:o}", mode);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_directory_at_the_address_does_not_stop_the_save() {
        let base = scratch("dir-at-address");
        let content = "payload";
        let dir = base.join("spill").join("sess-10");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(dir.join(format!("bash-{}.txt", digest(content)))).unwrap();

        let spill = save_text_in(&base, "sess-10", "bash", content).unwrap();
        assert_eq!(std::fs::read_to_string(&spill.path).unwrap(), content);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn an_unwritable_base_degrades_to_none() {
        let base = scratch("unwritable").join("blocked");
        std::fs::write(&base, "not a directory").unwrap();
        assert!(save_text_in(&base, "sess", "bash", "body").is_none());
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }

    #[test]
    fn the_digest_is_stable_and_content_dependent() {
        assert_eq!(digest("abc"), digest("abc"));
        assert_ne!(digest("abc"), digest("abd"));
        assert_eq!(digest("abc").len(), DIGEST_CHARS);
        assert!(digest("abc").chars().all(|c| c.is_ascii_hexdigit()));
    }
}
