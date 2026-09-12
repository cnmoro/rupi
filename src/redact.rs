//! Redaction of credential-shaped values in text that outlives its context.
//!
//! Compaction is the one operation that would otherwise age a secret out of the
//! conversation. The region holding it is deleted, so if the checkpoint copies the
//! secret forward it is promoted from something transient into something that rides
//! every later request and lands in the session file.
//!
//! The summarizer is explicitly told to preserve commands verbatim, which is right
//! for resuming work and wrong for credentials. Redaction resolves that by keeping
//! the shape and dropping the value: `curl -H 'Authorization: Bearer [redacted]'`
//! still says what ran.
//!
//! Matching is on the shape of a VALUE, never on a keyword alone. A keyword rule
//! would blank a line of prose about token buckets or password policy, which is
//! both useless and alarming. This is best-effort by nature: it catches the common
//! credential formats and the common `KEY=value` shapes, and it cannot catch a
//! secret that looks like ordinary text.

use regex::Regex;
use std::sync::OnceLock;

/// What replaces a redacted value.
pub const REDACTED: &str = "[redacted]";

/// Patterns whose whole match is a credential, replaced entirely.
///
/// These are formats with a recognizable prefix, so the prefix is kept for
/// readability and only the body is dropped.
fn whole_value_patterns() -> &'static [(Regex, &'static str)] {
    static PATTERNS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let rules: Vec<(&str, &str)> = vec![
            // OpenAI and Anthropic style keys.
            (r"\bsk-[A-Za-z0-9_\-]{16,}", "sk-"),
            // GitHub personal access tokens and their relatives.
            (r"\bgh[pousr]_[A-Za-z0-9]{16,}", "gh_"),
            (r"\bgithub_pat_[A-Za-z0-9_]{20,}", "github_pat_"),
            // Slack.
            (r"\bxox[baprs]-[A-Za-z0-9\-]{10,}", "xox-"),
            // AWS access key id.
            (r"\bAKIA[0-9A-Z]{16}\b", "AKIA"),
            // Google API key.
            (r"\bAIza[A-Za-z0-9_\-]{30,}", "AIza"),
            // A JSON web token.
            (
                r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
                "jwt:",
            ),
            // OpenRouter.
            (r"\bsk-or-[A-Za-z0-9_\-]{16,}", "sk-or-"),
        ];
        rules
            .into_iter()
            .filter_map(|(pattern, label)| Regex::new(pattern).ok().map(|re| (re, label)))
            .collect()
    })
}

/// Patterns where capture group 1 is a keyword prefix and group 2 is the value.
///
/// Only the value is replaced, so the reader still sees which setting was set.
fn keyed_value_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let rules = [
            // Authorization: Bearer <value>  /  Bearer <value>
            r"(?i)((?:authorization\s*:\s*)?bearer\s+)([A-Za-z0-9._\-+/=]{12,})",
            // KEY=value and KEY: value, for any identifier whose NAME contains a
            // credential word. The surrounding identifier characters are part of
            // the name, so `AWS_SECRET_ACCESS_KEY` matches as readily as `token`.
            // A value is required after `=` or `:`, which is what keeps this off
            // ordinary prose that merely mentions passwords or tokens.
            r#"(?i)([A-Za-z0-9_.\-]*(?:api[_\-]?key|secret|password|passwd|token|credential)[A-Za-z0-9_.\-]*\s*[=:]\s*)["']?([^\s"'&;]{8,})"#,
            // curl -u user:password
            r"(-u\s+[^\s:]+:)([^\s]{4,})",
            // --password value / --token value
            r"(?i)(--(?:password|token|api-key|secret)[=\s]+)([^\s]{4,})",
        ];
        rules.iter().filter_map(|p| Regex::new(p).ok()).collect()
    })
}

/// Replace credential-shaped values in `text`.
///
/// Returns the redacted text and how many values were replaced. The count is for
/// telling the reader that redaction happened, not for any security decision.
pub fn redact_secrets(text: &str) -> (String, usize) {
    let mut out = text.to_string();
    let mut count = 0usize;

    for (pattern, label) in whole_value_patterns() {
        let found = pattern.find_iter(&out).count();
        if found > 0 {
            count += found;
            out = pattern
                .replace_all(&out, format!("{}{}", label, REDACTED))
                .into_owned();
        }
    }

    for pattern in keyed_value_patterns() {
        let found = pattern.find_iter(&out).count();
        if found > 0 {
            count += found;
            out = pattern
                .replace_all(&out, |caps: &regex::Captures| {
                    format!("{}{}", &caps[1], REDACTED)
                })
                .into_owned();
        }
    }

    (out, count)
}

/// Whether `text` carries anything that looks like a credential.
pub fn has_secret(text: &str) -> bool {
    redact_secrets(text).1 > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redacted(text: &str) -> String {
        redact_secrets(text).0
    }

    #[test]
    fn common_key_formats_are_caught() {
        let cases = [
            ("sk-proj-abcdefghijklmnopqrstuvwxyz012345", "sk-"),
            ("ghp_abcdefghijklmnopqrstuvwxyz0123", "gh_"),
            ("github_pat_11ABCDEFG0abcdefghijklmnop", "github_pat_"),
            ("xoxb-123456789012-abcdefghijklm", "xox-"),
            ("AKIAIOSFODNN7EXAMPLE", "AKIA"),
            ("AIzaSyD-abcdefghijklmnopqrstuvwxyz01234", "AIza"),
        ];
        for (secret, label) in cases {
            let out = redacted(&format!("run with {} please", secret));
            assert!(!out.contains(secret), "{} survived: {}", secret, out);
            assert!(out.contains(label), "label missing for {}: {}", secret, out);
            assert!(out.contains(REDACTED), "{}", out);
        }
    }

    #[test]
    fn a_bearer_header_keeps_its_shape() {
        let out =
            redacted("curl -H 'Authorization: Bearer sk-live-DEADBEEFCAFEBABE12345' https://api");
        assert!(!out.contains("DEADBEEF"), "{}", out);
        // The reader must still be able to tell what the command did.
        assert!(out.contains("curl"), "{}", out);
        assert!(out.contains("https://api"), "{}", out);
        assert!(out.to_lowercase().contains("bearer"), "{}", out);
    }

    #[test]
    fn keyed_assignments_keep_the_key() {
        for text in [
            "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY",
            "API_KEY=abcdefghijklmnop",
            "password: hunter2hunter2",
            "--token abcdefghijkl",
            "curl -u admin:s3cr3tpassword https://x",
        ] {
            let out = redacted(text);
            assert!(out.contains(REDACTED), "not redacted: {} -> {}", text, out);
        }
        // The name of the setting survives, so the checkpoint still says what was set.
        assert!(redacted("API_KEY=abcdefghijklmnop").contains("API_KEY"));
    }

    #[test]
    fn ordinary_prose_is_left_alone() {
        // A keyword rule would blank all of these. Matching on value shape must not.
        for text in [
            "the token bucket rate limiter needs tuning",
            "we discussed the password policy for the admin panel",
            "src/auth.rs handles the secret rotation schedule",
            "the API key format is documented in README.md",
            "returns a token from the lexer",
        ] {
            assert_eq!(redacted(text), text, "prose was altered: {}", text);
        }
    }

    #[test]
    fn code_and_paths_survive() {
        for text in [
            "fn parse_token(input: &str) -> Token { todo!() }",
            "src/provider/openai.rs:203 sets tool_choice",
            "cargo test --lib -- --nocapture",
            "git commit -m 'add token parsing'",
        ] {
            assert_eq!(redacted(text), text, "altered: {}", text);
        }
    }

    #[test]
    fn a_short_value_is_not_a_credential() {
        // Avoids blanking things like `token: 3` or `password: x` in prose.
        assert_eq!(redacted("token: 3"), "token: 3");
        assert_eq!(redacted("password: ab"), "password: ab");
    }

    #[test]
    fn several_secrets_in_one_text_are_all_caught() {
        let text = "sk-abcdefghijklmnopqrstuvwx and ghp_abcdefghijklmnopqrstuv";
        let (out, count) = redact_secrets(text);
        assert_eq!(count, 2, "{}", out);
        assert!(!out.contains("abcdefghijklmnopqrstuvwx"), "{}", out);
    }

    #[test]
    fn redaction_is_idempotent() {
        let once = redacted("Authorization: Bearer sk-abcdefghijklmnopqrstuvwx");
        let twice = redacted(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn has_secret_agrees_with_redaction() {
        assert!(has_secret("API_KEY=abcdefghijklmnop"));
        assert!(!has_secret("the token bucket algorithm"));
    }

    #[test]
    fn empty_and_huge_inputs_are_safe() {
        assert_eq!(redacted(""), "");
        let huge = "a".repeat(200_000);
        assert_eq!(redacted(&huge).len(), huge.len());
    }

    #[test]
    fn multibyte_text_is_not_corrupted() {
        let text = "のログ API_KEY=abcdefghijklmnop 日本語";
        let out = redacted(text);
        assert!(out.contains("のログ"), "{}", out);
        assert!(out.contains("日本語"), "{}", out);
        assert!(out.contains(REDACTED), "{}", out);
    }
}
