//! Task anchor: the structural guarantee that the request which started a
//! session survives compaction.
//!
//! Before this module the originating prompt was ordinary history. `find_cut_point`
//! keeps only the recent tail, so after a few hundred tool calls the prompt sat far
//! outside that tail and compaction deleted it. It came back only if the summarizer
//! chose to restate it — a request to a model, not an invariant. Each further
//! compaction re-summarized the restatement and lost more of the original wording.
//!
//! The anchor makes the prompt a first-class value the session owns. Compaction
//! re-emits it verbatim below the checkpoint, so the exact bytes the user typed are
//! present in every request no matter how many summaries came before.

/// Opening tag of a rendered anchor block. Also the detection marker: a message
/// whose content starts with this is an anchor and never a user turn to answer.
pub const ANCHOR_OPEN: &str = "<active-task>";

/// Closing tag of a rendered anchor block.
pub const ANCHOR_CLOSE: &str = "</active-task>";

/// Framing that stops the model from treating a re-emitted anchor as new work.
///
/// Without it the anchor reads as a fresh user turn arriving after a checkpoint
/// that says the work is done, and the model restarts the task from the top.
const ANCHOR_PREAMBLE: &str =
    "This is a verbatim restatement of the request that started this session. \
It is NOT a new request and NOT a repeat instruction. Do not restart work that the conversation \
above records as finished. Treat the workspace and the tool results as authoritative, and inspect \
them instead of assuming earlier narration is still current. Continue from where the work stands.";

/// Maximum characters of the original request kept in a rendered anchor.
///
/// A pasted log or a large file body can make the prompt itself the thing that
/// blows the context. The head and the tail carry the request; the middle of an
/// oversized paste does not.
pub const MAX_ANCHOR_CHARS: usize = 6000;

/// Separator between the originating request and a later user instruction.
pub const LATER_INSTRUCTION_SEPARATOR: &str = "\n\n--- later instruction from the user ---\n";

/// Trim an over-long request, keeping the opening and closing text.
///
/// Public because the session clamps the stored anchor too, not only the rendered
/// one. A session with many follow-ups would otherwise grow the stored string
/// without bound. Head-and-tail is the right cut for both: the head is the request
/// that started the work, and the tail is the most recent instruction.
pub fn clamp(text: &str) -> String {
    if text.len() <= MAX_ANCHOR_CHARS {
        return text.to_string();
    }
    let head = MAX_ANCHOR_CHARS * 7 / 10;
    let tail = MAX_ANCHOR_CHARS - head;
    // Snap both cuts to character boundaries so multi-byte input cannot panic.
    let head_end = floor_boundary(text, head);
    let tail_start = ceil_boundary(text, text.len() - tail);
    format!(
        "{}\n... [anchor truncated: {} chars] ...\n{}",
        &text[..head_end],
        tail_start - head_end,
        &text[tail_start..]
    )
}

/// Largest character boundary at or below `index`.
fn floor_boundary(text: &str, index: usize) -> usize {
    let mut i = index.min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest character boundary at or above `index`.
fn ceil_boundary(text: &str, index: usize) -> usize {
    let mut i = index.min(text.len());
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Render the anchor block that carries the originating request.
///
/// `round` is the ordinal of this emission, used only to tell the model that a
/// repeated block is the same task and not a second copy of the work.
pub fn render(request: &str, emission: u32) -> String {
    format!(
        "{}\nemission: {}\n{}\n\n--- original request ---\n{}\n{}",
        ANCHOR_OPEN,
        emission,
        ANCHOR_PREAMBLE,
        clamp(request.trim()),
        ANCHOR_CLOSE
    )
}

/// Whether a message body is a rendered anchor block.
pub fn is_anchor(content: &str) -> bool {
    content.trim_start().starts_with(ANCHOR_OPEN)
}

/// Recover the original request text from a rendered anchor block.
///
/// Resume reads history back from disk with no in-memory state, so the anchor has
/// to be readable out of its own rendering.
pub fn extract_request(content: &str) -> Option<String> {
    if !is_anchor(content) {
        return None;
    }
    const SEPARATOR: &str = "--- original request ---\n";
    let start = content.find(SEPARATOR)? + SEPARATOR.len();
    let end = content.rfind(ANCHOR_CLOSE).unwrap_or(content.len());
    if end <= start {
        return None;
    }
    Some(content[start..end].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_contains_the_request_verbatim() {
        let block = render("refactor the parser and add tests", 1);
        assert!(block.starts_with(ANCHOR_OPEN));
        assert!(block.ends_with(ANCHOR_CLOSE));
        assert!(block.contains("refactor the parser and add tests"));
    }

    #[test]
    fn render_marks_the_block_as_not_a_new_request() {
        let block = render("do the thing", 3);
        assert!(block.contains("NOT a new request"));
        assert!(block.contains("emission: 3"));
    }

    #[test]
    fn is_anchor_detects_a_rendered_block() {
        assert!(is_anchor(&render("x", 1)));
        assert!(!is_anchor("please fix the build"));
        assert!(!is_anchor("[Compacted conversation history]\n## Summary"));
    }

    #[test]
    fn extract_request_round_trips() {
        let original = "fix the disappearing prompt during compaction";
        let block = render(original, 2);
        assert_eq!(extract_request(&block).as_deref(), Some(original));
    }

    #[test]
    fn extract_request_round_trips_multiline() {
        let original = "line one\nline two\n\nline four";
        let block = render(original, 1);
        assert_eq!(extract_request(&block).as_deref(), Some(original));
    }

    #[test]
    fn extract_request_rejects_a_plain_message() {
        assert_eq!(extract_request("just a normal message"), None);
    }

    #[test]
    fn clamp_keeps_head_and_tail_of_an_oversized_request() {
        let long = format!("{}{}", "A".repeat(MAX_ANCHOR_CHARS), "ZZZEND");
        let block = render(&long, 1);
        assert!(block.contains("anchor truncated"));
        assert!(block.contains("ZZZEND"));
        assert!(block.contains("AAAA"));
        assert!(block.len() < MAX_ANCHOR_CHARS + 1000);
    }

    #[test]
    fn clamp_does_not_split_multibyte_characters() {
        // Every char is 3 bytes, so a naive byte cut lands mid-character.
        let long = "の".repeat(MAX_ANCHOR_CHARS);
        let block = render(&long, 1);
        assert!(block.contains("anchor truncated"));
        assert!(block.contains("の"));
    }

    #[test]
    fn clamp_keeps_the_original_request_and_the_latest_instruction() {
        let combined = format!(
            "ORIGINAL REQUEST{}{}{}LATEST INSTRUCTION",
            "\n filler".repeat(2000),
            LATER_INSTRUCTION_SEPARATOR,
            "\n more filler".repeat(2000)
        );
        let clamped = clamp(&combined);
        assert!(clamped.starts_with("ORIGINAL REQUEST"));
        assert!(clamped.ends_with("LATEST INSTRUCTION"));
        assert!(clamped.contains("anchor truncated"));
    }

    #[test]
    fn short_request_is_not_truncated() {
        let block = render("short", 1);
        assert!(!block.contains("anchor truncated"));
    }
}
