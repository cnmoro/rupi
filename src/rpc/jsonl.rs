use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

/// Serialize a value as a JSONL record (JSON + newline).
pub fn serialize_json_line(value: &impl serde::Serialize) -> String {
    let mut s = serde_json::to_string(value).expect("serialization should not fail");
    s.push('\n');
    s
}

/// Read JSONL records from an async reader, calling `on_line` for each parsed value.
pub async fn read_jsonl_lines<R, F, T>(reader: R, mut on_line: F)
where
    R: AsyncRead + Unpin + Send + 'static,
    F: FnMut(T),
    T: serde::de::DeserializeOwned,
{
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        match buf_reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    if let Ok(val) = serde_json::from_str::<T>(trimmed) {
                        on_line(val);
                    }
                }
            }
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_json_line() {
        let line = serialize_json_line(&serde_json::json!({"type": "ping"}));
        assert_eq!(line, "{\"type\":\"ping\"}\n");
    }

    #[test]
    fn test_serialize_with_unicode() {
        let line = serialize_json_line(&serde_json::json!({"text": "a\u{2028}b\u{2029}c"}));
        assert!(line.contains("a\u{2028}b\u{2029}c"));
        assert!(line.ends_with('\n'));
        let parsed: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["text"], "a\u{2028}b\u{2029}c");
    }

    #[tokio::test]
    async fn test_read_jsonl_lines() {
        let input = "{\"a\":1}\n{\"b\":2}\n";
        let reader = input.as_bytes();
        let mut results = Vec::new();
        read_jsonl_lines::<_, _, serde_json::Value>(reader, |val| {
            results.push(val);
        })
        .await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["a"], 1);
        assert_eq!(results[1]["b"], 2);
    }

    #[tokio::test]
    async fn test_read_skips_empty_lines() {
        let input = "{\"a\":1}\n\n{\"b\":2}\n";
        let reader = input.as_bytes();
        let mut count = 0;
        read_jsonl_lines::<_, _, serde_json::Value>(reader, |_| {
            count += 1;
        })
        .await;
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_read_handles_crlf() {
        let input = "{\"a\":1}\r\n{\"b\":2}\r\n";
        let reader = input.as_bytes();
        let mut results = Vec::new();
        read_jsonl_lines::<_, _, serde_json::Value>(reader, |val| {
            results.push(val);
        })
        .await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_read_final_line_no_lf() {
        let input = "{\"a\":1}";
        let reader = input.as_bytes();
        let mut results = Vec::new();
        read_jsonl_lines::<_, _, serde_json::Value>(reader, |val| {
            results.push(val);
        })
        .await;
        assert_eq!(results.len(), 1);
    }
}
