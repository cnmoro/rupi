use std::fmt;

#[derive(Debug)]
pub enum AgentError {
    Io(std::io::Error),
    Serde(serde_json::Error),
    Http(reqwest::Error),
    Api { message: String, status_code: u16 },
    Config(String),
    Timeout,
    Cancelled,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::Io(e) => write!(f, "IO error: {}", e),
            AgentError::Serde(e) => write!(f, "JSON error: {}", e),
            AgentError::Http(e) => write!(f, "HTTP error: {}", e),
            AgentError::Api {
                message,
                status_code,
            } => {
                write!(f, "API error ({}): {}", status_code, message)
            }
            AgentError::Config(msg) => write!(f, "Config error: {}", msg),
            AgentError::Timeout => write!(f, "Request timed out"),
            AgentError::Cancelled => write!(f, "Operation cancelled"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<std::io::Error> for AgentError {
    fn from(e: std::io::Error) -> Self {
        AgentError::Io(e)
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::Serde(e)
    }
}

impl From<reqwest::Error> for AgentError {
    fn from(e: reqwest::Error) -> Self {
        AgentError::Http(e)
    }
}

/// Phrases a provider uses when the request does not fit the model's window.
///
/// Matched on the response body, which is the only place this arrives: every
/// OpenAI-compatible server words it differently, and several answer 400 with no
/// machine-readable code at all.
const CONTEXT_OVERFLOW_MARKERS: [&str; 9] = [
    "context_length_exceeded",
    "context length",
    "maximum context",
    "context window",
    "prompt is too long",
    "too many tokens",
    "exceeds the available context",
    "reduce the length of the messages",
    "input length and `max_tokens` exceed context limit",
];

impl AgentError {
    /// Whether the provider refused the request because it did not fit.
    ///
    /// Compaction decides when to run from an ESTIMATE of the token count. An
    /// estimate is sometimes wrong, and the provider then answers 400. Without
    /// this the turn ended there, which stops an unattended run for a reason the
    /// agent can fix by itself.
    pub fn is_context_overflow(&self) -> bool {
        let AgentError::Api { message, .. } = self else {
            return false;
        };
        let text = message.to_lowercase();
        CONTEXT_OVERFLOW_MARKERS
            .iter()
            .any(|marker| text.contains(&marker.to_lowercase()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api(message: &str) -> AgentError {
        AgentError::Api {
            message: message.to_string(),
            status_code: 400,
        }
    }

    #[test]
    fn the_real_refusals_are_recognized() {
        // Bodies as the providers actually word them.
        let bodies = [
            r#"{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 131204 tokens. Please reduce the length of the messages.","code":"context_length_exceeded"}}"#,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 215000 tokens > 200000 maximum"}}"#,
            r#"{"error":{"message":"input length and `max_tokens` exceed context limit: 7800 + 1000 > 8192"}}"#,
            r#"{"error":{"message":"the request exceeds the available context size, try increasing it"}}"#,
            r#"{"object":"error","message":"This model's maximum context length is 8192 tokens","code":400}"#,
        ];
        for body in bodies {
            assert!(
                api(body).is_context_overflow(),
                "not recognized as an overflow: {}",
                body
            );
        }
    }

    #[test]
    fn other_failures_are_not_mistaken_for_an_overflow() {
        // Compacting on an unrelated error throws away context for nothing.
        let bodies = [
            r#"{"error":{"message":"Incorrect API key provided","code":"invalid_api_key"}}"#,
            r#"{"error":{"message":"Rate limit reached for gpt-4 in organization org-x","code":"rate_limit_exceeded"}}"#,
            r#"{"error":{"message":"The model `gpt-9` does not exist","code":"model_not_found"}}"#,
            r#"{"error":{"message":"tool call arguments were not valid json"}}"#,
        ];
        for body in bodies {
            assert!(
                !api(body).is_context_overflow(),
                "wrongly read as an overflow: {}",
                body
            );
        }
        assert!(!AgentError::Timeout.is_context_overflow());
        assert!(!AgentError::Cancelled.is_context_overflow());
        assert!(!AgentError::Config("context length".into()).is_context_overflow());
    }
}
