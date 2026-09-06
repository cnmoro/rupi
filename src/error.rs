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
