use clap::Parser;

/// rupi - RPC coding agent for OpenAI-compatible APIs.
#[derive(Parser, Debug, Clone)]
#[command(name = "rupi", version, about)]
pub struct Cli {
    /// Base URL for the OpenAI-compatible API (overrides config file)
    #[arg(long, env = "RUPI_BASE_URL")]
    pub base_url: Option<String>,

    /// API key (overrides config file)
    #[arg(long = "api-key", env = "RUPI_API_KEY")]
    pub api_key: Option<String>,

    /// Model tag to use (overrides config file)
    #[arg(long, env = "RUPI_MODEL")]
    pub model: Option<String>,

    /// Context window size
    #[arg(long, default_value_t = 128000)]
    pub context_window: u64,

    /// Maximum execution time in seconds (default: no limit)
    #[arg(long, default_value_t = 0)]
    pub timeout: u64,

    /// Enable reasoning model
    #[arg(long, default_value_t = false)]
    pub reasoning: bool,

    /// Disable yolo mode: prompt for approval on every tool execution
    #[arg(long)]
    pub disable_yolo: bool,

    /// Enable persistent memory: the agent reads/writes ~/.config/rupi/MEMORY.md
    #[arg(long)]
    pub memory: bool,

    /// Run in RPC mode (JSONL protocol over stdin/stdout)
    #[arg(long)]
    pub rpc: bool,

    /// Run in raw mode (print SSE chunks as JSON lines to stdout)
    #[arg(long)]
    pub raw: bool,

    /// Resume a session by ID (loads ~/.config/rupi_sessions/<ID>.jsonl)
    #[arg(long)]
    pub session: Option<String>,

    /// Directory holding session transcripts (default: ~/.config/rupi_sessions)
    #[arg(long, env = "RUPI_SESSIONS_DIR")]
    pub sessions_dir: Option<String>,

    /// Ceiling for the bash tool's per-command timeout, in seconds
    #[arg(long, env = "RUPI_BASH_TIMEOUT_MAX", default_value_t = 120)]
    pub bash_timeout_max: u64,

    /// Timeout applied when the model does not ask for one, in seconds
    #[arg(long, env = "RUPI_BASH_TIMEOUT_DEFAULT", default_value_t = 30)]
    pub bash_timeout_default: u64,

    /// Seconds of silence mid-stream before the provider request is failed
    #[arg(long, env = "RUPI_STREAM_IDLE_TIMEOUT", default_value_t = 120)]
    pub stream_idle_timeout: u64,

    /// List all saved sessions with IDs and metadata
    #[arg(long)]
    pub list_sessions: bool,

    /// List available models from Opencode (requires opencode_api_key in config)
    #[arg(long)]
    pub list_opencode_models: bool,

    /// Opencode provider variant: "go" (default) or "zen"
    #[arg(long, default_value = "go")]
    pub opencode_provider: String,
}

impl Cli {
    pub fn mode(&self) -> &str {
        if self.rpc {
            "rpc"
        } else if self.raw {
            "raw"
        } else {
            "interactive"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mode_is_interactive() {
        let cli = Cli::parse_from(&["rupi"]);
        assert_eq!(cli.mode(), "interactive");
    }

    #[test]
    fn test_rpc_mode() {
        let cli = Cli::parse_from(&["rupi", "--rpc"]);
        assert_eq!(cli.mode(), "rpc");
    }

    #[test]
    fn test_raw_mode() {
        let cli = Cli::parse_from(&["rupi", "--raw"]);
        assert_eq!(cli.mode(), "raw");
    }

    #[test]
    fn test_cli_overrides() {
        let cli = Cli::parse_from(&[
            "rupi",
            "--base-url", "https://custom.api.com",
            "--api-key", "sk-custom",
            "--model", "custom-model",
        ]);
        assert_eq!(cli.base_url.as_deref(), Some("https://custom.api.com"));
        assert_eq!(cli.api_key.as_deref(), Some("sk-custom"));
        assert_eq!(cli.model.as_deref(), Some("custom-model"));
    }

    #[test]
    fn test_default_values() {
        let cli = Cli::parse_from(&["rupi"]);
        assert_eq!(cli.context_window, 128000);
        assert!(!cli.reasoning);
        assert!(!cli.rpc);
        assert!(!cli.raw);
    }

    #[test]
    fn test_env_var_names() {
        let cli = Cli::parse_from(&[
            "rupi",
            "--base-url", "https://api.example.com",
            "--api-key", "sk-test",
            "--model", "gpt-4",
        ]);
        assert_eq!(cli.base_url.as_deref(), Some("https://api.example.com"));
        assert_eq!(cli.api_key.as_deref(), Some("sk-test"));
        assert_eq!(cli.model.as_deref(), Some("gpt-4"));
    }
}
