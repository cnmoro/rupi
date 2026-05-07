use std::sync::Arc;
use tokio::sync::Mutex;

use rupi::agent::session::{self as agent_session, ApprovalFn, AgentSession};
use rupi::cli::Cli;
use rupi::config::RupiConfig;
use rupi::provider::openai::OpenAIConfig;
use rupi::rpc::handler::RpcHandler;
use rupi::rpc::types::RpcCommand;
use rupi::skills;
use rupi::skills::Skill;

use clap::Parser;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, stdin, stdout};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    // Load config from file, then override with CLI/env
    let file_config = RupiConfig::load().unwrap_or_else(|_| {
        if cli.base_url.is_none() || cli.api_key.is_none() || cli.model.is_none() {
            eprintln!(
                "Error: No config file found at ~/.config/rupi.json and --base-url/--api-key/--model not provided.\n\
                 Create ~/.config/rupi.json with:\n\
                 {{\n  \"base_url\": \"...\",\n  \"api_key\": \"...\",\n  \"model_tag\": \"...\"\n}}"
            );
            std::process::exit(1);
        }
        RupiConfig {
            base_url: String::new(),
            api_key: String::new(),
            model_tag: String::new(),
        }
    });

    let base_url = cli.base_url.as_deref().unwrap_or(&file_config.base_url);
    let api_key = cli.api_key.as_deref().unwrap_or(&file_config.api_key);
    let model = cli.model.as_deref().unwrap_or(&file_config.model_tag);

    if base_url.is_empty() || api_key.is_empty() || model.is_empty() {
        eprintln!("Error: Missing configuration. Provide --base-url, --api-key, --model or create ~/.config/rupi.json");
        std::process::exit(1);
    }

    let openai_config = OpenAIConfig {
        base_url: base_url.to_string(),
        api_key: api_key.to_string(),
        model: model.to_string(),
        context_window: cli.context_window,
        reasoning: cli.reasoning,
    };

    // Load skills
    let loaded_skills: Vec<Skill> = rupi::config::skills_dir()
        .map(|d| skills::load_skills(&d))
        .unwrap_or_default();

    // Load context files
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let context_files = agent_session::load_context_files(&cwd);

    match cli.mode() {
        "rpc" => run_rpc_mode(openai_config, loaded_skills, context_files).await,
        "raw" => run_raw_mode(openai_config, loaded_skills, context_files).await,
        _ => run_interactive_mode(openai_config, loaded_skills, context_files, cli.disable_yolo).await,
    }
}

async fn run_rpc_mode(
    config: OpenAIConfig,
    skills: Vec<Skill>,
    context_files: Vec<agent_session::ContextFile>,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = AgentSession::from_config_with(config, cwd, skills, context_files);
    let handler = RpcHandler::new(session);

    let (output_tx, mut output_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let _writer = tokio::spawn(async move {
        let mut out = stdout();
        while let Some(line) = output_rx.recv().await {
            let _ = out.write_all(line.as_bytes()).await;
            let _ = out.flush().await;
        }
    });

    let mut stdin_reader = BufReader::new(stdin());
    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        match stdin_reader.read_line(&mut line_buf).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = line_buf.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        let command: RpcCommand = match serde_json::from_str(&trimmed) {
            Ok(cmd) => cmd,
            Err(e) => {
                let error_resp = rupi::rpc::types::RpcResponse::error(
                    None,
                    "parse",
                    format!("Failed to parse command: {}", e),
                );
                let _ = output_tx.send(error_resp.to_json_line());
                continue;
            }
        };

        handler.handle(command, output_tx.clone()).await;
    }
}

async fn run_interactive_mode(
    config: OpenAIConfig,
    skills: Vec<Skill>,
    context_files: Vec<agent_session::ContextFile>,
    disable_yolo: bool,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = Arc::new(Mutex::new(AgentSession::from_config_with(
        config, cwd, skills, context_files,
    )));

    if disable_yolo {
        let sess = session.lock().await;
        use std::io::Write;
        let approval: ApprovalFn = Arc::new(|tool_name: &str, args: &str| {
            let mut line = String::new();
            let prompt = format!("\n[APPROVAL] Allow tool '{}({})'? [y/N] ", tool_name, args);
            let _ = std::io::Write::write(&mut std::io::stdout(), prompt.as_bytes());
            let _ = std::io::stdout().flush();
            line.clear();
            match std::io::stdin().read_line(&mut line) {
                Ok(_) => line.trim().eq_ignore_ascii_case("y") || line.trim().eq_ignore_ascii_case("yes"),
                Err(_) => false,
            }
        });
        sess.set_approval_fn(Some(approval));
    }

    rupi::modes::interactive::run_interactive(session).await;
}

async fn run_raw_mode(
    config: OpenAIConfig,
    skills: Vec<Skill>,
    context_files: Vec<agent_session::ContextFile>,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = Arc::new(Mutex::new(AgentSession::from_config_with(
        config, cwd, skills, context_files,
    )));
    rupi::modes::raw::run_raw(session).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_default_mode() {
        let cli = Cli::parse_from(&["rupi"]);
        assert_eq!(cli.mode(), "interactive");
    }

    #[test]
    fn test_cli_rpc_mode() {
        let cli = Cli::parse_from(&["rupi", "--rpc"]);
        assert_eq!(cli.mode(), "rpc");
    }

    #[test]
    fn test_cli_raw_mode() {
        let cli = Cli::parse_from(&["rupi", "--raw"]);
        assert_eq!(cli.mode(), "raw");
    }
}
