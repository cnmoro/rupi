use std::sync::Arc;
use tokio::sync::Mutex;

use rupi::agent::session::{self as agent_session, ApprovalFn, AgentSession};
use rupi::cli::Cli;
use rupi::config::RupiConfig;
use rupi::provider::openai::OpenAIConfig;
use rupi::rpc::handler::RpcHandler;
use rupi::rpc::types::RpcCommand;
use rupi::sessions;
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

    // Load config from file
    let file_config = match RupiConfig::load() {
        Ok(c) => c,
        Err(e) => {
            // If opencode provider was specified via CLI, we can work without the full config
            if cli.list_opencode_models {
                eprintln!("rupi: no config file found, but --list-opencode-models specified");
                eprintln!("rupi: create ~/.config/rupi.json with: {{\"opencode_api_key\": \"...\"}}");
                // We still need the API key — try env var
                RupiConfig {
                    base_url: None,
                    api_key: None,
                    model_tag: None,
                    opencode_api_key: std::env::var("OPENCODE_API_KEY").ok(),
                    opencode_provider: Some(cli.opencode_provider.clone()),
                }
            } else {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    };

    // Handle --list-opencode-models early
    if cli.list_opencode_models {
        let provider_id = match file_config.opencode_provider.as_deref().unwrap_or(&cli.opencode_provider) {
            "zen" => "opencode",
            _ => "opencode-go",
        };
        let models = rupi::opencode_models::fetch_opencode_models(provider_id)
            .unwrap_or_else(|e| {
                eprintln!("Error fetching opencode models: {}", e);
                std::process::exit(1);
            });
        println!("Opencode models ({}):", provider_id);
        rupi::opencode_models::print_models(&models);
        return;
    }

    // Resolve base_url, api_key, model — opencode config takes priority if set
    let base_url = if file_config.opencode_api_key.is_some() {
        file_config.effective_base_url().to_string()
    } else if let Some(url) = cli.base_url.as_deref() {
        url.to_string()
    } else {
        file_config.base_url.as_deref().unwrap_or("").to_string()
    };

    let api_key = if file_config.opencode_api_key.is_some() {
        rupi::auth::resolve_api_key(file_config.effective_api_key())
    } else if let Some(cli_key) = cli.api_key.as_deref() {
        rupi::auth::resolve_api_key(cli_key)
    } else {
        rupi::auth::resolve_api_key(file_config.api_key.as_deref().unwrap_or(""))
    };

    let model = if file_config.opencode_api_key.is_some() {
        cli.model.as_deref().unwrap_or("deepseek-v4-flash").to_string()
    } else if let Some(m) = cli.model.as_deref() {
        m.to_string()
    } else {
        file_config.model_tag.as_deref().unwrap_or("").to_string()
    };

    if api_key.is_empty() {
        eprintln!("Error: Missing API key. Provide --api-key, set opencode_api_key in config, or set OPENCODE_API_KEY env var.");
        std::process::exit(1);
    }

    if base_url.is_empty() {
        eprintln!("Error: Missing base URL. Provide --base-url or create ~/.config/rupi.json");
        std::process::exit(1);
    }

    let openai_config = OpenAIConfig {
        base_url: base_url.to_string(),
        api_key: api_key.to_string(),
        model: model.to_string(),
        context_window: cli.context_window,
        reasoning: cli.reasoning,
        timeout_secs: cli.timeout,
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

    let memory = cli.memory;
    let session_id = cli.session.as_deref();

    match cli.mode() {
        "rpc" => run_rpc_mode(openai_config, loaded_skills, context_files, session_id).await,
        "raw" => run_raw_mode(openai_config, loaded_skills, context_files, session_id).await,
        _ => run_interactive_mode(openai_config, loaded_skills, context_files, cli.disable_yolo, memory, session_id).await,
    }
}

async fn resolve_session(
    config: &OpenAIConfig,
    session_id: Option<&str>,
    cwd: &str,
    skills: &[Skill],
    context_files: &[agent_session::ContextFile],
    memory: bool,
) -> AgentSession {
    if let Some(sid) = session_id {
        let sid = sid.trim();
        if !sid.is_empty() {
            if let Some(path) = sessions::find_session_path(sid) {
                eprintln!("rupi: resuming session {}", sid);
                match AgentSession::from_session(
                    config.clone(), path, cwd.to_string(),
                    skills.to_vec(), context_files.to_vec(), memory,
                ).await {
                    Ok(session) => return session,
                    Err(e) => eprintln!("rupi: failed to resume session: {}", e),
                }
            } else {
                eprintln!("rupi: session '{}' not found, starting fresh", sid);
            }
        }
    }
    AgentSession::from_config_with(config.clone(), cwd.to_string(), skills.to_vec(), context_files.to_vec(), memory)
}

async fn run_rpc_mode(
    config: OpenAIConfig,
    skills: Vec<Skill>,
    context_files: Vec<agent_session::ContextFile>,
    session_id: Option<&str>,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = resolve_session(&config, session_id, &cwd, &skills, &context_files, false).await;
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
    memory: bool,
    session_id: Option<&str>,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = Arc::new(Mutex::new(
        resolve_session(&config, session_id, &cwd, &skills, &context_files, memory).await,
    ));

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
        sess.set_approval_fn(Some(approval)).await;
    }

    rupi::modes::interactive::run_interactive(session).await;
}

async fn run_raw_mode(
    config: OpenAIConfig,
    skills: Vec<Skill>,
    context_files: Vec<agent_session::ContextFile>,
    session_id: Option<&str>,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = Arc::new(Mutex::new(
        resolve_session(&config, session_id, &cwd, &skills, &context_files, false).await,
    ));
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
