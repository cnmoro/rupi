/// Resolve an API key from multiple sources.
/// Supports:
/// - Plain text keys
/// - `!command` syntax (runs command and uses stdout as key)
/// - Environment variables
pub fn resolve_api_key(input: &str) -> String {
    let input = input.trim();

    // Command-backed key: !echo mykey or !cat /path/to/key
    if let Some(cmd) = input.strip_prefix('!') {
        match std::process::Command::new("bash")
            .arg("-c")
            .arg(cmd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
        {
            Ok(output) => {
                let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !key.is_empty() {
                    return key;
                }
                // Fallback: return original if command produced no output
                input.to_string()
            }
            Err(_) => input.to_string(),
        }
    } else {
        input.to_string()
    }
}

/// Resolve API key from config, env var override, or command.
pub fn resolve_api_key_from_sources(
    config_key: &str,
    env_var_name: &str,
    cli_override: Option<&str>,
) -> String {
    // 1. CLI override takes precedence
    if let Some(key) = cli_override {
        if !key.is_empty() {
            return resolve_api_key(key);
        }
    }

    // 2. Environment variable
    if let Ok(key) = std::env::var(env_var_name) {
        if !key.is_empty() {
            return resolve_api_key(&key);
        }
    }

    // 3. Config file value
    if !config_key.is_empty() {
        return resolve_api_key(config_key);
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_key() {
        assert_eq!(resolve_api_key("sk-test"), "sk-test");
    }

    #[test]
    fn test_command_key() {
        let result = resolve_api_key("!echo sk-from-command");
        assert_eq!(result, "sk-from-command");
    }

    #[test]
    fn test_command_with_spaces() {
        let result = resolve_api_key("!echo hello-world");
        assert_eq!(result, "hello-world");
    }

    #[test]
    fn test_empty_command() {
        let result = resolve_api_key("!");
        // The command is just "!", which might fail, fallback to original
        assert!(!result.is_empty());
    }

    #[test]
    fn test_resolve_from_sources_cli() {
        let result = resolve_api_key_from_sources("config-key", "TEST_ENV_KEY", Some("cli-key"));
        assert_eq!(result, "cli-key");
    }

    #[test]
    fn test_resolve_from_sources_env() {
        std::env::set_var("TEST_AUTH_ENV", "env-key");
        let result = resolve_api_key_from_sources("config-key", "TEST_AUTH_ENV", None);
        assert_eq!(result, "env-key");
        std::env::remove_var("TEST_AUTH_ENV");
    }

    #[test]
    fn test_resolve_from_sources_config() {
        let result = resolve_api_key_from_sources("config-key", "NONEXISTENT_ENV_VAR_12345", None);
        assert_eq!(result, "config-key");
    }

    #[test]
    fn test_resolve_all_empty() {
        let result = resolve_api_key_from_sources("", "NONEXISTENT_ENV_67890", None);
        assert!(result.is_empty());
    }
}
