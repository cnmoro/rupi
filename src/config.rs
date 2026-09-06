use std::path::PathBuf;

/// Configuration loaded from ~/.config/rupi.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RupiConfig {
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub model_tag: Option<String>,
    #[serde(rename = "opencode_api_key", default)]
    pub opencode_api_key: Option<String>,
    #[serde(rename = "opencode_provider", default)]
    pub opencode_provider: Option<String>,
}

impl RupiConfig {
    pub fn load() -> Result<Self, String> {
        let path = config_path().ok_or_else(|| "Could not determine home directory".to_string())?;
        if !path.exists() {
            return Err(format!(
                "Config file not found at {}. Create it with:\n{{\n  \"base_url\": \"...\",\n  \"api_key\": \"...\",\n  \"model_tag\": \"...\"\n}}",
                path.display()
            ));
        }
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let config: RupiConfig = serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        // If opencode_api_key is set, skip standard field validation
        if config.opencode_api_key.is_some() {
            return Ok(config);
        }

        if config.base_url.as_deref().unwrap_or("").is_empty() {
            return Err("base_url cannot be empty in config".into());
        }
        if config.api_key.as_deref().unwrap_or("").is_empty() {
            return Err("api_key cannot be empty in config".into());
        }
        if config.model_tag.as_deref().unwrap_or("").is_empty() {
            return Err("model_tag cannot be empty in config".into());
        }
        Ok(config)
    }

    /// Resolve the effective API key — uses opencode_api_key if present.
    pub fn effective_api_key(&self) -> &str {
        self.opencode_api_key
            .as_deref()
            .unwrap_or(self.api_key.as_deref().unwrap_or(""))
    }

    /// Resolve the effective base URL — uses opencode URL if opencode_api_key is set.
    pub fn effective_base_url(&self) -> &str {
        if self.opencode_api_key.is_some() {
            match self.opencode_provider.as_deref() {
                Some("zen") => "https://opencode.ai/zen/v1",
                _ => "https://opencode.ai/zen/go/v1",
            }
        } else {
            self.base_url.as_deref().unwrap_or("")
        }
    }

    /// Get the opencode provider variant ("go" or "zen").
    pub fn opencode_variant(&self) -> &str {
        match self.opencode_provider.as_deref() {
            Some("zen") => "zen",
            _ => "go",
        }
    }
}

/// Return the path to ~/.config/rupi.json
pub fn config_path() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".config").join("rupi.json"))
}

/// Return the path to ~/.config/rupi/skills/
pub fn skills_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".config").join("rupi").join("skills"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_config_path() {
        let path = config_path();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.ends_with(".config/rupi.json"));
    }

    #[test]
    fn test_skills_dir() {
        let path = skills_dir();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.ends_with(".config/rupi/skills"));
    }

    #[test]
    fn test_load_valid_config() {
        let dir = std::env::temp_dir().join(format!("rupi-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("rupi.json");
        fs::write(
            &config_path,
            r#"{"base_url":"https://api.example.com","api_key":"sk-test","model_tag":"gpt-4"}"#,
        )
        .unwrap();

        // Temporarily redirect config_path to our test file
        let contents = fs::read_to_string(&config_path).unwrap();
        let config: RupiConfig = serde_json::from_str(&contents).unwrap();
        assert_eq!(config.base_url.as_deref(), Some("https://api.example.com"));
        assert_eq!(config.api_key.as_deref(), Some("sk-test"));
        assert_eq!(config.model_tag.as_deref(), Some("gpt-4"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_partial_config_succeeds_with_defaults() {
        // With only base_url, the other fields default to None
        let config: RupiConfig =
            serde_json::from_str(r#"{"base_url":"https://example.com"}"#).unwrap();
        assert_eq!(config.base_url.as_deref(), Some("https://example.com"));
        assert_eq!(config.api_key, None);
        assert_eq!(config.model_tag, None);
        assert_eq!(config.opencode_api_key, None);
    }

    #[test]
    fn test_config_empty_validation() {
        let config = RupiConfig {
            base_url: Some("".into()),
            api_key: Some("sk-test".into()),
            model_tag: Some("gpt-4".into()),
            opencode_api_key: None,
            opencode_provider: None,
        };
        assert!(config.base_url.as_deref().unwrap_or("").is_empty());
    }

    #[test]
    fn test_opencode_config_overrides() {
        let config = RupiConfig {
            base_url: Some("https://old.example.com".into()),
            api_key: Some("old-key".into()),
            model_tag: Some("old-model".into()),
            opencode_api_key: Some("oc_key".into()),
            opencode_provider: Some("go".into()),
        };
        assert_eq!(config.effective_api_key(), "oc_key");
        assert_eq!(config.effective_base_url(), "https://opencode.ai/zen/go/v1");
        assert_eq!(config.opencode_variant(), "go");
    }

    #[test]
    fn test_opencode_config_omits_standard_fields() {
        // Simulate a config that has ONLY opencode fields (what the user would write)
        let config = RupiConfig {
            base_url: None,
            api_key: None,
            model_tag: None,
            opencode_api_key: Some("oc_key".into()),
            opencode_provider: Some("zen".into()),
        };
        assert_eq!(config.effective_api_key(), "oc_key");
        assert_eq!(config.effective_base_url(), "https://opencode.ai/zen/v1");
        assert_eq!(config.opencode_variant(), "zen");
    }

    #[test]
    fn test_opencode_config_defaults() {
        let config = RupiConfig {
            base_url: Some("https://old.example.com".into()),
            api_key: Some("old-key".into()),
            model_tag: Some("old-model".into()),
            opencode_api_key: None,
            opencode_provider: None,
        };
        assert_eq!(config.effective_api_key(), "old-key");
        assert_eq!(config.effective_base_url(), "https://old.example.com");
        assert_eq!(config.opencode_variant(), "go");
    }

    #[test]
    fn test_opencode_config_default_provider_is_go() {
        let config = RupiConfig {
            base_url: None,
            api_key: None,
            model_tag: None,
            opencode_api_key: Some("k".into()),
            opencode_provider: None,
        };
        assert_eq!(config.effective_base_url(), "https://opencode.ai/zen/go/v1");
    }
}
