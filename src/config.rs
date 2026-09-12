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
        Self::load_from(&path)
    }

    /// Load and validate the config at an explicit path.
    ///
    /// Split out from `load` so the real resolution, parsing, and validation can be
    /// tested. Previously the only test hand-rolled a `serde_json::from_str` and
    /// never called this code at all, so every error message and every validation
    /// rule here was uncovered.
    pub fn load_from(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!(
                "Config file not found at {}. Create it with:\n{{\n  \"base_url\": \"...\",\n  \"api_key\": \"...\",\n  \"model_tag\": \"...\"\n}}",
                path.display()
            ));
        }
        let contents = std::fs::read_to_string(path)
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

    fn scratch_config(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rupi-config-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rupi.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn load_from_accepts_a_complete_config() {
        let path = scratch_config(
            "valid",
            r#"{"base_url":"https://api.test/v1","api_key":"sk-test","model_tag":"m"}"#,
        );
        let config = RupiConfig::load_from(&path).unwrap();
        assert_eq!(config.base_url.as_deref(), Some("https://api.test/v1"));
        assert_eq!(config.api_key.as_deref(), Some("sk-test"));
        assert_eq!(config.model_tag.as_deref(), Some("m"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn load_from_reports_a_missing_file_with_a_usable_message() {
        let missing = std::env::temp_dir().join("rupi-config-does-not-exist.json");
        let _ = std::fs::remove_file(&missing);
        let error = RupiConfig::load_from(&missing).unwrap_err();
        assert!(error.contains("Config file not found"), "{}", error);
        // The message has to show the user what to write, not just complain.
        assert!(error.contains("base_url"), "{}", error);
        assert!(error.contains("api_key"), "{}", error);
        assert!(error.contains("model_tag"), "{}", error);
    }

    #[test]
    fn load_from_rejects_malformed_json() {
        let path = scratch_config("malformed", "{not json");
        let error = RupiConfig::load_from(&path).unwrap_err();
        assert!(error.contains("Failed to parse"), "{}", error);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn load_from_validates_every_required_field() {
        let cases = [
            (
                r#"{"base_url":"","api_key":"k","model_tag":"m"}"#,
                "base_url",
            ),
            (r#"{"api_key":"k","model_tag":"m"}"#, "base_url"),
            (
                r#"{"base_url":"u","api_key":"","model_tag":"m"}"#,
                "api_key",
            ),
            (r#"{"base_url":"u","model_tag":"m"}"#, "api_key"),
            (
                r#"{"base_url":"u","api_key":"k","model_tag":""}"#,
                "model_tag",
            ),
            (r#"{"base_url":"u","api_key":"k"}"#, "model_tag"),
        ];
        for (body, expected) in cases {
            let path = scratch_config("invalid", body);
            let error = RupiConfig::load_from(&path).unwrap_err();
            assert!(
                error.contains(expected),
                "{} did not report {}: {}",
                body,
                expected,
                error
            );
            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }
    }

    #[test]
    fn an_opencode_key_bypasses_the_standard_fields() {
        let path = scratch_config("opencode", r#"{"opencode_api_key":"oc-test"}"#);
        let config = RupiConfig::load_from(&path).unwrap();
        assert_eq!(config.opencode_api_key.as_deref(), Some("oc-test"));
        assert!(config.base_url.is_none());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
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
