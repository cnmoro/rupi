use std::collections::HashMap;

/// A provider entry from the models.dev API.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelsDevProvider {
    pub id: String,
    pub name: String,
    pub api: Option<String>,
    pub models: HashMap<String, ModelsDevModel>,
}

/// A model entry from the models.dev API.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelsDevModel {
    pub id: String,
    pub name: Option<String>,
    pub cost: Option<ModelCost>,
    pub limit: Option<ModelLimit>,
    pub reasoning: Option<bool>,
    pub tool_call: Option<bool>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelLimit {
    pub context: Option<u64>,
    pub output: Option<u64>,
}

/// Fetch models from the models.dev API, with local snapshot fallback.
pub fn fetch_opencode_models(provider_id: &str) -> Result<Vec<ModelsDevModel>, String> {
    // Try to fetch from models.dev API first (with timeout)
    let result = try_fetch_from_api(provider_id);

    match result {
        Ok(models) => {
            if models.is_empty() {
                eprintln!("rupi: models.dev returned empty list for '{}', using snapshot", provider_id);
                Ok(parse_snapshot(provider_id))
            } else {
                Ok(models)
            }
        }
        Err(e) => {
            eprintln!("rupi: failed to fetch models from models.dev ({}), using snapshot", e);
            Ok(parse_snapshot(provider_id))
        }
    }
}

fn try_fetch_from_api(provider_id: &str) -> Result<Vec<ModelsDevModel>, String> {
    let url = "https://models.dev/api.json";
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("HTTP client: {}", e))?;

    let resp = client
        .get(url)
        .send()
        .map_err(|e| format!("HTTP request: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let providers: HashMap<String, ModelsDevProvider> = resp
        .json()
        .map_err(|e| format!("JSON parse: {}", e))?;

    match providers.get(provider_id) {
        Some(p) => {
            let mut models: Vec<ModelsDevModel> = p.models.values().cloned().collect();
            models.sort_by(|a, b| a.id.cmp(&b.id));
            Ok(models)
        }
        None => Err(format!("Provider '{}' not found in models.dev", provider_id)),
    }
}

/// Parse the bundled snapshot for opencode providers.
fn parse_snapshot(provider_id: &str) -> Vec<ModelsDevModel> {
    let snapshot = match provider_id {
        "opencode-go" => OPENCODE_GO_SNAPSHOT,
        "opencode" => OPENCODE_ZEN_SNAPSHOT,
        _ => return vec![],
    };

    let mut models: Vec<ModelsDevModel> = snapshot
        .iter()
        .map(|(id, name, input_cost, output_cost, ctx, reasoning, tool_call)| ModelsDevModel {
            id: id.to_string(),
            name: Some(name.to_string()),
            cost: Some(ModelCost {
                input: *input_cost,
                output: *output_cost,
            }),
            limit: Some(ModelLimit {
                context: Some(*ctx),
                output: Some(4096),
            }),
            reasoning: Some(*reasoning),
            tool_call: Some(*tool_call),
            status: None,
        })
        .collect();

    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

/// Format and print models to stdout.
pub fn print_models(models: &[ModelsDevModel]) {
    for m in models {
        let name = m.name.as_deref().unwrap_or(&m.id);
        let cost_str = m.cost.as_ref().map(|c| format!("  ${:.2}/$M in, ${:.2}/$M out", c.input, c.output)).unwrap_or_default();
        let ctx_str = m.limit.as_ref().and_then(|l| l.context).map(|c| format!("  ctx: {}", c)).unwrap_or_default();
        let reasoning_str = if m.reasoning.unwrap_or(false) { "  reasoning" } else { "" };
        let tool_str = if m.tool_call.unwrap_or(false) { "  tools" } else { "" };
        println!("{}{}{}{}{}", m.id, cost_str, ctx_str, reasoning_str, tool_str);
    }
}

// ── Bundled snapshots ──────────────────────────────────────────────────────

/// Opencode Go models (provider id: opencode-go)
const OPENCODE_GO_SNAPSHOT: &[(&str, &str, f64, f64, u64, bool, bool)] = &[
    ("deepseek-v4-flash", "DeepSeek V4 Flash", 0.30, 1.20, 204800, false, true),
    ("deepseek-v4-pro", "DeepSeek V4 Pro", 2.00, 8.00, 204800, false, true),
    ("glm-5", "GLM-5", 0.30, 1.20, 128000, false, true),
    ("glm-5-flash", "GLM-5 Flash", 0.10, 0.40, 128000, false, true),
    ("kimi-k2.5", "Kimi K2.5", 2.00, 8.00, 128000, false, true),
    ("minimax-m2.7", "MiniMax M2.7", 2.00, 8.00, 1048576, false, true),
    ("mimo-v2.5-pro", "MiMo V2.5 Pro", 2.00, 8.00, 128000, false, true),
];

/// Opencode Zen models (provider id: opencode)
const OPENCODE_ZEN_SNAPSHOT: &[(&str, &str, f64, f64, u64, bool, bool)] = &[
    ("claude-haiku-4-5", "Claude Haiku 4.5", 1.00, 5.00, 200000, false, true),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", 3.00, 15.00, 200000, false, true),
    ("gemini-2-5-flash", "Gemini 2.5 Flash", 0.15, 0.60, 1048576, false, true),
    ("gemini-3-1-pro", "Gemini 3.1 Pro", 2.00, 10.00, 2097152, false, true),
    ("gpt-5-1-codex-max", "GPT 5.1 Codex Max", 5.00, 25.00, 131072, true, true),
    ("gpt-5-nano", "GPT 5 Nano", 0.50, 2.50, 131072, false, true),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_go_snapshot() {
        let models = parse_snapshot("opencode-go");
        assert!(!models.is_empty(), "Should have Go models");
        assert!(models.iter().any(|m| m.id.contains("deepseek")));
    }

    #[test]
    fn test_parse_zen_snapshot() {
        let models = parse_snapshot("opencode");
        assert!(!models.is_empty(), "Should have Zen models");
        assert!(models.iter().any(|m| m.id.contains("gpt")));
    }

    #[test]
    fn test_parse_empty_snapshot() {
        let models = parse_snapshot("nonexistent");
        assert!(models.is_empty());
    }

    #[test]
    fn test_print_models_doesnt_crash() {
        let models = parse_snapshot("opencode-go");
        // Just verify it doesn't panic
        print_models(&models);
    }
}
