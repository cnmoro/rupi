/// A known provider definition.
#[derive(Debug, Clone)]
pub struct ProviderDef {
    /// Short identifier (e.g. "openai-compatible")
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
}

/// All known providers.
pub static ALL_PROVIDERS: &[ProviderDef] = &[
    ProviderDef {
        id: "openai-compatible",
        name: "OpenAI Compatible",
    },
];

/// Find a provider by ID.
pub fn find_provider(id: &str) -> Option<&'static ProviderDef> {
    ALL_PROVIDERS.iter().find(|p| p.id == id)
}

/// Get a list of provider IDs and names for display.
pub fn provider_list() -> Vec<(&'static str, &'static str)> {
    ALL_PROVIDERS.iter().map(|p| (p.id, p.name)).collect()
}
