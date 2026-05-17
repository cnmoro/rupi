use std::path::PathBuf;

/// A skill loaded from a SKILL.md file.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub body: String,
}

/// Parse frontmatter from markdown content.
/// Expects content between `---` markers at the start of the file.
pub fn parse_frontmatter(content: &str) -> (serde_yaml::Value, &str) {
    let content = content.trim_start();
    if !content.starts_with("---") {
        return (serde_yaml::Value::Mapping(serde_yaml::Mapping::new()), content);
    }

    let end = content[3..].find("---").map(|i| i + 3);
    match end {
        Some(end_pos) => {
            let yaml_str = &content[3..3 + end_pos - 3];
            let body = &content[3 + end_pos..];
            match serde_yaml::from_str::<serde_yaml::Value>(yaml_str) {
                Ok(frontmatter) => (frontmatter, body.trim()),
                Err(_) => (serde_yaml::Value::Mapping(serde_yaml::Mapping::new()), content),
            }
        }
        None => (serde_yaml::Value::Mapping(serde_yaml::Mapping::new()), content),
    }
}

/// Load all skills from the skills directory.
/// Skills are SKILL.md files in ~/.config/rupi/skills/.
pub fn load_skills(skills_dir: &PathBuf) -> Vec<Skill> {
    let mut skills = Vec::new();
    if !skills_dir.exists() || !skills_dir.is_dir() {
        return skills;
    }

    let entries = match std::fs::read_dir(skills_dir) {
        Ok(entries) => entries,
        Err(_) => return skills,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Check for SKILL.md inside subdirectory
            let skill_file = path.join("SKILL.md");
            if skill_file.exists() {
                if let Some(skill) = load_skill_from_file(&skill_file) {
                    skills.push(skill);
                }
            }
        } else {
            // Load any .md file at the root of the skills directory
            let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if fname.ends_with(".md") {
                if let Some(skill) = load_skill_from_file(&path) {
                    skills.push(skill);
                }
            }
        }
    }

    skills
}

/// Load a single skill from a file.
/// Uses frontmatter if present, otherwise derives name from filename.
fn load_skill_from_file(path: &PathBuf) -> Option<Skill> {
    let contents = std::fs::read_to_string(path).ok()?;
    let (frontmatter, body) = parse_frontmatter(&contents);

    // Try frontmatter name/description first
    let name = frontmatter
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            // Fallback: use filename stem as the name
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

    let description = frontmatter
        .get("description")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("")
        .to_string();

    let skill_body = if frontmatter.as_mapping().map(|m| m.is_empty()).unwrap_or(true) {
        // No frontmatter — entire file is the body
        contents.trim().to_string()
    } else {
        body.to_string()
    };

    Some(Skill {
        name,
        description,
        file_path: path.clone(),
        body: skill_body,
    })
}

/// Format skills into XML for inclusion in the system prompt.
pub fn format_skills_for_prompt(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }

    let mut result = String::from(
        "\n\n<available_skills>\n",
    );
    for skill in skills {
        result.push_str(&format!(
            "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    <location>{}</location>\n  </skill>\n",
            skill.name,
            skill.description,
            skill.file_path.display()
        ));
    }
    result.push_str("</available_skills>");
    result
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_skill(dir: &std::path::Path, name: &str, description: &str, body: &str) {
        fs::create_dir_all(dir).unwrap();
        let content = format!(
            "---\nname: {}\ndescription: {}\n---\n\n{}",
            name, description, body
        );
        fs::write(dir.join("SKILL.md"), content).unwrap();
    }

    #[test]
    fn test_parse_frontmatter_basic() {
        let content = "---\nname: test\ndescription: a test skill\n---\n\n# Skills body";
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(fm["name"].as_str().unwrap(), "test");
        assert_eq!(fm["description"].as_str().unwrap(), "a test skill");
        assert_eq!(body.trim(), "# Skills body");
    }

    #[test]
    fn test_parse_frontmatter_no_frontmatter() {
        let content = "# Just markdown\nno frontmatter here";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.as_mapping().unwrap().is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn test_parse_frontmatter_invalid_yaml() {
        let content = "---\nname: [unclosed\n---\nbody";
        let (fm, body) = parse_frontmatter(content);
        // Invalid YAML should result in empty frontmatter
        assert!(fm.as_mapping().unwrap().is_empty() || body == content);
    }

    #[test]
    fn test_load_skills_from_dir() {
        let dir = std::env::temp_dir().join(format!("rupi-skills-test-{}", std::process::id()));
        let skills_dir = dir.join("skills");
        fs::create_dir_all(&skills_dir).unwrap();

        create_test_skill(&skills_dir, "test-skill", "A test skill", "Do something");
        create_test_skill(&skills_dir.join("nested"), "nested-skill", "Nested skill", "Do nested");

        let skills = load_skills(&dir.join("skills"));
        assert!(!skills.is_empty(), "Should find at least one skill");
        assert!(skills.iter().any(|s| s.name == "test-skill"));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_skills_empty_dir() {
        let dir = std::env::temp_dir().join(format!("rupi-skills-empty-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let skills = load_skills(&dir);
        assert!(skills.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_skills_nonexistent_dir() {
        let dir = std::env::temp_dir().join("nonexistent-skills-dir-12345");
        let skills = load_skills(&dir);
        assert!(skills.is_empty());
    }

    #[test]
    fn test_format_skills_empty() {
        let result = format_skills_for_prompt(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_skills_nonempty() {
        let skills = vec![Skill {
            name: "test".into(),
            description: "A test".into(),
            file_path: PathBuf::from("/tmp/test/SKILL.md"),
            body: "body".into(),
        }];
        let result = format_skills_for_prompt(&skills);
        assert!(result.contains("<name>test</name>"));
        assert!(result.contains("<description>A test</description>"));
        assert!(result.contains("<available_skills>"));
        assert!(result.contains("</available_skills>"));
    }

    #[test]
    fn test_skill_no_frontmatter_uses_filename() {
        let dir = std::env::temp_dir().join(format!("rupi-skill-nofm-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("my-skill.md"), "# My Skill\n\nThis is the skill content.").unwrap();
        let skills = load_skills(&dir);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "my-skill");
        assert!(skills[0].body.contains("My Skill"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_multiple_md_files() {
        let dir = std::env::temp_dir().join(format!("rupi-skills-multi-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        // Create ONE.md, TWO.md, THREE.md at root
        for (name, desc) in &[("one", "first skill"), ("two", "second skill"), ("three", "third skill")] {
            let content = format!("---\nname: {}\ndescription: {}\n---\n\nbody of {}", name, desc, name);
            fs::write(dir.join(format!("{}.md", name.to_uppercase())), content).unwrap();
        }

        let skills = load_skills(&dir);
        assert_eq!(skills.len(), 3);
        assert!(skills.iter().any(|s| s.name == "one"));
        assert!(skills.iter().any(|s| s.name == "two"));
        assert!(skills.iter().any(|s| s.name == "three"));

        fs::remove_dir_all(dir).unwrap();
    }
}
