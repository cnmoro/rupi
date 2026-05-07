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
        } else if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
            if let Some(skill) = load_skill_from_file(&path) {
                skills.push(skill);
            }
        }
    }

    skills
}

/// Load a single skill from a SKILL.md file.
fn load_skill_from_file(path: &PathBuf) -> Option<Skill> {
    let contents = std::fs::read_to_string(path).ok()?;
    let (frontmatter, body) = parse_frontmatter(&contents);

    let name = frontmatter.get("name")?.as_str()?.to_string();
    let description = frontmatter.get("description")?.as_str()?.to_string();

    if name.is_empty() || description.is_empty() {
        return None;
    }

    Some(Skill {
        name,
        description,
        file_path: path.clone(),
        body: body.to_string(),
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

/// Build a system prompt with skills included.
pub fn build_system_prompt(skills: &[Skill], cwd: &str) -> String {
    let mut prompt = format!(
        "You are rupi, an AI coding assistant.\n\
        Current date: {}\n\
        Current working directory: {}\n",
        chrono_now(),
        cwd
    );

    let skills_xml = format_skills_for_prompt(skills);
    if !skills_xml.is_empty() {
        prompt.push_str(&format!(
            "\nThe following skills are available. \
            Use them to perform specialized tasks.\n{}",
            skills_xml
        ));
    }

    prompt
}

fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    format!("{}", secs)
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
    fn test_skill_missing_description_not_loaded() {
        let dir = std::env::temp_dir().join(format!("rupi-skill-missing-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let content = "---\nname: no-desc\n---\n\nbody";
        fs::write(dir.join("SKILL.md"), content).unwrap();
        let skills = load_skills(&dir);
        assert!(skills.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }
}
