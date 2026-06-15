use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use model2vec_rs::model::StaticModel;
use tree_sitter::{Language, Parser};

use crate::error::AgentError;

/// A single chunk of code from a file.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub language: Option<String>,
}

/// A search result with score.
#[derive(Debug, Clone)]
pub struct CodeSearchResult {
    pub chunk: CodeChunk,
    pub score: f32,
}

/// Code search index built from a directory.
pub struct CodeSearchIndex {
    chunks: Vec<CodeChunk>,
    embeddings: Array2<f32>,
}

static MODEL: Mutex<Option<StaticModel>> = Mutex::new(None);

fn get_model() -> Result<std::sync::MutexGuard<'static, Option<StaticModel>>, AgentError> {
    let mut guard = MODEL.lock().map_err(|_| AgentError::Config("Model lock poisoned".into()))?;
    if guard.is_none() {
        *guard = Some(
            StaticModel::from_pretrained(
                "minishlab/potion-code-16M",
                None::<&str>,
                None,
                None::<&str>,
            )
            .map_err(|e| {
                AgentError::Config(
                    format!(
                        "Failed to load code search model: {}. \
                         The model will be downloaded on first use from HuggingFace Hub.",
                        e
                    ),
                )
            })?,
        );
    }
    Ok(guard)
}

/// Known code file extensions and their language names.
fn language_for_ext(ext: &str) -> Option<&'static str> {
    match ext {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" | "jsx" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "rb" => Some("ruby"),
        "c" | "h" => Some("c"),
        "cpp" | "hpp" | "cc" | "cxx" => Some("cpp"),
        "cs" => Some("csharp"),
        "swift" => Some("swift"),
        "kt" | "kts" => Some("kotlin"),
        "scala" => Some("scala"),
        "php" => Some("php"),
        "r" => Some("r"),
        "m" => Some("matlab"),
        "lua" => Some("lua"),
        "sh" | "bash" | "zsh" => Some("shell"),
        "sql" => Some("sql"),
        "html" => Some("html"),
        "css" | "scss" | "less" => Some("css"),
        "toml" => Some("toml"),
        "yaml" | "yml" => Some("yaml"),
        "json" => Some("json"),
        "md" => Some("markdown"),
        "dockerfile" | "Dockerfile" => Some("dockerfile"),
        _ => None,
    }
}

/// Get a tree-sitter Language for a given language name.
/// Returns None for unsupported languages (falls back to line-based chunking).
fn tree_sitter_language(lang: &str) -> Option<Language> {
    match lang {
        "rust" => Some(Language::new(tree_sitter_rust::LANGUAGE)),
        "python" => Some(Language::new(tree_sitter_python::LANGUAGE)),
        "javascript" => Some(Language::new(tree_sitter_javascript::LANGUAGE)),
        "typescript" => Some(Language::new(tree_sitter_typescript::LANGUAGE_TYPESCRIPT)),
        "go" => Some(Language::new(tree_sitter_go::LANGUAGE)),
        "c" => Some(Language::new(tree_sitter_c::LANGUAGE)),
        "cpp" => Some(Language::new(tree_sitter_cpp::LANGUAGE)),
        "java" => Some(Language::new(tree_sitter_java::LANGUAGE)),
        "ruby" => Some(Language::new(tree_sitter_ruby::LANGUAGE)),
        "csharp" => Some(Language::new(tree_sitter_c_sharp::LANGUAGE)),
        _ => None,
    }
}

/// Standard file extensions for code files.
const CODE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "rb", "c", "h", "cpp", "hpp", "cc",
    "cs", "swift", "kt", "kts", "scala", "php", "r", "lua", "sh", "bash", "zsh", "sql",
];

/// Directories to always skip.
const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "build", "dist", ".next", "venv", ".venv",
    "__pycache__", ".cache", ".svelte-kit", ".nuxt", "out", ".idea", ".vscode",
    ".claude", "coverage", "vendor", ".gradle", ".tox", ".eggs", "egg-info",
    "site-packages", ".bzr", ".hg", ".svn", "CVS",
];

/// Desired chunk length in characters (matching seemb's 1500).
const DESIRED_CHUNK_LENGTH: usize = 1500;

/// Walk files in a directory matching code extensions.
pub fn walk_code_files(path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_dir(path, path, &mut files);
    files
}

fn walk_dir(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();
        if file_name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if !SKIP_DIRS.contains(&file_name.as_str()) {
                walk_dir(root, &path, files);
            }
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if CODE_EXTENSIONS.contains(&ext) {
                    files.push(path);
                }
            }
        }
    }
}

/// A chunk boundary (byte offsets into source).
#[derive(Debug, Clone)]
struct ChunkBoundary {
    start: usize,
    end: usize,
}

/// Recursively merge child nodes up to desired_length.
/// Mirrors seemb's _merge_node_inner algorithm.
fn merge_node_inner(node: &tree_sitter::Node, desired_length: usize) -> Vec<ChunkBoundary> {
    if node.child_count() == 0 {
        return vec![ChunkBoundary {
            start: node.start_byte(),
            end: node.end_byte(),
        }];
    }

    let mut groups: Vec<ChunkBoundary> = Vec::new();
    let mut index = 0;
    let children: Vec<_> = (0..node.child_count()).filter_map(|i| node.child(i)).collect();

    while index < children.len() {
        let child = &children[index];
        let start = child.start_byte();
        let mut end = child.end_byte();
        let mut length = end - start;
        index += 1;

        // If this single chunk is too large, recurse to split it
        if length > desired_length {
            groups.extend(merge_node_inner(child, desired_length));
            continue;
        }

        // Merge adjacent children while they fit
        while index < children.len() {
            let next = &children[index];
            let next_len = next.end_byte() - next.start_byte();
            if length + next_len > desired_length {
                break;
            }
            end = next.end_byte();
            length = end - start;
            index += 1;
        }

        groups.push(ChunkBoundary { start, end });
    }

    groups
}

/// Merge adjacent chunks up to desired_length.
/// Mirrors seemb's _merge_adjacent_chunks.
fn merge_adjacent_chunks(chunks: &[ChunkBoundary], desired_length: usize) -> Vec<ChunkBoundary> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut current_start = chunks[0].start;
    let mut current_end = chunks[0].end;
    let mut current_length = current_end - current_start;

    for group in chunks.iter().skip(1) {
        let length = group.end - group.start;

        if current_length + length > desired_length {
            merged.push(ChunkBoundary { start: current_start, end: current_end });
            current_start = group.start;
            current_end = group.end;
            current_length = length;
        } else {
            current_end = group.end;
            current_length += length;
        }
    }

    merged.push(ChunkBoundary { start: current_start, end: current_end });
    merged
}

/// AST-aware chunking using tree-sitter.
/// Mirrors seemb's chunk() function: parse, recursively merge nodes, return byte boundaries.
fn chunk_ast(source: &str, language: &Language, desired_length: usize) -> Vec<ChunkBoundary> {
    let stripped = source.trim();
    if stripped.is_empty() {
        return Vec::new();
    }

    let mut parser = Parser::new();
    parser.set_language(language).ok();
    let Some(tree) = parser.parse(source.as_bytes(), None) else {
        return Vec::new();
    };

    let root = tree.root_node();
    let raw_chunks = merge_node_inner(&root, desired_length);
    merge_adjacent_chunks(&raw_chunks, desired_length)
}

/// Line-based chunking fallback for unsupported languages.
/// Mirrors seemb's chunk_lines().
fn chunk_lines(source: &str, desired_length: usize) -> Vec<ChunkBoundary> {
    let stripped = source.trim();
    if stripped.is_empty() {
        return Vec::new();
    }

    let mut lines: Vec<ChunkBoundary> = Vec::new();
    let mut index = 0;
    for line in source.split('\n') {
        let line_len = line.len();
        lines.push(ChunkBoundary {
            start: index,
            end: index + line_len,
        });
        // +1 for the '\n' we split on
        index += line_len + 1;
    }

    merge_adjacent_chunks(&lines, desired_length)
}

/// Convert byte offset to 1-indexed line number.
fn byte_to_line(source: &str, byte_offset: usize) -> usize {
    source[..byte_offset.min(source.len())].lines().count().max(1)
}

/// Split a file's content into chunks using AST-aware or line-based chunking.
pub fn chunk_file(file_path: &str, content: &str, language: Option<String>) -> Vec<CodeChunk> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Try AST chunking if language is supported
    let boundaries = language.as_deref()
        .and_then(tree_sitter_language)
        .map(|lang| chunk_ast(content, &lang, DESIRED_CHUNK_LENGTH))
        .unwrap_or_else(|| chunk_lines(content, DESIRED_CHUNK_LENGTH));

    if boundaries.is_empty() {
        return Vec::new();
    }

    let source_len = content.len();
    boundaries.into_iter().map(|b| {
        let end = b.end.min(source_len);
        CodeChunk {
            file_path: file_path.to_string(),
            start_line: byte_to_line(content, b.start),
            end_line: byte_to_line(content, end.saturating_sub(1)),
            content: content[b.start..end].to_string(),
            language: language.clone(),
        }
    }).collect()
}

/// Find all chunks in a directory.
pub fn index_path(path: &Path) -> Vec<CodeChunk> {
    let files = walk_code_files(path);
    let mut all_chunks = Vec::new();
    for file_path in &files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if content.len() > 1_000_000 {
            continue; // skip files > 1MB
        }
        let rel = file_path
            .strip_prefix(path)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();
        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let language = language_for_ext(ext).map(|s| s.to_string());
        let chunks = chunk_file(&rel, &content, language);
        all_chunks.extend(chunks);
    }
    all_chunks
}

impl CodeSearchIndex {
    /// Build an index from a directory.
    pub fn build(path: &Path) -> Result<Self, AgentError> {
        let guard = get_model()?;
        let model = guard.as_ref().ok_or_else(|| AgentError::Config("Model not loaded".into()))?;
        let chunks = index_path(path);
        if chunks.is_empty() {
            return Err(AgentError::Config("No code files found to index".into()));
        }

        // Embed all chunks
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let raw = model.encode(&texts);
        let n = raw.len();
        let dim = if n == 0 { 0 } else { raw[0].len() };
        let flat: Vec<f32> = raw.into_iter().flatten().collect();
        let embeddings = if n > 0 && dim > 0 {
            Array2::from_shape_vec((n, dim), flat)
                .map_err(|e| AgentError::Config(format!("Failed to build embedding matrix: {}", e)))?
        } else {
            Array2::from_shape_vec((0, 0), vec![])
                .map_err(|e| AgentError::Config(format!("Empty index: {}", e)))?
        };

        Ok(CodeSearchIndex { chunks, embeddings })
    }

    /// Search with a natural language query.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<CodeSearchResult> {
        if self.chunks.is_empty() || query.trim().is_empty() {
            return Vec::new();
        }

        let guard = match get_model() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let model = match guard.as_ref() {
            Some(m) => m,
            None => return Vec::new(),
        };

        let query_emb = model.encode_single(query);
        if query_emb.is_empty() {
            return Vec::new();
        }

        let n = self.embeddings.nrows();
        let dim = self.embeddings.ncols();
        if dim == 0 || query_emb.len() != dim {
            return Vec::new();
        }

        // Compute cosine similarity scores
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let dot: f32 = self.embeddings.row(i).iter().zip(query_emb.iter()).map(|(a, b)| a * b).sum();
                (i, dot) // embeddings and query are already L2-normalized by model2vec
            })
            .collect();

        // Apply ranking boosts
        for (i, score) in scored.iter_mut() {
            let chunk = &self.chunks[*i];

            // Definition boost: lines starting with fn/def/class/struct/impl/trait
            let has_def = chunk.content.lines().any(|l| {
                let t = l.trim();
                t.starts_with("fn ")
                    || t.starts_with("def ")
                    || t.starts_with("class ")
                    || t.starts_with("struct ")
                    || t.starts_with("impl ")
                    || t.starts_with("trait ")
                    || t.starts_with("enum ")
                    || t.starts_with("type ")
                    || t.starts_with("interface ")
                    || t.starts_with("function ")
                    || t.starts_with("public ")
                    || t.starts_with("async ")
                    || t.starts_with("pub ")
            });
            if has_def {
                *score *= 1.15;
            }

            // Noise penalty for test files
            if chunk.file_path.contains("/test") || chunk.file_path.contains("/tests/") || chunk.file_path.starts_with("test") {
                *score *= 0.85;
            }

            // Noise penalty for examples
            if chunk.file_path.contains("/example") || chunk.file_path.contains("/examples/") {
                *score *= 0.9;
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let k = top_k.min(scored.len());
        scored[..k]
            .iter()
            .map(|&(i, score)| CodeSearchResult {
                chunk: self.chunks[i].clone(),
                score,
            })
            .collect()
    }

    /// Total number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
}

/// Format search results for display.
pub fn format_results(query: &str, results: &[CodeSearchResult]) -> String {
    if results.is_empty() {
        return format!("No results found for: {}\n", query);
    }

    let mut out = format!("Search results for: {}\n\n", query);
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. {}:{}-{} (score: {:.4})\n",
            i + 1,
            r.chunk.file_path,
            r.chunk.start_line,
            r.chunk.end_line,
            r.score
        ));
        // Show first 5 lines of the chunk
        for line in r.chunk.content.lines().take(5) {
            out.push_str(&format!("  {}\n", line));
        }
        if r.chunk.content.lines().count() > 5 {
            out.push_str("  ...\n");
        }
        out.push('\n');
    }
    out
}

/// Extract keyword tokens from a string for BM25-like matching.
/// Simple: split on non-alphanumeric (except underscore), lowercase, filter short words.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_lowercase())
        .collect()
}

/// Simple BM25 scoring for keyword-based code search.
/// Used as a fallback when the model isn't loaded yet.
pub fn search_keyword(chunks: &[CodeChunk], query: &str, top_k: usize) -> Vec<CodeSearchResult> {
    if chunks.is_empty() || query.trim().is_empty() {
        return Vec::new();
    }

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return Vec::new();
    }

    // Build per-chunk token sets
    let n = chunks.len();
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);

    for (i, chunk) in chunks.iter().enumerate() {
        let tokens = tokenize(&chunk.content);
        let total = tokens.len() as f32;
        if total == 0.0 {
            continue;
        }

        // Count query token occurrences in this chunk
        let mut score = 0.0f32;
        for qt in &query_tokens {
            let tf = tokens.iter().filter(|t| *t == qt).count() as f32;
            if tf > 0.0 {
                // TF with normalization
                score += (1.0 + tf.log10()) / (1.0 + total.log10());
            }
        }

        if score > 0.0 {
            scored.push((i, score));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(scored.len());
    scored[..k]
        .iter()
        .map(|&(i, score)| CodeSearchResult {
            chunk: chunks[i].clone(),
            score,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_file_small() {
        let content = "line1\nline2\nline3\n";
        let chunks = chunk_file("test.txt", content, None);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert!(chunks[0].file_path == "test.txt");
    }

    #[test]
    fn test_chunk_file_empty() {
        let chunks = chunk_file("empty.rs", "", Some("rust".into()));
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_file_rust_ast() {
        let content = r#"
fn main() {
    println!("hello");
}

fn helper() -> i32 {
    42
}

struct Config {
    name: String,
}

impl Config {
    fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }
}
"#;
        let chunks = chunk_file("test.rs", content, Some("rust".into()));
        // AST chunking should produce meaningful chunks
        assert!(!chunks.is_empty());
        // Should contain the function definitions
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("fn main"));
        assert!(all_content.contains("fn helper"));
    }

    #[test]
    fn test_chunk_file_python_ast() {
        let content = r#"
def hello():
    print("hello")

def world():
    print("world")

class Foo:
    def bar(self):
        return 42
"#;
        let chunks = chunk_file("test.py", content, Some("python".into()));
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("def hello"));
        assert!(all_content.contains("class Foo"));
    }

    #[test]
    fn test_chunk_file_fallback_to_lines() {
        // Unsupported language (markdown) should fall back to line-based chunking
        let content = "# Title\n\nSome text here\n\nMore content\n";
        let chunks = chunk_file("test.md", content, Some("markdown".into()));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_file_large_ast() {
        // Generate a file with many large functions - should split into multiple chunks
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!(
                "fn func_{}(x: i32, y: i32, z: i32, a: i32, b: i32) -> i32 {{\n    let result = x + y + z + a + b + {};\n    println!(\"computed {{}}\", result);\n    result\n}}\n\n",
                i, i
            ));
        }
        let chunks = chunk_file("test.rs", &content, Some("rust".into()));
        // Should produce multiple chunks for a large file
        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());
    }

    #[test]
    fn test_language_for_ext() {
        assert_eq!(language_for_ext("rs"), Some("rust"));
        assert_eq!(language_for_ext("py"), Some("python"));
        assert_eq!(language_for_ext("js"), Some("javascript"));
        assert_eq!(language_for_ext("nonexistent"), None);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("hello world, foo_bar!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"foo_bar".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // too short
    }

    #[test]
    fn test_walk_code_files() {
        let dir = std::env::temp_dir().join("rupi-code-search-test-walk");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.join("lib.py"), "def foo(): pass").unwrap();
        std::fs::write(dir.join("readme.md"), "# readme").unwrap();
        std::fs::write(dir.join("notes.txt"), "notes").unwrap();

        let files = walk_code_files(&dir);
        assert!(files.iter().any(|f| f.ends_with("main.rs")));
        assert!(files.iter().any(|f| f.ends_with("lib.py")));
        assert!(!files.iter().any(|f| f.ends_with("readme.md"))); // not a code ext
        assert!(!files.iter().any(|f| f.ends_with("notes.txt"))); // not a code ext
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_walk_code_files_skips_noise_dirs() {
        let dir = std::env::temp_dir().join("rupi-code-search-test-noise");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("node_modules")).unwrap();
        std::fs::write(dir.join("node_modules/pkg.js"), "const x = 1;").unwrap();
        std::fs::write(dir.join("main.js"), "const x = 1;").unwrap();

        let files = walk_code_files(&dir);
        assert!(!files.iter().any(|f| f.to_string_lossy().contains("node_modules")));
        assert!(files.iter().any(|f| f.ends_with("main.js")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_search_keyword() {
        let chunks = vec![
            CodeChunk {
                file_path: "auth.rs".into(),
                start_line: 1,
                end_line: 3,
                content: "fn login(username: &str, password: &str) -> bool { true }".into(),
                language: Some("rust".into()),
            },
            CodeChunk {
                file_path: "db.rs".into(),
                start_line: 1,
                end_line: 3,
                content: "fn connect() -> Connection { Connection::new() }".into(),
                language: Some("rust".into()),
            },
        ];

        let results = search_keyword(&chunks, "login", 5);
        assert_eq!(results.len(), 1);
        assert!(results[0].chunk.file_path.contains("auth"));
    }

    #[test]
    fn test_format_results_empty() {
        let formatted = format_results("test", &[]);
        assert!(formatted.contains("No results"));
    }

    #[test]
    fn test_format_results_non_empty() {
        let results = vec![CodeSearchResult {
            chunk: CodeChunk {
                file_path: "src/main.rs".into(),
                start_line: 1,
                end_line: 5,
                content: "fn main() {\n    println!(\"hello\");\n}\n".into(),
                language: Some("rust".into()),
            },
            score: 0.95,
        }];
        let formatted = format_results("main function", &results);
        assert!(formatted.contains("src/main.rs"));
        assert!(formatted.contains("fn main()"));
    }

    #[test]
    fn test_index_path_empty_dir() {
        let dir = std::env::temp_dir().join("rupi-code-search-test-empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let chunks = index_path(&dir);
        assert!(chunks.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_index_path_with_files() {
        let dir = std::env::temp_dir().join("rupi-code-search-test-index");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.rs"), "fn greet() { println!(\"hi\"); }").unwrap();
        std::fs::write(dir.join("utils.py"), "def helper(): pass").unwrap();

        let chunks = index_path(&dir);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.file_path.ends_with("hello.rs")));
        assert!(chunks.iter().any(|c| c.file_path.ends_with("utils.py")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_definition_boost_scoring() {
        let chunks = vec![
            CodeChunk {
                file_path: "def.rs".into(),
                start_line: 1,
                end_line: 3,
                content: "fn authenticate(user: &str) -> bool {\n    user == \"admin\"\n}".into(),
                language: Some("rust".into()),
            },
            CodeChunk {
                file_path: "use.rs".into(),
                start_line: 1,
                end_line: 3,
                content: "let is_auth = authenticate(\"admin\");".into(),
                language: Some("rust".into()),
            },
        ];

        let results = search_keyword(&chunks, "authenticate", 5);
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.chunk.file_path.contains("def")));
    }
}
