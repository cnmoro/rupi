use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use model2vec_rs::model::StaticModel;

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

/// Chunk size in lines.
const CHUNK_LINES: usize = 50;
/// Overlap between adjacent chunks.
const CHUNK_OVERLAP: usize = 10;

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

/// Split a file's content into overlapping line-based chunks.
pub fn chunk_file(file_path: &str, content: &str, language: Option<String>) -> Vec<CodeChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    if total == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let step = CHUNK_LINES.saturating_sub(CHUNK_OVERLAP);
    let mut start = 0;

    while start < total {
        let end = (start + CHUNK_LINES).min(total);
        let chunk_content = lines[start..end].join("\n");
        chunks.push(CodeChunk {
            file_path: file_path.to_string(),
            start_line: start + 1, // 1-indexed
            end_line: end,
            content: chunk_content,
            language: language.clone(),
        });
        if end == total {
            break;
        }
        start += step;
        if start + CHUNK_LINES > total {
            start = total.saturating_sub(CHUNK_LINES);
            if chunks.last().map_or(false, |c| c.start_line == start + 1) {
                break;
            }
        }
    }

    chunks
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
        let chunks = chunk_file("test.rs", content, Some("rust".into()));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
        assert_eq!(chunks[0].file_path, "test.rs");
    }

    #[test]
    fn test_chunk_file_large() {
        let lines: Vec<String> = (0..120).map(|i| format!("line_{}", i)).collect();
        let content = lines.join("\n");
        let chunks = chunk_file("test.rs", &content, Some("rust".into()));
        // 120 lines, 50 line chunks, 10 overlap: 50-10=40 step
        // chunk 1: 0-49, chunk 2: 40-89, chunk 3: 80-119
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].end_line - chunks[0].start_line >= 49);
        assert!(chunks[1].end_line - chunks[1].start_line >= 49);
    }

    #[test]
    fn test_chunk_file_empty() {
        let chunks = chunk_file("empty.rs", "", Some("rust".into()));
        assert!(chunks.is_empty());
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
        // Test that keyword search ranks definition chunks higher
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
        // Both chunks match "authenticate"
        assert!(results.iter().any(|r| r.chunk.file_path.contains("def")));
    }
}
