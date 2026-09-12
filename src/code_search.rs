use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use model2vec_rs::model::StaticModel;
use ndarray::Array2;
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
    boosts: Vec<f32>,
}

static MODEL: Mutex<Option<StaticModel>> = Mutex::new(None);

fn get_model() -> Result<std::sync::MutexGuard<'static, Option<StaticModel>>, AgentError> {
    let mut guard = MODEL
        .lock()
        .map_err(|_| AgentError::Config("Model lock poisoned".into()))?;
    if guard.is_none() {
        *guard = Some(
            StaticModel::from_pretrained(
                "minishlab/potion-code-16M",
                None::<&str>,
                None,
                None::<&str>,
            )
            .map_err(|e| {
                AgentError::Config(format!(
                    "Failed to load code search model: {}. \
                         The model will be downloaded on first use from HuggingFace Hub.",
                    e
                ))
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
    "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "rb", "c", "h", "cpp", "hpp", "cc", "cs",
    "swift", "kt", "kts", "scala", "php", "r", "lua", "sh", "bash", "zsh", "sql",
];

/// Directories to always skip.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "build",
    "dist",
    ".next",
    "venv",
    ".venv",
    "__pycache__",
    ".cache",
    ".svelte-kit",
    ".nuxt",
    "out",
    ".idea",
    ".vscode",
    ".claude",
    "coverage",
    "vendor",
    ".gradle",
    ".tox",
    ".eggs",
    "egg-info",
    "site-packages",
    ".bzr",
    ".hg",
    ".svn",
    "CVS",
];

/// Desired chunk length in characters (matching seemb's 1500).
const DESIRED_CHUNK_LENGTH: usize = 1500;

/// Walk files in a directory matching code extensions.
pub fn walk_code_files(path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_dir(path, &mut files);
    files
}

fn walk_dir(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();
        if file_name.starts_with('.') {
            continue;
        }
        if entry.file_type().map_or(true, |kind| kind.is_symlink()) {
            continue;
        }
        if path.is_dir() {
            if !SKIP_DIRS.contains(&file_name.as_str()) {
                walk_dir(&path, files);
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

/// Deepest AST nesting this will descend into.
///
/// Recursion here follows the syntax tree, so nesting depth is attacker-chosen: a
/// 30 KB file of nested brackets drove it past the 2 MiB stack of the blocking
/// worker that runs `search_code`. A Rust stack overflow aborts the process — it is
/// not a catchable panic — so a file placed in any indexed repository could take
/// down the whole agent, every session in it, from a tool call that only reads code.
///
/// Past this depth the node is emitted whole rather than split. A chunk larger than
/// the target is a worse search result; a dead process is a worse outcome than that.
const MAX_AST_DEPTH: usize = 200;

/// Recursively merge child nodes up to desired_length.
/// Mirrors seemb's _merge_node_inner algorithm.
fn merge_node_inner(node: &tree_sitter::Node, desired_length: usize) -> Vec<ChunkBoundary> {
    merge_node_depth(node, desired_length, 0)
}

fn merge_node_depth(
    node: &tree_sitter::Node,
    desired_length: usize,
    depth: usize,
) -> Vec<ChunkBoundary> {
    if depth >= MAX_AST_DEPTH {
        return vec![ChunkBoundary {
            start: node.start_byte(),
            end: node.end_byte(),
        }];
    }
    if node.child_count() == 0 {
        return vec![ChunkBoundary {
            start: node.start_byte(),
            end: node.end_byte(),
        }];
    }

    let mut groups: Vec<ChunkBoundary> = Vec::new();
    let mut index = 0;
    let children: Vec<_> = (0..node.child_count())
        .filter_map(|i| node.child(i))
        .collect();

    while index < children.len() {
        let child = &children[index];
        let start = child.start_byte();
        let mut end = child.end_byte();
        let mut length = end - start;
        index += 1;

        // If this single chunk is too large, recurse to split it
        if length > desired_length {
            groups.extend(merge_node_depth(child, desired_length, depth + 1));
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
            merged.push(ChunkBoundary {
                start: current_start,
                end: current_end,
            });
            current_start = group.start;
            current_end = group.end;
            current_length = length;
        } else {
            current_end = group.end;
            current_length += length;
        }
    }

    merged.push(ChunkBoundary {
        start: current_start,
        end: current_end,
    });
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
    source[..byte_offset.min(source.len())]
        .lines()
        .count()
        .max(1)
}

/// Split a file's content into chunks using AST-aware or line-based chunking.
pub fn chunk_file(file_path: &str, content: &str, language: Option<String>) -> Vec<CodeChunk> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Try AST chunking if language is supported
    let boundaries = language
        .as_deref()
        .and_then(tree_sitter_language)
        .map(|lang| chunk_ast(content, &lang, DESIRED_CHUNK_LENGTH))
        .unwrap_or_else(|| chunk_lines(content, DESIRED_CHUNK_LENGTH));

    if boundaries.is_empty() {
        return Vec::new();
    }

    let source_len = content.len();
    boundaries
        .into_iter()
        .map(|b| {
            let end = b.end.min(source_len);
            CodeChunk {
                file_path: file_path.to_string(),
                start_line: byte_to_line(content, b.start),
                end_line: byte_to_line(content, end.saturating_sub(1)),
                content: content[b.start..end].to_string(),
                language: language.clone(),
            }
        })
        .collect()
}

/// Find all chunks in a directory.
pub fn index_path(path: &Path) -> Vec<CodeChunk> {
    let files = walk_code_files(path);
    let mut all_chunks = Vec::new();
    for file_path in &files {
        if std::fs::metadata(file_path).map_or(true, |m| m.len() > 1_000_000) {
            continue;
        }
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

#[derive(Debug, PartialEq, Eq)]
/// Identity of a file revision, used to decide whether its chunks can be reused.
///
/// Modification time and length alone are not enough. A checkout, `tar -x`,
/// `rsync -a`, and `cp -p` all restore the recorded mtime, and coarse-granularity
/// filesystems collide on it anyway — so two different revisions of the same size
/// compared equal, and search then served content that no longer existed while
/// missing what did.
///
/// Status change time closes that. Unlike mtime it is set by the kernel on every
/// write and cannot be restored by the tools above. The inode catches a file
/// replaced wholesale. Both come from the same `metadata` call, so this costs no
/// extra I/O — which matters, because the whole point of the stamp is to avoid
/// reading the file.
///
/// Residual: an edit that lands inside the filesystem's own timestamp granularity
/// still looks unchanged. Catching that needs the content itself, which is the
/// read this exists to avoid; git handles the same case with its racy-timestamp
/// rule. The realistic cases — a checkout, `rsync -a`, `cp -p`, `tar -x` — all
/// happen far outside that window and are caught.
struct FileStamp {
    modified: Option<std::time::SystemTime>,
    len: u64,
    /// Seconds and nanoseconds of the last status change, on unix.
    changed: Option<(i64, i64)>,
    /// Inode, on unix.
    inode: Option<u64>,
}

/// Read the unix-only parts of a stamp.
#[cfg(unix)]
fn unix_identity(metadata: &std::fs::Metadata) -> (Option<(i64, i64)>, Option<u64>) {
    use std::os::unix::fs::MetadataExt;
    (
        Some((metadata.ctime(), metadata.ctime_nsec())),
        Some(metadata.ino()),
    )
}

/// Off unix there is no status change time, so only the creation time is used.
///
/// That identifies a file replaced by a new one, which is what most editors and
/// archive tools do. It does not identify a same-size edit made in place with the
/// modification time restored afterwards, so that case stays undetected on Windows.
/// The content digest cannot help: it prevents needless re-embedding once a read
/// happens, and here no read is prompted.
#[cfg(not(unix))]
fn unix_identity(metadata: &std::fs::Metadata) -> (Option<(i64, i64)>, Option<u64>) {
    // `created` rather than a platform-specific accessor: it is stable everywhere
    // and needs no per-target API I cannot compile here to verify.
    let created = metadata
        .created()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|since| (since.as_secs() as i64, since.subsec_nanos() as i64));
    (created, None)
}

struct CachedFile {
    stamp: FileStamp,
    /// Digest of the content these chunks were built from.
    ///
    /// The stamp decides whether to LOOK at a file; this decides whether to
    /// re-embed it. Status change time moves on `chmod`, `chown`, `touch`, and on
    /// every file of a tree restored by `rsync -a` or `tar -x` — none of which
    /// alter a byte. Without this, each of those forced a full re-embed of the
    /// whole repository, which is a worse problem than the stale results the
    /// stricter stamp was added to prevent.
    content_hash: String,
    chunks: Vec<CodeChunk>,
    embeddings: Vec<Vec<f32>>,
}

/// Digest of a file's content, used to decide whether embeddings can be kept.
fn content_digest(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Default)]
struct CachedRepository {
    files: BTreeMap<PathBuf, CachedFile>,
    index: Option<Arc<CodeSearchIndex>>,
}

/// One repository's cache, with its own lock.
///
/// The whole map used to sit behind a single mutex held for the entire refresh —
/// the directory walk, every file read, and the embedding call. A search of a
/// small repository then waited on an unrelated large one being indexed, which
/// matters because several sessions share this process. The map lock is now held
/// only long enough to find the slot.
struct CacheSlot {
    /// When this repository was last searched, for eviction order. Read while
    /// holding the map lock, so it lives outside the per-repository mutex.
    last_used: std::sync::atomic::AtomicU64,
    repo: Mutex<CachedRepository>,
}

impl Default for CacheSlot {
    fn default() -> Self {
        CacheSlot {
            last_used: std::sync::atomic::AtomicU64::new(0),
            repo: Mutex::new(CachedRepository::default()),
        }
    }
}

/// Ticks once per search, so the cache can order its entries by recency.
static CACHE_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl CachedRepository {
    fn refresh(
        &mut self,
        root: &Path,
        mut encode: impl FnMut(&[String]) -> Result<Vec<Vec<f32>>, AgentError>,
    ) -> Result<Arc<CodeSearchIndex>, AgentError> {
        let mut paths = walk_code_files(root);
        paths.sort();
        let mut changed = false;
        let old_len = self.files.len();
        self.files
            .retain(|path, _| paths.binary_search(path).is_ok());
        changed |= old_len != self.files.len();
        if changed {
            self.index = None;
        }
        for path in paths {
            let metadata = std::fs::metadata(&path)?;
            let (status_changed, inode) = unix_identity(&metadata);
            let stamp = FileStamp {
                modified: metadata.modified().ok(),
                len: metadata.len(),
                changed: status_changed,
                inode,
            };
            // On unix the stamp carries a status change time, which moves on every
            // write, so a matching stamp is trustworthy and costs no read.
            //
            // Off unix there is no such field: a same-size edit made in place with
            // the modification time restored compares equal, which is the bug this
            // whole stamp exists to prevent. There the content is read and digested
            // instead, bounded by the same size limit the indexer already applies —
            // a read per file is far cheaper than serving results for code that is
            // no longer on disk.
            let stamp_matches = self
                .files
                .get(&path)
                .is_some_and(|file| file.stamp == stamp);
            if stamp_matches && cfg!(unix) {
                continue;
            }
            let content = if stamp.len > 1_000_000 {
                String::new()
            } else {
                std::fs::read_to_string(&path).unwrap_or_default()
            };
            let content_hash = content_digest(&content);

            // The stamp changed but the bytes did not: keep the work. This is the
            // ordinary outcome of a permission change or a tree restore, and
            // re-embedding there costs the whole repository for nothing.
            if let Some(existing) = self.files.get_mut(&path) {
                if existing.content_hash == content_hash {
                    existing.stamp = stamp;
                    continue;
                }
            }

            let chunks = if content.is_empty() {
                Vec::new()
            } else {
                let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy();
                let language = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .and_then(language_for_ext)
                    .map(str::to_owned);
                chunk_file(&relative, &content, language)
            };
            let texts: Vec<_> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
            let embeddings = if texts.is_empty() {
                Vec::new()
            } else {
                encode(&texts)?
            };
            if embeddings.len() != chunks.len() {
                return Err(AgentError::Config(
                    "Embedding count does not match chunks".into(),
                ));
            }
            self.index = None;
            self.files.insert(
                path,
                CachedFile {
                    stamp,
                    content_hash,
                    chunks,
                    embeddings,
                },
            );
            changed = true;
        }
        if changed || self.index.is_none() {
            let chunks: Vec<_> = self
                .files
                .values()
                .flat_map(|file| file.chunks.clone())
                .collect();
            let rows: Vec<_> = self
                .files
                .values()
                .flat_map(|file| file.embeddings.iter())
                .collect();
            let dim = rows.first().map_or(0, |row| row.len());
            let embeddings = Array2::from_shape_vec(
                (rows.len(), dim),
                rows.into_iter().flatten().copied().collect(),
            )
            .map_err(|e| AgentError::Config(format!("Invalid embedding matrix: {e}")))?;
            let boosts = chunks.iter().map(ranking_boost).collect();
            self.index = Some(Arc::new(CodeSearchIndex {
                chunks,
                embeddings,
                boosts,
            }));
        }
        Ok(self.index.as_ref().expect("index initialized").clone())
    }
}

/// A bounded cache keyed by canonical repository path. Unchanged files keep
/// their chunks and embeddings; edits, additions, and deletions refresh the index.
pub fn cached_index(path: &Path) -> Result<Arc<CodeSearchIndex>, AgentError> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<CacheSlot>>>> = OnceLock::new();
    let root = path.canonicalize()?;

    // Hold the map lock only to find the slot, then release it before any I/O or
    // embedding work, so an unrelated repository is never blocked behind this one.
    let slot = {
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| AgentError::Config("Search cache lock poisoned".into()))?;
        if cache.len() >= 4 && !cache.contains_key(&root) {
            // Evict the least recently used, not whichever bucket the hasher
            // happens to yield first. Dropping the repository that was just
            // searched sends the next query back into a full re-embed of it.
            let victim = cache
                .iter()
                .min_by_key(|(_, slot)| slot.last_used.load(std::sync::atomic::Ordering::Relaxed))
                .map(|(key, _)| key.clone());
            if let Some(key) = victim {
                cache.remove(&key);
            }
        }
        let slot = cache.entry(root.clone()).or_default().clone();
        slot.last_used.store(
            CACHE_CLOCK.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
        slot
    };

    let mut repo = slot
        .repo
        .lock()
        .map_err(|_| AgentError::Config("Search cache lock poisoned".into()))?;
    repo.refresh(&root, |texts| {
        let guard = get_model()?;
        let model = guard
            .as_ref()
            .ok_or_else(|| AgentError::Config("Model not loaded".into()))?;
        Ok(model.encode(texts))
    })
}

fn ranking_boost(chunk: &CodeChunk) -> f32 {
    let has_def = chunk.content.lines().any(|line| {
        let line = line.trim();
        [
            "fn ",
            "def ",
            "class ",
            "struct ",
            "impl ",
            "trait ",
            "enum ",
            "type ",
            "interface ",
            "function ",
            "public ",
            "async ",
            "pub ",
        ]
        .iter()
        .any(|prefix| line.starts_with(prefix))
    });
    let mut boost = if has_def { 1.15 } else { 1.0 };
    if chunk.file_path.contains("/test") || chunk.file_path.starts_with("test") {
        boost *= 0.85;
    }
    if chunk.file_path.contains("/example") {
        boost *= 0.9;
    }
    boost
}

fn select_top_k(scored: &mut Vec<(usize, f32)>, k: usize) {
    let compare = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if k < scored.len() {
        scored.select_nth_unstable_by(k, compare);
        scored.truncate(k);
    }
    scored.sort_by(compare);
}

impl CodeSearchIndex {
    /// Build an index from a directory.
    pub fn build(path: &Path) -> Result<Self, AgentError> {
        let guard = get_model()?;
        let model = guard
            .as_ref()
            .ok_or_else(|| AgentError::Config("Model not loaded".into()))?;
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
            Array2::from_shape_vec((n, dim), flat).map_err(|e| {
                AgentError::Config(format!("Failed to build embedding matrix: {}", e))
            })?
        } else {
            Array2::from_shape_vec((0, 0), vec![])
                .map_err(|e| AgentError::Config(format!("Empty index: {}", e)))?
        };

        let boosts = chunks.iter().map(ranking_boost).collect();
        Ok(CodeSearchIndex {
            chunks,
            embeddings,
            boosts,
        })
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
                let dot: f32 = self
                    .embeddings
                    .row(i)
                    .iter()
                    .zip(query_emb.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (i, dot) // embeddings and query are already L2-normalized by model2vec
            })
            .collect();

        for (i, score) in &mut scored {
            *score *= self.boosts[*i];
        }
        select_top_k(&mut scored, top_k);

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

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
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

    select_top_k(&mut scored, top_k);
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
    fn cached_index_reuses_embeddings_and_refreshes_changes() {
        let root = std::env::temp_dir().join(format!("rupi-cache-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("one.rs"), "fn one() {}\n").unwrap();
        std::fs::write(root.join("two.rs"), "fn two() {}\n").unwrap();
        let mut cache = CachedRepository::default();
        let mut encoded = 0;
        let mut encode = |texts: &[String]| -> Result<Vec<Vec<f32>>, AgentError> {
            encoded += texts.len();
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        };
        let first = cache.refresh(&root, &mut encode).unwrap();
        let second = cache.refresh(&root, &mut encode).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        std::fs::write(root.join("one.rs"), "fn changed_name() {}\n").unwrap();
        let third = cache.refresh(&root, &mut encode).unwrap();
        assert!(!Arc::ptr_eq(&second, &third));
        assert!(third
            .chunks
            .iter()
            .any(|c| c.content.contains("changed_name")));
        std::fs::remove_file(root.join("two.rs")).unwrap();
        let fourth = cache.refresh(&root, &mut encode).unwrap();
        assert_eq!(fourth.len(), 1);
        assert_eq!(encoded, 3, "only the edited file should be re-embedded");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn top_k_matches_full_sort() {
        let all = vec![(0, 0.3), (1, 0.9), (2, 0.9), (3, -0.1), (4, 0.5)];
        for k in 0..=all.len() + 1 {
            let mut scored = all.clone();
            select_top_k(&mut scored, k);
            let mut expected = all.clone();
            expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            expected.truncate(k);
            assert_eq!(scored, expected);
        }
    }

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
    fn unchanged_content_keeps_its_embeddings() {
        // A permission change, a `touch`, or a tree restored by `rsync -a` moves
        // the status change time on every file without altering a byte. The stamp
        // is deliberately strict enough to notice, so the digest has to stop that
        // from re-embedding the whole repository for nothing.
        let body = "fn alpha() {}\nfn beta() {}\n";
        assert_eq!(content_digest(body), content_digest(body));
        assert_ne!(
            content_digest(body),
            content_digest("fn alpha() {}\nfn gamma() {}\n")
        );
        // Same length, different bytes — the case a length check cannot see.
        assert_ne!(content_digest("aaaa"), content_digest("aaab"));
        assert_eq!(content_digest(""), content_digest(""));
    }

    #[test]
    #[cfg(unix)]
    fn a_same_size_edit_with_a_restored_mtime_is_not_missed() {
        // The realistic case: a checkout, `tar -x`, `rsync -a` or `cp -p` restores
        // the recorded mtime, and a coarse filesystem collides on it anyway. Two
        // revisions of the same size then compared equal, so search served content
        // that no longer existed and missed what did.
        let dir = std::env::temp_dir().join(format!("rupi-stale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.rs");

        std::fs::write(&path, "fn alpha_marker() {}\n").unwrap();
        let first = std::fs::metadata(&path).unwrap();
        let (first_changed, first_inode) = unix_identity(&first);
        let before = FileStamp {
            modified: first.modified().ok(),
            len: first.len(),
            changed: first_changed,
            inode: first_inode,
        };

        // A real checkout or restore happens later, and some filesystems record
        // timestamps coarsely, so give the clock room to move.
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Same byte length, and the modification time put back exactly.
        std::fs::write(&path, "fn omega_marker() {}\n").unwrap();
        let restored = std::fs::File::options().write(true).open(&path).unwrap();
        restored.set_modified(first.modified().unwrap()).unwrap();
        drop(restored);

        let second = std::fs::metadata(&path).unwrap();
        let (second_changed, second_inode) = unix_identity(&second);
        let after = FileStamp {
            modified: second.modified().ok(),
            len: second.len(),
            changed: second_changed,
            inode: second_inode,
        };

        assert_eq!(
            before.modified, after.modified,
            "the test must restore the mtime"
        );
        assert_eq!(
            before.len, after.len,
            "the two revisions must be the same size"
        );
        assert_ne!(
            before, after,
            "a changed file compared equal, so its chunks would be reused"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn an_untouched_file_still_compares_equal() {
        // The stamp must not become so strict that nothing is ever reused.
        let dir = std::env::temp_dir().join(format!("rupi-stable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.rs");
        std::fs::write(&path, "fn stable() {}\n").unwrap();

        let read_stamp = || {
            let m = std::fs::metadata(&path).unwrap();
            let (changed, inode) = unix_identity(&m);
            FileStamp {
                modified: m.modified().ok(),
                len: m.len(),
                changed,
                inode,
            }
        };
        assert_eq!(read_stamp(), read_stamp());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deep_nesting_does_not_abort_the_process() {
        // A stack overflow in Rust aborts; it cannot be caught. A file like this in
        // any indexed repository used to take the whole agent down from a tool call
        // that only reads code.
        for depth in [1_000usize, 20_000, 60_000] {
            let content = format!("x = {}1{};\n", "(".repeat(depth), ")".repeat(depth));
            let chunks = chunk_file("deep.js", &content, Some("javascript".to_string()));
            assert!(!chunks.is_empty(), "depth {} produced no chunks", depth);
        }
    }

    #[test]
    fn test_chunk_file_rust_ast() {
        // Large enough to exceed DESIRED_CHUNK_LENGTH, so chunking really splits.
        // A small file legitimately produces one chunk, which is why the earlier
        // version of this test could not tell AST chunking from a line chunker.
        let mut content = String::new();
        for i in 0..30 {
            content.push_str(&format!(
                "fn function_{i}() -> i32 {{\n    // body marker {i}\n    let a = {i};\n    let b = a * 2;\n    b + {i}\n}}\n\n"
            ));
        }
        let chunks = chunk_file("test.rs", &content, Some("rust".to_string()));

        assert!(
            chunks.len() >= 2,
            "expected a split, got {} chunk(s)",
            chunks.len()
        );

        // The property that distinguishes AST chunking from arbitrary line cuts:
        // a function's signature and its body stay in the same chunk.
        for i in 0..30 {
            let signature = format!("fn function_{i}()");
            let marker = format!("// body marker {i}");
            let holding: Vec<usize> = chunks
                .iter()
                .enumerate()
                .filter(|(_, c)| c.content.contains(&signature))
                .map(|(index, _)| index)
                .collect();
            assert_eq!(
                holding.len(),
                1,
                "function_{i} appears in {} chunks",
                holding.len()
            );
            assert!(
                chunks[holding[0]].content.contains(&marker),
                "function_{i} was split from its own body"
            );
        }
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
        let all_content: String = chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
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
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
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
        assert!(!files
            .iter()
            .any(|f| f.to_string_lossy().contains("node_modules")));
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
