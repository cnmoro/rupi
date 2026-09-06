//! Todo list: a model-owned plan that stays at the tail of the context.
//!
//! The anchor holds the goal. This holds the decomposition. Both matter for a long
//! run, and they fail in different ways: the anchor is host-owned and exact, while
//! the todo list is model-owned and current. The list is whole-list replacement, so
//! the newest copy always states the complete plan and its progress — and because it
//! arrives as the newest tool result, it sits where the model's attention is
//! strongest rather than hundreds of tool calls back.
//!
//! The store is owned by the session, not by the process. An earlier draft used a
//! global, which is how `bash` carries its timeouts, but a plan is conversation
//! state: two sessions in one process would have silently shared and overwritten
//! each other's list.

use std::sync::RwLock;

/// Lifecycle state of one todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pending,
    InProgress,
    Completed,
}

impl Status {
    /// Parse a model-supplied status string.
    pub fn parse(value: &str) -> Option<Status> {
        match value {
            "pending" => Some(Status::Pending),
            "in_progress" => Some(Status::InProgress),
            "completed" => Some(Status::Completed),
            _ => None,
        }
    }

    /// The wire spelling of this status.
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pending => "pending",
            Status::InProgress => "in_progress",
            Status::Completed => "completed",
        }
    }

    /// The glyph used in the rendered list.
    fn marker(self) -> &'static str {
        match self {
            Status::Pending => "[ ]",
            Status::InProgress => "[~]",
            Status::Completed => "[x]",
        }
    }
}

/// One entry in the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoItem {
    pub content: String,
    pub status: Status,
}

/// The plan for one session.
#[derive(Debug, Default)]
pub struct TodoStore {
    items: RwLock<Vec<TodoItem>>,
}

impl TodoStore {
    /// A store with no tracked tasks.
    pub fn new() -> Self {
        TodoStore::default()
    }

    /// Discard the current list. Called when a session resets.
    pub fn reset(&self) {
        if let Ok(mut list) = self.items.write() {
            list.clear();
        }
    }

    /// A copy of the current list.
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.items
            .read()
            .map(|list| list.clone())
            .unwrap_or_default()
    }

    /// Replace the whole list.
    ///
    /// Rejects a call that marks more than one item `in_progress`, because a plan
    /// with two active steps tells the reader nothing about what the agent is doing
    /// now.
    pub fn replace(&self, items: Vec<TodoItem>) -> Result<String, String> {
        let active = items
            .iter()
            .filter(|i| i.status == Status::InProgress)
            .count();
        if active > 1 {
            return Err(format!(
                "Rejected: {} todos are in_progress. Keep AT MOST ONE in_progress at a time. \
Send the whole list again with a single active item.",
                active
            ));
        }
        if items.iter().any(|i| i.content.trim().is_empty()) {
            return Err("Rejected: every todo needs non-empty content.".to_string());
        }
        let rendered = render_list(&items);
        match self.items.write() {
            Ok(mut list) => *list = items,
            Err(_) => return Err("Rejected: todo state is poisoned.".to_string()),
        }
        Ok(rendered)
    }

    /// Render the current list, or `None` when nothing is tracked.
    pub fn render_current(&self) -> Option<String> {
        let items = self.snapshot();
        if items.is_empty() {
            return None;
        }
        Some(render_list(&items))
    }

    /// Whether every tracked item is complete. Vacuously false for an empty list.
    pub fn all_complete(&self) -> bool {
        let items = self.snapshot();
        !items.is_empty() && items.iter().all(|i| i.status == Status::Completed)
    }
}

/// Render a list for the model.
fn render_list(items: &[TodoItem]) -> String {
    if items.is_empty() {
        return "Todo list cleared. No tasks are tracked.".to_string();
    }
    let done = items
        .iter()
        .filter(|i| i.status == Status::Completed)
        .count();
    let mut out = format!("Todo list updated ({}/{} complete):\n", done, items.len());
    for item in items {
        out.push_str(&format!("{} {}\n", item.status.marker(), item.content));
    }
    if items.iter().all(|i| i.status == Status::Completed) {
        out.push_str("\nEvery task is complete. Confirm the result, then report back.");
    } else if !items.iter().any(|i| i.status == Status::InProgress) {
        out.push_str("\nNo task is in_progress. Mark the one being worked on before continuing.");
    }
    out.trim_end().to_string()
}

/// Parse the `todos` argument of a `todo_write` call.
pub fn parse_items(value: &serde_json::Value) -> Result<Vec<TodoItem>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| "Rejected: `todos` must be an array.".to_string())?;
    let mut items = Vec::with_capacity(array.len());
    for (index, entry) in array.iter().enumerate() {
        let content = entry
            .get("content")
            .and_then(|c| c.as_str())
            .ok_or_else(|| format!("Rejected: todo {} has no `content` string.", index))?;
        let raw_status = entry
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("pending");
        let status = Status::parse(raw_status).ok_or_else(|| {
            format!(
                "Rejected: todo {} has status {:?}. Use pending, in_progress, or completed.",
                index, raw_status
            )
        })?;
        items.push(TodoItem {
            content: content.to_string(),
            status,
        });
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(content: &str, status: Status) -> TodoItem {
        TodoItem {
            content: content.to_string(),
            status,
        }
    }

    #[test]
    fn replace_rejects_two_active_items() {
        let store = TodoStore::new();
        let result = store.replace(vec![
            item("a", Status::InProgress),
            item("b", Status::InProgress),
        ]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("AT MOST ONE"));
        assert!(
            store.snapshot().is_empty(),
            "a rejected call must not mutate the list"
        );
    }

    #[test]
    fn replace_rejects_empty_content() {
        let store = TodoStore::new();
        let result = store.replace(vec![item("   ", Status::Pending)]);
        assert!(result.is_err());
        assert!(store.snapshot().is_empty());
    }

    #[test]
    fn render_list_shows_progress_and_markers() {
        let text = render_list(&[
            item("read the code", Status::Completed),
            item("write the fix", Status::InProgress),
            item("run the tests", Status::Pending),
        ]);
        assert!(text.contains("1/3 complete"));
        assert!(text.contains("[x] read the code"));
        assert!(text.contains("[~] write the fix"));
        assert!(text.contains("[ ] run the tests"));
    }

    #[test]
    fn render_list_warns_when_nothing_is_active() {
        let text = render_list(&[item("a", Status::Pending)]);
        assert!(text.contains("No task is in_progress"));
    }

    #[test]
    fn render_list_confirms_a_finished_plan() {
        let text = render_list(&[item("a", Status::Completed)]);
        assert!(text.contains("Every task is complete"));
    }

    #[test]
    fn render_list_handles_an_empty_plan() {
        assert!(render_list(&[]).contains("cleared"));
    }

    #[test]
    fn parse_items_reads_a_well_formed_list() {
        let items = parse_items(&json!([
            {"content": "one", "status": "completed"},
            {"content": "two", "status": "in_progress"},
            {"content": "three"}
        ]))
        .unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].status, Status::Completed);
        assert_eq!(items[1].status, Status::InProgress);
        assert_eq!(items[2].status, Status::Pending);
    }

    #[test]
    fn parse_items_rejects_an_unknown_status() {
        let error = parse_items(&json!([{"content": "one", "status": "done"}])).unwrap_err();
        assert!(error.contains("pending, in_progress, or completed"));
    }

    #[test]
    fn parse_items_rejects_a_non_array() {
        assert!(parse_items(&json!({"content": "one"})).is_err());
    }

    #[test]
    fn status_round_trips() {
        for status in [Status::Pending, Status::InProgress, Status::Completed] {
            assert_eq!(Status::parse(status.as_str()), Some(status));
        }
    }

    #[test]
    fn a_store_tracks_its_own_list() {
        let store = TodoStore::new();
        store
            .replace(vec![item("only", Status::InProgress)])
            .unwrap();
        assert_eq!(store.snapshot().len(), 1);
        assert!(!store.all_complete());
        assert!(store.render_current().is_some());

        store
            .replace(vec![item("only", Status::Completed)])
            .unwrap();
        assert!(store.all_complete());

        store.reset();
        assert!(!store.all_complete());
        assert!(store.render_current().is_none());
    }

    #[test]
    fn two_stores_do_not_share_state() {
        let a = TodoStore::new();
        let b = TodoStore::new();
        a.replace(vec![item("a work", Status::InProgress)]).unwrap();
        assert_eq!(a.snapshot().len(), 1);
        assert!(b.snapshot().is_empty(), "sessions must not share a plan");
    }

    #[test]
    fn replace_is_whole_list_replacement() {
        let store = TodoStore::new();
        store
            .replace(vec![
                item("one", Status::Pending),
                item("two", Status::Pending),
            ])
            .unwrap();
        store
            .replace(vec![item("three", Status::InProgress)])
            .unwrap();
        let items = store.snapshot();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "three");
    }
}
