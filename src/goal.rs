//! Goal rounds and completion authority.
//!
//! Goal mode used to end a round by asking a second model, out of band and with only
//! the last six messages, whether the work was done. That judge saw no tools, no
//! workspace, and no earlier context, and it cost one extra request per round.
//!
//! Here the working model decides, but only where it has the standing to: a `goal`
//! tool call is admitted as a completion when it happens inside the exact round the
//! driver opened. A call from any other position is recorded and rejected, so the
//! model cannot declare victory from a stray turn. This is the authority rule from
//! the deepseek harness, and it costs no extra API call.
//!
//! The registry belongs to the session. Goal state is conversation state, so two
//! sessions in one process must not see each other's objective or round number.

use std::sync::RwLock;

/// Where the current goal stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    /// Work continues.
    Active,
    /// The model marked the objective achieved inside an admitted round.
    Complete,
    /// The model reported that it cannot proceed.
    Blocked,
}

/// Durable state of one goal.
#[derive(Debug, Clone)]
pub struct GoalState {
    pub objective: String,
    pub max_rounds: u32,
    /// The round currently open, or 0 before the driver admits the first one.
    pub admitted_round: u32,
    pub status: GoalStatus,
    /// Reason recorded by a `block` call.
    pub block_reason: Option<String>,
    /// The round in which an admitted completion or block landed.
    pub decided_round: u32,
}

/// Default number of continuation rounds before the driver gives up.
pub const DEFAULT_MAX_ROUNDS: u32 = 5;

/// Outcome of a model-issued `goal` tool call.
pub type GoalToolResult = Result<String, String>;

/// The goal owned by one session.
#[derive(Debug, Default)]
pub struct GoalRegistry {
    state: RwLock<Option<GoalState>>,
}

impl GoalRegistry {
    /// A registry with no goal set.
    pub fn new() -> Self {
        GoalRegistry::default()
    }

    /// Start tracking a new goal, discarding any previous one.
    pub fn set(&self, objective: Option<String>, max_rounds: u32) {
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        *guard = objective.map(|objective| GoalState {
            objective,
            max_rounds: max_rounds.max(1),
            admitted_round: 0,
            status: GoalStatus::Active,
            block_reason: None,
            decided_round: 0,
        });
    }

    /// A copy of the current goal state.
    pub fn current(&self) -> Option<GoalState> {
        self.state.read().ok().and_then(|g| g.clone())
    }

    /// The active objective, if a goal is set and still running.
    pub fn active_objective(&self) -> Option<String> {
        self.current().filter(|g| g.status == GoalStatus::Active).map(|g| g.objective)
    }

    /// Open round `round` for the current goal. Only the driver calls this.
    pub fn admit_round(&self, round: u32) {
        if let Ok(mut guard) = self.state.write() {
            if let Some(state) = guard.as_mut() {
                state.admitted_round = round;
            }
        }
    }

    /// Whether the current goal reached a terminal state inside an admitted round.
    pub fn is_decided(&self) -> bool {
        matches!(
            self.current().map(|g| g.status),
            Some(GoalStatus::Complete) | Some(GoalStatus::Blocked)
        )
    }

    /// Report the current goal to the model.
    pub fn read_goal(&self) -> GoalToolResult {
        match self.current() {
            None => {
                Err("No goal is set. The goal tool only applies while a goal is active.".to_string())
            }
            Some(state) => Ok(format!(
                "Objective: {}\nRound: {}/{}\nStatus: {}",
                state.objective,
                state.admitted_round,
                state.max_rounds,
                match state.status {
                    GoalStatus::Active => "active",
                    GoalStatus::Complete => "complete",
                    GoalStatus::Blocked => "blocked",
                }
            )),
        }
    }

    /// Mark the current goal complete, if the caller has the standing to.
    ///
    /// `round` is the round the model read out of the open `<goal_round>` block. A
    /// call that names any other round is rejected with the reason, so the model
    /// learns why instead of failing silently.
    pub fn complete(&self, round: u32) -> GoalToolResult {
        self.decide(round, GoalStatus::Complete, None)
    }

    /// Mark the current goal blocked, if the caller has the standing to.
    pub fn block(&self, round: u32, reason: &str) -> GoalToolResult {
        if reason.trim().is_empty() {
            return Err("Rejected: `reason` is required when blocking a goal.".to_string());
        }
        self.decide(round, GoalStatus::Blocked, Some(reason.to_string()))
    }

    /// Shared authority check and state transition for `complete` and `block`.
    fn decide(&self, round: u32, status: GoalStatus, reason: Option<String>) -> GoalToolResult {
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(_) => return Err("Rejected: goal state is poisoned.".to_string()),
        };
        let state = match guard.as_mut() {
            Some(state) => state,
            None => return Err("Rejected: no goal is set.".to_string()),
        };
        if state.admitted_round == 0 {
            return Err(
                "Rejected: no goal round is open. Only the goal driver can open a round."
                    .to_string(),
            );
        }
        if round != state.admitted_round {
            return Err(format!(
                "Rejected: this call names round {}, but round {} is the open one. \
Keep working and decide inside the current round.",
                round, state.admitted_round
            ));
        }
        if state.status != GoalStatus::Active {
            return Err(format!(
                "Rejected: the goal is already {}.",
                match state.status {
                    GoalStatus::Complete => "complete",
                    GoalStatus::Blocked => "blocked",
                    GoalStatus::Active => "active",
                }
            ));
        }
        state.status = status;
        state.block_reason = reason;
        state.decided_round = round;
        Ok(match status {
            GoalStatus::Complete => {
                format!("Goal marked complete in round {}. The driver will stop.", round)
            }
            GoalStatus::Blocked => {
                format!("Goal marked blocked in round {}. The driver will stop.", round)
            }
            GoalStatus::Active => unreachable!("decide is never called with Active"),
        })
    }
}

/// Render the continuation prompt for one goal round.
///
/// The round counter tells the model how much budget is left. The instruction to
/// re-inspect the workspace is the load-bearing part after a compaction: narration is
/// exactly what a summary degrades, so the model must not trust it.
pub fn render_round_prompt(objective: &str, round: u32, max_rounds: u32) -> String {
    format!(
        "<goal_round>\n\
Objective: {}\n\
Round: {}/{}\n\n\
Continue working toward the objective in this same session. Treat the current workspace, the \
tool results, and the files on disk as authoritative. Inspect them instead of assuming that \
earlier narration is still current. Make concrete progress and verify the result.\n\n\
Before you claim completion, gather evidence that the WHOLE objective is achieved, then call the \
`goal` tool with operation \"complete\" and this round number. If work remains, leave the goal \
active and keep going. If you cannot proceed, call the `goal` tool with operation \"block\" and \
give the reason.\n\
</goal_round>",
        objective, round, max_rounds
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(objective: &str, round: u32) -> GoalRegistry {
        let registry = GoalRegistry::new();
        registry.set(Some(objective.to_string()), 5);
        registry.admit_round(round);
        registry
    }

    #[test]
    fn complete_is_rejected_without_an_open_round() {
        let registry = GoalRegistry::new();
        registry.set(Some("ship it".into()), 5);
        let error = registry.complete(1).unwrap_err();
        assert!(error.contains("no goal round is open"), "{}", error);
        assert!(!registry.is_decided());
    }

    #[test]
    fn complete_is_rejected_from_the_wrong_round() {
        let registry = running("ship it", 2);
        let error = registry.complete(1).unwrap_err();
        assert!(error.contains("round 2 is the open one"), "{}", error);
        assert!(!registry.is_decided());
    }

    #[test]
    fn complete_is_accepted_inside_the_admitted_round() {
        let registry = running("ship it", 3);
        assert!(registry.complete(3).is_ok());
        assert!(registry.is_decided());
        let state = registry.current().unwrap();
        assert_eq!(state.status, GoalStatus::Complete);
        assert_eq!(state.decided_round, 3);
    }

    #[test]
    fn a_second_decision_is_rejected() {
        let registry = running("ship it", 1);
        assert!(registry.complete(1).is_ok());
        assert!(registry.complete(1).unwrap_err().contains("already complete"));
        assert!(registry.block(1, "changed my mind").unwrap_err().contains("already complete"));
    }

    #[test]
    fn block_requires_a_reason() {
        let registry = running("ship it", 1);
        assert!(registry.block(1, "   ").unwrap_err().contains("`reason` is required"));
        assert!(registry.block(1, "the API key is missing").is_ok());
        let state = registry.current().unwrap();
        assert_eq!(state.status, GoalStatus::Blocked);
        assert_eq!(state.block_reason.as_deref(), Some("the API key is missing"));
    }

    #[test]
    fn tools_are_rejected_when_no_goal_is_set() {
        let registry = GoalRegistry::new();
        assert!(registry.read_goal().is_err());
        assert!(registry.complete(1).unwrap_err().contains("no goal is set"));
        assert!(!registry.is_decided());
    }

    #[test]
    fn active_objective_hides_a_decided_goal() {
        let registry = running("ship it", 1);
        assert_eq!(registry.active_objective().as_deref(), Some("ship it"));
        registry.complete(1).unwrap();
        assert_eq!(registry.active_objective(), None);
    }

    #[test]
    fn read_goal_reports_the_open_round() {
        let registry = running("make the tests pass", 2);
        let text = registry.read_goal().unwrap();
        assert!(text.contains("Objective: make the tests pass"));
        assert!(text.contains("Round: 2/5"));
        assert!(text.contains("Status: active"));
    }

    #[test]
    fn two_registries_do_not_share_state() {
        let a = running("goal a", 1);
        let b = GoalRegistry::new();
        a.complete(1).unwrap();
        assert!(a.is_decided());
        assert!(!b.is_decided(), "sessions must not share a goal");
        assert_eq!(b.active_objective(), None);
    }

    #[test]
    fn setting_a_goal_clears_the_previous_decision() {
        let registry = running("first", 1);
        registry.complete(1).unwrap();
        registry.set(Some("second".into()), 5);
        assert!(!registry.is_decided());
        assert_eq!(registry.active_objective().as_deref(), Some("second"));
        assert_eq!(registry.current().unwrap().admitted_round, 0);
    }

    #[test]
    fn max_rounds_is_never_zero() {
        let registry = GoalRegistry::new();
        registry.set(Some("x".into()), 0);
        assert_eq!(registry.current().unwrap().max_rounds, 1);
    }

    #[test]
    fn round_prompt_carries_the_objective_and_the_budget() {
        let text = render_round_prompt("make the tests pass", 2, 5);
        assert!(text.starts_with("<goal_round>"));
        assert!(text.contains("Objective: make the tests pass"));
        assert!(text.contains("Round: 2/5"));
        assert!(text.contains("authoritative"));
        assert!(text.contains("\"complete\""));
        assert!(text.ends_with("</goal_round>"));
    }
}
