//! Budgeted search over partial reasoning states.
//!
//! The tree is model-agnostic. A generation adapter proposes next states;
//! a process-reward adapter estimates their value and uncertainty; hard
//! verifiers may accept or reject terminal answers. This separation avoids
//! treating a language model's self-confidence as an independent verifier.
//!
//! The default value estimate is deliberately uncalibrated. Deployments must
//! inject a trained/calibrated PRM before interpreting values as probabilities.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueEstimate {
    /// Estimated value in [0, 1]. This is not a probability unless calibrated.
    pub mean: f64,
    /// Epistemic uncertainty in [0, 1].
    pub uncertainty: f64,
    /// True only when the provider was calibrated on held-out outcomes.
    pub calibrated: bool,
}

impl ValueEstimate {
    #[must_use]
    pub fn uncalibrated() -> Self {
        Self {
            mean: 0.5,
            uncertainty: 1.0,
            calibrated: false,
        }
    }

    #[must_use]
    pub fn new(mean: f64, uncertainty: f64, calibrated: bool) -> Self {
        Self {
            mean: finite_unit(mean, 0.5),
            uncertainty: finite_unit(uncertainty, 1.0),
            calibrated,
        }
    }
}

impl Default for ValueEstimate {
    fn default() -> Self {
        Self::uncalibrated()
    }
}

/// A process-level evaluator. Implementations should score a partial state,
/// not merely judge a completed answer. A remote PRM adapter can implement
/// this trait without coupling the search policy to a specific provider.
pub trait ProcessRewardModel: Send + Sync {
    fn evaluate(&self, partial_state: &str) -> ValueEstimate;
}

/// Safe fallback when no independently trained PRM is configured.
#[derive(Clone, Copy, Debug, Default)]
pub struct UncalibratedValueModel;

impl ProcessRewardModel for UncalibratedValueModel {
    fn evaluate(&self, _partial_state: &str) -> ValueEstimate {
        ValueEstimate::uncalibrated()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verification {
    /// No independent verifier was applicable.
    Unknown,
    /// A deterministic or independently trusted verifier passed.
    Passed,
    /// A hard verifier rejected this state.
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeStatus {
    Open,
    Expanded,
    Terminal,
    Rejected,
}

#[derive(Clone, Debug)]
pub struct SearchProposal {
    /// The complete partial-state summary after taking this step.
    pub state: String,
    /// An optional final answer. Present only for a terminal proposal.
    pub answer: Option<String>,
    /// Relative proposal prior in [0, 1].
    pub prior: f64,
    /// Process value and uncertainty estimate.
    pub value: ValueEstimate,
    /// Hard-verifier result for terminal proposals, otherwise Unknown.
    pub verification: Verification,
    /// Estimated state tokens charged to the search budget (not provider usage).
    pub estimated_state_tokens: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub max_nodes: usize,
    pub max_depth: usize,
    pub max_expansions: usize,
    pub max_tokens: usize,
    pub branch_factor: usize,
    pub exploration: f64,
    pub uncertainty_bonus: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_nodes: 64,
            max_depth: 12,
            max_expansions: 48,
            max_tokens: 16_384,
            branch_factor: 4,
            exploration: 1.4,
            uncertainty_bonus: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    VerifiedAnswer,
    ExpansionBudget,
    NodeBudget,
    TokenBudget,
    NoExpandableState,
    DepthLimit,
}

#[derive(Clone, Debug)]
pub struct SearchNode {
    pub id: usize,
    pub parent: Option<usize>,
    pub depth: usize,
    pub step: String,
    pub state: String,
    pub answer: Option<String>,
    pub prior: f64,
    pub visits: u32,
    pub value: ValueEstimate,
    pub status: NodeStatus,
    pub verification: Verification,
    pub children: Vec<usize>,
    pub estimated_state_tokens: usize,
    value_sum: f64,
}

impl SearchNode {
    fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            self.value.mean
        } else {
            finite_unit(self.value_sum / f64::from(self.visits), self.value.mean)
        }
    }
}

/// Search diagnostics emitted alongside the selected answer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SearchStats {
    pub nodes: usize,
    pub expansions: usize,
    pub estimated_state_tokens: usize,
    pub max_depth_reached: usize,
    pub verified_terminals: usize,
    pub rejected_nodes: usize,
    pub transpositions: usize,
    pub prm_calibrated_nodes: usize,
    pub best_value: f64,
    pub best_uncertainty: f64,
}

/// Search tree with PUCT selection, uncertainty bonuses, transposition
/// deduplication, backpropagation and a token-aware exploration schedule.
#[derive(Clone, Debug)]
pub struct SearchTree {
    config: SearchConfig,
    nodes: Vec<SearchNode>,
    transpositions: HashMap<String, usize>,
    expansions: usize,
    tokens: usize,
    max_depth_reached: usize,
    transposition_hits: usize,
    stop_reason: Option<StopReason>,
}

impl SearchTree {
    #[must_use]
    pub fn new(root_state: impl Into<String>, config: SearchConfig) -> Self {
        let root_state = root_state.into();
        let mut transpositions = HashMap::new();
        transpositions.insert(normalize_state(&root_state), 0);
        Self {
            config: SearchConfig {
                max_nodes: config.max_nodes.max(1),
                max_depth: config.max_depth.max(1),
                max_expansions: config.max_expansions.max(1),
                max_tokens: config.max_tokens.max(1),
                branch_factor: config.branch_factor.clamp(1, 16),
                exploration: config.exploration.max(0.0),
                uncertainty_bonus: config.uncertainty_bonus.max(0.0),
            },
            nodes: vec![SearchNode {
                id: 0,
                parent: None,
                depth: 0,
                step: String::new(),
                state: root_state,
                answer: None,
                prior: 1.0,
                visits: 1,
                value: ValueEstimate::uncalibrated(),
                status: NodeStatus::Open,
                verification: Verification::Unknown,
                children: Vec::new(),
                estimated_state_tokens: 0,
                value_sum: 0.5,
            }],
            transpositions,
            expansions: 0,
            tokens: 0,
            max_depth_reached: 0,
            transposition_hits: 0,
            stop_reason: None,
        }
    }

    #[must_use]
    pub fn nodes(&self) -> &[SearchNode] {
        &self.nodes
    }

    #[must_use]
    pub const fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    #[must_use]
    pub fn remaining_tokens(&self) -> usize {
        self.config.max_tokens.saturating_sub(self.tokens)
    }

    #[must_use]
    pub fn remaining_expansions(&self) -> usize {
        self.config.max_expansions.saturating_sub(self.expansions)
    }

    /// Early search is broad; as the token budget shrinks it narrows branching
    /// and reduces the exploration coefficient, following budget-guided MCTS.
    #[must_use]
    pub fn current_branch_limit(&self) -> usize {
        let remaining = self.remaining_tokens() as f64 / self.config.max_tokens as f64;
        let factor = (1.0 + (self.config.branch_factor - 1) as f64 * remaining).ceil();
        (factor as usize).clamp(1, self.config.branch_factor)
    }

    #[must_use]
    pub fn current_exploration(&self) -> f64 {
        let remaining = self.remaining_tokens() as f64 / self.config.max_tokens as f64;
        self.config.exploration * (0.25 + 0.75 * remaining)
    }

    /// Select an expandable leaf with PUCT and an uncertainty bonus.
    /// None means the hard search budget is exhausted or all paths are closed.
    pub fn select_leaf(&mut self) -> Option<usize> {
        if let Some(reason) = self.budget_stop_reason() {
            self.stop_reason = Some(reason);
            return None;
        }
        let mut current = 0usize;
        loop {
            let status = self.nodes.get(current)?.status;
            match status {
                NodeStatus::Rejected | NodeStatus::Terminal => {
                    if current == 0 {
                        return None;
                    }
                    current = self.nodes[current].parent?;
                }
                NodeStatus::Open => {
                    if self.nodes[current].depth >= self.config.max_depth {
                        self.nodes[current].status = NodeStatus::Terminal;
                        self.stop_reason = Some(StopReason::DepthLimit);
                        if current == 0 {
                            return None;
                        }
                        current = self.nodes[current].parent?;
                        continue;
                    }
                    self.nodes[current].visits = self.nodes[current].visits.saturating_add(1);
                    return Some(current);
                }
                NodeStatus::Expanded => {
                    let parent_visits = f64::from(self.nodes[current].visits.max(1)).sqrt();
                    let eligible: Vec<usize> = self.nodes[current]
                        .children
                        .iter()
                        .copied()
                        .filter(|id| {
                            self.nodes.get(*id).is_some_and(|child| {
                                matches!(child.status, NodeStatus::Open | NodeStatus::Expanded)
                            })
                        })
                        .collect();
                    if eligible.is_empty() {
                        // This subtree is exhausted. Close it and backtrack so
                        // PUCT can still select a viable sibling at an ancestor.
                        self.nodes[current].status = NodeStatus::Terminal;
                        if current == 0 {
                            return None;
                        }
                        current = self.nodes[current].parent?;
                        continue;
                    }
                    let exploration = self.current_exploration();
                    let selected = eligible.into_iter().max_by(|left, right| {
                        let left_score = self.puct_score(*left, parent_visits, exploration);
                        let right_score = self.puct_score(*right, parent_visits, exploration);
                        left_score.total_cmp(&right_score)
                    })?;
                    current = selected;
                }
            }
        }
    }

    /// Expand one open node. Duplicate states are collapsed into the existing
    /// transposition; rejected proposals are recorded but never selectable.
    pub fn expand(
        &mut self,
        parent_id: usize,
        mut proposals: Vec<SearchProposal>,
    ) -> Result<Vec<usize>, &'static str> {
        if let Some(reason) = self.budget_stop_reason() {
            self.stop_reason = Some(reason);
            return Err("search budget exhausted");
        }
        let parent = self.nodes.get(parent_id).ok_or("parent node not found")?;
        if parent.status != NodeStatus::Open {
            return Err("only open nodes can be expanded");
        }
        if parent.depth >= self.config.max_depth {
            self.nodes[parent_id].status = NodeStatus::Terminal;
            self.stop_reason = Some(StopReason::DepthLimit);
            return Err("maximum search depth reached");
        }

        proposals.retain(|proposal| !proposal.state.trim().is_empty());
        proposals
            .sort_by(|left, right| proposal_priority(right).total_cmp(&proposal_priority(left)));
        proposals.truncate(self.current_branch_limit());

        let parent_depth = parent.depth;
        let mut created = Vec::new();
        for proposal in proposals {
            if self.nodes.len() >= self.config.max_nodes {
                self.stop_reason = Some(StopReason::NodeBudget);
                break;
            }
            if proposal.estimated_state_tokens > self.remaining_tokens() {
                self.stop_reason = Some(StopReason::TokenBudget);
                break;
            }
            let normalized_state = normalize_state(&proposal.state);
            if self.transpositions.contains_key(&normalized_state) {
                self.transposition_hits = self.transposition_hits.saturating_add(1);
                continue;
            }
            let id = self.nodes.len();
            let depth = parent_depth.saturating_add(1);
            let value = sanitize_estimate(proposal.value);
            let status = if proposal.verification == Verification::Failed {
                NodeStatus::Rejected
            } else if proposal.answer.is_some() || proposal.verification == Verification::Passed {
                NodeStatus::Terminal
            } else {
                NodeStatus::Open
            };
            self.tokens = self.tokens.saturating_add(proposal.estimated_state_tokens);
            self.nodes.push(SearchNode {
                id,
                parent: Some(parent_id),
                depth,
                step: proposal.state.clone(),
                state: proposal.state.clone(),
                answer: proposal.answer,
                prior: finite_unit(proposal.prior, 0.0),
                visits: 0,
                value,
                status,
                verification: proposal.verification,
                children: Vec::new(),
                estimated_state_tokens: proposal.estimated_state_tokens,
                value_sum: value.mean,
            });
            self.transpositions.insert(normalized_state, id);
            self.nodes[parent_id].children.push(id);
            self.max_depth_reached = self.max_depth_reached.max(depth);
            if status == NodeStatus::Terminal && proposal.verification == Verification::Passed {
                self.backpropagate(id, 1.0);
            } else {
                self.backpropagate(id, value.mean);
            }
            created.push(id);
        }
        self.nodes[parent_id].status = NodeStatus::Expanded;
        self.expansions = self.expansions.saturating_add(1);
        if created.is_empty() && self.nodes[parent_id].children.is_empty() {
            self.nodes[parent_id].status = NodeStatus::Rejected;
            self.stop_reason
                .get_or_insert(StopReason::NoExpandableState);
        }
        Ok(created)
    }

    /// Add a process-level evaluation to an open node and propagate its value.
    pub fn evaluate_node(
        &mut self,
        node_id: usize,
        estimate: ValueEstimate,
        verification: Verification,
    ) -> Result<(), &'static str> {
        let node = self.nodes.get_mut(node_id).ok_or("node not found")?;
        if node.status == NodeStatus::Rejected || node.status == NodeStatus::Terminal {
            return Err("closed nodes cannot be evaluated");
        }
        node.value = sanitize_estimate(estimate);
        node.verification = verification;
        if verification == Verification::Failed {
            node.status = NodeStatus::Rejected;
        } else if verification == Verification::Passed {
            node.status = NodeStatus::Terminal;
        }
        let score = if verification == Verification::Passed {
            1.0
        } else {
            node.value.mean
        };
        self.backpropagate(node_id, score);
        Ok(())
    }

    /// Return the best verified terminal answer, or the strongest unverified
    /// terminal if no hard verifier applies. Unverified answers remain explicit.
    #[must_use]
    pub fn best_answer(&self) -> Option<(usize, &str, Verification)> {
        self.nodes
            .iter()
            .filter(|node| node.answer.is_some() && node.status != NodeStatus::Rejected)
            .max_by(|left, right| answer_priority(left).total_cmp(&answer_priority(right)))
            .and_then(|node| {
                node.answer
                    .as_deref()
                    .map(|answer| (node.id, answer, node.verification))
            })
    }

    #[must_use]
    pub fn stats(&self) -> SearchStats {
        let verified_terminals = self
            .nodes
            .iter()
            .filter(|node| {
                node.status == NodeStatus::Terminal && node.verification == Verification::Passed
            })
            .count();
        let rejected_nodes = self
            .nodes
            .iter()
            .filter(|node| node.status == NodeStatus::Rejected)
            .count();
        let prm_calibrated_nodes = self
            .nodes
            .iter()
            .filter(|node| node.value.calibrated)
            .count();
        let best = self.best_answer().and_then(|(id, _, _)| self.nodes.get(id));
        SearchStats {
            nodes: self.nodes.len(),
            expansions: self.expansions,
            estimated_state_tokens: self.tokens,
            max_depth_reached: self.max_depth_reached,
            verified_terminals,
            rejected_nodes,
            transpositions: self.transposition_hits,
            prm_calibrated_nodes,
            best_value: best.map_or(0.0, SearchNode::mean_value),
            best_uncertainty: best.map_or(1.0, |node| node.value.uncertainty),
        }
    }

    pub fn finish(&mut self) -> StopReason {
        let reason = if self.nodes.iter().any(|node| {
            node.status == NodeStatus::Terminal && node.verification == Verification::Passed
        }) {
            StopReason::VerifiedAnswer
        } else {
            self.budget_stop_reason()
                .unwrap_or(StopReason::NoExpandableState)
        };
        self.stop_reason = Some(reason);
        reason
    }

    fn puct_score(&self, node_id: usize, parent_visits: f64, exploration: f64) -> f64 {
        let Some(node) = self.nodes.get(node_id) else {
            return f64::NEG_INFINITY;
        };
        if node.status == NodeStatus::Rejected {
            return f64::NEG_INFINITY;
        }
        let q = node.mean_value();
        let u = exploration * node.prior * parent_visits / (1.0 + f64::from(node.visits));
        let uncertainty = self.config.uncertainty_bonus * node.value.uncertainty
            / (1.0 + f64::from(node.visits)).sqrt();
        q + u + uncertainty
    }

    fn backpropagate(&mut self, node_id: usize, value: f64) {
        let value = finite_unit(value, 0.5);
        let mut cursor = Some(node_id);
        while let Some(id) = cursor {
            let Some(node) = self.nodes.get_mut(id) else {
                break;
            };
            node.visits = node.visits.saturating_add(1);
            node.value_sum += value;
            cursor = node.parent;
        }
    }

    fn budget_stop_reason(&self) -> Option<StopReason> {
        if self.expansions >= self.config.max_expansions {
            Some(StopReason::ExpansionBudget)
        } else if self.nodes.len() >= self.config.max_nodes {
            Some(StopReason::NodeBudget)
        } else if self.tokens >= self.config.max_tokens {
            Some(StopReason::TokenBudget)
        } else {
            None
        }
    }
}

/// Proposal priority used to cap branch width before insertion.
#[must_use]
pub fn proposal_priority(proposal: &SearchProposal) -> f64 {
    let estimate = sanitize_estimate(proposal.value);
    let verifier_bonus = match proposal.verification {
        Verification::Passed => 2.0,
        Verification::Failed => -2.0,
        Verification::Unknown => 0.0,
    };
    finite_unit(proposal.prior, 0.0) * 0.15 + estimate.mean + verifier_bonus
        - estimate.uncertainty * 0.1
}

fn answer_priority(node: &SearchNode) -> f64 {
    let hard = match node.verification {
        Verification::Passed => 2.0,
        Verification::Unknown => 0.0,
        Verification::Failed => -2.0,
    };
    hard + node.mean_value() - node.value.uncertainty * 0.1
}

fn sanitize_estimate(estimate: ValueEstimate) -> ValueEstimate {
    ValueEstimate::new(estimate.mean, estimate.uncertainty, estimate.calibrated)
}

fn finite_unit(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

fn normalize_state(state: &str) -> String {
    state
        .split_whitespace()
        .flat_map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}


#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn proposal(state: &str, value: f64, uncertainty: f64, prior: f64) -> SearchProposal {
        SearchProposal {
            state: state.to_string(),
            answer: None,
            prior,
            value: ValueEstimate::new(value, uncertainty, true),
            verification: Verification::Unknown,
            estimated_state_tokens: 4,
        }
    }

    #[test]
    fn unit__puct__expands_distinct_high_value_branches() {
        let mut tree = SearchTree::new(
            "root",
            SearchConfig {
                max_nodes: 8,
                max_depth: 4,
                max_expansions: 6,
                max_tokens: 100,
                branch_factor: 3,
                ..SearchConfig::default()
            },
        );
        let ids = tree
            .expand(
                0,
                vec![
                    proposal("branch-a", 0.9, 0.1, 0.8),
                    proposal("branch-b", 0.7, 0.2, 0.5),
                    proposal("branch-c", 0.4, 0.8, 0.2),
                ],
            )
            .unwrap();
        assert_eq!(ids.len(), 3);
        assert!(tree.select_leaf().is_some());
        assert_eq!(tree.stats().nodes, 4);
    }

    #[test]
    fn unit__transpositions__deduplicates_equivalent_states() {
        let mut tree = SearchTree::new("root", SearchConfig::default());
        let ids = tree
            .expand(
                0,
                vec![
                    proposal("same  partial state", 0.6, 0.2, 0.5),
                    proposal("SAME partial STATE", 0.7, 0.1, 0.7),
                ],
            )
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(tree.stats().transpositions, 1);
    }

    #[test]
    fn unit__puct__backtracks_from_exhausted_subtree_to_open_sibling() {
        let mut tree = SearchTree::new("root", SearchConfig::default());
        let branches = tree
            .expand(
                0,
                vec![
                    proposal("promising branch", 0.95, 0.1, 0.95),
                    proposal("alternative branch", 0.45, 0.2, 0.45),
                ],
            )
            .unwrap();
        let promising = branches[0];
        let alternative = branches[1];
        tree.expand(
            promising,
            vec![SearchProposal {
                state: "closed final answer".into(),
                answer: Some("done".into()),
                prior: 1.0,
                value: ValueEstimate::new(0.5, 1.0, false),
                verification: Verification::Unknown,
                estimated_state_tokens: 4,
            }],
        )
        .unwrap();

        // The higher-priority subtree is terminal; selection must backtrack
        // to the still-open sibling instead of terminating the whole search.
        assert_eq!(tree.select_leaf(), Some(alternative));
    }

    #[test]
    fn unit__hard_verifier__passed_terminal_outranks_higher_unverified_value() {
        let mut tree = SearchTree::new("root", SearchConfig::default());
        tree.expand(
            0,
            vec![
                SearchProposal {
                    state: "unverified branch".into(),
                    answer: Some("plausible".into()),
                    prior: 1.0,
                    value: ValueEstimate::new(0.99, 0.05, true),
                    verification: Verification::Unknown,
                    estimated_state_tokens: 4,
                },
                SearchProposal {
                    state: "verified branch".into(),
                    answer: Some("correct".into()),
                    prior: 0.5,
                    value: ValueEstimate::new(0.5, 0.2, true),
                    verification: Verification::Passed,
                    estimated_state_tokens: 4,
                },
            ],
        )
        .unwrap();
        assert_eq!(tree.best_answer().unwrap().1, "correct");
        assert_eq!(tree.finish(), StopReason::VerifiedAnswer);
    }

    #[test]
    fn unit__budget_schedule__narrows_exploration_as_tokens_deplete() {
        let mut tree = SearchTree::new(
            "root",
            SearchConfig {
                max_tokens: 100,
                branch_factor: 4,
                ..SearchConfig::default()
            },
        );
        let initial_branch = tree.current_branch_limit();
        let initial_exploration = tree.current_exploration();
        tree.expand(
            0,
            vec![
                proposal("a", 0.9, 0.1, 0.8),
                proposal("b", 0.8, 0.1, 0.6),
                proposal("c", 0.7, 0.1, 0.4),
                proposal("d", 0.6, 0.1, 0.2),
            ],
        )
        .unwrap();
        let leaf = tree.select_leaf().unwrap();
        tree.expand(
            leaf,
            vec![proposal("a1", 0.7, 0.1, 0.5), proposal("a2", 0.6, 0.1, 0.3)],
        )
        .unwrap();
        assert!(tree.remaining_tokens() < 100);
        assert!(tree.current_exploration() < initial_exploration);
        assert!(tree.current_branch_limit() <= initial_branch);
    }

    #[test]
    fn unit__process_value__sanitizes_nan_and_unbounded_scores() {
        let estimate = ValueEstimate::new(f64::NAN, 2.0, false);
        assert_eq!(estimate.mean, 0.5);
        assert_eq!(estimate.uncertainty, 1.0);
        assert!(!estimate.calibrated);
    }

    #[test]
    fn unit__failed_branches__are_never_selected() {
        let mut tree = SearchTree::new("root", SearchConfig::default());
        let ids = tree
            .expand(
                0,
                vec![
                    SearchProposal {
                        verification: Verification::Failed,
                        ..proposal("bad", 1.0, 0.0, 1.0)
                    },
                    proposal("good", 0.5, 0.2, 0.5),
                ],
            )
            .unwrap();
        assert_eq!(tree.nodes()[ids[0]].status, NodeStatus::Rejected);
        assert_eq!(tree.nodes()[ids[1]].status, NodeStatus::Open);
        assert_eq!(tree.select_leaf(), Some(ids[1]));
    }
}
