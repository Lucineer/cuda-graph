/*!
# cuda-graph

Property graph for agent knowledge.

Agents build knowledge graphs — entities, relationships, attributes.
This crate provides a property graph with traversal, pattern matching,
and index-based lookup.

- Nodes with typed labels and properties
- Directed edges with types and properties
- Index-based lookup (by label, property)
- BFS/DFS traversal
- Pattern matching (node-edge-node triples)
- Subgraph extraction
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

pub type Props = HashMap<String, String>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: Props,
}

impl GNode {
    pub fn new(id: &str) -> Self { GNode { id: id.to_string(), labels: vec![], properties: HashMap::new() } }
    pub fn with_label(mut self, label: &str) -> Self { self.labels.push(label.to_string()); self }
    pub fn with_prop(mut self, k: &str, v: &str) -> Self { self.properties.insert(k.to_string(), v.to_string()); self }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub properties: Props,
}

impl GEdge {
    pub fn new(id: &str, from: &str, to: &str, edge_type: &str) -> Self { GEdge { id: id.to_string(), from: from.to_string(), to: to.to_string(), edge_type: edge_type.to_string(), properties: HashMap::new() } }
    pub fn with_prop(mut self, k: &str, v: &str) -> Self { self.properties.insert(k.to_string(), v.to_string()); self }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// A property graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PropertyGraph {
    pub nodes: HashMap<String, GNode>,
    pub edges: Vec<GEdge>,
    pub out_edges: HashMap<String, Vec<usize>>, // node_id → edge indices
    pub in_edges: HashMap<String, Vec<usize>>,
    pub label_index: HashMap<String, Vec<String>>, // label → [node_ids]
    pub next_edge_id: u64,
}

impl PropertyGraph {
    pub fn new() -> Self { PropertyGraph { nodes: HashMap::new(), edges: vec![], out_edges: HashMap::new(), in_edges: HashMap::new(), label_index: HashMap::new(), next_edge_id: 0 } }

    pub fn add_node(&mut self, node: GNode) {
        let id = node.id.clone();
        for label in &node.labels {
            self.label_index.entry(label.clone()).or_default().push(id.clone());
        }
        self.nodes.insert(id, node);
    }

    pub fn add_edge(&mut self, mut edge: GEdge) -> usize {
        if edge.id.is_empty() { edge.id = format!("e{}", self.next_edge_id); }
        self.next_edge_id += 1;
        let from = edge.from.clone();
        let to = edge.to.clone();
        let idx = self.edges.len();
        self.out_edges.entry(from).or_default().push(idx);
        self.in_edges.entry(to).or_default().push(idx);
        self.edges.push(edge);
        idx
    }

    pub fn get_node(&self, id: &str) -> Option<&GNode> { self.nodes.get(id) }

    pub fn neighbors(&self, id: &str) -> Vec<&str> {
        self.out_edges.get(id).map(|indices| indices.iter().filter_map(|&i| self.edges.get(i)).map(|e| e.to.as_str()).collect()).unwrap_or_default()
    }

    pub fn out_degree(&self, id: &str) -> usize { self.out_edges.get(id).map(|v| v.len()).unwrap_or(0) }

    pub fn in_degree(&self, id: &str) -> usize { self.in_edges.get(id).map(|v| v.len()).unwrap_or(0) }

    pub fn nodes_by_label(&self, label: &str) -> Vec<&GNode> {
        self.label_index.get(label).map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect()).unwrap_or_default()
    }

    /// BFS traversal returning node IDs
    pub fn bfs(&self, start: &str, max_depth: usize) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = vec![];
        visited.insert(start.to_string());
        queue.push_back((start.to_string(), 0));
        while let Some((node, depth)) = queue.pop_front() {
            result.push(node.clone());
            if depth >= max_depth { continue; }
            for neighbor in self.neighbors(&node) {
                if visited.insert(neighbor.to_string()) {
                    queue.push_back((neighbor.to_string(), depth + 1));
                }
            }
        }
        result
    }

    /// Pattern match: find all (subject, predicate, object) triples
    pub fn match_pattern(&self, subject_label: Option<&str>, predicate: Option<&str>, object_label: Option<&str>) -> Vec<Triple> {
        let subject_ids: HashSet<&str> = match subject_label {
            Some(label) => self.nodes_by_label(label).iter().map(|n| n.id.as_str()).collect(),
            None => self.nodes.keys().map(|k| k.as_str()).collect(),
        };
        let mut results = vec![];
        for &sid in &subject_ids {
            for &eidx in self.out_edges.get(sid).unwrap_or(&vec![]) {
                if let Some(edge) = self.edges.get(eidx) {
                    if predicate.map_or(true, |p| p == edge.edge_type) {
                        if let Some(target) = self.nodes.get(&edge.to) {
                            if object_label.map_or(true, |l| target.labels.contains(&l.to_string())) {
                                results.push(Triple { subject: sid.to_string(), predicate: edge.edge_type.clone(), object: edge.to.clone() });
                            }
                        }
                    }
                }
            }
        }
        results
    }

    /// Subgraph extraction (BFS from a set of seeds)
    pub fn subgraph(&self, seeds: &[&str], max_depth: usize) -> PropertyGraph {
        let reachable: HashSet<String> = seeds.iter().flat_map(|&s| self.bfs(s, max_depth)).collect();
        let mut sg = PropertyGraph::new();
        for id in &reachable {
            if let Some(node) = self.nodes.get(id) { sg.add_node(node.clone()); }
        }
        for edge in &self.edges {
            if reachable.contains(&edge.from) && reachable.contains(&edge.to) { sg.add_edge(edge.clone()); }
        }
        sg
    }

    pub fn summary(&self) -> String {
        format!("Graph: {} nodes, {} edges, {} labels", self.nodes.len(), self.edges.len(), self.label_index.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_social_graph() -> PropertyGraph {
        let mut g = PropertyGraph::new();
        g.add_node(GNode::new("alice").with_label("Person").with_prop("name", "Alice"));
        g.add_node(GNode::new("bob").with_label("Person").with_prop("name", "Bob"));
        g.add_node(GNode::new("rust").with_label("Language").with_prop("name", "Rust"));
        g.add_edge(GEdge::new("", "alice", "bob", "knows"));
        g.add_edge(GEdge::new("", "alice", "rust", "likes"));
        g
    }

    #[test]
    fn test_add_and_get() {
        let g = make_social_graph();
        assert!(g.get_node("alice").is_some());
        assert!(g.get_node("missing").is_none());
    }

    #[test]
    fn test_neighbors() {
        let g = make_social_graph();
        let n = g.neighbors("alice");
        assert!(n.contains(&"bob"));
        assert!(n.contains(&"rust"));
    }

    #[test]
    fn test_degree() {
        let g = make_social_graph();
        assert_eq!(g.out_degree("alice"), 2);
        assert_eq!(g.in_degree("bob"), 1);
    }

    #[test]
    fn test_nodes_by_label() {
        let g = make_social_graph();
        let people = g.nodes_by_label("Person");
        assert_eq!(people.len(), 2);
    }

    #[test]
    fn test_bfs() {
        let g = make_social_graph();
        let result = g.bfs("alice", 1);
        assert!(result.contains(&"alice".to_string()));
        assert!(result.contains(&"bob".to_string()));
    }

    #[test]
    fn test_match_pattern() {
        let g = make_social_graph();
        let triples = g.match_pattern(Some("Person"), Some("knows"), Some("Person"));
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "alice");
    }

    #[test]
    fn test_match_pattern_partial() {
        let g = make_social_graph();
        let triples = g.match_pattern(None, Some("likes"), None);
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_subgraph() {
        let g = make_social_graph();
        let sg = g.subgraph(&["alice"], 1);
        assert!(sg.get_node("alice").is_some());
        assert!(sg.get_node("bob").is_some());
        assert!(sg.get_node("rust").is_some());
    }

    #[test]
    fn test_summary() {
        let g = PropertyGraph::new();
        assert!(g.summary().contains("0 nodes"));
    }
}
