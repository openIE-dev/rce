//! GraphBuilder: scope-aware construction of FunctionGraphs.

use std::collections::HashMap;

use crate::graph::{FunctionGraph, GraphNode, Param, Wire, WireKind};

/// Builds a `FunctionGraph` by accumulating nodes and wires with lexical scope tracking.
///
/// Variables are tracked as node indices in a scope stack. When a variable is referenced,
/// the builder walks the stack top-down to find the defining node. This flattens Lux's
/// nested lexical scopes into the graph's flat node+wire representation.
pub struct GraphBuilder {
    nodes: Vec<GraphNode>,
    wires: Vec<Wire>,
    /// Scope stack: each level maps variable names to their producing node index + type.
    scopes: Vec<HashMap<String, (usize, String)>>,
    /// Tracks which variables are mutated inside the current context (for loop var detection).
    mutations: Vec<String>,
    /// Function signature registry: name → (param_types, return_type).
    /// Populated at module level so Call expressions can infer return types.
    func_sigs: HashMap<String, (Vec<String>, String)>,
    /// Counter for generating unique lambda names.
    lambda_counter: usize,
    /// Lifted lambda functions accumulated during lowering.
    pub lifted_functions: Vec<FunctionGraph>,
    /// Trait method dispatch: maps (method_name) → vec of (trait_name, target_type)
    /// so method calls can resolve `obj.method()` to `Trait__method`.
    trait_dispatch: HashMap<String, Vec<(String, String)>>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            wires: Vec::new(),
            scopes: vec![HashMap::new()],
            mutations: Vec::new(),
            func_sigs: HashMap::new(),
            lambda_counter: 0,
            lifted_functions: Vec::new(),
            trait_dispatch: HashMap::new(),
        }
    }

    /// Append a node, return its index.
    pub fn push_node(&mut self, node: GraphNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Append a wire from `from` node to `to` node at `port`.
    pub fn push_wire(&mut self, from: usize, to: usize, port: i32, kind: WireKind) {
        self.wires.push(Wire { from, to, port_in_idx: port, wire_kind: kind });
    }

    /// Enter a new lexical scope.
    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Exit the current lexical scope.
    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    /// Bind a variable name to a node index in the current scope.
    pub fn bind_var(&mut self, name: &str, node_idx: usize, rce_type: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), (node_idx, rce_type.to_string()));
        }
    }

    /// Look up a variable by walking the scope stack top-down.
    /// Returns (node_index, rce_type_string).
    pub fn lookup_var(&self, name: &str) -> Option<(usize, String)> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.get(name) {
                return Some(entry.clone());
            }
        }
        None
    }

    /// Update a variable's binding to a new node (for mutation/SSA).
    pub fn rebind_var(&mut self, name: &str, node_idx: usize, rce_type: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), (node_idx, rce_type.to_string()));
                return;
            }
        }
        self.bind_var(name, node_idx, rce_type);
    }

    /// Take a snapshot of parent scope bindings (for detecting mutations in child scopes).
    /// Returns all bindings from scopes below the current top scope.
    pub fn snapshot_bindings(&self) -> HashMap<String, (usize, String)> {
        let mut snap = HashMap::new();
        // Walk all scopes except the top one (which is about to be a new child scope)
        for scope in &self.scopes {
            for (k, v) in scope {
                snap.insert(k.clone(), v.clone());
            }
        }
        snap
    }

    /// Record that a variable was mutated (for loop-var detection).
    pub fn record_mutation(&mut self, name: &str) {
        if !self.mutations.contains(&name.to_string()) {
            self.mutations.push(name.to_string());
        }
    }

    /// Add a function parameter as the next node.
    pub fn add_param(&mut self, name: &str, rce_type: &str) -> usize {
        let idx = self.push_node(GraphNode::Param(Param {
            name: name.to_string(),
            data_type: rce_type.to_string(),
        }));
        self.bind_var(name, idx, rce_type);
        idx
    }

    /// Current node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Register a function signature for type inference during Call lowering.
    pub fn register_func_sig(&mut self, name: &str, param_types: Vec<String>, return_type: String) {
        self.func_sigs.insert(name.to_string(), (param_types, return_type));
    }

    /// Look up a registered function's return type.
    pub fn func_return_type(&self, name: &str) -> Option<&str> {
        self.func_sigs.get(name).map(|(_, rt)| rt.as_str())
    }

    /// Generate a unique name for a lambda function.
    pub fn fresh_lambda_name(&mut self) -> String {
        let n = self.lambda_counter;
        self.lambda_counter += 1;
        format!("__lambda_{n}")
    }

    /// Get the function signature registry (for passing to child builders).
    pub fn func_sigs(&self) -> &HashMap<String, (Vec<String>, String)> {
        &self.func_sigs
    }

    /// Set the function signature registry (for inheriting from parent).
    pub fn set_func_sigs(&mut self, sigs: HashMap<String, (Vec<String>, String)>) {
        self.func_sigs = sigs;
    }

    /// Register a trait impl: method_name can be resolved to trait_name__method_name
    /// when called on an object of target_type.
    pub fn register_trait_impl(&mut self, trait_name: &str, target_type: &str, method_name: &str) {
        self.trait_dispatch
            .entry(method_name.to_string())
            .or_default()
            .push((trait_name.to_string(), target_type.to_string()));
    }

    /// Resolve a method call: given a method name and the receiver type,
    /// return the mangled function name if it matches a trait impl.
    pub fn resolve_method(&self, method_name: &str, receiver_type: &str) -> Option<String> {
        if let Some(impls) = self.trait_dispatch.get(method_name) {
            // Try exact type match first
            for (trait_name, target_type) in impls {
                if target_type == receiver_type {
                    return Some(format!("{trait_name}__{method_name}"));
                }
            }
            // If only one impl exists for this method, use it (monomorphic dispatch).
            // Must use the same `Trait__method` mangling as the exact-match path
            // and as lowering (lower_item.rs registers `{trait}__{method}`).
            if impls.len() == 1 {
                return Some(format!("{}__{}", impls[0].0, method_name));
            }
        }
        None
    }

    /// Get the trait dispatch table (for passing to child builders).
    pub fn trait_dispatch(&self) -> &HashMap<String, Vec<(String, String)>> {
        &self.trait_dispatch
    }

    /// Set the trait dispatch table (for inheriting from parent).
    pub fn set_trait_dispatch(&mut self, dispatch: HashMap<String, Vec<(String, String)>>) {
        self.trait_dispatch = dispatch;
    }

    /// Build the final `FunctionGraph`.
    pub fn build(self, name: &str, output_node: usize, return_type: &str) -> FunctionGraph {
        FunctionGraph {
            name: name.to_string(),
            nodes: self.nodes,
            wires: self.wires,
            output_node,
            return_type: return_type.to_string(),
        }
    }

    /// Extract nodes and wires (consumes builder) — for building sub-graphs.
    pub fn into_parts(self) -> (Vec<GraphNode>, Vec<Wire>) {
        (self.nodes, self.wires)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_method_mangling_is_consistent_across_paths() {
        // Lowering registers trait-impl functions as `{trait}__{method}`
        // (see lower_item.rs). Both resolve paths must produce that exact name.
        let mut b = GraphBuilder::new();
        b.register_trait_impl("Greet", "Person", "hello");
        // Exact-type match.
        assert_eq!(b.resolve_method("hello", "Person").as_deref(), Some("Greet__hello"));
        // Monomorphic fallback (single impl, non-matching receiver type) must
        // produce the SAME double-underscore name, not "Greet_hello".
        assert_eq!(b.resolve_method("hello", "Robot").as_deref(), Some("Greet__hello"));
    }

    #[test]
    fn resolve_method_ambiguous_returns_none() {
        let mut b = GraphBuilder::new();
        b.register_trait_impl("Greet", "Person", "hello");
        b.register_trait_impl("Wave", "Robot", "hello");
        // Two impls, neither matching the receiver → no monomorphic shortcut.
        assert_eq!(b.resolve_method("hello", "Alien"), None);
    }
}
