//! Structured ingestion — derive a language's construct coverage from its grammar.
//!
//! The deterministic per-language knowledge already exists at scale: tree-sitter grammars
//! (`node-types.json`), ASTs, vendor manuals, and VS Code language extensions. Rather than
//! hand-authoring LUTs (a random walk), we ingest the grammar's node types and map them onto the
//! universal construct taxonomy (`structure::CONSTRUCTS`). This yields, per language, exactly which
//! constructs its grammar realizes — the structural basis for completing its LUT to 100%.

use std::collections::BTreeSet;

use serde_json::Value;

/// Map a tree-sitter named node type to the universal constructs it realizes.
fn node_constructs(t: &str, out: &mut BTreeSet<&'static str>) {
    match t {
        "integer" | "number" | "int" | "integer_literal" => { out.insert("int_lit"); }
        "float" | "float_literal" | "decimal" => { out.insert("float_lit"); }
        "string" | "string_literal" | "str" => { out.insert("str_lit"); }
        "true" | "false" | "boolean" | "bool_literal" => { out.insert("bool_lit"); }
        "identifier" | "name" | "ident" | "scoped_identifier" | "type_identifier"
        | "field_identifier" | "self" | "super" | "crate" | "metavariable" => { out.insert("var"); }
        "binary_expression" | "binary_operator" => {
            for c in ["add","sub","mul","div","mod","eq","ne","lt","gt","le","ge","and","or"] { out.insert(c); }
        }
        "add_expr" => { out.insert("add"); }
        "sub_expr" => { out.insert("sub"); }
        "mul_expr" => { out.insert("mul"); }
        "div_expr" => { out.insert("div"); }
        "mod_expr" => { out.insert("mod"); }
        "cmp_expr" => { for c in ["eq","ne","lt","gt","le","ge"] { out.insert(c); } }
        "and_expr" => { out.insert("and"); }
        "or_expr" => { out.insert("or"); }
        "unary_expression" | "negation" | "neg" => { out.insert("neg"); }
        "not_expression" | "not_operator" | "logical_not" | "not_op" | "bitnot" => { out.insert("not"); }
        "if_expression" | "conditional_expression" | "if_statement" | "ternary_expression" | "if_expr" | "if_stmt"
        | "match_expression" | "when_expression" | "cond_expression" => { out.insert("if_expr"); }
        "call_expression" | "call" | "function_call" | "invocation" | "macro_invocation"
        | "method_invocation" | "invoke_expression" => { out.insert("call"); }
        "subscript_expression" | "index_expression" | "subscript" | "index" => { out.insert("index"); }
        "attribute" | "field_expression" | "member_expression" | "selector" | "member" | "tuple_member"
        | "scoped_type_identifier" => { out.insert("member"); }
        "list" | "array" | "array_expression" | "list_expression" | "tuple" | "list_comp"
        | "tuple_expression" | "vector_expression" => { out.insert("list"); }
        // Whole-program (statement/declaration) layer — the IR-kind expansion.
        "let_declaration" | "variable_declaration" | "val_declaration" | "let_binding"
        | "let_expression" | "var_declaration" => { out.insert("let"); }
        "assignment_expression" | "assignment" | "assign_expression" | "set_expression"
        | "assignment_statement" => { out.insert("assign"); }
        "return_statement" | "return_expression" | "return" => { out.insert("return"); }
        "while_expression" | "while_statement" | "loop_expression" | "while" => { out.insert("while"); }
        "for_expression" | "for_statement" | "for_in_expression" | "foreach_statement"
        | "for_in_statement" | "for" => { out.insert("for"); }
        "use_declaration" | "import_statement" | "import_declaration" | "include_directive"
        | "require_expression" | "open_statement" | "use_item" | "import" => { out.insert("import"); }
        "type_identifier" | "type_declaration" | "struct_item" | "enum_item" | "trait_item"
        | "impl_item" | "class_declaration" | "interface_declaration" | "type_alias_declaration"
        | "struct_specifier" | "enum_specifier" | "class_specifier" | "type_item" => { out.insert("type_def"); }
        "function_item" | "function_declaration" | "function_definition" | "method_declaration"
        | "method_definition" | "fn_item" | "function" | "def_declaration" | "fun_declaration" => { out.insert("fn"); }
        "if_statement" | "if_expression" => { out.insert("if_stmt"); }
        "match_expression" | "match_arm" | "switch_statement" | "switch_expression"
        | "case_expression" | "pattern" | "match_pattern" | "case_clause" => { out.insert("match"); }
        "closure_expression" | "lambda_expression" | "arrow_function" | "function_expression"
        | "anonymous_function" | "lambda" | "closure" => { out.insert("fn"); }
        "type_parameters" | "generic_type" | "type_parameter" | "generic_function"
        | "type_arguments" | "generic_parameter" => { out.insert("generic"); }
        "async_function_declaration" | "async_function" | "async_block" | "async" => { out.insert("async"); }
        "await_expression" | "await" => { out.insert("await"); }
        "attribute_item" | "attribute" | "annotation" | "decorator" | "annotation_argument"
        | "inner_attribute_item" => { out.insert("attribute"); }
        "macro_invocation" | "macro_definition" | "macro_rules" | "macro_call"
        | "macro" => { out.insert("macro"); }
        _ => {}
    }
}

/// Ingest a tree-sitter `node-types.json` value; return the sorted universal constructs the
/// grammar realizes. This is the structural coverage basis for the language.
pub fn constructs_from_node_types(nt: &Value) -> Vec<&'static str> {
    let mut out = BTreeSet::new();
    if let Some(arr) = nt.as_array() {
        for n in arr {
            if let Some(t) = n.get("type").and_then(|v| v.as_str()) {
                node_constructs(t, &mut out);
            }
        }
    }
    out.into_iter().collect()
}

/// Per-language grammar coverage: (mapped, total, fraction) over the grammar's NAMED node
/// types. The construct matrix is 100%; this is the measured per-language remainder —
/// which node types still need a taxonomy mapping / LUT row.
pub fn grammar_coverage(nt: &Value) -> (usize, usize, f64) {
    let (mut total, mut mapped) = (0, 0);
    if let Some(arr) = nt.as_array() {
        for n in arr {
            if !n.get("named").and_then(|v| v.as_bool()).unwrap_or(false) {
                continue;
            }
            total += 1;
            if let Some(t) = n.get("type").and_then(|v| v.as_str()) {
                let mut s = BTreeSet::new();
                node_constructs(t, &mut s);
                if !s.is_empty() {
                    mapped += 1;
                }
            }
        }
    }
    let frac = if total == 0 { 0.0 } else { mapped as f64 / total as f64 };
    (mapped, total, frac)
}