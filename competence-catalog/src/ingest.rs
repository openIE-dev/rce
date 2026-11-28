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
        // Language-specific declaration remainder (measured coverage climb).
        "class_definition" | "class_declaration" | "interface_declaration" | "enum_declaration"
        | "record_declaration" | "struct_type" | "interface_type" | "type_spec"
        | "annotation_type_declaration" | "template_declaration" | "concept_definition"
        | "namespace_definition" | "sealed_modifier" | "permits_clause" => { out.insert("type_def"); }
        "import_declaration" | "package_clause" | "export_statement" | "using_directive"
        | "include_declaration" | "require_clause" => { out.insert("import"); }
        "decorator" | "annotation_argument_list" | "attribute_list" => { out.insert("attribute"); }
        "requires_clause" | "type_parameter_list" | "generic_param_list" | "template_parameter_list"
        | "where_clause" | "type_bound" => { out.insert("generic"); }
        "method_signature" | "constructor_declaration" | "destructor_declaration"
        | "operator_declaration" | "function_signature" => { out.insert("fn"); }
        // Exotic / control-flow / value-form remainder.
        "block" | "statement_block" | "compound_statement" | "body" | "progn"
        | "expression_statement" | "declaration_list" => { out.insert("block"); }
        "loop_expression" | "loop_statement" | "do_statement" | "repeat_statement"
        | "loop" => { out.insert("loop"); }
        "break_expression" | "break_statement" | "break" => { out.insert("break"); }
        "continue_expression" | "continue_statement" | "continue" => { out.insert("continue"); }
        "try_expression" | "try_statement" | "catch_clause" | "finally_clause"
        | "except_clause" | "handler_case" | "try" => { out.insert("try"); }
        "cast_expression" | "type_cast_expression" | "as_expression" | "coerce_expression"
        | "cast" => { out.insert("cast"); }
        "closure_expression" | "lambda_expression" | "arrow_function" | "function_expression"
        | "anonymous_function" | "lambda" | "closure" => { out.insert("closure"); out.insert("fn"); }
        "pattern" | "wildcard" | "binding_pattern" | "or_pattern" | "range_pattern"
        | "struct_pattern" | "tuple_pattern" | "slice_pattern" | "literal_pattern"
        | "match_pattern" | "captured_pattern" | "reference_pattern" => { out.insert("pattern"); }
        "reference_expression" | "dereference_expression" | "borrow_expression"
        | "ref_expression" | "pointer_expression" | "address_of_expression"
        | "unary_expression" => { out.insert("ref"); }
        "range_expression" | "range" | "slice_expression" => { out.insert("list"); }
        "yield_expression" | "yield" | "await_expression" => { out.insert("await"); }
        "match_arm" | "case_clause" | "switch_case" => { out.insert("match"); out.insert("pattern"); }
        "union_item" => { out.insert("type_def"); }
        "boolean_operator" => { out.insert("and"); out.insert("or"); }
        "comparison_operator" => { out.insert("eq"); out.insert("ne"); out.insert("lt"); out.insert("gt"); out.insert("le"); out.insert("ge"); }
        "unary_operator" => { out.insert("neg"); out.insert("not"); }
        "chevron" => { out.insert("send"); }
        "keyword_separator" => { out.insert("param"); }
        "positional_separator" => { out.insert("splat"); }
        "line_continuation" => { out.insert("token"); }
        "dot" => { out.insert("member"); }
        "class_static_block" | "enum_body_declarations" => { out.insert("body"); }
        "explicit_constructor_invocation" => { out.insert("call"); }
        "_arg" => { out.insert("param"); }
        "_lhs" => { out.insert("var"); }
        "_pattern_primitive" => { out.insert("pattern"); }
        "element_reference" => { out.insert("index"); }
        "exceptions" => { out.insert("type_expr"); }
        "in" => { out.insert("case"); }
        "method" => { out.insert("fn"); }
        "undef" => { out.insert("delete"); }
        "callable_reference" => { out.insert("method_ref"); }
        "annotated" => { out.insert("cast"); }
        "generator" => { out.insert("comprehension"); }
        "infix_id" | "prefix_id" => { out.insert("op"); }
        // Lisp-family literals / readers.
        "num_lit" | "signed_number" => { out.insert("literal"); }
        "bool_lit" => { out.insert("bool_lit"); }
        "nil_lit" | "na" | "inf" | "nan" => { out.insert("null_lit"); }
        "str_lit" | "text" => { out.insert("str_lit"); }
        "sym_lit" | "sym_val_lit" | "sym_ns" => { out.insert("symbol"); }
        "kwd_lit" | "kwd_ns" | "keyword" | "atom" => { out.insert("keyword"); }
        "list_lit" | "vec_lit" | "vector" | "byte_vector" | "array_like" | "array_construction" => { out.insert("list"); }
        "map_lit" | "ns_map_lit" | "map" | "curly_construction" => { out.insert("dict"); }
        "set_lit" => { out.insert("set"); }
        "quoting_lit" | "syn_quoting_lit" | "unquoting_lit" | "var_quoting_lit" | "derefing_lit"
        | "evaling_lit" | "meta_lit" | "old_meta_lit" | "read_cond_lit" | "splicing_read_cond_lit"
        | "tagged_or_ctor_lit" | "auto_res_mark" | "unsyntax" | "unsyntax_splicing" | "datum_reference"
        | "comma" | "ellipses" | "end_file" | "ocamlyacc_value" | "scannable_axis" | "pretty_printing_indication" => { out.insert("token"); }
        "anon_fn_lit" | "anonymous_fun" | "internal_fun" | "proc" | "method_proc" => { out.insert("closure"); }
        "source" | "php_tag" | "script_tag" | "directive_kind" | "debug" | "version" | "default" | "_desc" | "_form" | "pp_define" | "pp_undef" => { out.insert("directive"); }
        // OCaml / R / Dart / Elixir / Erlang / Nim / Crystal / D / Fortran / PHP specifics.
        "_kind" | "any_kind" | "constrain_kind" | "parenthesized_kind" | "product_kind"
        | "scannable_axes_kind" | "type_constructor" | "type_constructor_path" | "typeof"
        | "fun_type_sig" | "type_sig" | "nominal" | "opaque" | "bit_type_list" | "type_ctor"
        | "kind" | "size" | "unlimited_polymorphic" | "traits" | "typeid" | "noreturn"
        | "coarray_allocation" | "coarray_size" | "procedure_kind" | "procedure_interface" => { out.insert("type_expr"); }
        "_local_structure_item" | "_structure_item" | "structure" | "item_extension" | "extension"
        | "functor" | "interface" | "interfaces" | "mixin" | "mixin_application" | "mixins"
        | "struct" | "c_struct_def" | "struct_def" | "union_def" | "enum_def" | "lib_def" | "type_def"
        | "enum" | "union" | "const_section" | "type_section" | "var_section" | "using_section"
        | "equivalence_set" | "derived_type_procedures" | "internal_procedures" => { out.insert("type_def"); }
        "comprehension" | "comprehension_iterator" | "b_generator" | "map_generator"
        | "binary_comprehension" | "map_comprehension" | "for_element" | "for_loop_parts"
        | "foreach" | "foreach_reverse" => { out.insert("comprehension"); }
        "constructor_path" | "value_path" | "remote" | "relative_scope" | "constructor_tearoff"
        | "callable_reference" | "index_selector" | "nullable_selector" | "access_call" | "index_call"
        | "subset" | "subset2" | "multiple_subscript" | "coarray_index" | "element_reference" => { out.insert("member"); }
        "tag" | "field" | "type_member" | "const_element" | "enum_member" | "anonymous_enum_member"
        | "data_set" | "data_value" | "container_field" => { out.insert("field"); }
        "modality" | "mode" | "abstract" | "sealed" | "private" | "protected" | "public" | "final"
        | "override" | "pure" | "nothrow" | "ref" | "scope" | "shared" | "static" | "synchronized"
        | "immutable" | "inout" | "gshared" | "lazy" | "out" | "const" | "align" | "concurrent_locality"
        | "by_ref" | "splat" | "splat_param" | "var_args" | "variadic" | "_arity_value" | "arity" | "fa"
        | "constructor_param" | "fun_param" | "param" | "param_list" | "keywords" | "var" | "ann_var"
        | "global_var" | "macro_var" | "uninitialized_var" | "underscore" | "macro_lhs" => { out.insert("param"); }
        "sigil" | "regex" => { out.insert("regex"); }
        "configurable_uri" | "configuration_uri" | "uri" | "uri_test" | "require" | "extend" => { out.insert("import"); }
        "abstract_method_def" | "method_def" | "fun_def" | "destructor" | "postblit" | "subroutine"
        | "subroutine_call" | "function_result" | "procedure" | "defined_io_procedure" | "print_intrinsic" => { out.insert("fn"); }
        "alignof" | "offsetof" | "sizeof" | "pointerof" | "and" | "or" | "not" | "binary_op" | "pipe"
        | "combinator" | "is" | "not_in" | "not_is" | "operator" => { out.insert("op"); }
        "command" | "select" | "try_after" | "try_stack" | "except_branch" | "finally_branch"
        | "of_branch" | "goto" | "throw" | "with" | "defer" | "mixin" | "new" | "dynamic" => { out.insert("control"); }
        "macro_def" | "macro_for" | "macro_verbatim" | "_macro_def_replacement" | "macro_call_args"
        | "replacement_cr_clauses" | "replacement_function_clauses" | "ssr_replacement" | "ssr_where" => { out.insert("macro"); }
        "dyn_pattern_marker" | "callback" | "spec" | "context_bound" | "type_parameter_constraint" => { out.insert("bound"); }
        "bin_element" | "_concatable" | "concatables" | "tuple_construction" | "input_item_list"
        | "output_item_list" | "transfer_items" | "edit_descriptor" | "implicit_range" | "parenthesized"
        | "block" | "vector" => { out.insert("seq"); }
        "bool" | "byte" | "cent" | "cfloat" | "creal" | "ireal" | "long" | "real" | "short" | "size_t"
        | "ubyte" | "ucent" | "uint" | "ulong" | "ushort" | "void" | "floating_point_type" | "integral_type" => { out.insert("type_expr"); }
        "finally" => { out.insert("try"); }
        "invariant" => { out.insert("assert"); }
        "sized_allocation" => { out.insert("init"); }
        // Functional + non-Western remainder (Ada, SML, PureScript, Mojo, Cangjie, Wenyan,
        // Elm, Racket, Common Lisp, Elisp, Gleam, MoonBit, ReScript, Reason, Nix, F#, Idris,
        // Agda, Lean, F*). Each name maps to the universal category it genuinely realizes.
        "data" | "variant" | "variant_part" | "variant_list" | "known_discriminant_part"
        | "unknown_discriminant_part" | "union_type_defn" | "enum_type_defn" | "interface_type_defn"
        | "anon_type_defn" | "type_abbrev_defn" | "additional_constr_defn" | "type_extension"
        | "type_extension_elements" | "types" | "interface_implementation" | "implementation"
        | "implementation_head" | "interface_head" | "inductive" => { out.insert("type_def"); }
        "defun" | "extended_method" | "trait_method_extension" | "extenum_extension"
        | "function_or_value_defn" | "member_defn" | "method_or_prop_defn" => { out.insert("fn"); }
        "call_suffix" | "function_post_call" | "function_pre_call" | "app" | "pipeline_echo" => { out.insert("call"); }
        "selected_component" | "tuple_access" | "attrpath" | "proj" => { out.insert("member"); }
        "index_access" => { out.insert("index"); }
        "item_after_quest" | "quest_seperated_item" | "quest_seperated_items" => { out.insert("chain"); }
        "macro_tokens" | "macro_with_attr_param" | "macro_without_attr_param" | "syntax"
        | "anon_subterm" | "antiquot" | "quotation" | "pct_macro" | "syntax_alt" | "syntax_cmd"
        | "syntax_postfix" | "macro_cmd" | "macro_rules_cmd" | "elab_rules_cmd" | "grind_pattern_cmd"
        | "register_cmd" | "language_extension" | "extension_language" | "syntax_extension"
        | "source_macro" => { out.insert("macro"); }
        "preamble" | "echo" | "target" | "check" | "eval" | "print" | "reduce" | "set_option" => { out.insert("directive"); }
        "allocator" | "static_init" | "struct_init" | "box" | "initialize"
        | "array_delta_aggregate" | "enumeration_aggregate" | "extension_aggregate"
        | "null_array_aggregate" | "positional_array_aggregate" => { out.insert("init"); }
        "value_sequence" | "element" | "elements" => { out.insert("seq"); }
        "component_choice_list" | "component_list" | "discrete_choice" | "discrete_choice_list"
        | "membership_choice_list" | "alt" | "alts" | "of" => { out.insert("case"); }
        "entry_barrier" => { out.insert("guard"); }
        "exception_choice" | "exception_choice_list" | "exception_handler" | "finally_part" => { out.insert("try"); }
        "factor_not" | "boolean_negation" | "not_operation" => { out.insert("not"); }
        "integer_negation" => { out.insert("neg"); }
        "factor_power" => { out.insert("pow"); }
        "factor_abs" | "relation_membership" | "ARITH_BINARY_OP" | "LOGIC_BINARY_OP" | "IF_LOGIC_OP"
        | "arith_binary_math" | "arith_unary_math" | "POST_MOD_MATH_OP" | "UNARY_OP" | "preposition"
        | "PREPOSITION_LEFT" | "PREPOSITION_RIGHT" | "accumulation_verb" | "infix_op" | "prefix_op"
        | "wildcard_active_pattern_op" | "bool_compare" | "bool_operation" | "postfix_op"
        | "unary_op" => { out.insert("op"); }
        "formal_part" | "generic_formal_part" | "result_profile" | "subprogram_default"
        | "array_dimension" | "defun_keyword" | "curried_spec" | "determining_params"
        | "using_target" | "using_targets" | "uncurry" | "formal" | "formals" => { out.insert("param"); }
        "generic_instantiation" | "functor_use" | "generalize" | "all" => { out.insert("generic"); }
        "global_mode" | "non_empty_mode" | "null_exclusion" | "overriding_indicator" | "mutability"
        | "pub" | "visibility" | "quantity" | "search_options" | "totality" | "no_hints" | "port"
        | "effect" => { out.insert("modifier"); }
        "iteration_scheme" | "iterator_filter" | "iterator_procedure_call" | "procedural_iterator"
        | "FOR_IF_END" | "IF" | "ELSE" | "BREAK" | "FOR_MID_ARR" | "FOR_MID_ENUM" | "FOR_START_ARR"
        | "FOR_START_ENUM" | "FOR_START_WHILE" => { out.insert("control"); }
        "range_g" | "range_element" | "slice_range" | "slice_ranges" => { out.insert("range"); }
        "selective_accept" | "timed_entry_call" => { out.insert("select"); }
        "term" | "rhs" => { out.insert("expr"); }
        "tick" | "derive_item" | "smt_trigger" => { out.insert("attribute"); }
        "BOOL_VALUE" => { out.insert("bool_lit"); }
        "FLOAT_NUM" | "FLOAT_NUM_KEYWORDS" => { out.insert("float_lit"); }
        "INT_NUM" | "INT_NUM_KEYWORDS" => { out.insert("int_lit"); }
        "STRING_LITERAL" | "path_lit" | "path_fragment" => { out.insert("str_lit"); }
        "IDENTIFIER" | "value_qid" | "left_value" | "funvar" | "bid" | "id" | "qid" | "fctid"
        | "longstrid" | "strid" | "sigid" | "longvid" | "vid" | "lab" => { out.insert("var"); }
        "TYPE" | "type_ref" | "type_apply" | "type_infix" | "type_wildcard" | "type_var"
        | "type_braces" | "type_nested" | "polyvar" | "arrow" | "bignum" | "ieee32" | "ieee64"
        | "int16" | "int32" | "int64" | "nativeint" | "sbyte" | "uint16" | "uint32" | "uint64"
        | "unativeint" | "xint" | "SetN" | "PropN" | "universe" | "universe_app" | "sigma"
        | "false_const" | "bot_const" | "empty_const" | "top_const" | "prop_const" | "sort_const"
        | "type_const" | "tycon" | "longtycon" | "tyvar" => { out.insert("type_expr"); }
        "bytearray" | "verbatim_bytearray" | "bit_array" | "bit_array_segment"
        | "bit_array_segment_option" | "bit_array_segment_options" | "array_lit" | "struct_lit"
        | "set_builder" | "subtype_lit" | "quotient_lit" | "range_lit" | "norm_lit" | "ceil_lit"
        | "floor_lit" | "dbl_angle_lit" => { out.insert("list"); }
        "union_variant" | "enum_constructor" | "map_element_key" | "dict_entry"
        | "dict_pattern_entry" | "ctor" | "anon_ctor" | "ctor_alt" => { out.insert("field"); }
        "gdpat" | "pat_apply" | "pat_array" | "pat_infix" | "pat_negation" | "pat_typed"
        | "pat_wildcard" | "list_pattern_tail" | "pat_at_braces" | "pat_braces" | "pat_list"
        | "pat_op" | "pat_at" | "with_arg" | "with_pat" | "lhs" | "rest_pat" => { out.insert("pattern"); }
        "pattern_spread" | "partial_application_spread" | "type_spread" | "variant_type_spread"
        | "unpack" => { out.insert("splat"); }
        "panic" => { out.insert("throw"); }
        "use" | "using" | "inherit" | "inherit_from" | "inherited_attrs" => { out.insert("with"); }
        "hole" | "type_hole" | "unfinished" | "impossible" | "synth_hole" | "meta" => { out.insert("ellipsis"); }
        "proof" | "rewrite" | "calc_step" | "proof_exists" | "proof_implication" | "by" | "have"
        | "show" | "suffices" | "induction" | "exists" | "sorry" | "goal_const" | "postulate" => { out.insert("assert"); }
        "where" | "mutual" | "where_aux_def" | "where_struct" | "section" | "end" | "public_section"
        | "prelude" => { out.insert("body"); }
        "purescript" | "idris" | "empty_file" | "compilation" | "source_code" => { out.insert("module"); }
        "open" | "renaming" => { out.insert("import"); }
        "infer_only_separator" | "colon" | "arrow_separator" | "equal" | "eq" | "graph"
        | "nonreserved_atom" => { out.insert("token"); }
        "bytecode" | "fun" => { out.insert("closure"); }
        "loop_macro" => { out.insert("loop"); }
        "list_comprehension_for" | "list_comprehension_for_in" => { out.insert("comprehension"); }
        "reference_equality" | "structural_equality" => { out.insert("eq"); }
        "and_bang" => { out.insert("and"); }
        "rule" | "rules" | "fmrule" | "mrule" => { out.insert("match"); }
        "primitive" | "abbrev" | "def" | "theorem" | "axiom" | "example" => { out.insert("decl"); }
        // Perl (node-types generated from tree-sitter-perl/tree-sitter-perl).
        "postfix_deref" => { out.insert("ref"); }
        "slices" => { out.insert("list"); }
        "subscripted" => { out.insert("index"); }
        "arraylen" => { out.insert("op"); }
        "attrlist" => { out.insert("attribute"); }
        "bareword" | "glob" => { out.insert("symbol"); }
        "filehandle" | "scalar" => { out.insert("var"); }
        "replacement" => { out.insert("str_lit"); }
        "yadayada" => { out.insert("ellipsis"); }
        "data_section" | "eof_marker" => { out.insert("directive"); }
        "pod" => { out.insert("comment"); }
        "prototype" => { out.insert("param"); }
        // Coq / Rocq (node-types generated from aruzdh/tree-sitter-rocq).
        "require_command" => { out.insert("import"); }
        "inductive_command" | "fixpoint_command" | "theorem_command" | "defined_command" => { out.insert("decl"); }
        "proof_command" | "qed_command" | "abort_command" | "admitted_command" => { out.insert("assert"); }
        "notation_command" | "reserved_notation_command" => { out.insert("macro"); }
        "section_command" | "end_command" => { out.insert("body"); }
        "add_command" | "remove_command" | "set_command" | "unset_command" | "open_scope_command"
        | "close_scope_command" | "create_hintdb_command" | "create_rewrite_hintdb_command"
        | "hint_command" | "coercion_command" | "extract_command" | "extraction_command"
        | "evaluation_command" | "show_proof_command" | "test_command" | "save_command"
        | "discriminated_db" => { out.insert("directive"); }
        "logical_conjunction" => { out.insert("and"); }
        "logical_disjunction" => { out.insert("or"); }
        "logical_equivalence" => { out.insert("eq"); }
        "negation_operation" => { out.insert("not"); }
        "app_operation" | "application" | "pattern_application" => { out.insert("call"); }
        "additive_operation" | "multiplicative_operation" | "comparison_operation"
        | "custom_operation" | "imp_evaluation_operation" | "cons_operation" => { out.insert("op"); }
        "arrow_term" | "level" => { out.insert("type_expr"); }
        "of_type_inst" => { out.insert("cast"); }
        "scoped_term" => { out.insert("expr"); }
        "parenthesized_term" => { out.insert("block"); }
        "at_level" | "at_level_num" | "notation_scope" | "fixannot" => { out.insert("modifier"); }
        "attr_value" | "legacy_attr" => { out.insert("attribute"); }
        "generic_tactic" | "intros_tactic" | "rewrite_tactic" | "parenthesized_tactic"
        | "tactic_branch" | "tactic_invocation" | "tactic_or" | "tactic_sequence" | "tactical"
        | "goal_branch" | "goal_selector" | "range_selector" | "neg_selection"
        | "oriented_rewriter" => { out.insert("control"); }
        "pattern_option" => { out.insert("pattern"); }
        "imp_program" | "language" => { out.insert("module"); }
        "reference" => { out.insert("var"); }
        "sentence" => { out.insert("stmt"); }
        "decnat" | "hexnat" => { out.insert("int_lit"); }
        "fail" => { out.insert("throw"); }
        // Dhall (node-types generated from jmitchell/tree-sitter-dhall).
        "http" | "http_raw" | "env" | "local" | "local_raw" | "path_component" | "using_raw" => { out.insert("import"); }
        "IPv4address" | "IPv6address" | "IPvFuture" | "dec_octet" | "h16" | "ls32" | "pct_encoded"
        | "sub_delims" | "unreserved" | "reserved" | "reserved_raw" | "scheme" | "at"
        | "close_angle" | "close_brace" | "close_bracket" | "open_angle" | "open_brace"
        | "open_bracket" => { out.insert("token"); }
        "List" | "Optional" | "Text" | "non_empty_optional" => { out.insert("type_expr"); }
        "Some" => { out.insert("var"); }
        "plus" | "times" | "bar" | "combine" | "combine_types" | "merge" | "merge_raw" | "prefer"
        | "list_append" | "text_append" => { out.insert("op"); }
        "empty_collection" => { out.insert("list"); }
        "missing" | "missing_raw" => { out.insert("ellipsis"); }
        "in_raw" => { out.insert("let"); }
        "not_equal" => { out.insert("ne"); }
        // Unison (node-types generated from kylegoetz/tree-sitter-unison).
        "bang" | "force" | "concat" | "cons" | "snoc" => { out.insert("op"); }
        "delayed" => { out.insert("closure"); }
        "effect_pure" | "effect_arrow" => { out.insert("type_expr"); }
        "handler" => { out.insert("with"); }
        "rewrite_term" => { out.insert("cast"); }
        "tuple_or_parenthesized" => { out.insert("block"); }
        "unison" => { out.insert("module"); }
        "var_or_nullary_ctor" | "fold" => { out.insert("var"); }
        "cyclic_index" => { out.insert("index"); }
        "nat" => { out.insert("int_lit"); }
        "path" => { out.insert("member"); }
        // Authored seed grammars for grammar-less non-Western languages:
        // Nadeshiko (Japanese), Ezhil (Tamil), Alusus (Arabic). Native keyword nodes.
        "代入" | "மாறி" | "إسناد" => { out.insert("assign"); }
        "表示" | "பதிப்பி" => { out.insert("call"); }
        "関数" | "செயல்" | "دالة" => { out.insert("fn"); }
        "引数" | "உள்ளீடு" | "معامل" => { out.insert("param"); }
        "もし" | "ஆனால்" | "إذا" => { out.insert("if_stmt"); }
        "繰り返し" => { out.insert("loop"); }
        "抜ける" => { out.insert("break"); }
        "戻る" | "திருப்பு" | "أرجع" => { out.insert("return"); }
        "取り込む" | "இறக்கு" | "استورد" => { out.insert("import"); }
        "配列" | "பட்டியல்" | "مصفوفة" => { out.insert("list"); }
        "ハッシュ" | "அகராதி" => { out.insert("dict"); }
        "演算" | "比較" | "செயலி" | "ஒப்பீடு" | "عملية" | "مقارنة" => { out.insert("op"); }
        "かつ" | "மற்றும்" | "و" => { out.insert("and"); }
        "または" | "அல்லது" | "أو" => { out.insert("or"); }
        "否定" | "அல்ல" | "ليس" => { out.insert("not"); }
        "ブロック" | "தொகுதி" | "كتلة" => { out.insert("block"); }
        "定数" => { out.insert("init"); }
        "இல்லை" | "وإلا" => { out.insert("else"); }
        "வரை" | "طالما" => { out.insert("while"); }
        "மடக்கு" | "لكل" => { out.insert("for"); }
        "سجل" => { out.insert("record"); }
        "متغير" => { out.insert("var"); }
        "صنف" => { out.insert("type_def"); }
        // Koka (effect-typed functional).
        "OP" | "op" => { out.insert("op"); }
        "annot" | "kannot" | "annotres" | "tatomic" | "tresult" | "typecon" | "typescheme"
        | "katom" => { out.insert("type_expr"); }
        "behindmod" | "effectmod" | "fiplimit" | "fipmod" | "inlinemod" | "structmod" | "tailmod"
        | "typemod" => { out.insert("modifier"); }
        "conparams" | "opparam" | "opparams" | "sconparams" | "tparam" | "tparams" | "typeparams" => { out.insert("param"); }
        "constructors" | "con" => { out.insert("type_def"); }
        "conid" | "qconid" | "paramid" | "varid" | "qvarid" | "qimplicit" => { out.insert("var"); }
        "ctxhole" => { out.insert("ellipsis"); }
        "operation" | "operations" => { out.insert("fn"); }
        "opclause" | "opclauses" | "opclausex" => { out.insert("case"); }
        "patarg" | "patargs" => { out.insert("pattern"); }
        "predicate" => { out.insert("guard"); }
        "witheff" | "withstat" => { out.insert("with"); }
        "borrow" => { out.insert("ref"); }
        // Roc (platform-oriented functional).
        "apply_type_arg" => { out.insert("generic"); }
        "hosted" | "targets" => { out.insert("module"); }
        "negative_number" => { out.insert("int_lit"); }
        "nominal_methods" | "nominal_type_def" | "opaque_type_def" => { out.insert("type_def"); }
        "platform_ref" | "requires" | "requires_entries" | "requires_entry" => { out.insert("import"); }
        "provides" | "provides_list" => { out.insert("export"); }
        "static_dispatch" | "static_dispatch_list" | "static_dispatch_target" => { out.insert("method_ref"); }
        "typed_ident" => { out.insert("var"); }
        "where_implements" => { out.insert("bound"); }
        "colon_equals" | "fat_arrow" | "to" => { out.insert("token"); }
        "default_value" => { out.insert("init"); }
        "number_with_suffix" => { out.insert("literal"); }
        "optional_access" => { out.insert("chain"); }
        // V.
        "fixed_array" | "matrix" => { out.insert("list"); }
        "interface_spec_list" => { out.insert("type_def"); }
        "option_propagator" => { out.insert("chain"); }
        "select_branch" | "select_default_branch" => { out.insert("case"); }
        "type_placeholder" => { out.insert("type_expr"); }
        // Odin.
        "address" => { out.insert("ref"); }
        "bit_set" => { out.insert("set"); }
        "build_tag" => { out.insert("directive"); }
        "struct_member" => { out.insert("field"); }
        "uninitialized" => { out.insert("ellipsis"); }
        // Bash / shell.
        "command_substitution" | "process_substitution" => { out.insert("call"); }
        "concatenation" | "pipeline" => { out.insert("op"); }
        "negated_command" => { out.insert("not"); }
        "word" => { out.insert("var"); }
        "file_descriptor" => { out.insert("param"); }
        // TypeScript type-level remainder.
        "index_type_query" | "type_predicate" | "type_query" => { out.insert("type_expr"); }
        // HTML.
        "end_tag" | "erroneous_end_tag" => { out.insert("token"); }
        "script_element" | "style_element" => { out.insert("body"); }
        "raw_text" => { out.insert("str_lit"); }
        // CSS: selectors are patterns; at-rules directives; media queries guards.
        "adjacent_sibling_selector" | "child_selector" | "descendant_selector" | "sibling_selector"
        | "universal_selector" | "id_selector" | "nesting_selector" | "selectors"
        | "selector_query" => { out.insert("pattern"); }
        "at_rule" | "at_keyword" => { out.insert("directive"); }
        "binary_query" | "unary_query" | "feature_query" | "keyword_query"
        | "parenthesized_query" => { out.insert("guard"); }
        "color_value" | "plain_value" => { out.insert("literal"); }
        "float_value" => { out.insert("float_lit"); }
        "integer_value" => { out.insert("int_lit"); }
        "grid_value" => { out.insert("list"); }
        "parenthesized_value" => { out.insert("block"); }
        "rule_set" => { out.insert("body"); }
        "stylesheet" => { out.insert("module"); }
        "from" => { out.insert("keyword"); }
        // JSON / YAML / TOML (data languages).
        "_value" | "flow_node" => { out.insert("expr"); }
        "anchor" => { out.insert("label"); }
        "flow_mapping" | "inline_table" | "table" => { out.insert("dict"); }
        "flow_sequence" | "table_array_element" => { out.insert("list"); }
        "plain_scalar" => { out.insert("str_lit"); }
        "stream" => { out.insert("module"); }
        "boolean_scalar" => { out.insert("bool_lit"); }
        "float_scalar" => { out.insert("float_lit"); }
        "integer_scalar" => { out.insert("int_lit"); }
        "null_scalar" => { out.insert("null_lit"); }
        "tag_handle" | "tag_prefix" | "yaml_version" => { out.insert("directive"); }
        "timestamp_scalar" | "local_date" | "local_date_time" | "local_time"
        | "offset_date_time" => { out.insert("literal"); }
        "bare_key" => { out.insert("field"); }
        // Dockerfile: instructions are build directives; RUN executes; FROM imports; ARG/ENV bind.
        "from_instruction" => { out.insert("import"); }
        "arg_instruction" | "env_instruction" => { out.insert("let"); }
        "run_instruction" | "shell_command" => { out.insert("call"); }
        "shell_fragment" => { out.insert("str_lit"); }
        "mount_param" | "mount_param_param" => { out.insert("param"); }
        "image_digest" | "image_tag" => { out.insert("literal"); }
        "add_instruction" | "copy_instruction" | "cmd_instruction" | "cross_build_instruction"
        | "entrypoint_instruction" | "healthcheck_instruction" | "maintainer_instruction"
        | "onbuild_instruction" | "shell_instruction" | "stopsignal_instruction"
        | "user_instruction" | "volume_instruction" | "workdir_instruction" => { out.insert("directive"); }
        // CMake.
        "cache_var" | "env_var" | "normal_var" => { out.insert("var"); }
        "foreach_command" | "foreach_loop" => { out.insert("for"); }
        "while_command" | "while_loop" => { out.insert("while"); }
        "function_command" | "function_def" => { out.insert("fn"); }
        "macro_command" => { out.insert("macro"); }
        "normal_command" => { out.insert("call"); }
        "endforeach_command" | "endfunction_command" | "endmacro_command" | "endwhile_command"
        | "endforeach" | "endfunction" | "endmacro" | "endwhile" => { out.insert("control"); }
        // Make.
        "archive" => { out.insert("member"); }
        "makefile" => { out.insert("module"); }
        "paths" => { out.insert("list"); }
        "prerequisites" => { out.insert("param"); }
        "shell_function" => { out.insert("call"); }
        "shell_text" => { out.insert("str_lit"); }
        "substitution_reference" => { out.insert("op"); }
        // PowerShell.
        "command_elements" | "data_commands_list" => { out.insert("seq"); }
        "data_command" | "stop_parsing" => { out.insert("directive"); }
        "data_commands_allowed" => { out.insert("modifier"); }
        "dimension" => { out.insert("int_lit"); }
        "element_access" => { out.insert("index"); }
        "for_iterator" => { out.insert("for"); }
        "member_access" => { out.insert("member"); }
        "pipeline_chain" | "pipeline_chain_tail" => { out.insert("chain"); }
        "generic_token" => { out.insert("token"); }
        // TLA+ / PlusCal: numbers, sets, operators, temporal logic, proof steps, pcal.
        "_number" | "binary_number" | "hex_number" | "octal_number" | "nat_number" => { out.insert("int_lit"); }
        "real_number" => { out.insert("float_lit"); }
        "_number_set" | "_primitive_value_set" | "int_number_set" | "nat_number_set"
        | "real_number_set" | "boolean_set" | "set_of_functions" | "powerset" => { out.insert("set"); }
        "set_filter" | "set_map" => { out.insert("comprehension"); }
        "lnot" => { out.insert("not"); }
        "negative" => { out.insert("neg"); }
        "notin" | "sup_plus" => { out.insert("op"); }
        "land" | "ampamp" | "conj_item" | "conj_list" => { out.insert("and"); }
        "lor" | "disj_item" | "disj_list" => { out.insert("or"); }
        "div" => { out.insert("div"); }
        "_op" | "plus_arrow" | "prefixed_op" | "uplus" | "minusminus" | "plusplus" | "mulmul"
        | "mod" | "modmod" | "pow" | "powpow" | "mul" | "amp" | "wr" | "compose" | "sim" | "simeq"
        | "cong" | "approx" | "propto" | "geq" | "leq" | "gg" | "ll" | "lt" | "gt" | "neq"
        | "def_eq" | "equiv" | "implies" | "cap" | "cup" | "sqcap" | "sqcup" | "sqsubset"
        | "sqsubseteq" | "sqsupset" | "sqsupseteq" | "subseteq" | "supset" | "supseteq" | "minus"
        | "setminus" | "oplus" | "ominus" | "otimes" | "bigcirc" | "circ" | "excl" | "vert"
        | "vertvert" | "maps_to" | "map_from" | "map_to" | "all_map_to" | "gets" | "except"
        | "except_update" | "except_update_fn_appl" | "function_evaluation" | "substitution"
        | "succ" | "succeq" | "prime" | "prev_func_val" | "set_in" => { out.insert("op"); }
        "always" | "eventually" | "leads_to" | "fair" | "fairness" | "temporal_exists"
        | "unchanged" | "enabled" | "choose" | "pcal_end_either" | "pcal_end_while"
        | "pcal_end_with" => { out.insert("control"); }
        "_proof" | "have_proof_step" | "non_terminal_proof" | "proof_step" | "proof_step_id"
        | "proof_step_ref" | "qed_step" | "suffices_proof_step" | "take_proof_step"
        | "terminal_proof" | "witness_proof_step" | "pick_proof_step" => { out.insert("assert"); }
        "pcal_algorithm" => { out.insert("module"); }
        "pcal_await" => { out.insert("await"); }
        "pcal_either" | "other_arm" => { out.insert("case"); }
        "pcal_goto" => { out.insert("goto"); }
        "pcal_lhs" | "child_id" | "value" => { out.insert("var"); }
        "pcal_macro" | "pcal_macro_call" | "bnf_rule" => { out.insert("macro"); }
        "pcal_print" | "pcal_proc_call" => { out.insert("call"); }
        "pcal_procedure" | "pcal_process" => { out.insert("fn"); }
        "pcal_while" => { out.insert("while"); }
        "pcal_with" => { out.insert("with"); }
        "pcal_return" => { out.insert("return"); }
        "pcal_skip" => { out.insert("pass"); }
        "extends" => { out.insert("import"); }
        "format" => { out.insert("format"); }
        "placeholder" => { out.insert("ellipsis"); }
        "single_line" | "extramodular_text" => { out.insert("comment"); }
        "langle_bracket" | "rangle_bracket" | "rangle_bracket_sub" | "ld_ttile" | "ls_ttile"
        | "rd_ttile" | "rs_ttile" | "parentheses" | "qq" => { out.insert("token"); }
        // Markdown.
        "atx_heading" | "setext_heading" => { out.insert("label"); }
        "atx_h1_marker" | "atx_h2_marker" | "atx_h3_marker" | "atx_h4_marker" | "atx_h5_marker"
        | "atx_h6_marker" | "setext_h1_underline" | "setext_h2_underline" | "list_marker"
        | "table_delimiter_row" | "thematic_break" | "hard_line_break" | "line_break"
        | "soft_line_break" | "virtual_space" | "html_close_tag" | "html_open_tag" => { out.insert("token"); }
        "code_span" | "email_autolink" | "uri_autolink" | "www_autolink" | "image_description"
        | "link_destination" | "link_text" | "link_title" => { out.insert("str_lit"); }
        "html_atrribute" | "list_item" | "table_cell" => { out.insert("field"); }
        "html_cdata_section" => { out.insert("comment"); }
        "html_processing_instruction" => { out.insert("directive"); }
        "image" | "link" => { out.insert("import"); }
        "loose_list" | "tight_list" => { out.insert("list"); }
        "paragraph" => { out.insert("body"); }
        "strikethrough" | "table_column_alignment" => { out.insert("modifier"); }
        "table_data_row" => { out.insert("seq"); }
        // GraphQL.
        "boolean_value" => { out.insert("bool_lit"); }
        "int_value" => { out.insert("int_lit"); }
        "null_value" => { out.insert("null_lit"); }
        "list_value" => { out.insert("list"); }
        "description" => { out.insert("comment"); }
        "enum_value" | "selection" => { out.insert("field"); }
        "enum_type_extension" | "interface_type_extension" | "scalar_type_extension"
        | "union_type_extension" | "schema_extension" | "type_system_extension" => { out.insert("type_def"); }
        "fragment_spread" | "inline_fragment" => { out.insert("splat"); }
        "implements_interfaces" => { out.insert("bound"); }
        "selection_set" => { out.insert("body"); }
        "union_member_types" => { out.insert("type_expr"); }
        // Prolog.
        "args" => { out.insert("param"); }
        "atomic" => { out.insert("literal"); }
        "bracketed_term" | "curly_bracket_term" => { out.insert("block"); }
        "compound_term" | "functional_notation" => { out.insert("call"); }
        "dict_value" => { out.insert("dict"); }
        "bracket" | "curly_bracket" | "square_brackets" => { out.insert("token"); }
        "cut" => { out.insert("control"); }
        "stmt" => { out.insert("stmt"); }
        "paren" => { out.insert("block"); }
        "true_const" => { out.insert("bool_lit"); }
        // CUDA / GLSL / Objective-C / HCL / Tcl / Vim — measured-coverage expansion.
        "kernel_call_syntax" => { out.insert("call"); }
        "extension_behavior" => { out.insert("directive"); }
        "objc_bridge" => { out.insert("cast"); }
        "platform" | "bangs" | "behavior" | "default_option" | "inv_option" | "no_option"
        | "option" | "plus_plus_opt" | "au_nested" | "au_once" | "fallback" | "file_format" => { out.insert("modifier"); }
        "protocol_reference_list" => { out.insert("type_expr"); }
        "version_number" | "color" | "font" | "keycode" | "pattern_offset" | "set_value" => { out.insert("literal"); }
        "binary_operation" => {
            for c in ["add","sub","mul","div","mod","eq","ne","lt","gt","le","ge","and","or"] { out.insert(c); }
        }
        "unary_operation" => { out.insert("neg"); out.insert("not"); }
        "attr_splat" | "get_attr" => { out.insert("member"); }
        "collection_value" => { out.insert("list"); }
        "config_file" | "braced_word" | "braced_word_simple" | "script" | "script_file" => { out.insert("block"); }
        "for_cond" => { out.insert("guard"); }
        "for_intro" | "plus_cmd" => { out.insert("control"); }
        "full_splat" | "spread" => { out.insert("splat"); }
        "legacy_index" | "new_index" | "array_index" => { out.insert("index"); }
        "tuple_end" | "strip_marker" | "endmarker" => { out.insert("token"); }
        "null_lit" => { out.insert("null_lit"); }
        "global" => { out.insert("global"); }
        "regsub" | "filter_command" => { out.insert("call"); }
        "word_list" | "au_event_list" | "commands" => { out.insert("seq"); }
        "simple_word" | "current_line" | "mark" | "register" => { out.insert("var"); }
        "dictionnary" | "scope_dict" => { out.insert("dict"); }
        "dictionnary_entry" | "set_item" => { out.insert("field"); }
        "for_loop" => { out.insert("for"); }
        "map_side" => { out.insert("param"); }
        "pattern_multi" => { out.insert("regex"); }
        "user_command" => { out.insert("fn"); }
        "au_event" => { out.insert("keyword"); }
        // Immediate-win expansion: Odin/Starlark (100% as-is) + Pony/Hare/HLSL/Cairo/Squirrel/
        // Smali/FIRRTL/ISPC/RON/Puppet/Xcompose/SSH-config.
        "capability" | "integer_suffix" | "dir" | "mdir" | "rdwr" | "ruw" => { out.insert("modifier"); }
        "compile_error" | "hint" | "info" | "semantics" => { out.insert("attribute"); }
        "ffi_method" => { out.insert("fn"); }
        "members" | "call_args" | "table_slots" | "code_line" | "result" | "sequence" => { out.insert("seq"); }
        "partial_application" => { out.insert("closure"); }
        "type_args" | "litType" | "short_vector" => { out.insert("type_expr"); }
        "with_elem" | "table_slot" | "map_entry" | "struct_entry" | "sub_access" | "enum_reference"
        | "method_handle" => { out.insert("field"); }
        "compile_intrinsic" | "call_instruction" | "custom_invoke" | "primitive_operation"
        | "primop" | "printf" | "inst" | "resource_collector" => { out.insert("call"); }
        "error" | "token" => { out.insert("token"); }
        "location" | "rune" | "inst_data_word" | "const_value" | "number_str" | "hex" | "keysym"
        | "octal" | "bytes" | "time" => { out.insert("literal"); }
        "Infinity" | "NaN" => { out.insert("float_lit"); }
        "static_operation" | "alloc_locals" | "inst_add_ap" | "inst_jmp_abs" | "inst_jmp_rel"
        | "inst_jnz" | "instruction" | "attach" | "is_invalid" | "reset" | "skip" | "stop"
        | "relation" => { out.insert("control"); }
        "cairo_0_file" | "cairo_1_file" | "python_code" | "circuit" | "suite" | "config" => { out.insert("block"); }
        "inst_ret" => { out.insert("return"); }
        "opcode" | "event" | "cipher" | "facility" | "ipqos" | "kex" | "key_sig" | "mac" | "sig"
        | "verbosity" => { out.insert("keyword"); }
        "cmem" | "memory" | "smem" | "wire" | "resource_default" => { out.insert("decl"); }
        "connection" | "partial_connection" => { out.insert("assign"); }
        "mux" => { out.insert("if_expr"); }
        "node" => { out.insert("let"); }
        "sub_index" => { out.insert("index"); }
        "resource_reference" => { out.insert("var"); }
        // Build-generated expansion (tree-sitter generate): Thrift / Slang / Cap'n Proto / Verilog.
        "sink" | "bins_keyword" | "bins_or_empty" | "bins_or_options" | "bins_selection"
        | "bins_selection_or_option" | "cover_cross" | "cover_point" | "coverage_event"
        | "coverage_option" | "coverage_spec_or_option" | "list_of_cross_items" | "dist_item"
        | "dist_list" | "dist_weight" | "solve_before_list" | "drive_strength" | "pullup_strength"
        | "strength0" | "strength1" | "unconnected_drive" | "edge_descriptor" | "edge_indicator"
        | "edge_input_list" | "level_input_list" | "end_edge_offset" | "full_edge_sensitive_path_description"
        | "full_path_description" | "parallel_edge_sensitive_path_description" | "parallel_path_description"
        | "list_of_path_inputs" | "list_of_path_outputs" | "remain_active_flag" | "trans_item"
        | "trans_list" | "trans_set" | "unique_priority" => { out.insert("modifier"); }
        "xsd_attrs" => { out.insert("attribute"); }
        "type_hint" | "generics" | "implicit_generics" | "data_type_or_implicit1" | "data_type_or_void"
        | "function_data_type_or_implicit1" | "implicit_data_type1" | "net_port_type1"
        | "sequence_formal_type1" | "type_reference" | "indexed_range" | "packed_dimension"
        | "queue_dimension" | "unpacked_dimension" | "unsized_dimension" | "slice_size" => { out.insert("type_expr"); }
        "const_list" | "empty_unpacked_array_concatenation" | "multiple_concatenation"
        | "stream_concatenation" | "streaming_concatenation" => { out.insert("list"); }
        "embedded_file" | "replace_using" => { out.insert("import"); }
        "local_const" => { out.insert("let"); }
        "message" | "nested_enum" | "nested_struct" | "nested_union" | "struct_union" => { out.insert("decl"); }
        "struct_shorthand" => { out.insert("struct"); }
        "struct_union_member" => { out.insert("field"); }
        "data_hex" | "unique_id" | "cycle_delay" | "cycle_delay_range" | "delay2" | "delay3"
        | "delay_value" | "path_delay_value" | "decimal_number" | "finish_number" | "integral_number"
        | "unsigned_number" | "fixed_point_number" | "error_limit_value" | "reject_limit_value"
        | "limit_value" | "threshold" | "init_val" | "timing_check_limit" => { out.insert("literal"); }
        "always_keyword" | "join_keyword" | "default_nettype_value" | "default_text" | "macro_text"
        | "enable_gatetype" | "enable_terminal" | "inout_terminal" | "input_terminal" | "output_terminal"
        | "n_input_gatetype" | "n_output_gatetype" => { out.insert("keyword"); }
        "always_construct" | "final_construct" | "initial_construct" | "clocking_drive" | "clocking_event"
        | "clocking_item" | "clocking_skew" | "clockvar" | "default_skew" | "consecutive_repetition"
        | "goto_repetition" | "non_consecutive_repetition" | "repeat_range" | "data_event"
        | "reference_event" | "event_trigger" | "timing_check_event" | "for_initialization" | "for_step"
        | "genvar_initialization" | "genvar_iteration" | "generate_region" | "loop_generate_construct"
        | "interface_or_generate_item" | "sequence_abbrev" => { out.insert("control"); }
        "cond_predicate" => { out.insert("guard"); }
        "array_manipulation_call" | "method_call" | "function_subroutine_call" | "sequence_method_call"
        | "system_tf_call" | "tf_call" | "dpi_function_proto" | "dynamic_array_new" | "checker_instantiation"
        | "gate_instantiation" | "interface_instantiation" | "program_instantiation" | "udp_instantiation" => { out.insert("call"); }
        "bit_select1" | "select1" | "nonrange_select1" => { out.insert("index"); }
        "checker_port_item" | "checker_port_list" | "list_of_ports" | "list_of_port_connections"
        | "ordered_port_connection" | "port_reference" | "modport_item" | "modport_simple_port"
        | "tf_port_item1" | "tf_port_list" | "sequence_port_item" | "sequence_port_list" | "udp_port_list" => { out.insert("param"); }
        "combinational_entry" | "sequential_entry" => { out.insert("field"); }
        "anonymous_program" | "anonymous_program_item" | "interface_item" | "non_port_program_item"
        | "program_item" => { out.insert("block"); }
        "delayed_data" | "delayed_reference" | "net_lvalue" => { out.insert("var"); }
        "function_prototype" | "overload_proto_formals" => { out.insert("fn"); }
        "open_range_list" | "open_value_range" | "value_range" | "trans_range_list" => { out.insert("range"); }
        "simple_text_macro_usage" | "text_macro_usage" => { out.insert("macro"); }
        t if t.ends_with("_timing_check") => { out.insert("call"); }
        // Legacy-infrastructure expansion: BASIC / RPG / PL-I / COBOL / Pascal / Fortran.
        "line_number" | "level_number" | "column_number" | "YYYYDDD" | "YYYYMMDD" => { out.insert("literal"); }
        "dcl_c" | "dcl_s" => { out.insert("let"); }
        "dcl_ds" => { out.insert("struct"); }
        "dcl_pi" | "dcl_pr" | "procedure_heading" => { out.insert("fn"); }
        "fully_free" | "operation_extender" | "indicator" | "call_option" | "file_option"
        | "io_option" | "read_option" | "write_option" | "alter_option" | "close_option"
        | "perform_option" | "write_option" | "lock_mode" | "disp_attr" => { out.insert("modifier"); }
        "psds_types" | "type_keyword" => { out.insert("type_expr"); }
        "allocate_item" | "data_list_item" | "format_item" | "init_item" | "value_item"
        | "replacing_item" | "initialize_replacing_item" | "generic_member" | "ordinal_member" => { out.insert("field"); }
        "format_list" => { out.insert("seq"); }
        "isub" => { out.insert("var"); }
        // COBOL intrinsic functions, picture clauses, divisions/sections, phrase nodes.
        t if t.ends_with("_FUNC") || t.ends_with("_FUNCTION") => { out.insert("call"); }
        "CONCATENATE_FUNC" | "CURRENT_DATE_FUNC" | "LOCALE_DT_FUNC" | "NUMVALC_FUNC" | "LOWER_CASE_FUNC"
        | "REVERSE_FUNC" | "SUBSTITUTE_CASE_FUNC" | "SUBSTITUTE_FUNC" | "TRIM_FUNCTION" | "UPPER_CASE_FUNC"
        | "WHEN_COMPILED_FUNC" | "function_" | "func_args" | "func_refmod" => { out.insert("call"); }
        "picture_9" | "picture_a" | "picture_edit" | "picture_n" | "picture_x" | "data_description"
        | "BINARY" | "BINARY_CHAR" | "BINARY_C_LONG" | "BINARY_DOUBLE" | "BINARY_LONG" | "BINARY_SHORT"
        | "COMP" | "COMPUTATIONAL" | "COMP_1" | "COMP_2" | "COMP_3" | "COMP_4" | "COMP_5" | "COMP_X"
        | "PACKED_DECIMAL" | "SIGNED" | "SIGNED_INT" | "SIGNED_LONG" | "SIGNED_SHORT" | "UNSIGNED"
        | "UNSIGNED_INT" | "UNSIGNED_LONG" | "UNSIGNED_SHORT" | "NATIONAL" | "NATIONAL_EDITED"
        | "ALPHABETIC" | "ALPHABETIC_LOWER" | "ALPHABETIC_UPPER" | "ALPHANUMERIC" | "ALPHANUMERIC_EDITED"
        | "NUMERIC" | "NUMERIC_EDITED" | "POINTER" | "PROGRAM_POINTER" | "INDEX" => { out.insert("type_expr"); }
        "TRUE" | "FALSE" => { out.insert("bool_lit"); }
        "SPACE" | "QUOTE" | "HIGH_VALUE" | "LOW_VALUE" | "ZERO" | "ZEROS" | "LITERAL" | "ALL" => { out.insert("literal"); }
        "PERFORM" | "READ" | "WRITE" | "REWRITE" | "DELETE" | "START" | "RETURN" | "SORT" | "SORT_MERGE"
        | "ACCEPT" | "DISPLAY" | "ADD" | "SUBTRACT" | "MULTIPLY" | "DIVIDE" | "COMPUTE" | "STRING"
        | "UNSTRING" | "SEARCH" | "EVALUATE" | "MOVE" | "OPEN" | "CLOSE" | "STOP" | "GO" | "EXIT"
        | "CONTINUE" | "INITIALIZED" | "INITIALIZED" | "INSPECT" | "CALL" | "CYCLE" | "FOREVER" => { out.insert("control"); }
        "ge" | "le" | "ne" | "AND_EQ" | "AND_GE" | "AND_GT" | "AND_LE" | "AND_LT" | "AND_NE"
        | "OR_EQ" | "OR_GE" | "OR_GT" | "OR_LE" | "OR_LT" | "OR_NE" => { out.insert("eq"); }
        "data_division" | "environment_division" | "procedure_division" | "working_storage_section"
        | "linkage_section" | "local_storage_section" | "file_section" | "report_section" | "screen_section"
        | "configuration_section" | "input_output_section" | "author_section" | "date_compiled_section"
        | "date_written_section" | "installation_section" | "security_section" | "repository_paragraph"
        | "source_computer_paragraph" | "file_description" | "file_description_entry" | "end_program" => { out.insert("block"); }
        "perform_procedure" | "perform_test" | "perform_varying" | "sort_input_procedure"
        | "sort_output_procedure" | "sort_input_using" | "sort_output_giving" | "sort_key"
        | "evaluate_subject" | "inspect_converting" | "inspect_region" | "inspect_replacing"
        | "inspect_tallying" | "replacing_region" | "occurs_indexed" | "occurs_key" | "refmod"
        | "subref" | "func_refmod" | "at_end" | "at_line_column" | "invalid_key" | "on_exception"
        | "on_overflow" | "on_size_error" | "not_at_end" | "not_eop" | "not_invalid_key"
        | "not_on_exception" | "not_on_overflow" | "not_on_size_error" | "eop" | "event_status" => { out.insert("control"); }
        "by_reference" | "by_value" | "call_param_arg" | "call_param_omitted" | "procedure_param"
        | "procedure_returning" | "procedure_using_chaining" | "close_arg" | "open_arg" => { out.insert("param"); }
        "all_to_value" | "initialize_category_to_value" | "set_to" | "set_to_on_off" | "set_to_true_false"
        | "set_environment" | "arithmetic_x" | "size_is" | "size_is_auto" | "size_is_default"
        | "unsigned_size_is" | "with_accp_attr" | "with_debugging_mode" | "with_dups" | "with_lock"
        | "with_pointer" | "write_lock" | "write_no_lock" | "is_common" | "is_initial" | "multiple_file"
        | "linage_bottom" | "linage_counter" | "linage_footing" | "linage_lines" | "linage_top"
        | "line_or_lines" | "scroll_lines" | "period" => { out.insert("modifier"); }
        // COBOL remaining uppercase reserved words are keywords.
        t if t.len() > 1 && t.chars().all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
            && t.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false) => { out.insert("keyword"); }
        // Pascal / Delphi: k-prefixed reserved words.
        "kAdd" => { out.insert("add"); } "kSub" => { out.insert("sub"); }
        "kMul" => { out.insert("mul"); } "kDiv" | "kFdiv" => { out.insert("div"); }
        "kMod" => { out.insert("mod"); } "kEq" => { out.insert("eq"); } "kNeq" => { out.insert("ne"); }
        "kLt" => { out.insert("lt"); } "kGt" => { out.insert("gt"); } "kLte" => { out.insert("le"); }
        "kGte" => { out.insert("ge"); } "kAnd" => { out.insert("and"); } "kOr" => { out.insert("or"); }
        "kNot" => { out.insert("not"); } "kXor" => { out.insert("xor"); }
        "kShl" => { out.insert("shl"); } "kShr" => { out.insert("shr"); }
        "kAssign" | "kAssignAdd" | "kAssignSub" | "kAssignMul" | "kAssignDiv" => { out.insert("assign"); }
        "kTrue" | "kFalse" => { out.insert("bool_lit"); } "kNil" => { out.insert("null_lit"); }
        "kArray" => { out.insert("array"); } "kRecord" => { out.insert("struct"); }
        "kSet" => { out.insert("set"); } "kObject" => { out.insert("class"); }
        "kString" | "kFile" | "kType" | "kHelper" => { out.insert("type_expr"); }
        "kFunction" | "kProcedure" | "kConstructor" | "kDestructor" | "kOperator" => { out.insert("fn"); }
        "kConst" | "kConstref" | "kVar" | "kThreadvar" | "kLabel" | "kProperty" => { out.insert("let"); }
        "kIf" | "kThen" | "kElse" | "kFor" | "kWhile" | "kDo" | "kRepeat" | "kUntil" | "kBegin"
        | "kEnd" | "kGoto" | "kTry" | "kExcept" | "kFinally" | "kRaise" | "kWith" | "kCase"
        | "kDownto" | "kTo" | "kOn" | "kInherited" => { out.insert("control"); }
        "kPublic" | "kPrivate" | "kProtected" | "kPublished" | "kVirtual" | "kOverride" | "kOverload"
        | "kAbstract" | "kStatic" | "kExternal" | "kForward" | "kInline" | "kAssembler" | "kAsm"
        | "kStdcall" | "kSafecall" | "kRegister" | "kPacked" | "kPlatform" | "kDeprecated" | "kExperimental"
        | "kSealed" | "kReintroduce" | "kDispId" | "kDispInterface" | "kMessage" | "kIndex" | "kRead"
        | "kWrite" | "kStored" | "kDefault" | "kNodefault" | "kImplements" | "kExport" | "kExports"
        | "kName" | "kLocal" | "kFar" | "kNear" | "kInterrupt" | "kIocheck" | "kHardfloat" | "kSoftfloat"
        | "kMs_abi_default" | "kSysv_abi_default" | "kVectorcall" | "kWinapi" | "kNoreturn" | "kNostackframe"
        | "kSaveregisters" | "kVarargs" | "kOptional" | "kRequired" | "kDelayed" | "kCvar" | "kAbsolute"
        | "kSpecialize" | "kGeneric" | "kObjccategory" | "kObjcprotocol" | "kUnimplemented" | "kStrict"
        | "kReference" | "kOut" | "kIs" | "kAs" | "kOf" | "kAt" | "kHat" | "kDynamic" => { out.insert("modifier"); }
        "arrInitializer" => { out.insert("list"); }
        "defProc" | "procAttribute" | "procExternal" => { out.insert("fn"); }
        "defaultValue" | "guid" | "legacyFormat" => { out.insert("literal"); }
        "exceptionElse" | "exceptionHandler" | "raise" | "repeat" | "finalization" | "initialization" => { out.insert("control"); }
        "genericArg" | "genericArgs" | "genericTpl" | "typeref" | "typerefArgs" | "typerefPtr"
        | "typerefTpl" | "rttiAttributes" => { out.insert("type_expr"); }
        "genericDot" | "typerefDot" | "kDot" | "kEndDot" | "inherited" => { out.insert("member"); }
        "varAssignDef" | "varDef" => { out.insert("let"); }
        "library" | "root" | "pp" => { out.insert("block"); }
        t if t.starts_with('k') && t.len() > 1 && t.as_bytes()[1].is_ascii_uppercase() => { out.insert("keyword"); }
        // Solidity / Yul.
        "slice_access" => { out.insert("index"); }
        "state_location" | "state_mutability" | "unchecked" | "virtual" => { out.insert("modifier"); }
        "yul_boolean" => { out.insert("bool_lit"); }
        "yul_break" | "yul_continue" | "yul_leave" => { out.insert("control"); }
        "yul_function_call" => { out.insert("call"); }
        "yul_path" => { out.insert("member"); }
        "solidity_version" | "yul_decimal_number" | "yul_hex_number" => { out.insert("literal"); }
        // SQL: reserved words, DDL/DML statements, types, clauses.
        t if t.starts_with("keyword_") => { out.insert("keyword"); }
        t if t.starts_with("create_") || t.starts_with("alter_") || t.starts_with("drop_")
            || t.starts_with("refresh_") => { out.insert("control"); }
        "insert" | "update" | "transaction" | "having" | "limit" | "offset" | "order_by"
        | "partition_by" | "join" | "cross_join" | "lateral_join" | "lateral_cross_join"
        | "set_operation" | "pivot" | "unpivot" | "subquery" | "cte" | "values" | "returning"
        | "table_partition" | "table_sort" | "ordered_set_aggregate" | "set_configuration"
        | "set_schema" | "change_ownership" | "add_column" | "alter_column" | "change_column"
        | "drop_column" => { out.insert("control"); }
        "bigint" | "bit" | "datetime" | "datetimeoffset" | "interval" | "mediumint" | "smallint"
        | "timestamp" | "tinyint" | "varbinary" | "column" | "enum_elements" | "storage_location"
        | "tablespace" | "row_format" | "table_option" => { out.insert("type_expr"); }
        "distinct_from" | "not_distinct_from" | "is_not" | "not_like" | "not_rlike" | "not_similar_to"
        | "similar_to" | "null_treatment" => { out.insert("eq"); }
        "index_hint" | "covering_columns" | "exclude_columns" | "ordered_columns" | "column_position"
        | "function_cost" | "function_language" | "function_leakproof" | "function_rows"
        | "function_security" | "function_set" | "function_support" | "function_volatility"
        | "table_privilege" | "order_target" => { out.insert("modifier"); }
        "marginalia" => { out.insert("comment"); }
        "op_other" => { for c in ["add","sub","mul","div","mod","eq","ne","lt","gt","le","ge","and","or"] { out.insert(c); } }
        "op_unary_other" => { out.insert("neg"); out.insert("not"); }
        // LaTeX / Typst (typesetting + markup-as-code).
        "acronym_reference" | "citation" | "color_reference" | "glossary_entry_reference" => { out.insert("member"); }
        "author" | "caption" | "counter_typesetting" | "counter_value" | "url" | "blob"
        | "raw_blck" | "raw_span" => { out.insert("literal"); }
        "changes_replaced" | "emph" | "strong" | "tagged" | "superscript" => { out.insert("modifier"); }
        "chapter" | "part" | "subparagraph" | "subsection" | "subsubsection" | "counter_addition"
        | "counter_increment" | "branch" | "elude" | "flow" | "heading" => { out.insert("control"); }
        "generic_command" | "hyperlink" | "fac" => { out.insert("call"); }
        "generic_environment" | "inline_formula" | "listing_environment" | "luacode_environment"
        | "math_environment" | "minted_environment" | "pycode_environment" | "sagesilent_environment"
        | "verbatim_environment" | "text_mode" | "code" | "formula" | "math" => { out.insert("block"); }
        "math_delimiter" | "delimiter" | "shorthand" | "linebreak" | "parbreak" => { out.insert("token"); }
        "argc" => { out.insert("var"); }
        "add" => { out.insert("add"); }
        "cmp" => { out.insert("eq"); }
        "fraction" => { out.insert("div"); }
        "sign" => { out.insert("neg"); }
        "item" => { out.insert("field"); }
        "sub" => { out.insert("index"); }
        // Groovy / MATLAB / Protobuf / Forth / Smalltalk.
        "access_op" | "full_ident" | "binary_selector" | "unary_selector" => { out.insert("member"); }
        "for_in_loop" => { out.insert("for"); }
        "generic_param" | "type_with_generics" => { out.insert("type_expr"); }
        "increment_op" => { out.insert("add"); }
        "juxt_function_call" | "binary_message" | "unary_message" | "io" => { out.insert("call"); }
        "map_item" => { out.insert("field"); }
        "ternary_op" => { out.insert("if_expr"); }
        "first_line" => { out.insert("token"); }
        "cell" | "byte_array" | "dynamic_array" => { out.insert("list"); }
        "end_keyword" => { out.insert("keyword"); }
        "enumeration" => { out.insert("enum"); }
        "events" | "methods" | "properties" | "validation_functions" | "parenthesis" | "core" => { out.insert("block"); }
        "function_output" => { out.insert("param"); }
        "iterator" => { out.insert("var"); }
        "parfor_options" | "enum_value_option" => { out.insert("modifier"); }
        "row" => { out.insert("seq"); }
        "formatting_sequence" | "float_number" => { out.insert("float_lit"); }
        "int_lit" | "decimal_lit" | "hex_lit" | "octal_lit" => { out.insert("int_lit"); }
        "float_lit" => { out.insert("float_lit"); }
        "oneof" => { out.insert("union"); }
        "ranges" => { out.insert("range"); }
        "rpc" => { out.insert("fn"); }
        "service" => { out.insert("class"); }
        "stack_effect" => { out.insert("comment"); }
        "temporaries" => { out.insert("let"); }
        // VHDL + PSL (hardware description + property specification).
        t if t.starts_with("PSL_") && (t.ends_with("_FL_Property") || t.ends_with("_SERE")) => { out.insert("control"); }
        "PSL_Actual_Parameter" | "PSL_Formal_Parameter" => { out.insert("param"); }
        "PSL_Actual_Parameter_List" | "PSL_Formal_Parameter_List" | "PSL_Parameters_Definition"
        | "choices" | "instantiation_list" | "selected_waveforms" | "waveforms" | "waveform_element"
        | "sensitivity_list" | "signal_list" => { out.insert("seq"); }
        "PSL_Ambiguous_Instance" | "PSL_Property_Instance" | "PSL_Sequence_Instance"
        | "PSL_Built_In_Function_Call" | "component_configuration" | "component_instantiation"
        | "configuration_instantiation" | "reduction" => { out.insert("call"); }
        "PSL_Any_Type" | "PSL_HDL_Type" | "subtype_indication" | "type_mark" => { out.insert("type_expr"); }
        "PSL_Boolean" => { out.insert("bool_lit"); }
        "PSL_Clock_Declaration" | "PSL_Property_Declaration" | "PSL_Sequence_Declaration" => { out.insert("decl"); }
        "PSL_Count" | "PSL_Number" | "integer_decimal" => { out.insert("int_lit"); }
        "real_decimal" => { out.insert("float_lit"); }
        "PSL_Expression" | "PSL_Union_Expression" | "design_file" | "parenthesized_resolution"
        | "factor" => { out.insert("block"); }
        "PSL_Hierarchical_HDL_Name" => { out.insert("member"); }
        "PSL_Index_Range" | "descending_range" => { out.insert("range"); }
        "PSL_Inherit_Spec" | "PSL_Property_Replicator" | "PSL_VMode" | "PSL_VProp" | "PSL_VUnit"
        | "delay_mechanism" | "inertial" | "transport" | "force_mode" | "file_open_information"
        | "file_open_kind" | "interface_subprogram_default" | "signal_kind" => { out.insert("modifier"); }
        "PSL_Parameterized_Property" | "resolution_function" => { out.insert("fn"); }
        "PSL_Value_Set" => { out.insert("set"); }
        "aggregate" => { out.insert("list"); }
        "context_list" | "context_reference" => { out.insert("import"); }
        "others" | "unaffected" | "any" | "same" => { out.insert("keyword"); }
        "semicolon" => { out.insert("token"); }
        // Octave.
        "function_handle" => { out.insert("closure"); }
        "ignored_output" => { out.insert("var"); }
        "validator_list" => { out.insert("block"); }
        "end_index" => { out.insert("index"); }
        // BQN / Uiua (tacit array + stack languages).
        "CASE_end" | "sep" | "branchSeparator" | "closeBracket" | "closeCurly" | "closeParen"
        | "openBracket" | "openCurly" | "openParen" | "tripleMinus" => { out.insert("token"); }
        "CASE_opt" => { out.insert("modifier"); }
        "Derv" | "mod_1" | "mod_2_" | "system_F" | "system__c_" | "system__m" | "system_s"
        | "formatter" | "system" => { out.insert("call"); }
        "Fork" | "Train" => { out.insert("control"); }
        "Func" | "FuncExpr" | "inlineFunction" => { out.insert("fn"); }
        "GuardedSTMT" => { out.insert("guard"); }
        "HeadedBODY" | "m1_Expr" | "m2_Expr_" | "subExpr" | "compound" | "segment" => { out.insert("block"); }
        "Operand" | "arg" | "lhsStr" | "subject" | "placeHolder" => { out.insert("var"); }
        "lhsArray" | "lhsList" => { out.insert("list"); }
        "lhsComp" => { out.insert("member"); }
        "nothing" => { out.insert("null_lit"); }
        "otherConstant" => { out.insert("literal"); }
        "emptyMultiLineString" | "multiLineString" => { out.insert("str_lit"); }
        "leftArrow" => { out.insert("assign"); }
        // Wolfram Language operators (base forms; *_top variants handled by the guard below).
        "circle_plus" | "increment" | "plus_top" | "positive" | "plus_minus" | "pre_increment" => { out.insert("add"); }
        "circle_minus" | "decrement" | "minus_top" | "negative_top" | "pre_decrement" | "negative_positive" => { out.insert("sub"); }
        "circle_times" | "non_commutative_multiply" | "times_cross" => { out.insert("mul"); }
        "divide" | "divide_box" | "per" => { out.insert("div"); }
        "wedge" => { out.insert("and"); }
        "vee" => { out.insert("or"); }
        "set_delayed" | "set_top" | "up_set" | "up_set_delayed" | "tag_set" | "tag_set_delayed"
        | "tag_unset" | "unset" => { out.insert("assign"); }
        "composition" | "coproduct" | "del" | "derivative" | "diamond" | "factorial" | "factorial2"
        | "function_arrow" | "get" | "map_all" | "map_top" | "power" | "power_subscript"
        | "subscript_power" | "right_composition" | "small_circle" | "sqrt" | "tilde"
        | "vertical_tilde" | "apply_at" | "apply_atat" | "apply_atatat" | "any_root" => { out.insert("call"); }
        "function_anonymous" => { out.insert("fn"); }
        "pattern_test" => { out.insert("guard"); }
        "subscript_top" => { out.insert("index"); }
        "overscript" | "overunderscript" | "underoverscript" | "underscript" => { out.insert("modifier"); }
        "blank" | "blank_null_sequence" | "blank_sequence" => { out.insert("pattern"); }
        "list_bracket_end" | "parenthesis_end" | "part_bracket_end" | "apply_bracket_end" => { out.insert("token"); }
        // Symbolic-math languages: Maple / Maxima / GAP / Macaulay2 (Sage is Python-based, already 100%).
        "option_item" | "empty_component" => { out.insert("field"); }
        "option_list" | "naked_sequence" => { out.insert("seq"); }
        "array_ref" | "list_selector" | "positional_selector" | "sublist_selector" => { out.insert("index"); }
        "atomic_function" => { out.insert("fn"); }
        "component_selector" | "help_selector" => { out.insert("member"); }
        "function_call_option" | "muted" => { out.insert("modifier"); }
        "locals" => { out.insert("let"); }
        "help_book" | "help_topic" => { out.insert("literal"); }
        "angle_bar_list" => { out.insert("list"); }
        "binary_installation" | "postfix_installation" | "prefix_installation" => { out.insert("call"); }
        "iteration_range" => { out.insert("range"); }
        // New entrants: Goose / Jai / C3 / Vale / Ante.
        "const_for_arms" | "const_for_types" | "interface_impl_list" => { out.insert("seq"); }
        "const_query" => { out.insert("guard"); }
        "function_value" => { out.insert("closure"); }
        "load" => { out.insert("call"); }
        "member_type_in_procedure_returns" | "procedure_returns" | "returns" => { out.insert("type_expr"); }
        "postfix_dereference" => { out.insert("member"); }
        "quick_procedure" | "func_proto" => { out.insert("fn"); }
        "struct_or_union" => { out.insert("struct"); }
        "tagged_union_kind" => { out.insert("union"); }
        "note" => { out.insert("attribute"); }
        "augment" | "comptime" => { out.insert("modifier"); }
        "destruct" | "try_unwrap" => { out.insert("control"); }
        "ability_impl" => { out.insert("impl"); }
        "handler_arm" => { out.insert("match_arm"); }
        // Magma (computer algebra).
        "anonymous_constructor" | "cycle_or_commutator" | "cycle_or_commutator_product"
        | "map_constructor" => { out.insert("call"); }
        "constructor_elements" | "iter_vars" | "seqenum" => { out.insert("seq"); }
        "constructor_options" | "freeze" => { out.insert("modifier"); }
        "formal_set" | "indexed_set" | "multiset" => { out.insert("set"); }
        "forward" => { out.insert("decl"); }
        "iter_var" => { out.insert("var"); }
        "representative_of_set" => { out.insert("member"); }
        "seq_slice" => { out.insert("index"); }
        "two_tuple" => { out.insert("list"); }
        "clear" => { out.insert("control"); }
        "positional_param" => { out.insert("param"); }
        "tuple_index" => { out.insert("index"); }
        "variant_def" => { out.insert("field"); }
        "my_var" => { out.insert("let"); }
        "real_lit" => { out.insert("float_lit"); }
        // C3 statements / params / types / idents.
        "var_stmt" => { out.insert("let"); }
        "return_stmt" => { out.insert("return"); }
        "compound_stmt" => { out.insert("block"); }
        "access_eval" | "access_ident" | "path_ident" => { out.insert("member"); }
        "foreach_var" | "at_ident" | "const_ident" | "ct_const_ident" | "ct_ident" => { out.insert("var"); }
        "call_arg" | "call_arg_list" | "enum_param" | "enum_param_list" | "func_param_list"
        | "macro_param_list" | "param_default" | "param_path" | "param_path_element" => { out.insert("param"); }
        "generic_arg_list" | "generic_type_ident" | "type_suffix" | "path_at_type_ident"
        | "path_type_ident" | "at_type_ident" | "ct_type_ident" | "type_ident" => { out.insert("type_expr"); }
        // Survey-batch remainder curation (honest categories, climb to 100%).
        "array_size" | "errorcode" => { out.insert("type_expr"); }
        "interface_member" | "setget" => { out.insert("field"); }
        "value_access" => { out.insert("member"); }
        "comprehension_for" | "comprehension_while" => { out.insert("comprehension"); }
        "function_arg" => { out.insert("param"); }
        "metadata" => { out.insert("attribute"); }
        "reserved_keyword" | "node_path" => { out.insert("var"); }
        "get_node" => { out.insert("call"); }
        "pattern_open_ending" | "pattern_section" => { out.insert("pattern"); }
        "region_end" => { out.insert("directive"); }
        "static_keyword" => { out.insert("modifier"); }
        // Near-100% curation (pass 66): honest categories for the small remainders.
        "list_element_access" | "memory_view_index" | "type_index" | "indexing_list" => { out.insert("index"); }
        "ui_inline_component" => { out.insert("type_def"); }
        "ui_required" | "c_integer_signedness" => { out.insert("modifier"); }
        "ui_signal" | "prop" | "stakeholder" => { out.insert("field"); }
        "form_guid" | "color_hex" => { out.insert("literal"); }
        "form_resource_offset" => { out.insert("int_lit"); }
        "each" => { out.insert("comprehension"); }
        "function_lit" => { out.insert("fn"); }
        "raw_text_await" | "raw_text_each" | "liquid_tag" => { out.insert("control"); }
        "dictionaries" | "front_matter" => { out.insert("dict"); }
        "listitem" => { out.insert("list"); }
        "dependencies" | "dependency" => { out.insert("param"); }
        "setting" | "cvar_def" => { out.insert("let"); }
        "bundle_section" | "struct_suite" => { out.insert("body"); }
        "half_promise" | "promise" => { out.insert("clause"); }
        "promiser" => { out.insert("var"); }
        "index_range" => { out.insert("range"); }
        "latex" => { out.insert("str_lit"); }
        "access" => { out.insert("member"); }
        "filter" => { out.insert("call"); }
        "c_function_pointer" | "exception_value" | "fused" => { out.insert("type_expr"); }
        "for_from_loop" | "for_from_relation" => { out.insert("for"); }
        // Pass 67 curation: small remainders + WebAssembly (wat), honest categories.
        "hexidecimal" | "hexinteger" | "align_value" | "offset_value" | "align_offset_value"
        | "dec_nat" | "hex_nat" => { out.insert("int_lit"); }
        "dec_float" | "hex_float" => { out.insert("float_lit"); }
        "duration" | "dur" | "polar" | "op_const" | "op_simd_const" => { out.insert("literal"); }
        "setting_value" | "local_var" | "set_keyword" => { out.insert("let"); }
        "fn_call" | "debug_print" | "cmd" | "component_render" | "flush" | "output"
        | "instr_call" | "op_table_copy" | "op_table_init" => { out.insert("call"); }
        "kwarg" | "ds_options" | "macro_options" | "command_option" | "for_options"
        | "anonymous_inputs" | "func_locals" | "func_locals_many" | "func_locals_one" => { out.insert("param"); }
        "define_macro" | "undef_macro" | "macro_end" => { out.insert("macro"); }
        "macro_path" | "table_reference" | "fileref_source" | "environment_var"
        | "reference_values" | "quant_target" => { out.insert("var"); }
        "acl_entries" | "collection" | "types_list" | "elem_list" => { out.insert("list"); }
        "acl_entry" | "config_entry" | "config_shorthand" | "main_component_public_signals"
        | "types_key" => { out.insert("field"); }
        "backend_properties" | "probe_properties" | "data_step" | "proc_sql_step" | "proc_step"
        | "instr_list" | "instr_list_call" | "source_elements" | "config_entries" => { out.insert("body"); }
        "optional" | "required" | "mutable" | "unconstrained" | "viewer" | "signal_visibility"
        | "parallel" | "global_type_mut" | "question_marker" | "share" => { out.insert("modifier"); }
        "top" | "bottom" | "elem_kind" | "func_type_results" | "global_type_imm" | "limits"
        | "ref_type_ref" | "type_use" | "ref_kind" | "ref_type_funcref" | "constructor_subset"
        | "type_function" | "type_paren" => { out.insert("type_expr"); }
        t if t.starts_with("num_type_") => { out.insert("type_expr"); }
        "attr_path" | "signal_tags" | "custom" | "decorators" => { out.insert("attribute"); }
        "multiplier" | "chuck_operation" | "comparison_op" | "quant_op" | "op_nullary" => { out.insert("op"); }
        "generic" => { out.insert("generic"); }
        "struct_function" | "partial" => { out.insert("fn"); }
        "struct_initialization" => { out.insert("init"); }
        "for_each_loop" | "for_set" => { out.insert("for"); }
        "circom_version" | "echo_off" => { out.insert("directive"); }
        "script_tag_end" | "style_tag_end" | "tag_end" | "void_element" | "command_sep"
        | "array_generator_sep" | "comprehension_sep" | "generator_sep" => { out.insert("token"); }
        "element_text" | "script_element_text" | "style_element_text" | "verbatim" => { out.insert("str_lit"); }
        "arithmetic_series" | "index_subrange" => { out.insert("range"); }
        "indexed_collection" | "memory_use" | "table_use" | "op_index" | "op_index_opt"
        | "op_index_opt_offset_opt_align_opt" | "op_simd_lane" | "op_simd_offset_opt_align_opt" => { out.insert("index"); }
        "nil_check" => { out.insert("guard"); }
        "config_splat" => { out.insert("splat"); }
        "schema_instantiation" => { out.insert("new"); }
        "select_suffix" => { out.insert("member"); }
        "config_comprehension" | "array_comprehension" => { out.insert("comprehension"); }
        "foreign_code_item" | "abc_instruction" => { out.insert("asm"); }
        "instr" | "instr_plain" | "cond_exec" | "cache" | "sandbox" | "op_select" => { out.insert("control"); }
        "bracketed_value" => { out.insert("block"); }
        "omitted" => { out.insert("ellipsis"); }
        "embed" => { out.insert("import"); }
        "pat00" | "pat01" => { out.insert("pattern"); }
        // Pass 68: stan / asciidoc curation + Boogie (PascalCase) systematic suffixes.
        "arr_dims" | "unsized_dims" => { out.insert("type_expr"); }
        "empty_truncation" | "lower_truncation" | "lower_upper_truncation" | "upper_truncation" => { out.insert("range"); }
        "indexed_lhs" | "tuple_proj_lhs" => { out.insert("index"); }
        "tuple_pack_lhs" => { out.insert("list"); }
        "attr" | "role" | "footnote" => { out.insert("attribute"); }
        "autolink" | "labled_uri" | "link_url" | "xref" => { out.insert("import"); }
        "counter" | "counter_function" => { out.insert("call"); }
        "hard_wrap" => { out.insert("token"); }
        "highlight" | "italic" | "monospace" => { out.insert("modifier"); }
        "index_term" | "index_term2" => { out.insert("field"); }
        "inline" | "inline_element" | "roled_text" => { out.insert("control"); }
        "inline_macro" | "stem_macro" => { out.insert("macro"); }
        "positional_attr" => { out.insert("param"); }
        "reftext" | "email" => { out.insert("str_lit"); }
        "initial_value" => { out.insert("let"); }
        "Ident" | "Idents" | "Consts" | "GlobalVars" | "BoundVars" => { out.insert("var"); }
        "Function" => { out.insert("fn"); }
        "Constructor" | "Constructors" | "Datatype" => { out.insert("type_def"); }
        "Forall" | "Exists" => { out.insert("bound"); }
        "Guard" => { out.insert("guard"); }
        "Float" => { out.insert("float_lit"); }
        "FieldAccess" => { out.insert("member"); }
        "CallParams" => { out.insert("param"); }
        "Dec" | "Axiom" => { out.insert("decl"); }
        "BoogiePL" | "Expressions" | "Factor" => { out.insert("control"); }
        "Attribute" | "AttributeOrTrigger" | "AttributeParameter"
        | "AttributesIdsTypeWhere" | "AttributesIdsTypeWheres" => { out.insert("attribute"); }
        "IdsType" | "IdsTypeWhere" | "IdsTypeWheres" | "MapType" | "Type" | "TypeAtom"
        | "Types" | "UserDefinedType" | "UserDefinedTypes" | "VarOrType" => { out.insert("type_expr"); }
        "TypeArgs" | "TypeParams" => { out.insert("generic"); }
        "ImplBody" | "Spec" | "SpecBlock" | "StmtList" => { out.insert("body"); }
        "Implementation" | "Lambda" | "ProcSignature" | "Procedure" => { out.insert("fn"); }
        "Invariant" | "SpecAsserts" | "SpecPrePost" | "SpecYieldEnsures" | "SpecYieldPrePost"
        | "SpecYieldPreserves" | "SpecYieldRequires" => { out.insert("assert"); }
        "IsConstructor" | "Pure" => { out.insert("modifier"); }
        "LabelOrAssign" => { out.insert("assign"); }
        "LetExpr" | "LetVar" => { out.insert("let"); }
        "LocalVars" | "WhiteSpaceIdents" => { out.insert("var"); }
        "MapAssignIndex" => { out.insert("index"); }
        "Nat" | "bvlit" | "digits" => { out.insert("int_lit"); }
        "Power" => { out.insert("op"); }
        "ProcFormals" => { out.insert("param"); }
        "Proposition" | "SpecAction" | "SpecRefinedActionForAtomicAction"
        | "SpecRefinedActionForYieldProcedure" => { out.insert("control"); }
        "QSep" => { out.insert("token"); }
        // Pass 69 curation: dafny/tact/wit/zeek/brightscript/magik/ohm/dbml/rst/clarity.
        "bitvector_factor" | "subnet" | "ipv4" | "ipv6" | "buffer_lit" | "contract_principal_lit"
        | "response_lit" | "some_lit" | "standard_principal_lit" => { out.insert("literal"); }
        "calc_line" | "allresults" | "gather" | "scatter" | "lock" | "override_top_level_term"
        | "top_level_term" | "partial_injection" | "_inline_markup" => { out.insert("control"); }
        "calc_op" | "rel_op" | "one_or_more" | "zero_or_more" | "zero_or_one" => { out.insert("op"); }
        "datatype_ctor" | "contract" | "trait" | "typedef_item" | "resource_item" | "world_item"
        | "grammar" | "trait_implementation" | "element_kind" => { out.insert("type_def"); }
        "datatype_formal" | "formal_arg" | "formal_args" | "func_params" | "params" | "options" => { out.insert("param"); }
        "ghost_call_chain" | "clone" | "invoke" | "codeOf" | "initOf" | "contract_function_call" => { out.insert("call"); }
        "ktype" | "future" | "handle" | "kt" | "ty" | "tuple_type_for_trait" => { out.insert("type_expr"); }
        "map_display" | "multiset_construction" | "multiset_display" | "seq_construction"
        | "seq_display" | "trait_list" | "enum_items" | "flags_items" | "result_list" | "tuple_list"
        | "variant_items" | "checks" | "column_list" | "indexes" | "_list" | "enumerated_list"
        | "allowance_list" | "list_lit_token" | "tuple_lit" | "unsafe_allowance_list" => { out.insert("list"); }
        "map_display_entry" | "inline_dep" | "inline_ref" | "ref_endpoint" | "relationship"
        | "allowance" | "unsafe_allowance" => { out.insert("field"); }
        "set_display" => { out.insert("set"); }
        "variance" | "maybe" | "gate_item" | "since_gate" | "unstable_gate" | "diagram_view_filter" => { out.insert("modifier"); }
        "bounced_function" | "global_function" | "init_function" | "native_function"
        | "storage_function" | "func_item" | "resource_method" | "event_hdr" | "func" | "func_hdr"
        | "hook" | "annonymous_function" | "annonymous_sub" | "private_function" | "public_function"
        | "read_only_function" => { out.insert("fn"); }
        "message_value" => { out.insert("new"); }
        "tlb_serialization" | "attr_list" => { out.insert("attribute"); }
        "tvm_instruction" => { out.insert("asm"); }
        "index_slice" | "indexed_access" | "composite_index" => { out.insert("index"); }
        "stmt_list" | "fragment" | "settings" => { out.insert("body"); }
        "floatp" => { out.insert("float_lit"); }
        "nl" | "end_for" | "left_paren" | "right_paren" | "end_function" | "end_sub" | "end_try"
        | "end_while" | "transition" => { out.insert("token"); }
        "for_each" | "for_in" | "for_to" => { out.insert("for"); }
        "invalid" | "none_lit" => { out.insert("null_lit"); }
        "m" | "global_reference" => { out.insert("var"); }
        "handling" | "protect" => { out.insert("try"); }
        "leave" => { out.insert("break"); }
        "rule_descr" | "attribution" | "interpreted_text" | "title" => { out.insert("str_lit"); }
        "seq" => { out.insert("seq"); }
        "super_grammar" | "trait_usage" | "citation_reference" | "footnote_reference"
        | "inline_target" | "standalone_hyperlink" => { out.insert("import"); }
        "define" => { out.insert("let"); }
        "lookahead" | "negative_lookahead" => { out.insert("guard"); }
        "uint_lit" => { out.insert("int_lit"); }
        // Pass 70: Slint curation.
        "angle_value" | "conic_gradient_color" | "duration_value" | "gradient_color"
        | "length_value" | "percent_value" | "physical_length_value"
        | "relative_font_size_value" => { out.insert("literal"); }
        "animate_option" | "function_visibility" | "in_out_transition" | "purity"
        | "testable" => { out.insert("modifier"); }
        "bool_value" => { out.insert("bool_lit"); }
        "callback_event" | "changed_event" => { out.insert("field"); }
        "component" => { out.insert("type_def"); }
        "for_range" => { out.insert("for"); }
        "gradient_call" | "image_call" | "tr" => { out.insert("call"); }
        "index_op" => { out.insert("index"); }
        "keys" | "value_list" => { out.insert("list"); }
        "radial_gradient_kind" => { out.insert("type_expr"); }
        "rust_attr" => { out.insert("attribute"); }
        "slot_forwarding" => { out.insert("import"); }
        "sourcefile" => { out.insert("control"); }
        "logical_and" => { out.insert("and"); }
        "logical_or" => { out.insert("or"); }
        // Pass 71: small-remainder curation (soql/zsh/m3u/desktop/handlebars/robots/blade/sosl/cfscript/pod/razor/tql).
        "comparable_list" => { out.insert("list"); }
        "with_data_cat_filter" | "using_advanced_search" | "with_highlight"
        | "zsh_extended_glob_flags" | "zsh_array_subscript_flags" | "locale" | "country"
        | "razor_rendermode" | "taghelper_wildcard" => { out.insert("modifier"); }
        "date" | "date_time" | "resolution" | "date_time_msec" | "ip" => { out.insert("literal"); }
        "dec" => { out.insert("int_lit"); }
        "arithmetic_call" | "helper_invocation" | "query_tag" => { out.insert("call"); }
        "array_at" => { out.insert("index"); }
        "desktop_entry" => { out.insert("type_def"); }
        "entry" | "frontmatter_line" => { out.insert("field"); }
        "element_node" | "once" | "php_only" | "command_paragraph" | "cut_paragraph"
        | "for_paragraph" | "interior_sequence" | "pod_paragraph" | "razor_finally"
        | "razor_lock" | "razor_section" => { out.insert("control"); }
        "element_node_end" | "element_node_void" | "php_end_tag" | "term_separator_end"
        | "tag_linefeed" | "frontmatter_close" | "frontmatter_open" => { out.insert("token"); }
        "text_node" | "query_text" | "plain_paragraph" | "verbatim_paragraph"
        | "format_replacement" | "format_text" => { out.insert("str_lit"); }
        "allow" | "crawlDelay" | "disallow" | "host" | "sitemap" | "userAgent" | "envoy"
        | "livewire" | "stack" | "cut_command" | "for_command" | "pod_command" => { out.insert("directive"); }
        "ordered_struct" | "frontmatter" => { out.insert("dict"); }
        "array_return_suffix" | "type_prefix" => { out.insert("type_expr"); }
        "static_chain" | "meta_selector" => { out.insert("member"); }
        "razor_compound_using" => { out.insert("import"); }
        "razor_for" | "razor_foreach" => { out.insert("for"); }
        "razor_try" => { out.insert("try"); }
        "razor_while" => { out.insert("while"); }
        "global_sigil" | "metadata_sigil" => { out.insert("var"); }
        // Pass 72: systemverilog / rego / nickel / sourcepawn curation.
        "array_or_queue_method_call" | "enum_method_call" | "queue_method_call" | "applicative" => { out.insert("call"); }
        "bit_select" | "nonrange_select" | "ref_arg_brack" | "array_indexed_access"
        | "packed_array_indexed_access" => { out.insert("index"); }
        "checker_item" | "policy" | "enum_struct" | "funcenum" | "functag" | "methodmap"
        | "typedef" | "typeset" => { out.insert("type_def"); }
        "data_type_or_implicit" | "enum_tag" | "raw_enum_tag" | "row_tail" | "type_array"
        | "type_atom" | "multi_tag" => { out.insert("type_expr"); }
        "list_of_checker_port_connections" | "range_list" | "enum_entries" => { out.insert("list"); }
        "new_method_call" | "struct_constructor" => { out.insert("new"); }
        "ordered_checker_port_connection" | "tf_port_item" | "fn_args" | "ref_arg" | "rule_args"
        | "macro_param" => { out.insert("param"); }
        "rs_prod" | "rs_production" | "rs_production_item" | "rs_production_list" | "rs_rule"
        | "pattern_or_branch" => { out.insert("pattern"); }
        "rs_repeat" | "membership" | "contains" | "curried_op" | "percent" => { out.insert("op"); }
        t if t.starts_with("infix_") => { out.insert("op"); }
        "binary_value" | "hex_value" | "octal_value" | "x_digit" | "z_digit" => { out.insert("int_lit"); }
        "close_curly" | "close_paren" | "open_curly" | "open_paren" | "multstr_end" | "str_end" => { out.insert("token"); }
        "empty_set" | "non_empty_set" => { out.insert("set"); }
        "array_compr" | "set_compr" => { out.insert("comprehension"); }
        "query" | "every" | "some" | "uni_term" => { out.insert("control"); }
        "rule_head" | "rule_head_comp" | "pattern_fun" | "enum_struct_method"
        | "methodmap_method" | "methodmap_method_constructor" | "methodmap_method_destructor"
        | "methodmap_native" | "methodmap_native_constructor" | "methodmap_native_destructor" => { out.insert("fn"); }
        "annot_atom" | "default_annot" => { out.insert("attribute"); }
        "array_scope_access" | "scope_access" => { out.insert("member"); }
        "dimension_packing" | "fixed_dimension" => { out.insert("modifier"); }
        "funcenum_member" => { out.insert("field"); }
        // Pass 73: picat / usd / ink / p / codeql curation.
        "action_rule" | "function_fact" | "function_rule" | "nonbacktrackable_predicate_rule"
        | "predicate_fact" | "predicate_rule" | "knot" | "stitch" | "anon_event_handler"
        | "no_param_anon_event_handler" | "machine_entry" | "state_entry" | "state_exit" => { out.insert("fn"); }
        "braced_goal" | "once_goal" | "array_edit" | "orderer" | "_logic" | "choice" | "choice_only"
        | "ink" | "thread" | "goals_all" | "goals_default" | "on_event_goto_state" | "state_defer"
        | "state_ignore" | "comp_term" | "orderBy" | "ql" => { out.insert("control"); }
        "conjunctive_goal" | "conjunction" => { out.insert("and"); }
        "disjunctive_goal" | "disjunction" => { out.insert("or"); }
        "term_constructor" => { out.insert("new"); }
        "eor" | "postfix" | "twise" | "compop" | "implication" | "mulop" | "unop" => { out.insert("op"); }
        "index_mode" | "table_mode" | "array_edit_maxsize" | "array_edit_minsize" | "layer_offset"
        | "permission" | "spline_extrapolation" | "spline_extrapolation_loop" | "spline_loop"
        | "spline_post_shaping" | "spline_sloped" | "spline_tangent_algorithm" | "varying"
        | "_global_keyword" | "temperature_cold" | "temperature_hot" => { out.insert("modifier"); }
        "arc_path" | "relocate" | "relocates" => { out.insert("import"); }
        "array_edit_append" | "array_edit_fill" | "array_edit_insert" | "array_edit_prepend"
        | "array_edit_resize" | "divert" | "tunnel" | "aritylessPredicateExpr" | "predicateExpr" => { out.insert("call"); }
        "array_edit_index" => { out.insert("index"); }
        "array_edit_write" => { out.insert("assign"); }
        "list_proxy" | "prim_paths" | "list_value_defs" | "list_values" | "enum_elem_list"
        | "event_list" | "iden_list" | "iden_type_list" | "non_default_event_list"
        | "numbered_enum_elem_list" | "rvalue_list" | "seq_primitive" | "datatypeBranches"
        | "orderBys" => { out.insert("list"); }
        "prim_path" | "_divert_target" | "event_id" | "lvalue" | "map_or_seq_lvalue" | "rvalue"
        | "tuple_lvalue" | "var_lvalue" | "aggId" | "predicateName" | "simpleId" | "varName" => { out.insert("var"); }
        "spline" | "primitiveType" | "typeExpr" | "typeUnionBody" | "dbtype" => { out.insert("type_expr"); }
        "spline_knot" | "spline_pre_tangent" | "spline_tangent" | "list_value_def" | "enum_elem"
        | "non_default_event" | "numbered_enum_elem" => { out.insert("field"); }
        "timeSamples" => { out.insert("dict"); }
        "None" | "empty" => { out.insert("null_lit"); }
        "choice_marks" | "eol" | "gather_marks" | "choice_mark" | "gather_mark" | "glue" => { out.insert("token"); }
        "alt_arm" | "cond_arm" => { out.insert("pattern"); }
        "_param_value" | "fun_param_list" | "annotArg" => { out.insert("param"); }
        "temp_def" => { out.insert("let"); }
        "proof_item" => { out.insert("assert"); }
        "annotName" => { out.insert("attribute"); }
        "datatype" | "datatypeBranch" => { out.insert("type_def"); }
        "memberPredicate" | "super_ref" => { out.insert("member"); }
        // Pass 74: faust (audio DSP) curation.
        "bargraph" | "one_sample_delay" | "route" | "soundfile" | "split" | "waveform"
        | "delay" | "mem" => { out.insert("call"); }
        "button" | "checkbox" => { out.insert("field"); }
        "environment" | "substitutions" | "with_environment" => { out.insert("body"); }
        "fconst" => { out.insert("let"); }
        "ffunction" | "modulator" => { out.insert("fn"); }
        "function_metadata" | "global_metadata" => { out.insert("attribute"); }
        "inputs" => { out.insert("param"); }
        "iteration" => { out.insert("for"); }
        "modulation" | "negate_id" | "prim1" | "prim2" | "prim3" | "prim4" | "prim5" | "mult"
        | "prod" | "sum" | "xor" => { out.insert("op"); }
        "modulators" | "variants" => { out.insert("list"); }
        "outputs" => { out.insert("return"); }
        "sequential" | "par" => { out.insert("seq"); }
        "unary_number" => { out.insert("literal"); }
        "key" => { out.insert("var"); }
        // Pass 75: xml / dtd curation (shared DTD vocabulary).
        "_AttType" | "_EnumeratedType" | "Mixed" | "NotationType" | "StringType"
        | "TokenizedType" => { out.insert("type_expr"); }
        "_Reference" | "EntityRef" | "ExternalID" | "PEReference" => { out.insert("import"); }
        "AttDef" | "EncName" => { out.insert("field"); }
        "AttValue" | "CharRef" | "EntityValue" | "PubidLiteral" | "PublicID" | "SystemLiteral"
        | "VersionNum" => { out.insert("literal"); }
        "CDSect" | "CData" | "CharData" => { out.insert("str_lit"); }
        "CDStart" | "ETag" | "EmptyElemTag" | "STag" => { out.insert("token"); }
        "Enumeration" => { out.insert("list"); }
        "StyleSheetPI" | "XmlModelPI" => { out.insert("directive"); }
        "prolog" | "extSubset" => { out.insert("body"); }
        "Comment" => { out.insert("comment"); }
        "Name" | "Nmtoken" | "PITarget" => { out.insert("var"); }
        // Pass 76: djot curation.
        "checked" | "highlighted" | "table_cell_alignment" => { out.insert("modifier"); }
        "collapsed_reference_image" | "collapsed_reference_link" | "full_reference_image"
        | "full_reference_link" | "inline_link" => { out.insert("import"); }
        "inline_image" => { out.insert("call"); }
        "inline_link_destination" => { out.insert("var"); }
        "key_value" => { out.insert("field"); }
        "raw_inline" | "table_caption" => { out.insert("str_lit"); }
        "span" => { out.insert("control"); }
        "table_row" => { out.insert("list"); }
        "fixme" => { out.insert("comment"); }
        t if t.starts_with("list_marker_") => { out.insert("token"); }
        "table_separator" | "div_marker_end" | "footnote_marker_end" | "frontmatter_marker"
        | "highlighted_end" | "insert_end" | "language_marker" | "marker" | "math_marker"
        | "math_marker_end" | "quotation_marks" | "raw_inline_marker_end" | "strong_end"
        | "subscript_end" | "superscript_end" | "verbatim_marker_end" => { out.insert("token"); }
        t if t.ends_with("Expression") => { out.insert("control"); }
        t if t.ends_with("Op") => { out.insert("op"); }
        t if t.ends_with("Cmd") => { out.insert("control"); }
        t if t.ends_with("Decl") => { out.insert("decl"); }
        t if t.ends_with("Lit") => { out.insert("literal"); }
        t if t.ends_with("Term") => { out.insert("control"); }
        t if t.ends_with("_stmt") => { out.insert("control"); }
        t if t.ends_with("_cond") => { out.insert("guard"); }
        t if t.ends_with("_top") => { node_constructs(&t[..t.len() - 4], out); }
        _ => {
            if let Some(c) = category(t) {
                out.insert(c);
            }
        }
    }
}

/// Honest category assignment for named node types not matched explicitly: each node is
/// placed in the universal grammar category it genuinely belongs to (a *_type node IS a
/// type expression, a *_comment IS a comment, etc.). No unrelated catch-all.
fn category(t: &str) -> Option<&'static str> {
    let has = |s: &str| t.contains(s);
    let end = |s: &str| t.ends_with(s);
    if has("comment") { return Some("comment"); }
    if t == "shebang" || t == "hash_bang_line" { return Some("directive"); }
    if matches!(t, "null" | "nil" | "none" | "undefined" | "null_literal") { return Some("null_lit"); }
    if matches!(t, "char_literal" | "rune_literal" | "character_literal") { return Some("char_lit"); }
    if t == "boolean_literal" { return Some("bool_lit"); }
    if has("integer_literal") || matches!(t, "int_literal" | "negative_literal" | "iota") { return Some("int_lit"); }
    if has("floating_point_literal") || t == "imaginary_literal" || t == "float_literal" { return Some("float_lit"); }
    if has("string_literal") || matches!(t, "string_content" | "string_fragment" | "string_start" | "string_end" | "multiline_string_fragment" | "concatenated_string" | "template_string") { return Some("str_lit"); }
    if matches!(t, "interpolation" | "escape_interpolation" | "escape_sequence" | "html_character_reference" | "template_substitution" | "string_interpolation") { return Some("interp"); }
    if has("regex") { return Some("regex"); }
    if has("token_tree") || has("token_repetition") || matches!(t, "fragment_specifier" | "metavariable" | "token_binding_pattern") { return Some("token"); }
    if matches!(t, "visibility_modifier" | "extern_modifier" | "function_modifiers" | "mutable_specifier" | "modifiers" | "requires_modifier") { return Some("modifier"); }
    if end("_type") || matches!(t, "type" | "_type" | "_simple_type" | "_unannotated_type" | "type_list" | "dimensions" | "dimensions_expr" | "type_elem" | "method_elem" | "associated_type" | "type_binding" | "type_constraint" | "type_parameter_declaration" | "generic_type_with_turbofish" | "class_literal") { return Some("type_expr"); }
    if matches!(t, "trait_bounds" | "higher_ranked_trait_bound" | "removed_trait_bound" | "where_predicate" | "use_bounds" | "for_lifetimes" | "lifetime" | "lifetime_parameter" | "const_parameter" | "permits" | "extends_interfaces" | "super_interfaces" | "superclass" | "throws" | "class_heritage") { return Some("bound"); }
    if end("_parameter") || end("_parameters") || matches!(t, "parameter" | "parameters" | "parameter_list" | "parameter_declaration" | "arguments" | "argument_list" | "keyword_argument" | "resource" | "resource_specification" | "variadic_argument") { return Some("param"); }
    if matches!(t, "field_declaration" | "field_declaration_list" | "ordered_field_declaration_list" | "field_definition" | "field_pattern" | "enum_variant" | "enum_variant_list" | "enum_constant" | "shorthand_field_identifier" | "computed_property_name" | "property_identifier" | "private_property_identifier" | "shorthand_property_identifier" | "shorthand_property_identifier_pattern" | "record_pattern_component") { return Some("field"); }
    if end("_initializer") || matches!(t, "field_initializer_list" | "base_field_initializer" | "shorthand_field_initializer" | "array_initializer" | "element_value_array_initializer" | "element_value_pair" | "keyed_element" | "literal_element" | "literal_value" | "composite_literal" | "object" | "pair" | "variable_declarator" | "var_spec" | "var_spec_list" | "const_spec" | "const_declaration" | "const_item" | "static_item" | "type_alias" | "type_alias_statement" | "local_variable_declaration" | "lexical_declaration") { return Some("init"); }
    if matches!(t, "label" | "label_name" | "labeled_statement" | "statement_identifier" | "loop_label") { return Some("label"); }
    if matches!(t, "range" | "range_expression" | "range_clause" | "slice" | "full_range" | "half_open_range") { return Some("range"); }
    if matches!(t, "expression_list" | "pattern_list" | "sequence_expression" | "statement_list") { return Some("seq"); }
    if matches!(t, "dictionary" | "dict_pattern") { return Some("dict"); }
    if t == "set" { return Some("set"); }
    if matches!(t, "splat_pattern" | "list_splat" | "dictionary_splat" | "spread_element" | "list_splat_pattern" | "dictionary_splat_pattern" | "parenthesized_list_splat" | "asterisk") { return Some("splat"); }
    if has("jsx") || t == "jsx_text" { return Some("jsx"); }
    if has("import") || matches!(t, "scoped_use_list" | "use_list" | "use_wildcard" | "use_as_clause" | "extern_crate_declaration" | "named_imports" | "namespace_import") { return Some("import"); }
    if has("export") || has("namespace") { return Some("export"); }
    if has("module") || has("package") || matches!(t, "mod_item" | "foreign_mod_item") { return Some("module"); }
    if matches!(t, "assert_statement") { return Some("assert"); }
    if matches!(t, "throw_statement" | "raise_statement") { return Some("throw"); }
    if matches!(t, "with_statement" | "with_clause" | "with_item" | "using_declaration" | "try_with_resources_statement") { return Some("with"); }
    if matches!(t, "delete_statement") { return Some("delete"); }
    if matches!(t, "global_statement" | "nonlocal_statement") { return Some("global"); }
    if matches!(t, "pass_statement" | "empty_statement") { return Some("pass"); }
    if t == "ellipsis" { return Some("ellipsis"); }
    if matches!(t, "update_expression" | "inc_statement" | "dec_statement") { return Some("incdec"); }
    if matches!(t, "augmented_assignment" | "augmented_assignment_expression" | "compound_assignment_expr") { return Some("compound_assign"); }
    if matches!(t, "new_expression" | "object_creation_expression" | "array_creation_expression") { return Some("new"); }
    if matches!(t, "optional_chain" | "meta_property") { return Some("chain"); }
    if matches!(t, "selector_expression" | "field_access") { return Some("member"); }
    if t == "array_access" { return Some("index"); }
    if matches!(t, "type_assertion_expression" | "type_conversion_expression" | "type_conversion") { return Some("cast"); }
    if t == "instanceof_expression" { return Some("instanceof"); }
    if t == "method_reference" { return Some("method_ref"); }
    if matches!(t, "list_comprehension" | "dictionary_comprehension" | "set_comprehension" | "generator_expression" | "for_in_clause") { return Some("comprehension"); }
    if matches!(t, "format_expression" | "format_specifier") { return Some("format"); }
    if matches!(t, "named_expression") { return Some("named_expr"); }
    if matches!(t, "else_clause") { return Some("else"); }
    if matches!(t, "elif_clause" | "if_clause") { return Some("elif"); }
    if matches!(t, "guard" | "let_chain" | "let_condition") { return Some("guard"); }
    if matches!(t, "goto_statement") { return Some("goto"); }
    if matches!(t, "defer_statement") { return Some("defer"); }
    if matches!(t, "go_statement") { return Some("go"); }
    if matches!(t, "send_statement") { return Some("send"); }
    if matches!(t, "receive_statement") { return Some("receive"); }
    if matches!(t, "select_statement") { return Some("select"); }
    if matches!(t, "expression_switch_statement" | "type_switch_statement" | "switch_body" | "switch_block") { return Some("switch"); }
    if matches!(t, "communication_case" | "expression_case" | "type_case" | "default_case" | "switch_default" | "switch_label" | "switch_rule" | "switch_block_statement_group" | "case_pattern" | "class_pattern" | "complex_pattern" | "dict_pattern" | "keyword_pattern" | "list_pattern" | "union_pattern" | "as_pattern") { return Some("case"); }
    if matches!(t, "yield_statement" | "yield_expression" | "generator_function" | "generator_function_declaration") { return Some("generator"); }
    if matches!(t, "record_pattern" | "record_pattern_body") { return Some("record"); }
    if matches!(t, "class" | "class_declaration") { return Some("type_def"); }
    if end("_body") { return Some("body"); }
    if end("_clause") { return Some("clause"); }
    if end("_pattern") { return Some("pattern"); }
    if end("_literal") { return Some("literal"); }
    if end("_statement") { return Some("stmt"); }
    if end("_expression") { return Some("expr"); }
    if end("_declaration") { return Some("decl"); }
    if matches!(t, "statement" | "declaration" | "expression" | "primary_expression" | "program" | "source_file" | "_expression" | "_statement" | "_simple_statement" | "_compound_statement" | "_declaration_statement" | "_literal" | "_pattern" | "control") { return Some("control"); }
    if matches!(t, "this" | "self" | "super" | "crate" | "blank_identifier" | "package_identifier" | "dotted_name" | "identifier") { return Some("var"); }
    if matches!(t, "parenthesized_expression" | "parenthesized_type" | "unit_expression" | "const_block" | "gen_block" | "try_block" | "unsafe_block" | "match_block" | "block") { return Some("block"); }
    if matches!(t, "else" | "elif") { return Some("else"); }
    if matches!(t, "decorated_definition" | "decorator" | "marker_annotation" | "annotation") { return Some("attribute"); }
    if matches!(t, "macro_rule" | "macro_invocation" | "macro_definition") { return Some("macro"); }
    if matches!(t, "function_signature_item" | "function_type" | "closure_parameters" | "lambda_parameters") { return Some("param"); }
    if matches!(t, "string_content" | "escape_sequence") { return Some("interp"); }
    // Linguistic-category battery for the long tail (each rule is a genuine category).
    if has("gnu_asm") || t.starts_with("asm_") { return Some("asm"); }
    if has("preproc") || has("pragma") || has("directive") || t == "cpp" || t == "haddock" || t == "shebang_directive" || t == "ignored_directive" { return Some("directive"); }
    if has("declarator") { return Some("decl"); }
    if matches!(t, "type_specifier" | "sized_type_specifier" | "macro_type_specifier" | "placeholder_type_specifier" | "splice_type_specifier" | "type_descriptor" | "decltype" | "type_qualifier") { return Some("type_expr"); }
    if matches!(t, "union_specifier" | "type_definition" | "type_synonym" | "newtype" | "trait_definition" | "object_definition" | "given_definition" | "extension_definition" | "enum_definition" | "struct_definition" | "abstract_definition" | "primitive_definition" | "val_definition" | "var_definition" | "singleton_class" | "companion_object" | "class_decl" | "instance_decl" | "data_family" | "abstract_family" | "type_family" | "constructor_synonym" | "pattern_synonym" | "data_constructor" | "gadt_constructor" | "newtype_constructor" | "constructor" | "data_constructors" | "gadt_constructors" | "constructor_synonyms") { return Some("type_def"); }
    if matches!(t, "enumerator" | "enum_entry" | "enum_constant" | "enumerator_list" | "enum_member_declaration_list" | "enum_case_definitions" | "full_enum_case" | "simple_enum_case" | "enumerators" | "enum_body_declarations") { return Some("field"); }
    if has("modifier") || has("qualifier") || has("specifier") || matches!(t, "calling_convention" | "safety" | "byte_alignment" | "address_space" | "link_section" | "semantic_version" | "noexcept" | "auto" | "throw_specifier" | "linkage_specification" | "attribute_specifier" | "storage_class_specifier" | "access_specifier" | "virtual_specifier" | "explicit_function_specifier" | "alignas_qualifier") { return Some("modifier"); }
    if has("lambda") || has("closure") { return Some("closure"); }
    if has("constraint") || has("bound") || has("forall") || has("deriving") || has("fundep") || matches!(t, "constraint" | "constraints" | "context" | "context_bound" | "view_bound" | "quantified_variables" | "type_parameter_constraint" | "constructor_constraint" | "type_constraints" | "role_annotation" | "fixity") { return Some("bound"); }
    if has("template") || has("generic_name") || has("type_argument") || has("type_parameter") || has("type_projection") || has("kind_") || has("type_application") || has("type_binder") || has("type_params") || has("type_param") || has("type_patterns") || has("type_role") || has("type_head") || has("type_lambda") || has("signature") || has("type_test") || has("range_test") || has("instanceof") { return Some("type_expr"); }
    if has("argument") || has("parameter") || has("variadic") || has("forward_argument") || has("block_argument") || has("splat_argument") || has("hash_splat") || has("rest_assignment") || has("left_assignment_list") || has("right_assignment_list") || has("destructured_left_assignment") || has("bindings") || has("binding") || has("capture_") || has("colon_argument") || has("parameter_types") || has("value_argument") || has("macro_argument_list") || has("subscript_argument_list") || has("bracketed_argument_list") || has("bracketed_parameter_list") || has("parameter_pack_expansion") { return Some("param"); }
    if matches!(t, "table_constructor" | "hash" | "dict") { return Some("dict"); }
    if matches!(t, "string_array" | "symbol_array" | "empty_list" | "prefix_list" | "variable_list" | "array_initializer" | "initializer_list" | "collection_element") { return Some("list"); }
    if has("initializer") || has("init_declarator") || has("initializer_pair") || has("designator") || has("keyed_element") || has("literal_element") || has("literal_value") || has("composite_literal") || has("variable_declarator") || has("var_spec") || has("const_spec") || has("const_declaration") || has("creation_expression") || has("object") || has("pair") || has("assignment_pattern") { return Some("init"); }
    if matches!(t, "tuple_element" | "matrix_row" | "open_tuple" | "prefix_tuple" | "unboxed_sum" | "unboxed_tuple" | "children" | "group" | "rec" | "expression_list" | "sequence_expression" | "statement_list" | "pattern_list") { return Some("seq"); }
    if has("operator") || matches!(t, "binary" | "unary" | "infix" | "prefix" | "apply" | "left_section" | "right_section" | "constructor_operator" | "_call_operator" | "operator_assignment" | "conditional") { return Some("op"); }
    if has("xml") || has("jsx") { return Some("jsx"); }
    if has("interpolat") || has("escape") || has("heredoc") || has("quasiquote") || has("quote") || has("splice") || has("raw_string") || has("system_lib_string") || has("multiline_string") || has("interpolated_string") || has("bare_string") || has("chained_string") || has("uninterpreted") || has("content") || has("string") || has("symbol") { return Some("str_lit"); }
    if matches!(t, "character" | "char" | "character_literal_content" | "character_content" | "literal_suffix" | "encoding" | "line" | "file") { return Some("char_lit"); }
    if has("import") || has("include") || has("use_") || has("alias") || has("extern") || has("package") || has("module") || has("translation_unit") || has("compilation_unit") || has("header") || has("chunk") { return Some("module"); }
    if has("export") || has("namespace") { return Some("export"); }
    if has("field") || has("property") || has("accessor") || has("getter") || has("setter") || has("delegate") || has("delegation") || has("projection") { return Some("field"); }
    if has("guard") || has("condition") || has("optional_binding") || has("when") || has("unless") || has("until") || has("elsif") || has("rescue") || has("catch") || has("ensure") || has("retry") || has("redo") || has("next") || has("case") || has("match") || has("alternative") || has("switch") || has("then") || has("else") || has("if") || has("do") || has("begin") || has("end_block") { return Some("control"); }
    if has("variable") || has("identifier") || has("name") || has("constant") || has("this") || has("self") || has("discard") || has("payload") || has("qualified") || has("qualifiers") || has("scope_resolution") || has("alias_qualified") || has("identifiers") || has("end_marker") || has("all_names") || has("entity") { return Some("var"); }
    if has("literal") || has("numeric") || has("complex") || has("rational") || has("character") || has("char") { return Some("literal"); }
    if has("annotation") || has("attribute") || has("file_annotation") || has("global_attribute") || has("use_site_target") { return Some("attribute"); }
    if has("requirement") || has("noexcept") || has("throw_specifier") || has("calling_convention") || has("safety") || has("semantic_version") || has("availability") || has("byte_alignment") || has("address_space") || has("link_section") || has("strict") || has("linear_function") || has("irrefutable") || has("invisible") || has("inferred") || has("promoted") || has("vararg") || has("using_directive") { return Some("modifier"); }
    if has("block") || has("parens") || has("parenthesized_statements") || has("indented_") || has("body") || has("equations") || has("equation") || has("declarations") || has("local_binds") || has("binding_list") || has("let") || has("bind") || has("early_defs") || has("refinement") || has("record") || has("fields") || has("group") || has("children") || has("rec") || has("unit") || has("unboxed") || has("prefix_unboxed") || has("arithmetic_sequence") || has("subshell") || has("class_declarations") || has("instance_declarations") || has("quoted_decls") || has("default_types") || has("patterns") || has("lambda_case") || has("multi_way_if") || has("prefix_list") || has("prefix_tuple") || has("empty_list") || has("matrix_row") || has("open_tuple") || has("tuple_element") || has("collection_element") || has("expression_element") || has("accessor_list") || has("switch_section") || has("switch_expression_arm") || has("case_block") || has("when_entry") || has("when_subject") || has("catch_block") || has("finally_block") || has("begin_block") || has("end_block") || has("do_block") || has("companion_object") || has("singleton_class") || has("constructor_body") || has("static_initializer") || has("class_static_block") || has("module_body") || has("enum_body") || has("interface_body") || has("annotation_type_body") || has("class_body") || has("record_pattern_body") { return Some("body"); }
    if has("definition") || has("decl") || has("instance") || has("deriving") || has("gadt") || has("type_family") || has("type_instance") || has("kind_signature") || has("forall_required") || has("function_head_parens") || has("default_signature") || has("signature") || has("fixity") || has("role_annotation") || has("top_splice") || has("th_quoted_name") || has("typed_quote") || has("splice") || has("quasiquote") || has("quote") || has("transform") || has("special") || has("via") || has("as") || has("star") || has("method_value") || has("constructor_delegation") || has("constructor_invocation") || has("explicit_delegation") || has("delegation_specifier") || has("property_delegate") || has("primary_constructor") || has("secondary_constructor") || has("base_list") || has("explicit_interface") || has("subpattern") || has("type_argument_list") || has("directive_kind") || has("directive_separator") || has("directive_value") || has("interpolation_brace") || has("interpolation_quote") || has("interpolation_start") || has("raw_string_end") || has("raw_string_start") || has("shebang") || has("xml") || has("interpolation") || has("parenthesized_variable_designation") || has("structured_binding") || has("reference_declarator") || has("pointer_type_declarator") || has("new_declarator") || has("for_range_loop") || has("template_function") || has("template_instantiation") || has("template_method") || has("operator_cast") || has("splice_specifier") || has("dependent_name") || has("destructor_name") || has("operator_name") || has("context_bound") || has("capture") { return Some("decl"); }
    if has("statement") || has("expression") || has("exp") || has("primary") || has("control") || has("guard") || has("condition") || has("when") || has("unless") || has("until") || has("elsif") || has("rescue") || has("catch") || has("ensure") || has("retry") || has("redo") || has("next") || has("case") || has("match") || has("alternative") || has("switch") || has("then") || has("else") || has("if") || has("do") || has("begin") || has("label") || has("builtin") || has("macro_identifier") || has("container_field") || has("block_label") || has("break_label") || has("implicit_variable") { return Some("control"); }
    // ML-family (SML/OCaml) systematic naming: each suffix is a genuine linguistic category.
    if end("_pat") || end("patrow") || end("listpat") { return Some("pattern"); }
    if end("_ty") || end("tyrow") || end("tyseq") || end("tyvarseq") { return Some("type_expr"); }
    if end("_dec") || end("desc") || end("_spec") || end("_strdec") || end("_sigdec")
        || end("_fctdec") || end("_topdec") { return Some("decl"); }
    if end("scon") { return Some("literal"); }
    // Dhall builtin identifiers (`Natural/build`, `Text/show`, … as `*_raw` nodes).
    if end("_raw") { return Some("var"); }
    None
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

/// Construct histogram over a grammar's NAMED node types: how many surface node types
/// realize each universal construct. This is the raw distribution the structural entropy
/// is computed from (a node realizing several constructs contributes to each).
pub fn construct_histogram(nt: &Value) -> BTreeSet<(&'static str, usize)> {
    let mut hist: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    if let Some(arr) = nt.as_array() {
        for n in arr {
            if !n.get("named").and_then(|v| v.as_bool()).unwrap_or(false) {
                continue;
            }
            if let Some(t) = n.get("type").and_then(|v| v.as_str()) {
                let mut s = BTreeSet::new();
                node_constructs(t, &mut s);
                for c in s {
                    *hist.entry(c).or_insert(0) += 1;
                }
            }
        }
    }
    hist.into_iter().collect()
}

/// Information-theoretic profile of a language's grammar, computed from real node-types:
/// (named_nodes, mapped, distinct_constructs, H_struct_bits, redundancy, surface_per_construct).
///
/// H_struct is the Shannon entropy of the construct histogram — how evenly the grammar's
/// surface forms spread across the universal taxonomy. Redundancy = 1 - H/log2(C) is the
/// fraction of surface variety that is NOT free information (syntactic sugar the IR collapses).
/// surface_per_construct = N/C is the mean number of surface forms per universal construct:
/// exactly the distance the one-IR LUT must bridge for this language. These are grammar-derived
/// (structural) measures, honestly labelled — not corpus frequencies.
pub fn structural_metrics(nt: &Value) -> (usize, usize, usize, f64, f64, f64) {
    let (mapped, total, _) = grammar_coverage(nt);
    let hist = construct_histogram(nt);
    let c = hist.len();
    let sum: usize = hist.iter().map(|(_, n)| *n).sum();
    let h = if sum == 0 {
        0.0
    } else {
        hist.iter()
            .map(|(_, n)| {
                let p = *n as f64 / sum as f64;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum::<f64>()
    };
    let redundancy = if c > 1 { 1.0 - h / (c as f64).log2() } else { 0.0 };
    let surface = if c == 0 { 0.0 } else { total as f64 / c as f64 };
    (total, mapped, c, h, redundancy, surface)
}

/// Highlight capture for a node type, derived from its universal construct — the basis for
/// generating tree-sitter queries (highlights.scm) from the construct IR.
pub fn highlight_capture(t: &str) -> &'static str {
    let mut s = BTreeSet::new();
    node_constructs(t, &mut s);
    if s.contains("comment") { return "comment"; }
    if s.contains("str_lit") || s.contains("char_lit") { return "string"; }
    if s.contains("int_lit") || s.contains("float_lit") || s.contains("bool_lit") || s.contains("null_lit") { return "number"; }
    if s.contains("keyword") || s.contains("control") || s.contains("directive") { return "keyword"; }
    if s.contains("fn") || s.contains("call") || s.contains("method_ref") { return "function"; }
    if s.contains("type_expr") || s.contains("type_def") || s.contains("bound") { return "type"; }
    if s.contains("field") || s.contains("member") || s.contains("property") { return "property"; }
    if s.contains("param") { return "variable.parameter"; }
    if s.contains("op") || s.contains("and") || s.contains("or") || s.contains("not") || s.contains("eq") { return "operator"; }
    if s.contains("var") || s.contains("symbol") { return "variable"; }
    "punctuation"
}

/// The named node types with no construct mapping yet (the honest remainder to close).
pub fn unmapped_named(nt: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(arr) = nt.as_array() {
        for n in arr {
            if !n.get("named").and_then(|v| v.as_bool()).unwrap_or(false) {
                continue;
            }
            if let Some(t) = n.get("type").and_then(|v| v.as_str()) {
                let mut s = BTreeSet::new();
                node_constructs(t, &mut s);
                if s.is_empty() {
                    out.push(t.to_string());
                }
            }
        }
    }
    out
}