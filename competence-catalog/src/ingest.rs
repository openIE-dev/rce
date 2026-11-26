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
        "stmt" => { out.insert("stmt"); }
        "paren" => { out.insert("block"); }
        "true_const" => { out.insert("bool_lit"); }
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