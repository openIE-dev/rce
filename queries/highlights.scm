; Syntax highlighting for RCE. Capture names follow the tree-sitter
; conventions used by most editors (nvim-treesitter, Helix, Zed).

[
  "module" "use" "as" "intent" "example" "refused" "check"
  "const" "type" "struct" "state" "event" "error" "kind"
  "trait" "impl" "forall" "extern" "step" "on" "actor"
  "let" "var" "with" "arena" "device" "bound" "pass" "where"
  "chan" "iter" "spawn" "select" "recv" "send" "to" "from" "after" "default"
] @keyword

"fn" @keyword.function

[
  "if" "elif" "else" "then" "match"
] @keyword.conditional

[
  "while" "loop" "for" "break" "continue"
] @keyword.repeat

[
  "return" "emit" "yield"
] @keyword.return

[
  "fail" "try" "catch"
] @keyword.exception

[
  "and" "or" "not" "in"
] @keyword.operator

[
  (PUB) (LINEAR) (COMPTIME) (MUT) "comptime" "mut"
] @keyword.modifier

(STRING) @string
(INT) @number
(FLOAT) @number.float
[(true) (false) (true_pat) (false_pat)] @boolean
[(none) (none_pat)] @constant.builtin
(self_ref) @variable.builtin
[(self_param) (ref_self_param) (mut_self_param)] @variable.builtin
(wildcard) @variable.builtin

(fn_sig (NAME) @function)
(call (name (NAME) @function.call))
(call (member (member_name) @function.method.call))

(state_decl (NAME) @type)
(struct_decl (NAME) @type)
(kind_decl (NAME) @type)
(trait_decl (NAME) @type)
(actor_decl (NAME) @type)
(type_decl (NAME) @type)
(type_path (path (NAME) @type))
(gparam (NAME) @type)

; Version 0.8 indexed types (spec 5.16): context-typed text, secrets,
; capabilities and guarded fields.
((type_path (path (NAME) @type.builtin))
  (#any-of? @type.builtin "Text" "Html" "Url" "Sql" "Json" "Secret" "Cap" "Guarded"))

(event_decl (NAME) @constructor)
(error_decl (NAME) @constructor)
(variant (NAME) @constructor)
(ctor_pat (path (NAME) @constructor))
(event_call (NAME) @constructor)
(event_ref (NAME) @constructor)
(step_decl (NAME) @type)

(field (NAME) @property)
(field_item (NAME) @property)
(member (member_name) @property)
(arg (NAME) @label)
(pat_arg (NAME) @label)

(param (NAME) @variable.parameter)
(const_decl (NAME) @constant)
(module_header (path (NAME) @module))
(use_decl (path (NAME) @module))

[
  (assign_op) (cmp_op) (add_op) (mul_op) (shift_op) (range_op)
  "!" "?" "&" "|" "^" "~" "->" "=>" "@" "=" (SEND)
] @operator

["(" ")" "[" "]" "{" "}"] @punctuation.bracket
["," "." ":" ";"] @punctuation.delimiter
