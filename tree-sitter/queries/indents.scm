;; nvim-treesitter indent captures. RCE is indentation sensitive (the scanner
;; emits INDENT and DEDENT), so a new line after a block header indents one level.
[
  (block)
  (struct_decl)
  (state_decl)
  (kind_decl)
  (trait_decl)
  (impl_decl)
  (actor_decl)
  (protocol_decl)
  (intent_decl)
  (match_stmt)
  (select_stmt)
  (par_stmt)
  (supervisor_decl)
  (proto_offer)
  (proto_choose)
  (proto_loop)
] @indent.begin

(args) @indent.begin
(list) @indent.begin
(map) @indent.begin

[ ")" "]" "}" ] @indent.branch @indent.end
[ (elif_clause) (else_clause) ] @indent.branch
