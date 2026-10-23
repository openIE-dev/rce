"""Lift probe: Go CST (tree-sitter) + a semantics table -> an RCE Form skeleton.
Deterministic: no model, no heuristics beyond the table; every decision is a
CST node kind looked up in SEM. Scope: a struct with a pointer-receiver method
per event, guards as `if <field> { return Err }`, field assignments as updates."""
import sys, json, hashlib
import tree_sitter, tree_sitter_go
GO = tree_sitter.Language(tree_sitter_go.language())
P = tree_sitter.Parser(GO)

# the semantics table: Go spec facts the lift needs (Go spec §Numeric types, §Arithmetic)
SEM = {
  "types": {"bool": "bool", "int64": "i64", "int": "i64", "uint64": "u64", "string": "text", "float64": "f64"},
  "int_overflow": "wraps (Go spec: 'integer overflow ... wraps around'); RCE traps: emit must insert checked ops",
  "division": "truncates toward zero, remainder takes dividend sign (Go spec) = RCE 6.3",
  "error_return": "a function returning a non-nil error = a refused event (T8: state unchanged only if no write preceded the return)",
}
src = open(sys.argv[1], "rb").read()
tree = P.parse(src)
def text(n): return src[n.start_byte:n.end_byte].decode()
def kids(n, kind): return [c for c in n.children if c.type == kind]
def walk(n):
    yield n
    for c in n.children: yield from walk(c)

regions, errors, events, steps = {}, [], [], []
for n in walk(tree.root_node):
    if n.type == "type_declaration":
        for spec in kids(n, "type_spec"):
            name = text(spec.child_by_field_name("name")); ty = spec.child_by_field_name("type")
            if ty.type == "struct_type":
                fields = []
                for fd in walk(ty):
                    if fd.type == "field_declaration":
                        fname = text(fd.child_by_field_name("name")); ftype = text(fd.child_by_field_name("type"))
                        fields.append((fname, SEM["types"].get(ftype, "?"+ftype)))
                regions[name] = fields
    if n.type == "var_declaration":
        for vs in kids(n, "var_spec"):
            v = vs.child_by_field_name("value")
            if v and "errors.New" in text(v):
                errors.append(text(v).split('"')[1])
    if n.type == "method_declaration":
        recv = n.child_by_field_name("receiver"); name = text(n.child_by_field_name("name"))
        rtype = text([c for c in walk(recv) if c.type == "type_identifier"][0])
        body = n.child_by_field_name("body")
        guards, writes = [], []
        stmts = [s for sl in kids(body, 'statement_list') for s in sl.children]
        for st in stmts:
            if st.type == "if_statement":
                cond = text(st.child_by_field_name("condition")).replace("d.", "")
                ret = [r for r in walk(st) if r.type == "return_statement"]
                if ret and "Err" in text(ret[0]):
                    err = text(ret[0]).split("Err")[1].strip()
                    guards.append((cond, err))
            if st.type == "assignment_statement":
                lhs = text(st.child_by_field_name("left")).replace("d.", ""); rhs = text(st.child_by_field_name("right"))
                writes.append((lhs, rhs))
        events.append(name); steps.append((rtype, name, guards, writes))

out = [f'state {r}'] if False else []
lines = []
for r, fields in regions.items():
    lines.append(f"state {r}")
    for f, t in fields: lines.append(f"    {f}: {t} = {'false' if t=='bool' else '0' if t in ('i64','u64') else '\"\"'}")
lines.append("")
for e in events: lines.append(f"event {e}")
lines.append("")
for x in errors: lines.append(f"error {x}")
lines.append("")
for r, e, guards, writes in steps:
    for cond, err in guards:
        lines.append(f"step {r} on {e} if {cond}")
        lines.append(f"    fail {err}")
    lines.append(f"step {r} on {e}")
    for l, v in writes: lines.append(f"    {l} = {v}")
    if not writes: lines.append("    pass")
form = "\n".join(lines) + "\n"
print(form)
print("# lift sha256:", hashlib.sha256(form.encode()).hexdigest()[:16], file=sys.stderr)
