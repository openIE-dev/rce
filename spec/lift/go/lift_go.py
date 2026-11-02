#!/usr/bin/env python3
"""Lift probe: Go CST (tree-sitter) + a semantics table -> an RCE Form skeleton.

Usage (positional arguments only):

    lift_go.py lift FILE    the lifted Form on standard output (exit 0), or one
                            line per construct the probe does not carry on
                            standard output, `requires change FILE: line N: RULE: reason`
                            (exit 1); exit 3 when FILE does not parse as Go
    lift_go.py report FILE [RCE HAND]
                            the lift report (rce-lift-report/1, Appendix D.8 L5
                            and L6) on standard output: every claim of
                            spec/lift/authority.json for this probe, its required
                            authority, the digests it is bound to, its evidence and
                            its verdict; with RCE (the rce binary) and HAND (a
                            hand-written RCE program), GO-EQUIV gets the evidence of
                            `rce equiv json`. Exit 0, or 4 when a claim failed; 1, 3
                            and 2 as for lift
    lift_go.py rules        the rule ids and what each covers

Deterministic: no model, no heuristics beyond the table; every decision is a
CST node kind looked up in the rows or rules of table.json, and every row
cites its clause of the Go specification. The probe reads Go and runs no
Go; `report` with RCE runs `rce equiv` on the lifted Form.

Scope: a struct with a pointer-receiver method per event, guards as
`if <condition> { return Err }` before any write, field assignments as
updates, and `errors.New` variables as errors. Every other construct is
reported with its rule and line, never dropped: a construct inside one that
is not carried (a loop inside a function, a call inside a loop) is reported
too, so the report lists everything the Form would be missing."""
import sys, hashlib, json, os, re, subprocess, tempfile
import tree_sitter, tree_sitter_go
GO = tree_sitter.Language(tree_sitter_go.language())
P = tree_sitter.Parser(GO)

# the lift table: data, every row citing its clause of the Go specification (table.json)
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "table.json")
with open(TABLE_PATH, encoding="utf-8") as _f:
    TABLE = json.load(_f)
ROWS = {r["id"]: r for r in TABLE["rows"]}
TYPES = {r["go"]: r for r in TABLE["rows"] if r["id"].startswith("type.")}
OPS = {r["go"]: r for r in TABLE["rows"] if r["id"].startswith("op.") and r["id"] != "op.not"}
RULES = TABLE["rules"]
sys.path.insert(0, os.path.dirname(HERE))
import authority  # noqa: E402  (spec/lift/authority.py)

# node kinds reported wherever they occur inside a construct that is not carried
NESTED = {
  "function_declaration": "L-GO-FUNC", "func_literal": "L-GO-FUNC", "method_declaration": "L-GO-METHOD",
  "for_statement": "L-GO-FOR",
  "expression_switch_statement": "L-GO-SWITCH", "type_switch_statement": "L-GO-SWITCH", "select_statement": "L-GO-SWITCH",
  "if_statement": "L-GO-IF",
  "call_expression": "L-GO-CALL",
  "type_assertion_expression": "L-GO-ASSERT",
  "array_type": "L-GO-COLLECTION", "implicit_length_array_type": "L-GO-COLLECTION", "slice_type": "L-GO-COLLECTION",
  "map_type": "L-GO-COLLECTION", "channel_type": "L-GO-COLLECTION", "composite_literal": "L-GO-COLLECTION",
  "index_expression": "L-GO-COLLECTION", "slice_expression": "L-GO-COLLECTION",
  "go_statement": "L-GO-STMT", "defer_statement": "L-GO-STMT", "goto_statement": "L-GO-STMT",
  "labeled_statement": "L-GO-STMT", "send_statement": "L-GO-STMT",
}

STMT_NAMES = {
  "short_var_declaration": "a short variable declaration (:=)", "var_declaration": "a local var declaration",
  "const_declaration": "a local const declaration", "inc_statement": "an increment (++)", "dec_statement": "a decrement (--)",
  "expression_statement": "an expression statement", "block": "a nested block", "go_statement": "a go statement",
  "defer_statement": "a defer statement", "goto_statement": "a goto statement", "labeled_statement": "a labelled statement",
  "send_statement": "a channel send", "break_statement": "a break statement", "continue_statement": "a continue statement",
  "fallthrough_statement": "a fallthrough statement", "type_declaration": "a local type declaration",
}

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# RCE's reserved words (the list spec/lift/python/lift_py.py renames around); a Go name among them requires change
KW = {"and", "or", "not", "in", "if", "then", "else", "elif", "for", "while", "loop", "fn", "let", "var",
      "return", "break", "continue", "match", "emit", "fail", "true", "false", "none", "state", "event",
      "error", "step", "on", "kind", "struct", "type", "trait", "impl", "use", "module", "const", "pub",
      "check", "example", "intent", "pass", "bound", "self", "spawn", "select", "with", "device", "yield",
      "comptime", "try", "catch", "as", "actor", "extern", "linear", "forall", "recv", "send", "after", "default"}


class Lift:
    def __init__(self, src):
        self.src = src
        self.diags = []  # (start_byte, line, rule, reason)
        self.used = set()  # the ids of the table rows the Form is built from

    def row(self, rid):
        self.used.add(rid)
        return ROWS[rid]

    def text(self, n):
        return self.src[n.start_byte:n.end_byte].decode()

    def report(self, n, rule, reason):
        self.diags.append((n.start_byte, n.start_point[0] + 1, rule, reason))

    def drop(self, n, rule, reason):
        """n is not carried: report it, then every construct of NESTED inside it."""
        self.report(n, rule, reason)
        for c in n.named_children:
            self.inner(c)

    def inner(self, n):
        if n.type == "comment":
            return
        if n.type in NESTED:
            self.report(n, NESTED[n.type], self.describe(n))
        for c in n.named_children:
            self.inner(c)

    def describe(self, n):
        t = self.text(n).split("\n")[0].strip()
        if len(t) > 60:
            t = t[:57] + "..."
        return f"{n.type.replace('_', ' ')} `{t}`"

    # ---- expressions: field selectors of the receiver, literals, !, &&, ||, comparisons
    def expr(self, n, recv, fields):
        k = n.type
        if k in ("true", "false"):
            self.row("lit.bool")
            return k
        if k == "int_literal" and re.fullmatch(r"[0-9]+", self.text(n)) and (self.text(n) == "0" or not self.text(n).startswith("0")):
            self.row("lit.int")
            return self.text(n)
        if k == "interpreted_string_literal" and "\\" not in self.text(n):
            self.row("lit.string")
            return self.text(n)
        if k == "selector_expression":
            op, f = n.child_by_field_name("operand"), n.child_by_field_name("field")
            if op.type == "identifier" and self.text(op) == recv and self.text(f) in fields:
                self.row("expr.field")
                return self.text(f)
            return self.bad(n, "L-GO-EXPR", f"a selector that is not a field of the receiver: {self.describe(n)}")
        if k == "parenthesized_expression":
            inner = self.expr(n.named_children[0], recv, fields)
            self.row("expr.paren")
            return None if inner is None else f"({inner})"
        if k == "unary_expression":
            op = self.text(n.child_by_field_name("operator"))
            if op == "!":
                x = self.expr(n.child_by_field_name("operand"), recv, fields)
                return None if x is None else f"{self.row('op.not')['rce']} {x}"
            if op in ("&", "*"):
                return self.bad(n, "L-GO-POINTER", self.describe(n))
            return self.bad(n, "L-GO-INT", self.describe(n))
        if k == "binary_expression":
            op = self.text(n.child_by_field_name("operator"))
            if op not in OPS:
                rule = "L-GO-INT" if op in ("+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "&^") else "L-GO-EXPR"
                return self.bad(n, rule, self.describe(n))
            a = self.expr(n.child_by_field_name("left"), recv, fields)
            b = self.expr(n.child_by_field_name("right"), recv, fields)
            if a is None or b is None:
                return None
            return f"{a} {self.row(OPS[op]['id'])['rce']} {b}"
        if k in NESTED:
            return self.bad(n, NESTED[k], self.describe(n))
        if k == "identifier":
            return self.bad(n, "L-GO-EXPR", f"a name that is not a field of the receiver: `{self.text(n)}`")
        return self.bad(n, "L-GO-EXPR", self.describe(n))

    def name(self, n):
        if self.text(n) in KW:
            self.report(n, "L-GO-NAME", f"`{self.text(n)}` is an RCE keyword")

    def bad(self, n, rule, reason):
        self.drop(n, rule, reason)
        return None

    # ---- declarations
    def run(self, root):
        regions, errors, err_vars, events, steps = {}, [], {}, [], []
        methods = []
        for n in root.named_children:
            k = n.type
            if k in ("package_clause", "comment"):
                continue
            if k == "import_declaration":
                for s in [x for x in self.walk(n) if x.type == "import_spec"]:
                    path = self.text(s.child_by_field_name("path"))
                    if path != '"errors"' or s.child_by_field_name("name") is not None:
                        self.report(s, "L-GO-IMPORT", f"import {self.text(s)}")
            elif k == "type_declaration":
                for spec in n.named_children:
                    if spec.type != "type_spec" or spec.child_by_field_name("type_parameters") is not None:
                        self.drop(spec, "L-GO-TYPE", self.describe(spec))
                        continue
                    name = self.text(spec.child_by_field_name("name")); ty = spec.child_by_field_name("type")
                    if ty.type != "struct_type":
                        self.drop(spec, "L-GO-TYPE", f"type {name} is not a struct: {self.describe(ty)}")
                        continue
                    self.name(spec.child_by_field_name("name"))
                    fields = []
                    for fd in [x for x in self.walk(ty) if x.type == "field_declaration"]:
                        names = fd.children_by_field_name("name")
                        ftype = fd.child_by_field_name("type")
                        if not names:
                            self.drop(fd, "L-GO-TYPE", f"an embedded field: {self.describe(fd)}")
                            continue
                        trow = TYPES.get(self.text(ftype))
                        if trow is None:
                            self.drop(fd, "L-GO-TYPE", f"field type `{self.text(ftype)}` is outside the semantics table")
                            continue
                        self.row(trow["id"])
                        for nm in names:
                            self.name(nm)
                            fields.append((self.text(nm), trow["rce"], trow["zero"]))
                    self.row("decl.struct")
                    regions[name] = fields
            elif k == "var_declaration":
                for vs in [x for x in self.walk(n) if x.type == "var_spec"]:
                    names = vs.children_by_field_name("name"); v = vs.child_by_field_name("value")
                    vals = v.named_children if v is not None else []
                    ok = (len(names) == 1 and len(vals) == 1 and vs.child_by_field_name("type") is None
                          and vals[0].type == "call_expression" and self.text(vals[0].child_by_field_name("function")) == "errors.New")
                    if ok:
                        args = vals[0].child_by_field_name("arguments").named_children
                        msg = self.text(args[0])[1:-1] if len(args) == 1 and args[0].type == "interpreted_string_literal" else None
                        ok = msg is not None
                    if ok and (IDENT.match(msg) is None or msg in KW):
                        self.report(vs, "L-GO-NAME", f"error message `{msg}` is not an RCE name")
                        continue
                    if not ok:
                        self.drop(vs, "L-GO-DECL", f"var {self.describe(vs)}")
                        continue
                    self.row("decl.error")
                    err_vars[self.text(names[0])] = msg
                    errors.append(msg)
            elif k == "const_declaration":
                self.drop(n, "L-GO-DECL", self.describe(n))
            elif k == "method_declaration":
                methods.append(n)
            elif k == "function_declaration":
                self.drop(n, "L-GO-FUNC", f"func {self.text(n.child_by_field_name('name'))}")
            else:
                self.drop(n, "L-GO-STMT", self.describe(n))
        for n in methods:
            st = self.method(n, regions, err_vars, events)
            if st is not None:
                events.append(st[1]); steps.append(st)
        return regions, errors, events, steps

    def walk(self, n):
        yield n
        for c in n.children:
            yield from self.walk(c)

    def method(self, n, regions, err_vars, events):
        name = self.text(n.child_by_field_name("name"))
        params = n.child_by_field_name("receiver").named_children
        recv = rtype = None
        if len(params) == 1 and params[0].type == "parameter_declaration":
            p = params[0]; ty = p.child_by_field_name("type"); nm = p.child_by_field_name("name")
            if nm is not None and ty.type == "pointer_type" and self.text(ty)[1:].strip() in regions:
                recv, rtype = self.text(nm), self.text(ty)[1:].strip()
        result = n.child_by_field_name("result")
        why = None
        if recv is None:
            why = f"method {name}: the receiver is not a named pointer to a lifted struct"
        elif n.child_by_field_name("parameters").named_children:
            why = f"method {name} has parameters"
        elif result is None or self.text(result) != "error":
            why = f"method {name} does not return exactly error"
        elif name in events:
            why = f"a second method named {name}"
        if why:
            self.drop(n, "L-GO-METHOD", why)
            return None
        self.name(n.child_by_field_name("name"))
        fields = {f for f, _, _ in regions[rtype]}
        body = n.child_by_field_name("body")
        stmts = [s for sl in body.named_children if sl.type == "statement_list" for s in sl.named_children]
        stmts += [s for s in body.named_children if s.type not in ("statement_list", "comment")]
        stmts = [s for s in stmts if s.type != "comment"]
        guards, writes, before = [], [], len(self.diags)
        for i, s in enumerate(stmts):
            last = i == len(stmts) - 1
            if s.type == "if_statement":
                g = self.guard(s, recv, fields, err_vars)
                if g is None:
                    continue
                if writes:
                    self.drop(s, "L-GO-ORDER", f"method {name}: guard `if {g[0]}` after a write")
                    continue
                self.row("stmt.guard")
                guards.append(g)
            elif s.type == "assignment_statement":
                lhs, rhs = s.child_by_field_name("left").named_children, s.child_by_field_name("right").named_children
                op = self.text(s.child_by_field_name("operator"))
                if op != "=" or len(lhs) != 1 or len(rhs) != 1:
                    self.drop(s, "L-GO-INT" if op in ("+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=", "&^=") else "L-GO-ASSIGN", self.describe(s))
                    continue
                l = lhs[0]
                if not (l.type == "selector_expression" and self.text(l.child_by_field_name("operand")) == recv
                        and self.text(l.child_by_field_name("field")) in fields):
                    self.drop(s, "L-GO-ASSIGN", self.describe(s))
                    continue
                v = self.expr(rhs[0], recv, fields)
                if v is not None:
                    self.row("stmt.assign")
                    writes.append((self.text(l.child_by_field_name("field")), v))
            elif s.type == "return_statement":
                if not (last and self.text(s) == "return nil"):
                    self.drop(s, "L-GO-RETURN", self.describe(s))
                else:
                    self.row("stmt.return-nil")
            elif s.type in NESTED:
                self.drop(s, NESTED[s.type], self.describe(s))
            else:
                self.drop(s, "L-GO-STMT", STMT_NAMES.get(s.type, s.type.replace("_", " ")) + f": {self.describe(s)}")
        if not stmts or stmts[-1].type != "return_statement":
            self.report(n, "L-GO-RETURN", f"method {name} does not end with `return nil`")
        if len(self.diags) != before:
            return None
        self.row("decl.event")
        return (rtype, name, guards, writes)

    def guard(self, s, recv, fields, err_vars):
        if s.child_by_field_name("initializer") is not None or s.child_by_field_name("alternative") is not None:
            return self.bad(s, "L-GO-IF", self.describe(s))
        cons = [c for c in s.child_by_field_name("consequence").named_children if c.type != "comment"]
        cons = [x for c in cons for x in (c.named_children if c.type == "statement_list" else [c]) if x.type != "comment"]
        if len(cons) != 1 or cons[0].type != "return_statement":
            return self.bad(s, "L-GO-IF", self.describe(s))
        vals = [c for c in cons[0].named_children if c.type != "comment"]
        vals = [x for c in vals for x in (c.named_children if c.type == "expression_list" else [c])]
        if len(vals) != 1 or vals[0].type != "identifier" or self.text(vals[0]) not in err_vars:
            self.drop(cons[0], "L-GO-RETURN", f"a guard that does not return a declared error: {self.describe(cons[0])}")
            return None
        cond = self.expr(s.child_by_field_name("condition"), recv, fields)
        if cond is None:
            return None
        return (cond, err_vars[self.text(vals[0])])


def lift(src):
    form, diags, _ = lift_rows(src)
    return form, diags


def lift_rows(src):
    """(Form text, diagnostics, ids of the table rows used)."""
    tree = P.parse(src)
    if tree.root_node.has_error:
        return None, None, None
    L = Lift(src)
    regions, errors, events, steps = L.run(tree.root_node)
    if L.diags:
        return None, sorted(set(L.diags)), None
    lines = []
    for r, fields in regions.items():
        lines.append(f"state {r}")
        for f, t, z in fields: lines.append(f"    {f}: {t} = {z}")
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
    return "\n".join(lines) + "\n", [], sorted(L.used)


def report(path, src, rce=None, hand=None):
    """(the lift report, None), or (None, diagnostics) when the lift does not
    apply; (None, None) when src is not Go."""
    form, diags, used = lift_rows(src)
    if form is None:
        return None, diags
    out = (form + "\n").encode()  # the bytes `lift FILE` prints
    base = {"source": authority.sha256_bytes(src), "table": authority.sha256_file(TABLE_PATH),
            "lifter": authority.sha256_file(os.path.abspath(__file__)), "form": authority.sha256_bytes(out)}
    rce_d = authority.sha256_file(rce) if rce else None
    bindings = {
        "GO-ROWS": dict(base, checker=base["lifter"]),
        "GO-AGREE": dict(base, checker=None, recording=None),
        "GO-EQUIV": dict(base, checker=rce_d, counterpart=authority.sha256_file(hand) if hand else None),
    }
    records = []
    bare = [r for r in used if not ROWS[r]["clause"].strip() or not ROWS[r]["derivation"].strip()]
    detail = f"{len(used)} rows of table.json: {', '.join(used)}"
    if bare:
        detail += f"; without a clause or a derivation: {', '.join(bare)}"
    records.append(authority.record("GO-ROWS", "lift_go.py rows", "derived", "derived",
                                    "differs" if bare else "holds", bindings["GO-ROWS"], detail))
    if rce and hand:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "lifted.rce")
            with open(f, "wb") as g:
                g.write(out)
            p = subprocess.run([rce, "equiv", "json", f, hand], capture_output=True)
        try:
            e = json.loads(p.stdout)
        except ValueError:
            e = None
        if e is None or e.get("format") != "rce-equiv/1":
            res, why = "undecided", f"rce equiv exited {p.returncode} with no rce-equiv/1 report"
        elif e["equivalent"] and e["exact"]:
            res, why = "holds", f"equivalent, exact: {e['states'][0]} and {e['states'][1]} states over {len(e['alphabet'])} event instances"
        elif e["equivalent"] or e["undecided"]:
            res, why = "undecided", "an exploration reached its bound: equivalent to the explored bound only" if e["equivalent"] else "undecided"
        else:
            w = e.get("witness")
            res, why = "differs", ("after " + ", ".join(w["after"]) + f": {w['a']} / {w['b']}") if w else "different event instances explored"
        records.append(authority.record("GO-EQUIV", "rce equiv", "bisimilar", "derived", res,
                                        {k: v for k, v in bindings["GO-EQUIV"].items() if v is not None}, why))
    return authority.report("go", path, bindings, records), None


def main(argv):
    if len(argv) in (3, 5) and argv[1] == "report":
        path = argv[2]
        with open(path, "rb") as f:
            src = f.read()
        rep, diags = report(path, src, *(argv[3:5] if len(argv) == 5 else ()))
        if rep is None and diags is None:
            print(f"{path}: does not parse as Go", file=sys.stderr)
            return 3
        if rep is None:
            for _, line, rule, reason in diags:
                print(f"requires change {path}: line {line}: {rule}: {reason}")
            return 1
        sys.stdout.write(authority.dumps(rep))
        return 4 if rep["status"] == "failed" else 0
    if len(argv) == 2 and argv[1] == "rules":
        for k, v in RULES.items():
            print(f"{k}: {v}")
        return 0
    if len(argv) != 3 or argv[1] != "lift":
        print("\n\n".join(__doc__.strip().split("\n\n")[1:3]), file=sys.stderr)
        return 2
    path = argv[2]
    with open(path, "rb") as f:
        src = f.read()
    form, diags = lift(src)
    if form is None and diags is None:
        print(f"{path}: does not parse as Go", file=sys.stderr)
        return 3
    if diags:
        for _, line, rule, reason in diags:
            print(f"requires change {path}: line {line}: {rule}: {reason}")
        return 1
    print(form)
    print("# lift sha256:", hashlib.sha256(form.encode()).hexdigest()[:16], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
