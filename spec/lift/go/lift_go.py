#!/usr/bin/env python3
"""Lift probe: Go CST (tree-sitter) + a semantics table -> an RCE Form skeleton.

Usage (positional arguments only):

    lift_go.py lift FILE    the lifted Form on standard output (exit 0), or one
                            line per construct the probe does not carry on
                            standard output, `requires change FILE: line N: RULE: reason`
                            (exit 1); exit 3 when FILE does not parse as Go
    lift_go.py rules        the rule ids and what each covers

Deterministic: no model, no heuristics beyond the table; every decision is a
CST node kind looked up in SEM or RULES. The probe reads Go and runs
nothing. Scope: a struct with a pointer-receiver method per event, guards as
`if <condition> { return Err }` before any write, field assignments as
updates, and `errors.New` variables as errors. Every other construct is
reported with its rule and line, never dropped: a construct inside one that
is not carried (a loop inside a function, a call inside a loop) is reported
too, so the report lists everything the Form would be missing."""
import sys, hashlib, re
import tree_sitter, tree_sitter_go
GO = tree_sitter.Language(tree_sitter_go.language())
P = tree_sitter.Parser(GO)

# the semantics table: Go spec facts the lift needs (Go spec §Numeric types, §Arithmetic)
SEM = {
  "types": {"bool": "bool", "int64": "i64", "int": "i64", "uint64": "u64", "string": "text", "float64": "f64"},
  "zero": {"bool": "false", "i64": "0", "u64": "0", "text": '""', "f64": "0.0"},
  "int_overflow": "wraps (Go spec: 'integer overflow ... wraps around'); RCE traps: emit must insert checked ops",
  "division": "truncates toward zero, remainder takes dividend sign (Go spec) = RCE 6.3",
  "error_return": "a function returning a non-nil error = an event whose step keeps the state (T8: state unchanged only if no write preceded the return)",
  # Go operator -> RCE operator, for the operators whose meaning is the same in both
  "ops": {"&&": "and", "||": "or", "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="},
}

RULES = {
  "L-GO-IMPORT": "an import other than \"errors\": the standard library and other packages are not lifted",
  "L-GO-FUNC": "a function declaration or function literal: only pointer-receiver methods of a lifted struct are events",
  "L-GO-METHOD": "a method outside the event shape: a receiver that is not a pointer to a lifted struct, parameters, results other than error, or a second method of one name",
  "L-GO-TYPE": "a type declaration that is not a struct, an embedded field, or a field type outside the semantics table",
  "L-GO-DECL": "a const or var declaration other than `var ErrX = errors.New(\"Name\")`",
  "L-GO-FOR": "a for loop (loops and their count bounds are not lifted)",
  "L-GO-SWITCH": "a switch, type switch or select statement",
  "L-GO-IF": "an if statement other than `if <condition> { return ErrX }` with no init statement and no else",
  "L-GO-ORDER": "a guard after a field write: Go keeps the write when the method returns an error, an RCE fail keeps the state unchanged",
  "L-GO-RETURN": "a return other than `return ErrX` of a declared error in a guard or `return nil` as the last statement",
  "L-GO-ASSIGN": "an assignment other than `recv.field = <expression>` to a field of the receiver",
  "L-GO-CALL": "a call (library and user functions are not lifted)",
  "L-GO-ASSERT": "a type assertion",
  "L-GO-COLLECTION": "an array, slice, map or channel type, a composite literal, an index or a slice expression",
  "L-GO-POINTER": "a pointer type, an address-of or a dereference outside the receiver",
  "L-GO-INT": "integer arithmetic: Go's integer overflow wraps, RCE's traps",
  "L-GO-NAME": "a Go name that is an RCE keyword, or an error message that is not an RCE name",
  "L-GO-EXPR": "another expression form outside the table (named in the reason)",
  "L-GO-STMT": "another statement form outside the scope (named in the reason)",
}

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
            return k
        if k == "int_literal" and re.fullmatch(r"[0-9]+", self.text(n)) and (self.text(n) == "0" or not self.text(n).startswith("0")):
            return self.text(n)
        if k == "interpreted_string_literal" and "\\" not in self.text(n):
            return self.text(n)
        if k == "selector_expression":
            op, f = n.child_by_field_name("operand"), n.child_by_field_name("field")
            if op.type == "identifier" and self.text(op) == recv and self.text(f) in fields:
                return self.text(f)
            return self.bad(n, "L-GO-EXPR", f"a selector that is not a field of the receiver: {self.describe(n)}")
        if k == "parenthesized_expression":
            inner = self.expr(n.named_children[0], recv, fields)
            return None if inner is None else f"({inner})"
        if k == "unary_expression":
            op = self.text(n.child_by_field_name("operator"))
            if op == "!":
                x = self.expr(n.child_by_field_name("operand"), recv, fields)
                return None if x is None else f"not {x}"
            if op in ("&", "*"):
                return self.bad(n, "L-GO-POINTER", self.describe(n))
            return self.bad(n, "L-GO-INT", self.describe(n))
        if k == "binary_expression":
            op = self.text(n.child_by_field_name("operator"))
            if op not in SEM["ops"]:
                rule = "L-GO-INT" if op in ("+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "&^") else "L-GO-EXPR"
                return self.bad(n, rule, self.describe(n))
            a = self.expr(n.child_by_field_name("left"), recv, fields)
            b = self.expr(n.child_by_field_name("right"), recv, fields)
            if a is None or b is None:
                return None
            return f"{a} {SEM['ops'][op]} {b}"
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
                        rt = SEM["types"].get(self.text(ftype))
                        if rt is None:
                            self.drop(fd, "L-GO-TYPE", f"field type `{self.text(ftype)}` is outside the semantics table")
                            continue
                        for nm in names:
                            self.name(nm)
                            fields.append((self.text(nm), rt))
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
        fields = {f for f, _ in regions[rtype]}
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
                    writes.append((self.text(l.child_by_field_name("field")), v))
            elif s.type == "return_statement":
                if not (last and self.text(s) == "return nil"):
                    self.drop(s, "L-GO-RETURN", self.describe(s))
            elif s.type in NESTED:
                self.drop(s, NESTED[s.type], self.describe(s))
            else:
                self.drop(s, "L-GO-STMT", STMT_NAMES.get(s.type, s.type.replace("_", " ")) + f": {self.describe(s)}")
        if not stmts or stmts[-1].type != "return_statement":
            self.report(n, "L-GO-RETURN", f"method {name} does not end with `return nil`")
        if len(self.diags) != before:
            return None
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
    tree = P.parse(src)
    if tree.root_node.has_error:
        return None, None
    L = Lift(src)
    regions, errors, events, steps = L.run(tree.root_node)
    if L.diags:
        return None, sorted(set(L.diags))
    lines = []
    for r, fields in regions.items():
        lines.append(f"state {r}")
        for f, t in fields: lines.append(f"    {f}: {t} = {SEM['zero'][t]}")
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
    return "\n".join(lines) + "\n", []


def main(argv):
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
