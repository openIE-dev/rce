#!/usr/bin/env python3
"""Lift a defined subset of Python 3.13 into RCE text (route T of
spec/proposals/rce-form.md section 16).

Usage (positional arguments only):

    lift_py.py lift FILE         the lifted RCE program on standard output (exit 0),
                                 or one line per construct outside the subset on
                                 standard output, `requires change FILE: line N: RULE: reason`
                                 (exit 1); exit 3 when FILE is not Python
    lift_py.py functions FILE    one line per function of FILE: `lifted NAME` with its
                                 RCE text checked by the caller, or the requires-change
                                 lines of that function (function-level coverage)
    lift_py.py rules             the rule ids and what each covers

The stages are those of section 16: read (UTF-8; a PEP 263 coding line that
names another codec requires change), parse (CPython's own `ast`, the front
end of the pinned oracle's release), resolve (Python's scope rule: a name
bound in a function is local to it), types (one static type per name and
expression, found by unification; a name whose values have more than one
type requires change), bounds (rule B4: every loop gets a count bound from
a closed list of rules, or requires change) and printing (RCE text that
calls spec/lift/python/runtime.rce). Nothing is approximated: a construct
the lift cannot carry exactly is reported with its rule and line, and
never dropped. The lift reads no clock, environment or network, and its
output is a function of the source bytes.
"""
import ast
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(HERE, "runtime.rce")

RULES = {
    "L-PY-ENCODING": "the source is not UTF-8, or a PEP 263 coding line names another codec",
    "L-PY-IMPORT": "an import: the lift reads one module, and the standard library is not lifted",
    "L-PY-CLASS": "a class, or an attribute of an object (classes, methods and the object heap are not lifted)",
    "L-PY-EXC": "try, raise or assert: exceptions (transform B2) are not lifted",
    "L-PY-WITH": "a with statement (context managers)",
    "L-PY-DYN": "dynamic code or reflection: eval, exec, getattr, setattr, type, isinstance, id, hash, decorators, dunder names",
    "L-PY-CLOSURE": "a lambda, a nested function or nonlocal: closures (transform B3) are not lifted",
    "L-PY-GLOBAL": "a global statement, or a function that reads a module-level name that is rebound or mutated",
    "L-PY-GEN": "yield or a generator used as a value (step splitting, transform B5, is not lifted)",
    "L-PY-ASYNC": "async and await",
    "L-PY-MATCH": "a match statement",
    "L-PY-FLOAT": "a float or complex value: true division, a float literal, float()",
    "L-PY-BYTES": "bytes and bytearray values",
    "L-PY-SET": "a set: its iteration order is the hash table's, a choice the lift does not carry",
    "L-PY-NONE": "None used as a value (optional values)",
    "L-PY-INT": "an integer literal outside i64 (integers are lifted to i64, whose overflow traps)",
    "L-PY-TEXT": "a string literal with a lone surrogate, which text (UTF-8) cannot hold",
    "L-PY-TYPE": "a name or expression whose values do not have one static type",
    "L-PY-POLY": "a function whose parameter types no call site of the module fixes",
    "L-PY-UNBOUNDED": "a while loop whose count bound no rule of the closed list derives",
    "L-PY-ALIAS": "a list or dict shared by two names, or iterated, while one of them is mutated in place",
    "L-PY-CALL": "a call, method or argument form outside the runtime table (named in the reason)",
    "L-PY-FORMAT": "%-formatting, str.format, or an f-string conversion or format spec outside the table",
    "L-PY-SYNTAX": "another statement or expression form outside the subset (named in the reason)",
    "L-PY-CALLEE": "a function that calls a function which requires change",
}

KW = {"and", "or", "not", "in", "if", "then", "else", "elif", "for", "while", "loop", "fn", "let", "var",
      "return", "break", "continue", "match", "emit", "fail", "true", "false", "none", "state", "event",
      "error", "step", "on", "kind", "struct", "type", "trait", "impl", "use", "module", "const", "pub",
      "check", "example", "intent", "pass", "bound", "self", "spawn", "select", "with", "device", "yield",
      "comptime", "try", "catch", "as", "actor", "extern", "linear", "forall", "recv", "send", "after", "default"}


class Diag(Exception):
    def __init__(self, line, rule, reason):
        super().__init__(reason)
        self.line, self.rule, self.reason = line, rule, reason


# --- types -----------------------------------------------------------------

class TV:
    n = 0

    def __init__(self, origin="", line=0):
        TV.n += 1
        self.id = TV.n
        self.ref = None
        self.origin = origin
        self.line = line


class TC:
    def __init__(self, name, args=()):
        self.name, self.args = name, list(args)


INT, BOOL, STR, NONE = TC("int"), TC("bool"), TC("str"), TC("none")


def L(t): return TC("list", [t])
def D(k, v): return TC("dict", [k, v])
def T(ts): return TC("tuple", ts)


def prune(t):
    while isinstance(t, TV) and t.ref is not None:
        t = t.ref
    return t


def occurs(v, t):
    t = prune(t)
    if t is v:
        return True
    return isinstance(t, TC) and any(occurs(v, a) for a in t.args)


def unify(a, b):
    a, b = prune(a), prune(b)
    if a is b:
        return True
    if isinstance(a, TV):
        if occurs(a, b):
            return False
        a.ref = b
        return True
    if isinstance(b, TV):
        return unify(b, a)
    if a.name != b.name or len(a.args) != len(b.args):
        return False
    return all(unify(x, y) for x, y in zip(a.args, b.args))


def show(t):
    t = prune(t)
    if isinstance(t, TV):
        return "?"
    if t.name == "list":
        return f"list[{show(t.args[0])}]"
    if t.name == "dict":
        return f"dict[{show(t.args[0])}, {show(t.args[1])}]"
    if t.name == "tuple":
        return "tuple[" + ", ".join(show(a) for a in t.args) + "]"
    return t.name


def resolved(t):
    t = prune(t)
    return isinstance(t, TC) and all(resolved(a) for a in t.args)


def free_vars(t, out):
    t = prune(t)
    if isinstance(t, TV):
        out.append(t)
    else:
        for a in t.args:
            free_vars(a, out)


def is_mut(t):
    t = prune(t)
    return isinstance(t, TC) and (t.name in ("list", "dict") or (t.name == "tuple" and any(is_mut(a) for a in t.args)))


def scalar(t):
    t = prune(t)
    return isinstance(t, TC) and t.name in ("int", "bool", "none")


def rt(t):
    t = prune(t)
    if isinstance(t, TV):
        return "i64"   # only reached for a type nothing observes (see default_unobserved)
    n = t.name
    if n == "int":
        return "i64"
    if n == "bool":
        return "bool"
    if n == "str":
        return "text"
    if n == "list":
        return f"List[{rt(t.args[0])}]"
    if n == "dict":
        return f"(List[{rt(t.args[0])}], List[{rt(t.args[1])}])"
    if n == "tuple":
        if len(t.args) == 1:
            return f"({rt(t.args[0])},)"
        return "(" + ", ".join(rt(a) for a in t.args) + ")"
    raise AssertionError(n)


def mangle(t):
    t = prune(t)
    n = t.name if isinstance(t, TC) else "int"
    if n == "int":
        return "i"
    if n == "bool":
        return "b"
    if n == "str":
        return "s"
    if n == "list":
        return "L" + mangle(t.args[0])
    if n == "dict":
        return "D" + mangle(t.args[0]) + mangle(t.args[1])
    if n == "tuple":
        return f"T{len(t.args)}" + "".join(mangle(a) for a in t.args)
    return "u"


def default(t):
    t = prune(t)
    n = t.name if isinstance(t, TC) else "int"
    if n == "int":
        return "0"
    if n == "bool":
        return "false"
    if n == "str":
        return '""'
    if n == "list":
        return "[]"
    if n == "dict":
        return "([], [])"
    if n == "tuple":
        if len(t.args) == 1:
            return f"({default(t.args[0])},)"
        return "(" + ", ".join(default(a) for a in t.args) + ")"
    raise AssertionError(n)


def ordered(t):
    t = prune(t)
    return isinstance(t, TC) and t.name in ("int", "bool", "str", "list", "tuple") and all(ordered(a) for a in t.args)


# --- scopes ----------------------------------------------------------------

def assigned_names(body):
    """Names a block binds (Python's rule: a name bound anywhere in a
    function is local to all of it), not entering nested scopes."""
    out = []

    def tgt(t):
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                tgt(e)
        elif isinstance(t, ast.Starred):
            tgt(t.value)

    def walk(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                out.append(n.name)
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(n, ast.Assign):
            for t in n.targets:
                tgt(t)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            tgt(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            tgt(n.target)
        elif isinstance(n, ast.NamedExpr):
            tgt(n.target)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.append((a.asname or a.name).split(".")[0])
        for c in ast.iter_child_nodes(n):
            walk(c)

    for s in body:
        walk(s)
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def root_name(e):
    while isinstance(e, ast.Subscript):
        e = e.value
    return e.id if isinstance(e, ast.Name) else None


MUT_LIST = {"append", "extend", "insert", "pop", "remove", "sort", "reverse", "clear"}
MUT_DICT = {"pop", "setdefault", "update", "clear", "popitem"}


def mutations(body):
    """(root name, line) of every in-place mutation in a block: a mutating
    method call, an item assignment or deletion, or += on a list."""
    out = []
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (MUT_LIST | MUT_DICT):
            r = root_name(n.func.value)
            if r:
                out.append((r, n.lineno))
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete)):
            ts = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in ts:
                for x in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                    if isinstance(x, ast.Subscript):
                        r = root_name(x)
                        if r:
                            out.append((r, n.lineno))
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                out.append((n.target.id + "\0aug", n.lineno))
    return out


# --- the lifter ------------------------------------------------------------

class Fn:
    def __init__(self, node, name):
        self.node, self.name = node, name
        self.params = [a.arg for a in node.args.args] if node else []
        self.ptypes = [TV("param", node.lineno) for _ in self.params] if node else []
        self.ret = TV("return", node.lineno if node else 0)
        self.locals = {}
        self.mut_params = set()
        self.calls = set()
        self.diags = []
        self.uses_globals = set()
        self.has_value_return = False


class Lifter:
    def __init__(self, path, src_bytes):
        self.path = path
        self.src = src_bytes
        self.types = {}          # id(node) -> type
        self.pending = []        # (kind, node, types...) resolved by fixpoint
        self.fns = {}
        self.cur = None
        self.consts = {}         # module name -> (node, type)
        self.loop_bounds = {}    # id(While) -> rce bound text builder info
        self.mod_assign_count = {}

    # diagnostics
    def diag(self, node, rule, reason):
        line = getattr(node, "lineno", 0)
        d = (line, rule, reason)
        if d not in self.cur.diags:
            self.cur.diags.append(d)

    def need(self, ok, node, rule, reason):
        if not ok:
            self.diag(node, rule, reason)
        return ok

    def u(self, a, b, node, what):
        if not unify(a, b):
            self.diag(node, "L-PY-TYPE", f"{what}: {show(a)} and {show(b)} are different types")
            return False
        return True

    # --- resolve and infer -------------------------------------------------

    def run(self):
        try:
            text = self.src.decode("utf-8")
        except UnicodeDecodeError as e:
            return None, [(1, "L-PY-ENCODING", f"not UTF-8 at byte {e.start}")]
        m = re.match(rb"^(?:[^\n]*\n)?[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", self.src)
        if m and m.group(1).decode().lower().replace("_", "-") not in ("utf-8", "utf8"):
            return None, [(1, "L-PY-ENCODING", f"coding line names {m.group(1).decode()}")]
        try:
            self.tree = ast.parse(text, self.path)
        except SyntaxError as e:
            raise SystemExit(f"not Python: {self.path}: line {e.lineno}: {e.msg}") from None
        mod = Fn(None, "<module>")
        self.mod = mod
        self.fns["<module>"] = mod
        body = self.tree.body
        for s in body:
            if isinstance(s, ast.FunctionDef):
                f = Fn(s, s.name)
                self.fns[s.name] = f
        self.mod_names = assigned_names(body)
        self.imported = set()
        for s in ast.walk(self.tree):
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    self.imported.add((a.asname or a.name).split(".")[0])
        for n in self.mod_names:
            if n not in self.fns:
                mod.locals[n] = TV("name " + n)
        self.module_mutations = mutations(body)
        for s in body:
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            self.mod_assign_count[x.id] = self.mod_assign_count.get(x.id, 0) + 1
            elif isinstance(s, (ast.AugAssign, ast.AnnAssign, ast.For)):
                for x in ast.walk(s.target):
                    if isinstance(x, ast.Name):
                        self.mod_assign_count[x.id] = self.mod_assign_count.get(x.id, 0) + 2
            for x in ast.walk(s):
                if x is not s and isinstance(x, (ast.Assign, ast.AugAssign, ast.For, ast.AnnAssign)) and not isinstance(s, ast.FunctionDef):
                    for t in (x.targets if isinstance(x, ast.Assign) else [x.target]):
                        for y in ast.walk(t):
                            if isinstance(y, ast.Name):
                                self.mod_assign_count[y.id] = self.mod_assign_count.get(y.id, 0) + 2
        for name, f in self.fns.items():
            if f.node is not None:
                self.setup_fn(f)
        # infer: functions first, then the module's statements
        for name, f in self.fns.items():
            if f.node is not None:
                self.cur = f
                self.block(f.node.body)
        self.cur = mod
        for s in body:
            if isinstance(s, ast.FunctionDef):
                continue
            self.stmt(s)
        self.solve_pending()
        self.finish()
        return text, None

    def setup_fn(self, f):
        self.cur = f
        n = f.node
        a = n.args
        if n.decorator_list:
            self.diag(n, "L-PY-DYN", "a decorator")
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            self.diag(n, "L-PY-CALL", "*args, **kwargs, keyword-only or positional-only parameters")
        if a.defaults:
            self.diag(n, "L-PY-CALL", "default parameter values")
        if n.returns is not None or any(x.annotation is not None for x in a.args):
            pass  # annotations do not change what Python runs
        for x in ast.walk(n):
            if x is not n and isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                self.diag(x, "L-PY-CLOSURE", "a nested function" if not isinstance(x, ast.Lambda) else "a lambda")
            if isinstance(x, ast.Nonlocal):
                self.diag(x, "L-PY-CLOSURE", "nonlocal")
            if isinstance(x, ast.Global):
                self.diag(x, "L-PY-GLOBAL", "a global statement")
            if isinstance(x, (ast.Yield, ast.YieldFrom)):
                f.ret.origin = "error"   # a generator: its yield is reported
        loc = assigned_names(n.body)
        for p, t in zip(f.params, f.ptypes):
            f.locals[p] = t
        for x in a.kwonlyargs + a.posonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
            f.locals.setdefault(x.arg, TV("param", n.lineno))
        for x in loc:
            if x not in f.locals:
                f.locals[x] = TV("name " + x, n.lineno)
        muts = mutations(n.body)
        f.mut_params = {r for r, _ in muts if r in f.params}
        f.muts = muts
        f.rebound = set(loc)

    def lookup(self, node):
        name = node.id
        f = self.cur
        if name in self.imported and name not in (f.locals if f is not self.mod else {}):
            return TV("import")
        if name in f.locals:
            return f.locals[name]
        if f is not self.mod and name in self.mod.locals:
            f.uses_globals.add(name)
            return self.mod.locals[name]
        if name in self.fns and self.fns[name].node is not None:
            self.diag(node, "L-PY-CLOSURE", f"the function {name} used as a value")
            return TV()
        if name in ("True", "False", "None"):
            return TV()
        self.diag(node, "L-PY-SYNTAX", f"the name {name} is not bound in the module (a builtin used as a value)")
        return TV()

    def solve_pending(self):
        changed = True
        while changed:
            changed = False
            rest = []
            for p in self.pending:
                if self.try_pending(p):
                    changed = True
                else:
                    rest.append(p)
            self.pending = rest
        for p in self.pending:
            kind, node = p[0], p[1]
            self.cur = p[-1]
            if any(isinstance(prune(x), TV) and prune(x).origin in ("import", "error", "param") for x in p[2:5]):
                continue  # the import is reported
            self.diag(node, "L-PY-TYPE", f"the operand types of {kind} are not fixed by the program")

    def try_pending(self, p):
        kind, node, a, b, r, f = p
        A, B = prune(a), prune(b)
        self.cur = f
        if kind == "+":
            for X in (A, B):
                if isinstance(X, TC):
                    if X.name in ("int", "str", "list"):
                        self.u(a, b, node, "the operands of +")
                        self.u(r, a, node, "the result of +")
                    else:
                        self.diag(node, "L-PY-TYPE", f"+ on {show(X)}")
                    return True
            return False
        if kind == "*":
            if isinstance(A, TC) and isinstance(B, TC):
                if A.name == "int" and B.name == "int":
                    self.u(r, INT, node, "the result of *")
                elif A.name in ("str", "list") and B.name == "int":
                    self.u(r, a, node, "the result of *")
                elif A.name == "int" and B.name in ("str", "list"):
                    self.u(r, b, node, "the result of *")
                else:
                    self.diag(node, "L-PY-TYPE", f"* on {show(A)} and {show(B)}")
                return True
            if isinstance(A, TC) and A.name in ("str", "list"):
                self.u(b, INT, node, "the count of *")
                self.u(r, a, node, "the result of *")
                return True
            if isinstance(B, TC) and B.name in ("str", "list"):
                self.u(a, INT, node, "the count of *")
                self.u(r, b, node, "the result of *")
                return True
            if isinstance(A, TC) and A.name == "int" and isinstance(B, TC):
                return False
            return False
        if kind == "in":
            if isinstance(B, TC):
                if B.name == "str":
                    self.u(a, STR, node, "the left operand of in")
                elif B.name == "list":
                    self.u(a, B.args[0], node, "the left operand of in")
                elif B.name == "dict":
                    self.u(a, B.args[0], node, "the left operand of in")
                else:
                    self.diag(node, "L-PY-TYPE", f"in on {show(B)}")
                return True
            return False
        if kind == "iter":
            if isinstance(A, TC):
                if A.name == "str":
                    self.u(b, STR, node, "the item of a str")
                elif A.name == "list":
                    self.u(b, A.args[0], node, "the item of a list")
                elif A.name == "dict":
                    self.u(b, A.args[0], node, "the key of a dict")
                elif A.name == "tuple" and A.args:
                    for x in A.args:
                        self.u(b, x, node, "the items of an iterated tuple")
                else:
                    self.diag(node, "L-PY-TYPE", f"iteration over {show(A)}")
                return True
            return False
        if kind == "index":
            if isinstance(A, TC):
                if A.name == "str":
                    self.u(b, INT, node, "a str index")
                    self.u(r, STR, node, "a str item")
                elif A.name == "list":
                    self.u(b, INT, node, "a list index")
                    self.u(r, A.args[0], node, "a list item")
                elif A.name == "dict":
                    self.u(b, A.args[0], node, "a dict key")
                    self.u(r, A.args[1], node, "a dict value")
                elif A.name == "tuple":
                    idx = node.slice
                    if isinstance(idx, ast.Constant) and isinstance(idx.value, int) and not isinstance(idx.value, bool) and -len(A.args) <= idx.value < len(A.args):
                        self.u(r, A.args[idx.value], node, "a tuple item")
                    else:
                        self.diag(node, "L-PY-TYPE", "a tuple index that is not a constant in range")
                else:
                    self.diag(node, "L-PY-TYPE", f"indexing {show(A)}")
                return True
            return False
        if kind == "len":
            if isinstance(A, TC):
                if A.name not in ("str", "list", "dict", "tuple"):
                    self.diag(node, "L-PY-TYPE", f"len of {show(A)}")
                return True
            return False
        if kind == "slice":
            if isinstance(A, TC):
                if A.name not in ("str", "list"):
                    self.diag(node, "L-PY-TYPE", f"a slice of {show(A)}")
                return True
            return False
        raise AssertionError(kind)

    def pend(self, kind, node, a, b=None, r=None):
        p = (kind, node, a, b if b is not None else TV(), r if r is not None else TV(), self.cur)
        if not self.try_pending(p):
            self.pending.append(p)

    # statements

    def block(self, body):
        for s in body:
            self.stmt(s)

    def stmt(self, s):
        f = self.cur
        k = type(s).__name__
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                return  # a docstring or a bare string: no effect
            self.expr(s.value)
        elif isinstance(s, ast.Assign):
            if len(s.targets) > 1 and not all(isinstance(t, ast.Name) for t in s.targets):
                self.diag(s, "L-PY-SYNTAX", "a chained assignment to items")
            t = self.expr(s.value)
            for tg in s.targets:
                self.target(tg, t, s)
        elif isinstance(s, ast.AnnAssign):
            if s.value is None:
                return
            t = self.expr(s.value)
            self.target(s.target, t, s)
        elif isinstance(s, ast.AugAssign):
            if not isinstance(s.target, (ast.Name, ast.Subscript)):
                self.diag(s, "L-PY-SYNTAX", "an augmented assignment to this target")
                return
            lt = self.expr(s.target)
            rt_ = self.expr(s.value)
            self.binop(s, s.op, lt, rt_, lt)
        elif isinstance(s, ast.If):
            self.cond(s.test)
            self.block(s.body)
            self.block(s.orelse)
        elif isinstance(s, ast.While):
            self.cond(s.test)
            self.block(s.body)
            self.block(s.orelse)
            self.while_bound(s)
        elif isinstance(s, ast.For):
            et = self.iterable(s.iter, s)
            self.target(s.target, et, s)
            self.block(s.body)
            self.block(s.orelse)
            self.for_alias(s)
        elif isinstance(s, (ast.Break, ast.Continue, ast.Pass)):
            pass
        elif isinstance(s, ast.Return):
            if f is self.mod:
                self.diag(s, "L-PY-SYNTAX", "return outside a function")
                return
            if s.value is None:
                self.u(f.ret, NONE, s, "the result of the function")
            else:
                if isinstance(s.value, ast.Constant) and s.value.value is None:
                    self.u(f.ret, NONE, s, "the result of the function")
                    return
                f.has_value_return = True
                t = self.expr(s.value)
                self.u(f.ret, t, s, "the results of the function")
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                if isinstance(t, ast.Subscript) and not isinstance(t.slice, ast.Slice):
                    self.expr(t)
                else:
                    self.diag(s, "L-PY-SYNTAX", "del of a name or a slice")
        elif isinstance(s, ast.FunctionDef):
            if f is not self.mod:
                return  # reported by setup_fn
        elif isinstance(s, ast.ClassDef):
            self.diag(s, "L-PY-CLASS", f"class {s.name}")
        elif isinstance(s, (ast.Try, ast.TryStar)):
            self.diag(s, "L-PY-EXC", "try")
        elif isinstance(s, ast.Raise):
            self.diag(s, "L-PY-EXC", "raise")
        elif isinstance(s, ast.Assert):
            self.diag(s, "L-PY-EXC", "assert")
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            self.diag(s, "L-PY-WITH", "with")
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            names = ", ".join(a.name for a in s.names)
            mod = getattr(s, "module", None)
            self.diag(s, "L-PY-IMPORT", f"import of {mod + ': ' if mod else ''}{names}")
        elif isinstance(s, ast.Global):
            self.diag(s, "L-PY-GLOBAL", "a global statement")
        elif isinstance(s, ast.Nonlocal):
            self.diag(s, "L-PY-CLOSURE", "nonlocal")
        elif isinstance(s, ast.Match):
            self.diag(s, "L-PY-MATCH", "match")
        elif isinstance(s, (ast.AsyncFor, ast.AsyncFunctionDef)):
            self.diag(s, "L-PY-ASYNC", k)
        else:
            self.diag(s, "L-PY-SYNTAX", f"the statement {k}")

    def target(self, tg, t, s):
        if isinstance(tg, ast.Name):
            v = self.lookup(tg)
            self.u(v, t, tg, f"the values assigned to {tg.id}")
            self.types[id(tg)] = v
        elif isinstance(tg, (ast.Tuple, ast.List)):
            if any(isinstance(e, ast.Starred) for e in tg.elts):
                self.diag(tg, "L-PY-SYNTAX", "a starred assignment target")
                return
            ts = [TV() for _ in tg.elts]
            if not unify(t, T(ts)):
                P = prune(t)
                if isinstance(P, TC) and P.name == "list":
                    self.diag(tg, "L-PY-TYPE", "unpacking a list (its length is not a static fact)")
                else:
                    self.diag(tg, "L-PY-TYPE", f"unpacking {show(t)} into {len(ts)} names")
                return
            for e, et in zip(tg.elts, ts):
                self.target(e, et, s)
        elif isinstance(tg, ast.Subscript):
            if isinstance(tg.slice, ast.Slice):
                self.diag(tg, "L-PY-SYNTAX", "assignment to a slice")
                return
            it = self.expr(tg)
            self.u(it, t, tg, "the item assigned")
            B = prune(self.types.get(id(tg.value)))
            if isinstance(B, TC) and B.name == "str":
                self.diag(tg, "L-PY-TYPE", "item assignment to a str")
        elif isinstance(tg, ast.Attribute):
            self.diag(tg, "L-PY-CLASS", f"assignment to the attribute {tg.attr}")
        else:
            self.diag(tg, "L-PY-SYNTAX", "this assignment target")

    def cond(self, e):
        """A test: Python takes the truth value of any operand; and/or
        here only decide."""
        if isinstance(e, ast.BoolOp):
            for v in e.values:
                self.cond(v)
            self.types[id(e)] = BOOL
            self.types[("cond", id(e))] = True
            return BOOL
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            self.cond(e.operand)
            self.types[id(e)] = BOOL
            self.types[("cond", id(e))] = True
            return BOOL
        t = self.expr(e)
        self.truthable(t, e)
        return t

    def truthable(self, t, e):
        P = prune(t)
        if isinstance(P, TC) and P.name not in ("int", "bool", "str", "list", "dict"):
            self.diag(e, "L-PY-TYPE", f"the truth value of {show(P)}")

    def iterable(self, it, s):
        """The type of the items an iteration yields; records the form."""
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("range", "enumerate", "zip", "reversed", "sorted") and it.func.id not in self.cur.locals and it.func.id not in self.fns:
            fn = it.func.id
            if it.keywords and not (fn == "enumerate" and [k.arg for k in it.keywords] == ["start"]) and not (fn == "sorted" and [k.arg for k in it.keywords] == ["reverse"]):
                self.diag(it, "L-PY-CALL", f"{fn} with keyword arguments")
                return TV()
            if fn == "range":
                if not 1 <= len(it.args) <= 3:
                    self.diag(it, "L-PY-CALL", "range with this number of arguments")
                for a in it.args:
                    self.u(self.expr(a), INT, a, "a range bound")
                if len(it.args) == 3 and not self.const_int(it.args[2]):
                    self.diag(it, "L-PY-CALL", "range with a step that is not a constant")
                self.types[id(it)] = L(INT)
                return INT
            if fn == "enumerate":
                et = self.iterable(it.args[0], s) if it.args else TV()
                for k in it.keywords:
                    self.u(self.expr(k.value), INT, k.value, "the start of enumerate")
                if len(it.args) == 2:
                    self.u(self.expr(it.args[1]), INT, it.args[1], "the start of enumerate")
                return T([INT, et])
            if fn == "zip":
                if len(it.args) != 2:
                    self.diag(it, "L-PY-CALL", "zip of other than two iterables")
                    return TV()
                return T([self.iterable(it.args[0], s), self.iterable(it.args[1], s)])
            if fn in ("reversed", "sorted"):
                if len(it.args) != 1:
                    self.diag(it, "L-PY-CALL", f"{fn} with this number of arguments")
                    return TV()
                et = self.iterable(it.args[0], s)
                if fn == "sorted" and not ordered(et) and resolved(et):
                    self.diag(it, "L-PY-TYPE", f"sorting {show(et)}")
                for k in it.keywords:
                    self.need(isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool), k.value, "L-PY-CALL", "reverse that is not a constant")
                return et
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("items", "keys", "values") and not it.args:
            dt = self.expr(it.func.value)
            k, v = TV(), TV()
            self.u(dt, D(k, v), it, f"the receiver of .{it.func.attr}()")
            self.types[id(it)] = dt
            return T([k, v]) if it.func.attr == "items" else (k if it.func.attr == "keys" else v)
        t = self.expr(it)
        et = TV()
        self.pend("iter", it, t, et)
        return et

    def const_int(self, e):
        if isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool):
            return e.value
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant) and isinstance(e.operand.value, int):
            return -e.operand.value
        return None

    # expressions

    def expr(self, e):
        n0 = len(self.cur.diags)
        t = self._expr(e)
        P = prune(t)
        if len(self.cur.diags) > n0 and isinstance(P, TV) and not P.origin:
            P.origin = "error"   # its construct is reported; what follows from it is not reported again
        self.types[id(e)] = t
        return t

    def _expr(self, e):
        f = self.cur
        if isinstance(e, ast.Constant):
            v = e.value
            if isinstance(v, bool):
                return BOOL
            if isinstance(v, int):
                if not -2 ** 63 <= v < 2 ** 63:
                    self.diag(e, "L-PY-INT", f"the literal {v} is outside i64")
                return INT
            if isinstance(v, str):
                if any(0xD800 <= ord(c) <= 0xDFFF for c in v):
                    self.diag(e, "L-PY-TEXT", "a lone surrogate in a string literal")
                return STR
            if v is None:
                self.diag(e, "L-PY-NONE", "None as a value")
                return TV()
            if isinstance(v, (float, complex)):
                self.diag(e, "L-PY-FLOAT", f"the {type(v).__name__} literal {v!r}")
                return TV()
            if isinstance(v, bytes):
                self.diag(e, "L-PY-BYTES", "a bytes literal")
                return TV()
            self.diag(e, "L-PY-SYNTAX", f"the constant {v!r}")
            return TV()
        if isinstance(e, ast.Name):
            if e.id in ("True", "False"):
                return BOOL
            return self.lookup(e)
        if isinstance(e, ast.BinOp):
            a = self.expr(e.left)
            b = self.expr(e.right)
            r = TV()
            if isinstance(e.op, ast.Mod) and isinstance(e.left, (ast.Constant, ast.JoinedStr)) and isinstance(getattr(e.left, "value", None), str):
                self.diag(e, "L-PY-FORMAT", "%-formatting")
                return STR
            self.binop(e, e.op, a, b, r)
            return r
        if isinstance(e, ast.UnaryOp):
            if isinstance(e.op, ast.Not):
                self.cond(e.operand)
                return BOOL
            t = self.expr(e.operand)
            self.u(t, INT, e, f"the operand of unary {type(e.op).__name__}")
            return INT
        if isinstance(e, ast.BoolOp):
            ts = [self.expr(v) for v in e.values]
            for v, t in zip(e.values, ts):
                if not unify(t, BOOL):
                    self.diag(v, "L-PY-TYPE", f"{'and' if isinstance(e.op, ast.And) else 'or'} gives one of its operands, here of type {show(t)}, as a value")
            return BOOL
        if isinstance(e, ast.Compare):
            prev = self.expr(e.left)
            ops = list(zip(e.ops, e.comparators))
            if len(ops) > 1:
                for c in e.comparators[:-1]:
                    if any(isinstance(x, ast.Call) for x in ast.walk(c)):
                        self.diag(e, "L-PY-SYNTAX", "a chained comparison whose middle operand calls a function")
            left = e.left
            for op, c in ops:
                ct = self.expr(c)
                if isinstance(op, (ast.In, ast.NotIn)):
                    self.pend("in", e, prev, ct)
                elif isinstance(op, (ast.Is, ast.IsNot)):
                    self.diag(e, "L-PY-NONE" if (isinstance(c, ast.Constant) and c.value is None) else "L-PY-DYN", "is / is not (object identity)")
                elif isinstance(op, (ast.Eq, ast.NotEq)):
                    self.u(prev, ct, e, "the operands of ==")
                else:
                    self.u(prev, ct, e, "the operands of an order comparison")
                    if resolved(prev) and not ordered(prev):
                        self.diag(e, "L-PY-TYPE", f"ordering {show(prev)}")
                prev = ct
            return BOOL
        if isinstance(e, ast.IfExp):
            self.cond(e.test)
            a = self.expr(e.body)
            b = self.expr(e.orelse)
            self.u(a, b, e, "the branches of a conditional expression")
            return a
        if isinstance(e, ast.Call):
            return self.call(e)
        if isinstance(e, ast.Subscript):
            bt = self.expr(e.value)
            if isinstance(e.slice, ast.Slice):
                sl = e.slice
                for x in (sl.lower, sl.upper, sl.step):
                    if x is not None:
                        self.u(self.expr(x), INT, x, "a slice bound")
                if sl.step is not None and self.const_int(sl.step) is None:
                    self.diag(e, "L-PY-SYNTAX", "a slice step that is not a constant")
                self.pend("slice", e, bt)
                return bt
            it = self.expr(e.slice)
            r = TV()
            self.pend("index", e, bt, it, r)
            return r
        if isinstance(e, ast.List):
            et = TV("empty list", e.lineno)
            for x in e.elts:
                if isinstance(x, ast.Starred):
                    self.diag(x, "L-PY-SYNTAX", "a starred item in a list display")
                    continue
                self.u(et, self.expr(x), x, "the items of a list")
            return L(et)
        if isinstance(e, ast.Tuple):
            ts = []
            for x in e.elts:
                if isinstance(x, ast.Starred):
                    self.diag(x, "L-PY-SYNTAX", "a starred item in a tuple display")
                    ts.append(TV())
                    continue
                ts.append(self.expr(x))
            if not ts:
                self.diag(e, "L-PY-TYPE", "the empty tuple")
            return T(ts)
        if isinstance(e, ast.Dict):
            k, v = TV("empty dict", e.lineno), TV("empty dict", e.lineno)
            for kk, vv in zip(e.keys, e.values):
                if kk is None:
                    self.diag(e, "L-PY-SYNTAX", "** in a dict display")
                    continue
                self.u(k, self.expr(kk), kk, "the keys of a dict")
                self.u(v, self.expr(vv), vv, "the values of a dict")
            return D(k, v)
        if isinstance(e, ast.ListComp):
            return L(self.comp(e))
        if isinstance(e, ast.GeneratorExp):
            self.diag(e, "L-PY-GEN", "a generator expression used as a value")
            return TV()
        if isinstance(e, (ast.Set, ast.SetComp)):
            self.diag(e, "L-PY-SET", "a set")
            return TV()
        if isinstance(e, ast.DictComp):
            self.diag(e, "L-PY-SYNTAX", "a dict comprehension")
            return TV()
        if isinstance(e, ast.JoinedStr):
            for v in e.values:
                if isinstance(v, ast.FormattedValue):
                    self.fvalue(v)
            return STR
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id in self.imported:
            return TV("import")
        if isinstance(e, ast.Attribute):
            if e.attr in self.STR_METHODS | self.LIST_METHODS | self.DICT_METHODS:
                self.diag(e, "L-PY-CLOSURE", f"the method {e.attr} used as a value")
            elif e.attr.startswith("__"):
                self.diag(e, "L-PY-DYN", f"the attribute {e.attr}")
            else:
                self.diag(e, "L-PY-CLASS", f"the attribute {e.attr}")
            return TV()
        if isinstance(e, ast.Lambda):
            self.diag(e, "L-PY-CLOSURE", "a lambda")
            return TV()
        if isinstance(e, (ast.Yield, ast.YieldFrom)):
            self.diag(e, "L-PY-GEN", "yield")
            return TV()
        if isinstance(e, ast.Await):
            self.diag(e, "L-PY-ASYNC", "await")
            return TV()
        if isinstance(e, ast.NamedExpr):
            self.diag(e, "L-PY-SYNTAX", "an assignment expression (:=)")
            return TV()
        if isinstance(e, ast.Starred):
            self.diag(e, "L-PY-SYNTAX", "a starred expression")
            return TV()
        self.diag(e, "L-PY-SYNTAX", f"the expression {type(e).__name__}")
        return TV()

    def binop(self, node, op, a, b, r):
        if isinstance(op, ast.Add):
            self.pend("+", node, a, b, r)
        elif isinstance(op, ast.Mult):
            self.pend("*", node, a, b, r)
        elif isinstance(op, ast.Div):
            self.diag(node, "L-PY-FLOAT", "true division / gives a float")
        elif isinstance(op, (ast.Sub, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            nm = type(op).__name__
            self.u(a, INT, node, f"the left operand of {nm}")
            self.u(b, INT, node, f"the right operand of {nm}")
            self.u(r, INT, node, f"the result of {nm}")
            if isinstance(op, ast.Pow):
                k = self.const_int(node.right) if isinstance(node, ast.BinOp) else None
                if k is not None and k < 0:
                    self.diag(node, "L-PY-FLOAT", "a negative power gives a float")
        else:
            self.diag(node, "L-PY-SYNTAX", f"the operator {type(op).__name__}")

    def fvalue(self, v):
        t = self.expr(v.value)
        if v.conversion not in (-1, ord("r"), ord("s")):
            self.diag(v, "L-PY-FORMAT", f"the conversion !{chr(v.conversion)}")
        spec = ""
        if v.format_spec is not None:
            parts = v.format_spec.values
            if len(parts) == 1 and isinstance(parts[0], ast.Constant):
                spec = parts[0].value
            elif parts:
                self.diag(v, "L-PY-FORMAT", "a format spec computed at run time")
                return
        self.types[("spec", id(v))] = spec

    def comp(self, e):
        if len(e.generators) != 1:
            self.diag(e, "L-PY-SYNTAX", "a comprehension with more than one for clause")
            return TV()
        g = e.generators[0]
        if g.is_async:
            self.diag(e, "L-PY-ASYNC", "async comprehension")
        saved = dict(self.cur.locals)
        names = []
        for x in ast.walk(g.target):
            if isinstance(x, ast.Name):
                names.append(x.id)
        et = self.iterable(g.iter, e)
        for n in names:
            self.cur.locals[n] = TV("comprehension " + n)
        self.target(g.target, et, e)
        for c in g.ifs:
            self.cond(c)
        t = self.expr(e.elt)
        self.types[("compvars", id(e))] = {n: self.cur.locals[n] for n in names}
        self.cur.locals.clear()
        self.cur.locals.update(saved)
        return t

    def genarg(self, a):
        """A generator expression passed to a consumer is the list it yields;
        so is reversed(x), whose iterator the consumer runs to its end."""
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "reversed" and len(a.args) == 1 and not a.keywords and "reversed" not in self.cur.locals:
            et = self.iterable(a.args[0], a)
            t = L(et)
            self.types[id(a)] = t
            self.types[("reversed", id(a))] = True
            return t
        if isinstance(a, ast.GeneratorExp):
            if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in self.fns for x in ast.walk(a)):
                self.diag(a, "L-PY-GEN", "a generator whose items call a function (the lift evaluates it eagerly)")
            t = L(self.comp(a))
            self.types[id(a)] = t
            return t
        return self.expr(a)

    BUILTINS = {"print", "len", "str", "repr", "int", "abs", "min", "max", "sum", "sorted", "list", "reversed",
                "enumerate", "zip", "ord", "chr", "divmod", "pow", "any", "all", "range", "bool", "dict"}
    DYN = {"eval", "exec", "getattr", "setattr", "hasattr", "delattr", "type", "isinstance", "issubclass", "id",
           "hash", "globals", "locals", "vars", "__import__", "compile", "callable", "super", "object"}

    def call(self, e):
        f = self.cur
        fn = e.func
        if isinstance(fn, ast.Name) and fn.id in self.imported and fn.id not in self.fns:
            for a in e.args:
                self.expr(a)
            return TV("import")
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in self.imported:
            for a in e.args:
                self.expr(a)
            return TV("import")
        if isinstance(fn, ast.Name):
            name = fn.id
            if name in self.fns and self.fns[name].node is not None and name not in f.locals:
                return self.user_call(e, self.fns[name])
            if name in f.locals or (name in self.mod.locals and name not in self.BUILTINS):
                self.diag(e, "L-PY-CLOSURE", f"a call of the value {name}")
                return TV()
            if name in self.DYN:
                self.diag(e, "L-PY-DYN", f"{name}()")
                return TV()
            if name in ("float", "complex"):
                self.diag(e, "L-PY-FLOAT", f"{name}()")
                return TV()
            if name in ("bytes", "bytearray", "memoryview"):
                self.diag(e, "L-PY-BYTES", f"{name}()")
                return TV()
            if name in ("set", "frozenset"):
                self.diag(e, "L-PY-SET", f"{name}()")
                return TV()
            if name in ("open", "input"):
                self.diag(e, "L-PY-IMPORT", f"{name}(): input and output other than print")
                return TV()
            if name in ("iter", "next", "map", "filter", "tuple", "format", "round", "hex", "oct", "bin", "ascii", "slice"):
                self.diag(e, "L-PY-CALL", f"{name}()")
                return TV()
            if name not in self.BUILTINS:
                self.diag(e, "L-PY-CALL", f"{name}()")
                return TV()
            return self.builtin(e, name)
        if isinstance(fn, ast.Attribute):
            if isinstance(fn.value, ast.Constant) and isinstance(fn.value.value, str) and fn.attr == "format":
                self.diag(e, "L-PY-FORMAT", "str.format")
                return STR
            if isinstance(fn.value, ast.Name) and fn.value.id not in f.locals and fn.value.id not in self.mod.locals:
                self.diag(e, "L-PY-IMPORT" if fn.value.id not in ("str", "list", "dict", "int") else "L-PY-CALL",
                          f"{fn.value.id}.{fn.attr}()")
                return TV()
            return self.method(e, fn.value, fn.attr)
        self.expr(fn)
        self.diag(e, "L-PY-CLOSURE", "a call of a computed value")
        return TV()

    def user_call(self, e, g):
        f = self.cur
        f.calls.add(g.name)
        args = list(e.args)
        if any(isinstance(a, ast.Starred) for a in args):
            self.diag(e, "L-PY-CALL", f"a starred argument to {g.name}")
            return g.ret
        if e.keywords:
            byname = {k.arg: k.value for k in e.keywords}
            if None in byname or any(k not in g.params[len(args):] for k in byname):
                self.diag(e, "L-PY-CALL", f"these keyword arguments to {g.name}")
                return g.ret
            args = args + [byname[p] for p in g.params[len(args):] if p in byname]
            self.types[("args", id(e))] = args
        if len(args) != len(g.params):
            self.diag(e, "L-PY-CALL", f"{g.name} takes {len(g.params)} arguments, given {len(args)}")
            return g.ret
        for a, pt in zip(args, g.ptypes):
            self.u(pt, self.expr(a), a, f"an argument of {g.name}")
        return g.ret

    def builtin(self, e, name):
        args = e.args
        kws = {k.arg: k.value for k in e.keywords}
        if name != "print" and any(isinstance(a, ast.Starred) for a in args):
            self.diag(e, "L-PY-CALL", f"a starred argument to {name}")
            return TV()
        if name == "print":
            for k in kws:
                if k not in ("sep", "end"):
                    self.diag(e, "L-PY-IMPORT" if k == "file" else "L-PY-CALL", f"print(..., {k}=)")
                else:
                    self.u(self.expr(kws[k]), STR, kws[k], f"print's {k}")
            for a in args:
                if isinstance(a, ast.Starred):
                    t = self.expr(a.value)
                    self.types[id(a)] = t
                    P = prune(t)
                    if isinstance(P, TC) and P.name == "tuple":
                        continue
                    et = TV()
                    self.u(t, L(et), a, "a starred argument of print")
                else:
                    self.printable(self.expr(a), a)
            return NONE
        if kws and not (name in ("sorted",) and list(kws) == ["reverse"]) and not (name == "enumerate" and list(kws) == ["start"]):
            self.diag(e, "L-PY-CALL", f"{name} with keyword arguments {', '.join(map(str, kws))}")
            return TV()
        n = len(args)
        if name == "len":
            if n != 1:
                return self.bad(e, name)
            t = self.expr(args[0])
            self.pend("len", e, t)
            return INT
        if name in ("str", "repr"):
            if n != 1:
                return self.bad(e, name)
            self.printable(self.expr(args[0]), args[0])
            return STR
        if name == "int":
            if n != 1:
                return self.bad(e, name)
            t = self.expr(args[0])
            P = prune(t)
            if isinstance(P, TC) and P.name not in ("int", "str"):
                self.diag(e, "L-PY-TYPE", f"int() of {show(P)}")
            elif isinstance(P, TV):
                self.u(t, STR, e, "the argument of int()")
            return INT
        if name == "bool":
            if n != 1:
                return self.bad(e, name)
            self.cond(args[0])
            return BOOL
        if name in ("abs", "ord", "chr"):
            if n != 1:
                return self.bad(e, name)
            t = self.expr(args[0])
            self.u(t, STR if name == "ord" else INT, e, f"the argument of {name}")
            return STR if name == "chr" else INT
        if name in ("divmod", "pow"):
            if not (n == 2 or (name == "pow" and n == 3)):
                return self.bad(e, name)
            for a in args:
                self.u(self.expr(a), INT, a, f"an argument of {name}")
            if name == "pow" and n == 2:
                k = self.const_int(args[1])
                if k is not None and k < 0:
                    self.diag(e, "L-PY-FLOAT", "a negative power gives a float")
            return T([INT, INT]) if name == "divmod" else INT
        if name in ("min", "max"):
            if n == 0:
                return self.bad(e, name)
            if n == 1:
                t = self.genarg(args[0])
                et = TV()
                if not unify(t, L(et)):
                    P = prune(t)
                    if isinstance(P, TC) and P.name == "str":
                        et = STR
                    else:
                        self.diag(e, "L-PY-TYPE", f"{name} of {show(t)}")
                        return TV()
                if resolved(et) and not ordered(et):
                    self.diag(e, "L-PY-TYPE", f"{name} of {show(et)}")
                return et
            t = self.expr(args[0])
            for a in args[1:]:
                self.u(t, self.expr(a), a, f"the arguments of {name}")
            return t
        if name == "sum":
            if n != 1:
                return self.bad(e, name)
            self.u(self.genarg(args[0]), L(INT), e, "the argument of sum")
            return INT
        if name in ("any", "all"):
            if n != 1:
                return self.bad(e, name)
            t = self.genarg(args[0])
            self.u(t, L(BOOL), e, f"the argument of {name} (a list of bool)")
            return BOOL
        if name in ("sorted", "list", "reversed"):
            if n != 1:
                return self.bad(e, name)
            for k, v in kws.items():
                self.need(isinstance(v, ast.Constant) and isinstance(v.value, bool), v, "L-PY-CALL", "reverse that is not a constant")
            if isinstance(args[0], ast.GeneratorExp):
                t = self.genarg(args[0])
                et = prune(t).args[0] if isinstance(prune(t), TC) else TV()
            else:
                et = self.iterable(args[0], e)
            if name == "sorted" and resolved(et) and not ordered(et):
                self.diag(e, "L-PY-TYPE", f"sorting {show(et)}")
            if name == "reversed":
                self.diag(e, "L-PY-GEN", "reversed() used as a value (an iterator)")
            return L(et)
        if name in ("enumerate", "zip"):
            self.diag(e, "L-PY-GEN", f"{name}() used as a value (an iterator)")
            return TV()
        if name == "range":
            self.diag(e, "L-PY-CALL", "range() used as a value")
            return TV()
        if name == "dict":
            if n == 0 and not kws:
                return D(TV("empty dict"), TV("empty dict"))
            return self.bad(e, name)
        return self.bad(e, name)

    def bad(self, e, name):
        self.diag(e, "L-PY-CALL", f"{name} with these arguments")
        return TV()

    def printable(self, t, node):
        P = prune(t)
        if isinstance(P, TC) and P.name == "none":
            self.diag(node, "L-PY-NONE", "None as a value (the result of a function that returns nothing)")

    STR_METHODS = {"join", "upper", "lower", "strip", "split", "startswith", "endswith", "find", "index", "count", "replace",
                   "isdigit", "isalpha", "ljust", "rjust", "center"}
    LIST_METHODS = {"append", "extend", "insert", "pop", "index", "count", "remove", "reverse", "sort", "copy", "clear"}
    DICT_METHODS = {"get", "pop", "keys", "values", "items"}

    def method(self, e, recv, m):
        args = e.args
        n = len(args)
        if e.keywords and not (m == "sort" and [k.arg for k in e.keywords] == ["reverse"]):
            self.diag(e, "L-PY-CALL", f".{m}() with keyword arguments")
            return TV()
        rtp = self.expr(recv)
        P = prune(rtp)
        if isinstance(P, TV):
            # the receiver's type is fixed by the method when only one type has it
            if m in self.STR_METHODS - self.LIST_METHODS - self.DICT_METHODS:
                unify(rtp, STR)
            elif m in self.LIST_METHODS - self.STR_METHODS - self.DICT_METHODS:
                unify(rtp, L(TV("list", e.lineno)))
            elif m in self.DICT_METHODS - self.LIST_METHODS - self.STR_METHODS:
                unify(rtp, D(TV(), TV()))
            P = prune(rtp)
        if isinstance(P, TV) and P.origin in ("import", "error"):
            return TV(P.origin)
        if isinstance(P, TV):
            self.diag(e, "L-PY-TYPE", f"the receiver of .{m}() has no fixed type")
            return TV()
        kind = P.name
        self.types[("recv", id(e))] = kind
        if kind == "int":
            if m == "bit_length" and n == 0:
                return INT
            self.diag(e, "L-PY-CALL", f"int.{m}()")
            return TV()
        ats = [self.genarg(a) if (kind == "str" and m == "join") else self.expr(a) for a in args]
        if kind == "str":
            if m not in self.STR_METHODS:
                self.diag(e, "L-PY-CALL", f"str.{m}()")
                return TV()
            if m == "join":
                if n != 1:
                    return self.bad(e, "str.join")
                self.u(ats[0], L(STR), e, "the argument of str.join")
                return STR
            if m in ("upper", "lower", "isdigit", "isalpha"):
                if n:
                    return self.bad(e, "str." + m)
                return BOOL if m.startswith("is") else STR
            if m == "strip":
                if n:
                    return self.bad(e, "str.strip with characters")
                return STR
            if m == "split":
                if n > 1:
                    return self.bad(e, "str.split with maxsplit")
                if n:
                    self.u(ats[0], STR, e, "the separator of split")
                return L(STR)
            if m in ("startswith", "endswith", "find", "index", "count"):
                if n != 1:
                    return self.bad(e, "str." + m)
                self.u(ats[0], STR, e, f"the argument of str.{m}")
                return BOOL if m.endswith("with") else INT
            if m == "replace":
                if n != 2:
                    return self.bad(e, "str.replace")
                for a in ats:
                    self.u(a, STR, e, "an argument of str.replace")
                return STR
            if m in ("ljust", "rjust", "center"):
                if n not in (1, 2):
                    return self.bad(e, "str." + m)
                self.u(ats[0], INT, e, f"the width of str.{m}")
                if n == 2:
                    self.u(ats[1], STR, e, f"the fill of str.{m}")
                return STR
        if kind == "list":
            et = P.args[0]
            if m not in self.LIST_METHODS:
                self.diag(e, "L-PY-CALL", f"list.{m}()")
                return TV()
            if m == "append":
                if n != 1:
                    return self.bad(e, "list.append")
                self.u(et, ats[0], e, "the items of a list")
                return NONE
            if m == "extend":
                if n != 1:
                    return self.bad(e, "list.extend")
                self.u(P, ats[0], e, "the argument of list.extend (a list)")
                return NONE
            if m == "insert":
                if n != 2:
                    return self.bad(e, "list.insert")
                self.u(ats[0], INT, e, "the index of list.insert")
                self.u(et, ats[1], e, "the items of a list")
                return NONE
            if m == "pop":
                if n > 1:
                    return self.bad(e, "list.pop")
                if n:
                    self.u(ats[0], INT, e, "the index of list.pop")
                return et
            if m in ("index", "count", "remove"):
                if n != 1:
                    return self.bad(e, "list." + m)
                self.u(et, ats[0], e, f"the argument of list.{m}")
                return INT if m != "remove" else NONE
            if m in ("reverse", "clear"):
                return NONE if n == 0 else self.bad(e, "list." + m)
            if m == "sort":
                if n:
                    return self.bad(e, "list.sort with a key")
                for k in e.keywords:
                    self.need(isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool), k.value, "L-PY-CALL", "reverse that is not a constant")
                return NONE
            if m == "copy":
                return P if n == 0 else self.bad(e, "list.copy")
        if kind == "dict":
            kt, vt = P.args
            if m not in self.DICT_METHODS:
                self.diag(e, "L-PY-CALL", f"dict.{m}()")
                return TV()
            if m == "get":
                if n == 1:
                    self.diag(e, "L-PY-NONE", "dict.get without a default gives None for a missing key")
                    return vt
                if n != 2:
                    return self.bad(e, "dict.get")
                self.u(kt, ats[0], e, "the key of dict.get")
                self.u(vt, ats[1], e, "the default of dict.get")
                return vt
            if m == "pop":
                if n != 1:
                    return self.bad(e, "dict.pop with a default")
                self.u(kt, ats[0], e, "the key of dict.pop")
                return vt
            self.diag(e, "L-PY-GEN", f"dict.{m}() used as a value (a view)")
            return TV()
        self.diag(e, "L-PY-CALL", f"{show(P)}.{m}()")
        return TV()

    # --- the checks that need the whole scope -------------------------------

    def loop_body_lines(self, loop):
        return {n.lineno for n in ast.walk(loop) if hasattr(n, "lineno")}

    def for_alias(self, s):
        """Python iterates over the live list; the lift iterates over a
        snapshot. They agree unless the body mutates what is iterated."""
        it = s.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("items", "keys", "values"):
            it = it.func.value
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("enumerate", "reversed", "zip"):
            its = it.args
        else:
            its = [it]
        body_muts = mutations(s.body)
        for x in its:
            r = root_name(x)
            if r and any(m == r for m, _ in body_muts):
                self.diag(s, "L-PY-ALIAS", f"the loop mutates {r}, which it iterates over")

    def while_bound(self, s):
        """Rule B4 for a while loop: the test is a conjunction one of whose
        terms compares a variable v with a value the body does not change,
        and every path through the body that runs again moves the
        variables of those terms toward their limits by a positive
        constant, changing them in no other way. The count bound is then
        the sum of the distances, evaluated once before the loop."""
        terms = []
        t = s.test
        conj = t.values if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And) else [t]
        body_names = set(assigned_names(s.body))
        body_muts = {r for r, _ in mutations(s.body)}
        for c in conj:
            if isinstance(c, ast.Compare) and len(c.ops) == 1 and isinstance(c.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                l, op, rgt = c.left, c.ops[0], c.comparators[0]
                less = isinstance(op, (ast.Lt, ast.LtE))
                # v < lim and lim > v: v rises; v > lim and lim < v: v falls
                for v, lim, rising in ((l, rgt, less), (rgt, l, not less)):
                    if not isinstance(v, ast.Name):
                        continue
                    lim_names = {x.id for x in ast.walk(lim) if isinstance(x, ast.Name)}
                    if lim_names & (body_names | body_muts):
                        continue
                    if any(isinstance(x, ast.Call) and not (isinstance(x.func, ast.Name) and x.func.id == "len") for x in ast.walk(lim)):
                        continue
                    terms.append((v.id, lim, rising, isinstance(op, (ast.LtE, ast.GtE))))
                    break
        names = [x[0] for x in terms]
        if not terms or len(set(names)) != len(names):
            self.diag(s, "L-PY-UNBOUNDED", "the test compares no variable with a limit the loop does not change")
            return
        dirs = {x[0]: x[2] for x in terms}

        def step(st):
            # the constant change of a measured variable, or None when it changes otherwise
            if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id in dirs:
                k = self.const_int(st.value)
                if k is None or k <= 0 or not isinstance(st.op, (ast.Add, ast.Sub)):
                    return None
                return st.target.id, (k if isinstance(st.op, ast.Add) == dirs[st.target.id] else -k)
            return False

        bad = []

        def walk(stmts, acc):
            """acc: the minimum progress so far on this path; returns the
            minimum progress over the paths that reach the end of stmts or a
            continue, None when a path makes none."""
            res = []
            cur = acc
            for st in stmts:
                r = step(st)
                if r is None:
                    bad.append(st)
                    return []
                if r:
                    if r[1] < 0:
                        bad.append(st)
                        return []
                    cur = cur + r[1]
                    continue
                if isinstance(st, ast.If):
                    a = walk(st.body, cur)
                    b = walk(st.orelse, cur)
                    ends = a + b
                    if not ends:
                        return res
                    if any(x[1] == "continue" for x in ends):
                        res += [x for x in ends if x[1] == "continue"]
                    falls = [x[0] for x in ends if x[1] == "end"]
                    if not falls:
                        return res
                    cur = min(falls)
                    continue
                if isinstance(st, ast.Continue):
                    res.append((cur, "continue"))
                    return res
                if isinstance(st, (ast.Break, ast.Return)):
                    return res
                for x in ast.walk(st):
                    if isinstance(x, (ast.Assign, ast.AugAssign, ast.For, ast.NamedExpr)):
                        tg = x.targets if isinstance(x, ast.Assign) else [x.target]
                        for y in tg:
                            for z in ast.walk(y):
                                if isinstance(z, ast.Name) and z.id in dirs:
                                    bad.append(st)
                                    return []
            res.append((cur, "end"))
            return res

        ends = walk(s.body, 0)
        if bad:
            self.diag(s, "L-PY-UNBOUNDED", f"the body changes {', '.join(sorted(dirs))} other than by a constant step toward the limit")
            return
        if any(x[0] < 1 for x in ends):
            self.diag(s, "L-PY-UNBOUNDED", f"a path through the body does not move {', '.join(sorted(dirs))} toward the limit")
            return
        self.loop_bounds[id(s)] = terms

    def finish(self):
        # functions that take a list or dict they mutate take it by &mut, transitively
        changed = True
        while changed:
            changed = False
            for f in self.fns.values():
                if f.node is None:
                    continue
                for x in ast.walk(f.node):
                    if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in self.fns and x.func.id not in f.locals:
                        g = self.fns[x.func.id]
                        args = self.types.get(("args", id(x)), x.args)
                        for p, a in zip(g.params, args):
                            if p in g.mut_params:
                                r = root_name(a)
                                if r in f.params and r not in f.mut_params:
                                    f.mut_params.add(r)
                                    changed = True
        for f in self.fns.values():
            self.cur = f
            body = f.node.body if f.node else [s for s in self.tree.body if not isinstance(s, ast.FunctionDef)]
            self.alias_check(f, body)
            if f.node is not None:
                if f.has_value_return and not always_returns(f.node.body):
                    if not unify(f.ret, NONE):
                        self.diag(f.node, "L-PY-NONE", f"{f.name} returns a value on some paths and None where a path ends without return")
                elif not f.has_value_return:
                    unify(f.ret, NONE)
                for p, t in zip(f.params, f.ptypes):
                    if not resolved(t):
                        self.diag(f.node, "L-PY-POLY", f"the type of the parameter {p} of {f.name} is not fixed by any call")
                    elif p in f.mut_params and p in assigned_in(f.node.body):
                        self.diag(f.node, "L-PY-ALIAS", f"{f.name} mutates the argument {p} and also rebinds it")
                if not resolved(f.ret):
                    self.diag(f.node, "L-PY-POLY", f"the result type of {f.name} is not fixed")
                for g in f.uses_globals:
                    if g in self.imported:
                        continue
                    if not self.const_ok(g):
                        self.diag(f.node, "L-PY-GLOBAL", f"{f.name} reads the module-level name {g}, which is rebound or mutated, or not a literal")
                    else:
                        self.consts[g] = True
        # a name whose type stays open only through an empty list or dict that is
        # never filled is observed by nothing that depends on the item type
        # ([] and {} print the same whatever it is), so it is given int
        for f in self.fns.values():
            for name, t in f.locals.items():
                if not resolved(t):
                    vs = []
                    free_vars(t, vs)
                    for v in vs:
                        if v.origin in ("empty list", "empty dict"):
                            unify(v, INT)

    def const_ok(self, g):
        if self.mod_assign_count.get(g, 0) != 1:
            return False
        if any(r == g for r, _ in self.module_mutations):
            return False
        for f in self.fns.values():
            if f.node is not None and any(r == g for r, _ in f.muts):
                return False
        for s in self.tree.body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and s.targets[0].id == g:
                return literal(s.value)
        return False

    def alias_check(self, f, body):
        """A list or dict read from a name into a second name, a list item
        or a result shares one object in Python and is a copy in RCE. The
        two agree unless the object is mutated in place after the sharing
        while both names can still see it."""
        muts = mutations(body)
        mut_lines = {}
        for r, ln in muts:
            mut_lines.setdefault(r, []).append(ln)
        loops = [n for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, (ast.For, ast.While))]

        def enclosing(line):
            return [lp for lp in loops if lp.lineno <= line <= lp.end_lineno]

        def check(src, dst, node):
            t = self.types.get(id(src))
            if t is None or not is_mut(t):
                return
            r = root_name(src)
            if r is None:
                return
            share = node.lineno
            if dst and dst in mut_lines:
                self.diag(node, "L-PY-ALIAS", f"{dst} shares its list with {r} and is mutated in place")
                return
            if f.node is not None and r in f.params:
                if dst is None or r in f.mut_params:
                    self.diag(node, "L-PY-ALIAS", f"the argument {r} is stored or returned while the caller can still see it")
                    return
            for ln in mut_lines.get(r, []):
                if ln > share:
                    self.diag(node, "L-PY-ALIAS", f"{r} is shared at line {share} and mutated in place after it")
                    return
                encl = [lp for lp in enclosing(share) if lp.lineno <= ln <= lp.end_lineno]
                if encl:
                    lp = encl[-1]
                    first = lp.body[0]
                    rebinds = (isinstance(first, ast.Assign) and len(first.targets) == 1 and isinstance(first.targets[0], ast.Name)
                               and first.targets[0].id == r and root_name(first.value) is None)
                    if not (rebinds and first.lineno < ln):
                        self.diag(node, "L-PY-ALIAS", f"{r} is shared at line {share} and mutated in place in the same loop")
                        return

        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, (ast.FunctionDef, ast.Lambda)) and n is not f.node:
                continue
            if isinstance(n, ast.Assign):
                for tg in n.targets:
                    if isinstance(tg, ast.Name):
                        check(n.value, tg.id, n)
                    elif isinstance(tg, ast.Subscript):
                        check(n.value, None, n)
                    elif isinstance(tg, ast.Tuple) and isinstance(n.value, ast.Tuple):
                        for a, b in zip(tg.elts, n.value.elts):
                            check(b, a.id if isinstance(a, ast.Name) else None, n)
            elif isinstance(n, ast.Return) and n.value is not None:
                check(n.value, None, n)
            elif isinstance(n, (ast.List, ast.Tuple)) and isinstance(getattr(n, "ctx", None), ast.Load):
                for x in n.elts:
                    check(x, None, n)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("append", "insert", "extend"):
                for a in n.args:
                    check(a, None, n)
            elif isinstance(n, ast.For):
                # a loop variable holds an item of what is iterated, shared in Python
                it = n.iter
                srcs = it.args if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("enumerate", "zip", "reversed", "sorted")) else [it]
                for x in ast.walk(n.target):
                    if isinstance(x, ast.Name):
                        t = self.types.get(id(x))
                        if t is not None and is_mut(t):
                            if x.id in mut_lines:
                                self.diag(n, "L-PY-ALIAS", f"the loop variable {x.id} is an item of the list and is mutated in place")
                            for src in srcs:
                                r = root_name(src)
                                if r and any(ln > n.end_lineno for ln in mut_lines.get(r, [])):
                                    self.diag(n, "L-PY-ALIAS", f"the loop variable {x.id} shares an item of {r}, which is mutated in place after the loop")


def literal(e):
    if isinstance(e, ast.Constant):
        return isinstance(e.value, (int, str)) and not isinstance(e.value, bool) or isinstance(e.value, bool)
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
        return literal(e.operand)
    if isinstance(e, (ast.List, ast.Tuple)):
        return all(literal(x) for x in e.elts)
    if isinstance(e, ast.Dict):
        return all(k is not None and literal(k) and literal(v) for k, v in zip(e.keys, e.values))
    return False


def always_returns(body):
    if not body:
        return False
    s = body[-1]
    if isinstance(s, ast.Return):
        return True
    if isinstance(s, ast.If):
        return always_returns(s.body) and always_returns(s.orelse)
    if isinstance(s, ast.While) and isinstance(s.test, ast.Constant) and s.test.value is True:
        return not any(isinstance(x, ast.Break) for x in ast.walk(s))
    return False


def reads(body):
    """The names a block reads (an augmented assignment reads its target)."""
    out = set()
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.AugAssign):
            for x in ast.walk(n.target):
                if isinstance(x, ast.Name):
                    out.add(x.id)
    return out


def assigned_in(body):
    out = set()
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.For)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    out.add(t.id)
                elif isinstance(t, ast.Tuple):
                    out |= {x.id for x in t.elts if isinstance(x, ast.Name)}
    return out


# --- printing RCE text ------------------------------------------------------

FMT_RE = re.compile(r"^(?:(?P<fill>.)?(?P<align>[<>=^]))?(?P<sign>[-+ ])?(?P<alt>#)?(?P<zero>0)?(?P<width>\d+)?(?P<group>[,_])?(?:\.(?P<prec>\d+))?(?P<type>[bcdeEfFgGnosxX%])?$", re.S)


def rstr(s):
    o = ['"']
    for c in s:
        cp = ord(c)
        if c == '"':
            o.append('\\"')
        elif c == "\\":
            o.append("\\\\")
        elif c == "\n":
            o.append("\\n")
        elif c == "\t":
            o.append("\\t")
        elif c == "\r":
            o.append("\\r")
        elif cp < 32 or cp == 127 or not c.isprintable() or unicodedata.category(c) in ("Mn", "Me", "Mc"):
            o.append("\\u{%x}" % cp)
        else:
            o.append(c)
    o.append('"')
    return "".join(o)


class Printer:
    def __init__(self, lifter):
        self.L = lifter
        self.helpers = {}        # name -> text (monomorphic, generated)
        self.uses = set()        # runtime definitions called
        self.tmp = 0
        self.lines = []
        self.np_table = False
        self.read = set()

    def fresh(self, base="t"):
        self.tmp += 1
        return f"{base}_{self.tmp}"

    def ty(self, e):
        return prune(self.L.types[id(e)])

    def use(self, name):
        self.uses.add(name)
        return name

    # names
    def vname(self, n):
        return ("v_" if n in self.read else "_v_") + n

    def name_ref(self, n):
        if n in self.alias:
            return self.alias[n]
        f = self.f
        if n in f.locals or f is self.L.mod and n in self.L.mod.locals:
            if f is self.L.mod and n in self.L.consts:
                return "G_" + n
            return self.vname(n)
        if n in self.L.consts:
            return "G_" + n
        return self.vname(n)

    # values: every read of a list, dict, text or tuple is a copy
    def val(self, e):
        code = self.ex(e)
        if self.is_place(e) and not scalar(self.ty(e)):
            return f"copy({code})"
        return code

    def is_place(self, e):
        return isinstance(e, ast.Name) or (isinstance(e, ast.Subscript) and not isinstance(e.slice, ast.Slice) and self.is_place(e.value)
                                            and prune(self.L.types[id(e.value)]).name in ("list", "dict", "tuple"))

    def place(self, e):
        """An RCE place for a Python name or item."""
        if isinstance(e, ast.Name):
            return self.name_ref(e.id)
        bt = prune(self.L.types[id(e.value)])
        b = self.place(e.value)
        if bt.name == "list":
            return f"{b}[{self.use('py_ix')}({self.val(e.slice)}, {b}.len())]"
        if bt.name == "dict":
            return f"{b}.1[{self.use('py_dix')}(&{b}, {self.val(e.slice)})]"
        if bt.name == "tuple":
            k = self.L.const_int(e.slice)
            return f"{b}.{k % len(bt.args)}"
        raise AssertionError(bt.name)

    def borrow(self, e):
        """&x for a place, or a temporary bound by the caller."""
        return "&" + self.place(e)

    # expressions
    def ex(self, e):
        m = getattr(self, "ex_" + type(e).__name__)
        return m(e)

    def ex_Constant(self, e):
        v = e.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v) if v >= 0 else f"(0 - {-v})" if v == -2 ** 63 else str(v)
        if isinstance(v, str):
            return rstr(v)
        raise AssertionError(v)

    def ex_Name(self, e):
        if e.id == "True":
            return "true"
        if e.id == "False":
            return "false"
        return self.name_ref(e.id)

    def ex_BinOp(self, e):
        t = self.ty(e)
        a, b = self.val(e.left), self.val(e.right)
        op = e.op
        if isinstance(op, ast.Add):
            if t.name == "list":
                return f"{self.use('py_cat')}({a}, {b})"
            return f"({a} + {b})"
        if isinstance(op, ast.Mult):
            lt = self.ty(e.left)
            if t.name == "int":
                return f"({a} * {b})"
            if t.name == "str":
                return f"{self.use('py_rep_s')}({a}, {b})" if lt.name == "str" else f"{self.use('py_rep_s')}({b}, {a})"
            return f"{self.use('py_rep_l')}({a}, {b})" if lt.name == "list" else f"{self.use('py_rep_l')}({b}, {a})"
        if isinstance(op, ast.Sub):
            return f"({a} - {b})"
        if isinstance(op, ast.FloorDiv):
            return f"{self.use('py_floordiv')}({a}, {b})"
        if isinstance(op, ast.Mod):
            return f"{self.use('py_mod')}({a}, {b})"
        if isinstance(op, ast.Pow):
            return f"{self.use('py_pow')}({a}, {b})"
        if isinstance(op, ast.LShift):
            return f"{self.use('py_shl')}({a}, {b})"
        if isinstance(op, ast.RShift):
            return f"{self.use('py_shr')}({a}, {b})"
        sym = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}[type(op)]
        return f"({a} {sym} {b})"

    def ex_UnaryOp(self, e):
        if isinstance(e.op, ast.Not):
            return f"(not {self.truth(e.operand)})"
        a = self.val(e.operand)
        if isinstance(e.op, ast.USub):
            return f"(0 - {a})"
        if isinstance(e.op, ast.UAdd):
            return a
        return f"(~{a})"

    def truth(self, e):
        """The truth value of e, as a bool."""
        if isinstance(e, ast.BoolOp):
            op = " and " if isinstance(e.op, ast.And) else " or "
            return "(" + op.join(self.truth(v) for v in e.values) + ")"
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            return f"(not {self.truth(e.operand)})"
        t = self.ty(e)
        c = self.ex(e) if self.is_place(e) else self.val(e)
        if t.name == "bool":
            return c
        if t.name == "int":
            return f"({c} != 0)"
        if t.name == "str":
            return f"({c}.len_bytes() > 0)"
        if t.name == "list":
            return f"({c}.len() > 0)"
        if t.name == "dict":
            return f"({c}.0.len() > 0)"
        raise AssertionError(t.name)

    def ex_BoolOp(self, e):
        return self.truth(e)

    def ex_Compare(self, e):
        parts = []
        left = e.left
        for op, c in zip(e.ops, e.comparators):
            parts.append(self.cmp1(left, op, c))
            left = c
        return parts[0] if len(parts) == 1 else "(" + " and ".join(parts) + ")"

    def cmp1(self, a, op, b):
        if isinstance(op, (ast.In, ast.NotIn)):
            bt = self.ty(b)
            if bt.name == "str":
                r = f"{self.val(b)}.contains({self.val(a)})"
            elif bt.name == "list":
                r = f"{self.val(b)}.contains({self.val(a)})"
            else:
                r = f"({self.use('py_dfind')}({self.borrow(b) if self.is_place(b) else self.tmpref(b)}, {self.val(a)}) >= 0)"
            return f"(not {r})" if isinstance(op, ast.NotIn) else r
        sym = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}[type(op)]
        return f"({self.val(a)} {sym} {self.val(b)})"

    def tmpref(self, e):
        raise Diag(e.lineno, "L-PY-SYNTAX", "a dict that is not a variable or an item, used with in")

    def ex_IfExp(self, e):
        return f"(if {self.truth(e.test)} then {self.val(e.body)} else {self.val(e.orelse)})"

    def ex_Subscript(self, e):
        bt = self.ty(e.value)
        if isinstance(e.slice, ast.Slice):
            sl = e.slice
            lo = self.val(sl.lower) if sl.lower is not None else "0"
            hi = self.val(sl.upper) if sl.upper is not None else "0"
            st = self.val(sl.step) if sl.step is not None else "1"
            hl = "true" if sl.lower is not None else "false"
            hh = "true" if sl.upper is not None else "false"
            if bt.name == "str":
                return f"{self.use('py_slice_s')}({self.val(e.value)}, {lo}, {hi}, {st}, {hl}, {hh})"
            if self.is_place(e.value):
                return f"{self.use('py_slice')}(&{self.place(e.value)}, {lo}, {hi}, {st}, {hl}, {hh})"
            raise Diag(e.lineno, "L-PY-SYNTAX", "a slice of a computed list")
        if bt.name == "str":
            return f"{self.use('py_str_at')}({self.val(e.value)}, {self.val(e.slice)})"
        if self.is_place(e.value):
            return self.place(e)
        raise Diag(e.lineno, "L-PY-SYNTAX", "an item of a computed value")

    def ex_List(self, e):
        return "[" + ", ".join(self.val(x) for x in e.elts) + "]"

    def ex_Tuple(self, e):
        if len(e.elts) == 1:
            return f"({self.val(e.elts[0])},)"
        return "(" + ", ".join(self.val(x) for x in e.elts) + ")"

    def ex_Dict(self, e):
        if not e.keys:
            return "([], [])"
        return "([" + ", ".join(self.val(k) for k in e.keys) + "], [" + ", ".join(self.val(v) for v in e.values) + "])"

    def ex_ListComp(self, e):
        return self.comp(e)

    def ex_GeneratorExp(self, e):
        return self.comp(e)

    def comp(self, e):
        g = e.generators[0]
        cv = self.L.types[("compvars", id(e))]
        saved = dict(self.alias)
        used = {x.id for part in [e.elt] + list(g.ifs) for x in ast.walk(part) if isinstance(x, ast.Name)}
        for n in cv:
            self.alias[n] = self.fresh("c" if n in used else "_c")
        src, et = self.iter_list(g.iter)
        pat = self.pattern(g.target)
        cond = ""
        if g.ifs:
            cond = " if " + " and ".join(self.truth(c) for c in g.ifs)
        body = self.val(e.elt)
        self.alias = saved
        return f"[{body} for {pat} in {src}{cond}]"

    def pattern(self, t):
        if isinstance(t, ast.Name):
            return self.alias.get(t.id, self.vname(t.id))
        return "(" + ", ".join(self.pattern(x) for x in t.elts) + ")"

    def iter_list(self, it):
        """RCE code for a list holding the items an iteration yields."""
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            a = it.args
            if len(a) == 1:
                return f"{self.use('py_range')}(0, {self.val(a[0])}, 1)", INT
            if len(a) == 2:
                return f"{self.use('py_range')}({self.val(a[0])}, {self.val(a[1])}, 1)", INT
            return f"{self.use('py_range')}({self.val(a[0])}, {self.val(a[1])}, {self.val(a[2])})", INT
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate":
            start = self.val(it.args[1]) if len(it.args) == 2 else (self.val(it.keywords[0].value) if it.keywords else "0")
            s, et = self.iter_list(it.args[0])
            return f"{self.use('py_enumerate')}({s}, {start})", T([INT, et])
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "zip":
            a, at = self.iter_list(it.args[0])
            b, bt = self.iter_list(it.args[1])
            return f"{self.use('py_zip')}({a}, {b})", T([at, bt])
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "reversed":
            s, et = self.iter_list(it.args[0])
            return f"{s}.reverse()", et
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "sorted":
            s, et = self.iter_list(it.args[0])
            rev = any(k.arg == "reverse" and k.value.value for k in it.keywords)
            return f"{self.use('py_sorted_rev' if rev else 'py_sorted')}({s})", et
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("items", "keys", "values") and not it.args:
            dt = self.ty(it.func.value)
            d = self.place(it.func.value) if self.is_place(it.func.value) else None
            if d is None:
                raise Diag(it.lineno, "L-PY-SYNTAX", "a view of a computed dict")
            if it.func.attr == "items":
                return f"{self.use('py_ditems')}(&{d})", T(dt.args)
            if it.func.attr == "keys":
                return f"copy({d}.0)", dt.args[0]
            return f"copy({d}.1)", dt.args[1]
        t = self.ty(it)
        if t.name == "str":
            return f"{self.val(it)}.chars()", STR
        if t.name == "list":
            return self.val(it), t.args[0]
        if t.name == "dict":
            if self.is_place(it):
                return f"copy({self.place(it)}.0)", t.args[0]
            return f"{self.val(it)}.0", t.args[0]
        if t.name == "tuple":
            if isinstance(it, ast.Tuple):
                return "[" + ", ".join(self.val(x) for x in it.elts) + "]", t.args[0]
            if self.is_place(it):
                p = self.place(it)
                return "[" + ", ".join(f"copy({p}.{i})" for i in range(len(t.args))) + "]", t.args[0]
            raise Diag(it.lineno, "L-PY-SYNTAX", "iteration over a computed tuple")
        raise AssertionError(t.name)

    def ex_JoinedStr(self, e):
        parts = []
        for v in e.values:
            if isinstance(v, ast.Constant):
                parts.append(rstr(v.value))
            else:
                parts.append(self.fvalue(v))
        if not parts:
            return '""'
        return "(" + " + ".join(parts) + ")"

    def fvalue(self, v):
        spec = self.L.types.get(("spec", id(v)), "")
        t = self.ty(v.value)
        if v.conversion == ord("r"):
            s, t = self.repr_of(v.value), STR
        elif v.conversion == ord("s"):
            s, t = self.str_of(v.value), STR
        else:
            s = None
        if not spec:
            return s if s is not None else self.str_of(v.value)
        m = FMT_RE.match(spec)
        if not m:
            raise Diag(v.lineno, "L-PY-FORMAT", f"the format spec {spec!r}")
        g = m.groupdict()
        fill = g["fill"] or " "
        width = int(g["width"]) if g["width"] else 0
        if t.name == "int" and s is None:
            ty = g["type"] or "d"
            if ty not in "dxXob" or g["prec"] is not None:
                raise Diag(v.lineno, "L-PY-FORMAT", f"the format spec {spec!r} for an int")
            align = g["align"] or ">"
            if g["zero"] and not g["align"]:
                if g["group"]:
                    raise Diag(v.lineno, "L-PY-FORMAT", f"zero padding with grouping ({spec!r})")
                fill, align = "0", "="
            base = {"d": 10, "x": 16, "X": 16, "o": 8, "b": 2}[ty]
            if g["group"] == "," and base != 10:
                raise Diag(v.lineno, "L-PY-FORMAT", f"the format spec {spec!r}")
            return (f"{self.use('py_fmt_int')}({self.val(v.value)}, {rstr(fill)}, {rstr(align)}, {rstr(g['sign'] or '')}, "
                    f"{'true' if g['alt'] else 'false'}, {width}, {rstr(g['group'] or '')}, {base}, {'true' if ty == 'X' else 'false'})")
        if t.name == "str" or s is not None:
            if (g["type"] not in (None, "s")) or g["sign"] or g["alt"] or g["group"] or g["align"] == "=" or g["zero"]:
                raise Diag(v.lineno, "L-PY-FORMAT", f"the format spec {spec!r} for a str")
            prec = int(g["prec"]) if g["prec"] is not None else -1
            body = s if s is not None else self.val(v.value)
            return f"{self.use('py_fmt_str')}({body}, {rstr(fill)}, {rstr(g['align'] or '<')}, {width}, {prec})"
        raise Diag(v.lineno, "L-PY-FORMAT", f"a format spec for {show(t)}")

    # str() and repr() by type, as monomorphic helpers
    def str_of(self, e):
        t = self.ty(e)
        if t.name == "str":
            return self.val(e)
        return self.show_fn(t, False) + f"({self.val(e)})"

    def repr_of(self, e):
        t = self.ty(e)
        return self.show_fn(t, True) + f"({self.val(e)})"

    def show_fn(self, t, rep):
        t = prune(t)
        n = t.name
        if n == "int":
            return self.helper("py_str_i", "fn py_str_i(x: i64) -> text = x.tostr()")
        if n == "bool":
            return self.use("py_str_b")
        if n == "str":
            if rep:
                self.np_table = True
                return self.use("py_repr_s")
            return self.helper("py_str_s", "fn py_str_s(x: text) -> text = x")
        m = mangle(t)
        name = "py_repr_" + m
        if name in self.helpers:
            return name
        self.helpers[name] = None
        if n == "list":
            el = self.show_fn(t.args[0], True)
            src = (f"fn {name}(xs: {rt(t)}) -> text\n    var o = \"[\"\n    var first = true\n    for x in xs\n"
                   f"        if not first\n            o = o + \", \"\n        o = o + {el}(copy(x))\n        first = false\n    return o + \"]\"")
        elif n == "dict":
            kf = self.show_fn(t.args[0], True)
            vf = self.show_fn(t.args[1], True)
            src = (f"fn {name}(d: {rt(t)}) -> text\n    var o = \"{{\"\n    for i in 0..d.0.len()\n"
                   f"        if i > 0\n            o = o + \", \"\n        o = o + {kf}(copy(d.0[i])) + \": \" + {vf}(copy(d.1[i]))\n    return o + \"}}\"")
        elif n == "tuple":
            fs = [self.show_fn(a, True) for a in t.args]
            body = " + \", \" + ".join(f"{f}(copy(x.{i}))" for i, f in enumerate(fs))
            close = '",)"' if len(fs) == 1 else '")"'
            src = f"fn {name}(x: {rt(t)}) -> text = \"(\" + {body} + {close}"
        elif n == "none":
            raise Diag(0, "L-PY-NONE", "None as a value")
        else:
            raise AssertionError(n)
        self.helpers[name] = src
        return name

    def helper(self, name, src):
        self.helpers[name] = src
        return name

    def ex_Call(self, e):
        fn = e.func
        if self.L.types.get(("reversed", id(e))):
            s, _ = self.iter_list(e.args[0])
            return f"{s}.reverse()"
        if isinstance(fn, ast.Name):
            name = fn.id
            if name in self.L.fns and self.L.fns[name].node is not None and name not in self.f.locals:
                return self.user_call(e, self.L.fns[name])
            return self.builtin(e, name)
        return self.method(e, fn.value, fn.attr)

    def user_call(self, e, g):
        args = self.L.types.get(("args", id(e)), e.args)
        out = []
        for p, a, pt in zip(g.params, args, g.ptypes):
            if p in g.mut_params:
                if isinstance(a, ast.Name) and a.id in self.f.params and a.id in self.f.mut_params:
                    out.append(self.name_ref(a.id))
                elif self.is_place(a):
                    out.append("&mut " + self.place(a))
                else:
                    raise Diag(a.lineno, "L-PY-ALIAS", f"{g.name} mutates its argument {p}, here a computed value")
            else:
                out.append(self.val(a))
        return f"f_{g.name}({', '.join(out)})"

    def builtin(self, e, name):
        a = e.args
        if name == "len":
            t = self.ty(a[0])
            c = self.place(a[0]) if self.is_place(a[0]) else self.val(a[0])
            if t.name == "str":
                return f"({c}.len_chars() as i64)"
            if t.name == "dict":
                return f"({c}.0.len() as i64)"
            if t.name == "tuple":
                return str(len(t.args))
            return f"({c}.len() as i64)"
        if name == "str":
            return self.str_of(a[0])
        if name == "repr":
            return self.repr_of(a[0])
        if name == "int":
            if self.ty(a[0]).name == "int":
                return self.val(a[0])
            return f"{self.use('py_int_s')}({self.val(a[0])})"
        if name == "bool":
            return self.truth(a[0])
        if name == "abs":
            return f"({self.val(a[0])}).abs()"
        if name == "ord":
            return f"{self.use('py_ord')}({self.val(a[0])})"
        if name == "chr":
            return f"{self.use('py_chr')}({self.val(a[0])})"
        if name == "divmod":
            return f"{self.use('py_divmod')}({self.val(a[0])}, {self.val(a[1])})"
        if name == "pow":
            if len(a) == 3:
                return f"{self.use('py_powmod')}({self.val(a[0])}, {self.val(a[1])}, {self.val(a[2])})"
            return f"{self.use('py_pow')}({self.val(a[0])}, {self.val(a[1])})"
        if name in ("min", "max"):
            if len(a) == 1:
                t = self.ty(a[0])
                c = self.val(a[0])
                if t.name == "str":
                    c = f"{c}.chars()"
                return f"{self.use('py_' + name + '_l')}({c})"
            r = self.val(a[-1])
            for x in reversed(a[:-1]):
                r = f"{self.use('py_' + name + '2')}({self.val(x)}, {r})"
            # Python keeps the first of equal items; equal items are equal values here
            return r
        if name == "sum":
            c = self.val(a[0])
            if isinstance(a[0], ast.Name):
                return f"{self.use('py_sum')}(&{self.place(a[0])})"
            t = self.fresh()
            self.pre.append(f"let {t} = {c}")
            return f"{self.use('py_sum')}(&{t})"
        if name in ("any", "all"):
            return f"{self.use('py_' + name)}({self.val(a[0])})"
        if name in ("sorted", "list"):
            if isinstance(a[0], ast.GeneratorExp):
                s = self.comp(a[0])
            else:
                s, _ = self.iter_list(a[0])
            if name == "list":
                return s
            rev = any(k.arg == "reverse" and k.value.value for k in e.keywords)
            return f"{self.use('py_sorted_rev' if rev else 'py_sorted')}({s})"
        if name == "dict":
            return "([], [])"
        raise Diag(e.lineno, "L-PY-CALL", f"{name}()")

    def method(self, e, recv, m):
        kind = self.L.types[("recv", id(e))]
        a = e.args
        if kind == "int":
            return f"{self.use('py_bitlen')}({self.val(recv)})"
        if kind == "str":
            r = self.val(recv)
            if m == "join":
                arg = self.comp(a[0]) if isinstance(a[0], ast.GeneratorExp) else self.val(a[0])
                return f"{r}.join({arg})"
            simple = {"upper": "py_upper", "lower": "py_lower", "strip": "py_strip", "isdigit": "py_isdigit", "isalpha": "py_isalpha"}
            if m in simple:
                return f"{self.use(simple[m])}({r})"
            if m == "split":
                if not a:
                    return f"{self.use('py_split_ws')}({r})"
                return f"{self.use('py_split')}({r}, {self.val(a[0])})"
            if m in ("startswith", "endswith"):
                return f"{r}.{m}({self.val(a[0])})"
            if m == "find":
                return f"{self.use('py_find')}({r}, {self.val(a[0])})"
            if m == "index":
                return f"{self.use('py_index_s')}({r}, {self.val(a[0])})"
            if m == "count":
                return f"{self.use('py_count_s')}({r}, {self.val(a[0])})"
            if m == "replace":
                return f"{self.use('py_replace')}({r}, {self.val(a[0])}, {self.val(a[1])})"
            if m in ("ljust", "rjust", "center"):
                fill = self.val(a[1]) if len(a) == 2 else '" "'
                if len(a) == 2:
                    raise Diag(e.lineno, "L-PY-CALL", f"str.{m} with a fill character")
                align = {"ljust": "<", "rjust": ">", "center": "^"}[m]
                if m == "center":
                    raise Diag(e.lineno, "L-PY-CALL", "str.center (CPython puts the odd padding on the left when the width is odd)")
                return f"{self.use('py_pad')}({r}, {fill}, \"{align}\", {self.val(a[0])})"
        if kind == "list":
            if not self.is_place(recv):
                if m in ("index", "count", "copy"):
                    pass
                else:
                    raise Diag(e.lineno, "L-PY-SYNTAX", f".{m}() on a computed list")
            p = self.place(recv) if self.is_place(recv) else None
            pm = self.mref(recv) if p else None
            if m == "append":
                return f"{p}.push({self.val(a[0])})"
            if m == "extend":
                return f"{self.use('py_extend')}({pm}, {self.val(a[0])})"
            if m == "insert":
                return f"{self.use('py_insert')}({pm}, {self.val(a[0])}, {self.val(a[1])})"
            if m == "pop":
                if a:
                    return f"{self.use('py_pop_at')}({pm}, {self.val(a[0])})"
                return f"{self.use('py_pop_last')}({pm})"
            if m == "index":
                return f"{self.use('py_index')}({self.bref(recv)}, {self.val(a[0])})"
            if m == "count":
                return f"{self.use('py_count')}({self.bref(recv)}, {self.val(a[0])})"
            if m == "remove":
                return f"{self.use('py_remove')}({pm}, {self.val(a[0])})"
            if m == "reverse":
                return f"{p} = copy({p}).reverse()"
            if m == "sort":
                rev = any(k.arg == "reverse" and k.value.value for k in e.keywords)
                if rev:
                    return f"{p} = {self.use('py_sorted_rev')}(copy({p}))"
                return f"{p}.sort()"
            if m == "copy":
                return self.val(recv)
            if m == "clear":
                return f"{p} = []"
        if kind == "dict":
            if not self.is_place(recv):
                raise Diag(e.lineno, "L-PY-SYNTAX", f".{m}() on a computed dict")
            if m == "get":
                return f"{self.use('py_dget_or')}(&{self.place(recv)}, {self.val(a[0])}, {self.val(a[1])})"
            if m == "pop":
                return f"{self.use('py_dpop')}({self.mref(recv)}, {self.val(a[0])})"
        raise Diag(e.lineno, "L-PY-CALL", f"{kind}.{m}()")

    def mref(self, e):
        if isinstance(e, ast.Name) and self.f.node is not None and e.id in self.f.mut_params:
            return self.name_ref(e.id)
        return "&mut " + self.place(e)

    def bref(self, e):
        if isinstance(e, ast.Name) and self.f.node is not None and e.id in self.f.mut_params:
            return self.name_ref(e.id)
        if self.is_place(e):
            return "&" + self.place(e)
        t = self.fresh()
        self.pre.append(f"let {t} = {self.val(e)}")
        return "&" + t

    # statements
    def emit(self, line):
        self.out.append("    " * self.depth + line)

    def flush_pre(self):
        for p in self.pre:
            self.emit(p)
        self.pre = []

    def block(self, body):
        n0 = len(self.out)
        for s in body:
            self.stmt(s)
        if len(self.out) == n0:
            self.emit("pass")

    def stmt(self, s):
        self.pre = []
        lines_before = self.out
        self.out = []
        depth = self.depth
        try:
            self._stmt(s)
        finally:
            body = self.out
            self.out = lines_before
            for p in self.pre:
                self.out.append("    " * depth + p)
            self.out.extend(body)
            self.pre = []

    def _stmt(self, s):
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):
                return
            if isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) and s.value.func.id == "print" and s.value.func.id not in self.f.locals:
                return self.print_(s.value)
            c = self.ex(s.value)
            t = self.ty(s.value)
            if t.name == "none" or (isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) and s.value.func.attr in ("append", "reverse", "sort", "clear") and self.L.types.get(("recv", id(s.value))) == "list"):
                self.emit(c)
            else:
                self.emit(f"let _ = {c}")
        elif isinstance(s, (ast.Assign, ast.AnnAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            if self.f is self.L.mod and len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id in self.L.consts:
                return
            if len(targets) == 1:
                self.assign(targets[0], s.value)
            else:
                t = self.fresh()
                self.emit(f"let {t} = {self.val(s.value)}")
                for tg in targets:
                    self.emit(f"{self.place(tg)} = {t if scalar(self.ty(tg)) else 'copy(' + t + ')'}")
        elif isinstance(s, ast.AugAssign):
            self.augassign(s)
        elif isinstance(s, ast.If):
            self.emit(f"if {self.truth(s.test)}")
            self.depth += 1
            self.block(s.body)
            self.depth -= 1
            orelse = s.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                s2 = orelse[0]
                self.pre = []
                c = self.truth(s2.test)
                if self.pre:
                    break
                self.emit(f"elif {c}")
                self.depth += 1
                self.block(s2.body)
                self.depth -= 1
                orelse = s2.orelse
            if orelse:
                self.emit("else")
                self.depth += 1
                self.block(orelse)
                self.depth -= 1
        elif isinstance(s, ast.While):
            self.while_(s)
        elif isinstance(s, ast.For):
            self.for_(s)
        elif isinstance(s, ast.Break):
            if self.brk and self.brk[-1]:
                self.emit(f"{self.brk[-1]} = true")
            self.emit("break")
        elif isinstance(s, ast.Continue):
            self.emit("continue")
        elif isinstance(s, ast.Pass):
            self.emit("pass")
        elif isinstance(s, ast.Return):
            if s.value is None or (isinstance(s.value, ast.Constant) and s.value.value is None):
                self.emit("return")
            else:
                self.emit(f"return {self.val(s.value)}")
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                bt = self.ty(t.value)
                if bt.name == "list":
                    self.emit(f"let _ = {self.use('py_pop_at')}({self.mref(t.value)}, {self.val(t.slice)})")
                elif bt.name == "dict":
                    self.emit(f"let _ = {self.use('py_dpop')}({self.mref(t.value)}, {self.val(t.slice)})")
                else:
                    raise Diag(s.lineno, "L-PY-SYNTAX", f"del of an item of {show(bt)}")
        else:
            raise Diag(s.lineno, "L-PY-SYNTAX", type(s).__name__)

    def assign(self, tg, value):
        if isinstance(tg, ast.Name):
            self.emit(f"{self.name_ref(tg.id)} = {self.val(value)}")
            return
        if isinstance(tg, ast.Subscript):
            bt = self.ty(tg.value)
            if bt.name == "dict":
                self.emit(f"{self.use('py_dset')}({self.mref(tg.value)}, {self.val(tg.slice)}, {self.val(value)})")
                return
            self.emit(f"{self.place(tg)} = {self.val(value)}")
            return
        # tuple targets: evaluate the whole right side first, as Python does
        t = self.fresh()
        self.emit(f"let {t} = {self.val(value)}")
        self.unpack(tg, t)

    def unpack(self, tg, t):
        for i, x in enumerate(tg.elts):
            xt = self.ty(x) if id(x) in self.L.types else None
            item = f"{t}.{i}"
            if isinstance(x, ast.Name):
                self.emit(f"{self.name_ref(x.id)} = {item if scalar(xt) else 'copy(' + item + ')'}")
            elif isinstance(x, ast.Subscript):
                bt = self.ty(x.value)
                v = item if scalar(xt) else f"copy({item})"
                if bt.name == "dict":
                    self.emit(f"{self.use('py_dset')}({self.mref(x.value)}, {self.val(x.slice)}, {v})")
                else:
                    self.emit(f"{self.place(x)} = {v}")
            else:
                u = self.fresh()
                self.emit(f"let {u} = copy({item})")
                self.unpack(x, u)

    def augassign(self, s):
        tg = s.target
        t = self.ty(tg)
        fake = ast.BinOp(left=tg, op=s.op, right=s.value)
        ast.copy_location(fake, s)
        self.L.types[id(fake)] = t
        if isinstance(tg, ast.Name) and t.name == "list" and isinstance(s.op, ast.Add):
            self.emit(f"{self.use('py_extend')}({self.mref(tg)}, {self.val(s.value)})")
            return
        if isinstance(tg, ast.Subscript) and self.ty(tg.value).name == "dict":
            v = self.fresh()
            self.emit(f"let {v} = {self.ex_BinOp(fake)}")
            self.emit(f"{self.use('py_dset')}({self.mref(tg.value)}, {self.val(tg.slice)}, {v})")
            return
        if isinstance(tg, ast.Subscript) and t.name == "list" and isinstance(s.op, ast.Add):
            raise Diag(s.lineno, "L-PY-ALIAS", "+= on a list item (extends the shared list in place)")
        self.emit(f"{self.place(tg)} = {self.ex_BinOp(fake)}")

    def print_(self, call):
        kws = {k.arg: k.value for k in call.keywords}
        sep = self.val(kws["sep"]) if "sep" in kws else '" "'
        end = self.val(kws["end"]) if "end" in kws else '"\\n"'
        parts = []
        for a in call.args:
            if isinstance(a, ast.Starred) and prune(self.L.types[id(a)]).name == "tuple":
                tt = prune(self.L.types[id(a)])
                tv = self.fresh()
                self.emit(f"let {tv} = {self.val(a.value)}")
                parts.extend(self.show_fn(x, False) + f"(copy({tv}.{i}))" if prune(x).name != "str" else f"copy({tv}.{i})" for i, x in enumerate(tt.args))
            elif isinstance(a, ast.Starred):
                lt = prune(self.L.types[id(a)])
                f = self.show_fn(lt.args[0], False) if prune(lt.args[0]).name != "str" else None
                if f:
                    parts.append(f"{sep}.join([{f}(x) for x in {self.val(a.value)}])")
                else:
                    parts.append(f"{sep}.join({self.val(a.value)})")
            else:
                parts.append(self.str_of(a))
        if not parts:
            self.emit(f"emit {end}")
            return
        if sep == '" "' and all(not isinstance(a, ast.Starred) for a in call.args):
            body = ' + " " + '.join(parts)
        else:
            body = f" + {sep} + ".join(parts)
        self.emit(f"emit {body} + {end}")

    def bound_text(self, s):
        terms = self.L.loop_bounds[id(s)]
        ds = []
        for v, lim, rising, incl in terms:
            vv = self.name_ref(v)
            ll = self.val(lim)
            d = f"({ll} - {vv})" if rising else f"({vv} - {ll})"
            ds.append(d)
        # each run moves the sum of the distances by at least one, so the
        # loop runs at most that sum plus one times (the plus one covers <= and >=)
        b = self.fresh("b")
        self.emit(f"let {b}: i64 = " + " + ".join(f"{self.use('py_max2')}({d}, 0)" for d in ds) + " + 1")
        return b

    def while_(self, s):
        b = self.bound_text(s)
        flag = None
        if s.orelse:
            flag = self.fresh("brk")
            self.emit(f"var {flag} = false")
        self.emit(f"while {self.truth(s.test)} bound {b} as u64")
        self.brk.append(flag)
        self.depth += 1
        self.block(s.body)
        self.depth -= 1
        self.brk.pop()
        if s.orelse:
            self.emit(f"if not {flag}")
            self.depth += 1
            self.block(s.orelse)
            self.depth -= 1

    def for_(self, s):
        flag = None
        if s.orelse:
            flag = self.fresh("brk")
            self.emit(f"var {flag} = false")
        it = s.iter
        k = self.fresh("k")
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            a = [self.val(x) for x in it.args]
            if len(a) == 1:
                a = ["0", a[0]]
            step = self.L.const_int(it.args[2]) if len(it.args) == 3 else 1
            lo, hi = self.fresh(), self.fresh()
            self.emit(f"let {lo}: i64 = {a[0]}")
            self.emit(f"let {hi}: i64 = {a[1]}")
            if step == 1:
                self.emit(f"for {k} in {lo}..{hi}")
                item = k
            else:
                self.emit(f"for {k} in 0..{self.use('py_rlen')}({lo}, {hi}, {step})")
                item = f"({lo} + {k} * {step})"
            et = INT
        else:
            src, et = self.iter_list(it)
            lst = self.fresh()
            self.emit(f"let {lst} = {src}")
            self.emit(f"for {k} in 0..{lst}.len()")
            item = f"{lst}[{k}]" if scalar(et) else f"copy({lst}[{k}])"
        self.brk.append(flag)
        self.depth += 1
        if isinstance(s.target, ast.Name):
            self.emit(f"{self.name_ref(s.target.id)} = {item}")
        else:
            t = self.fresh()
            self.emit(f"let {t} = {item}")
            self.unpack(s.target, t)
        self.block(s.body)
        self.depth -= 1
        self.brk.pop()
        if s.orelse:
            self.emit(f"if not {flag}")
            self.depth += 1
            self.block(s.orelse)
            self.depth -= 1

    # functions
    def function(self, f):
        self.f = f
        self.read = reads(f.node.body)
        self.alias = {}
        self.brk = []
        self.out = []
        self.depth = 1
        recursive = self.in_cycle(f.name)
        params = []
        rebound = assigned_in(f.node.body)
        heads = []
        for p, t in zip(f.params, f.ptypes):
            if p in f.mut_params:
                params.append(f"{self.vname(p)}: &mut {rt(t)}")
            elif p in rebound:
                params.append(f"a_{p}: {rt(t)}")
                heads.append(f"var {self.vname(p)}: {rt(t)} = a_{p}")
            else:
                params.append(f"{self.vname(p)}: {rt(t)}")
        for n, t in f.locals.items():
            if n in f.params:
                continue
            heads.append(f"var {self.vname(n)}: {rt(t)} = {default(t)}")
        ret = prune(f.ret)
        sig = f"fn f_{f.name}({', '.join(params)})" + ("" if ret.name == "none" else f" -> {rt(ret)}") + (" bound 1000" if recursive else "")
        for h in heads:
            self.emit(h)
        self.block(f.node.body)
        return sig + "\n" + "\n".join(self.out)

    def in_cycle(self, name):
        seen, stack = set(), list(self.L.fns[name].calls)
        while stack:
            g = stack.pop()
            if g == name:
                return True
            if g in seen or g not in self.L.fns:
                continue
            seen.add(g)
            stack.extend(self.L.fns[g].calls)
        return False

    def main(self):
        f = self.L.mod
        self.f = f
        self.read = reads([s for s in self.L.tree.body if not isinstance(s, ast.FunctionDef)])
        self.alias = {}
        self.brk = []
        self.out = []
        self.depth = 1
        for n, t in f.locals.items():
            if n in self.L.consts:
                continue
            self.emit(f"var {self.vname(n)}: {rt(t)} = {default(t)}")
        body = [s for s in self.L.tree.body if not isinstance(s, ast.FunctionDef)]
        self.block(body)
        return "fn main()\n" + "\n".join(self.out)

    def const(self, g):
        t = self.L.mod.locals[g]
        for s in self.L.tree.body:
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id == g:
                self.f = self.L.mod
                self.alias = {}
                return f"const G_{g}: {rt(t)} = {self.val(s.value)}"
        raise AssertionError(g)


def runtime_defs():
    with open(RUNTIME, encoding="utf-8") as f:
        src = f.read()
    defs = {}
    cur, buf = None, []
    for line in src.split("\n"):
        m = re.match(r"^fn (\w+)", line)
        if m:
            if cur:
                defs[cur] = "\n".join(buf).rstrip()
            cur, buf = m.group(1), [line]
        elif cur and (line.startswith(" ") or line == ""):
            buf.append(line)
        else:
            if cur:
                defs[cur] = "\n".join(buf).rstrip()
            cur, buf = None, []
    if cur:
        defs[cur] = "\n".join(buf).rstrip()
    return defs


def np_table():
    if unicodedata.unidata_version != "15.1.0":
        raise SystemExit(f"the printable table must come from Unicode 15.1.0 (CPython 3.13); this Python has {unicodedata.unidata_version}")
    lo, hi, s = [], [], None
    for c in range(0x80, 0x110000):
        np = not chr(c).isprintable()
        if np and s is None:
            s = c
        if not np and s is not None:
            lo.append(s)
            hi.append(c - 1)
            s = None
    if s is not None:
        lo.append(s)
        hi.append(0x10FFFF)
    return lo, hi


def link(pr, fns, with_main):
    """The program text: the runtime definitions used (with their own
    callees), the generated helpers, constants, functions and main."""
    defs = runtime_defs()
    need = set(pr.uses)
    for h in pr.helpers.values():
        need |= set(re.findall(r"\b(py_\w+)\(", h))
    for t in fns:
        need |= set(re.findall(r"\b(py_\w+)\(", t))
    work = list(need)
    while work:
        n = work.pop()
        if n in defs:
            for m in re.findall(r"\b(py_\w+)\(", defs[n]):
                if m not in need:
                    need.add(m)
                    work.append(m)
    out = []
    for n in defs:
        if n in need:
            out.append(defs[n])
    for n, h in pr.helpers.items():
        out.append(h)
    if "py_nonprintable" in need:
        lo, hi = np_table()
        out.append("# The code points from U+0080 that str.isprintable() rejects under Unicode 15.1.0, as ranges")
        out.append("const PY_NP_LO: List[i64] = [" + ", ".join(map(str, lo)) + "]")
        out.append("const PY_NP_HI: List[i64] = [" + ", ".join(map(str, hi)) + "]")
    out.extend(fns)
    return "\n\n".join(out) + "\n"


def lift(path, src):
    L_ = Lifter(path, src)
    text, err = L_.run()
    if err:
        return None, err, L_
    diags = []
    for f in L_.fns.values():
        diags += f.diags
    if diags:
        return None, sorted(set(diags)), L_
    pr = Printer(L_)
    try:
        parts = [pr.const(g) for g in L_.consts]
        for name, f in L_.fns.items():
            if f.node is not None:
                parts.append(pr.function(f))
        parts.append(pr.main())
    except Diag as d:
        return None, [(d.line, d.rule, d.reason)], L_
    head = f"# Lifted from {os.path.basename(path)} by spec/lift/python/lift_py.py; do not edit.\n\n"
    return head + link(pr, parts, True), [], L_


def lift_functions(path, src):
    """Function-level coverage: each function of the module with the
    functions it calls, applied only when none of them requires change
    (section 16: the frontier is drawn at function granularity)."""
    L_ = Lifter(path, src)
    text, err = L_.run()
    if err:
        return [("<module>", None, err)]
    own = {}
    for name, f in L_.fns.items():
        if f.node is not None:
            own[name] = sorted(set(f.diags))
    res = []
    for name, f in L_.fns.items():
        if f.node is None:
            continue
        closure, stack = [], [name]
        while stack:
            g = stack.pop()
            if g in closure:
                continue
            closure.append(g)
            stack.extend(sorted(L_.fns[g].calls))
        diags = list(own[name])
        for g in closure[1:]:
            if own.get(g):
                diags.append((f.node.lineno, "L-PY-CALLEE", f"{name} calls {g}, which requires change"))
        if diags:
            res.append((name, None, diags))
            continue
        pr = Printer(L_)
        try:
            consts = sorted({c for g in closure for c in L_.fns[g].uses_globals})
            parts = [pr.const(c) for c in consts]
            for g in closure:
                parts.append(pr.function(L_.fns[g]))
        except Diag as d:
            res.append((name, None, [(d.line, d.rule, d.reason)]))
            continue
        res.append((name, link(pr, parts, False), []))
    return res


def main(argv):
    if len(argv) == 2 and argv[1] == "rules":
        for k, v in RULES.items():
            print(f"{k}: {v}")
        return 0
    if len(argv) != 3 or argv[1] not in ("lift", "functions"):
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    path = argv[2]
    with open(path, "rb") as f:
        src = f.read()
    try:
        if argv[1] == "lift":
            text, diags, _ = lift(path, src)
            if diags:
                for line, rule, reason in diags:
                    print(f"requires change {path}: line {line}: {rule}: {reason}")
                return 1
            sys.stdout.write(text)
            return 0
        for name, text, diags in lift_functions(path, src):
            if diags:
                for line, rule, reason in diags:
                    print(f"requires change {path}: {name}: line {line}: {rule}: {reason}")
            else:
                print(f"lifted {name}")
        return 0
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
