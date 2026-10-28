#!/usr/bin/env python3
"""Hand-written tests aimed at single clauses. The expected result of each is
read off the clause (the law), never off a run. The reference is consulted
only afterwards: a disagreement is written to findings/targeted.json and the
test keeps the law.  targeted.py TESTS_DIR"""
import json, os, pathlib, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
sys.path.insert(0, os.path.dirname(__file__))
import rcetest
from refclient import Ref
out = pathlib.Path(sys.argv[1]); ref = Ref(); findings = []; n = 0

T = []
def t(name, area, clauses, src, expect, **kw):
    T.append(dict(name=name, area=area, clauses=clauses, src=src.lstrip("\n"), expect=expect, **kw))

# --- 3.x resolved ambiguities R1..R9 (syntax)
t("r1-dangling-else", "lexing-parsing", ["R1"], """
fn f(a: bool, b: bool) -> i64
    if a
        if b
            return 1
        else
            return 2
    return 3
fn main()
    emit f(true, false)
    emit f(false, true)
""", "run", outputs=["emit 2", "emit 3"], outcome="done")
t("r2-try-in-if-expr", "lexing-parsing", ["R2"], """
fn f(x: i64?) -> i64 = if true then try x else 0 else 1
""", "fit", diagnostic="W-SYN-1")
t("r3-chained-comparison", "lexing-parsing", ["R3"], """
fn f(a: i64) -> bool = 0 < a < 10
""", "fit", diagnostic="W-SYN-1")
t("r3-in-range", "lexing-parsing", ["R3"], """
fn main()
    let a: i64 = 5
    emit a in 0..10
""", "run", outputs=["emit true"], outcome="done")
t("r4-assignment-in-expression", "lexing-parsing", ["R4"], """
fn main()
    var x: i64 = 0
    emit (x = 1)
""", "fit", diagnostic="W-SYN-1")
t("r5-generic-call", "lexing-parsing", ["R5"], """
fn id[T](x: T) -> T = x
fn main()
    emit id[i64](7)
""", "run", outputs=["emit 7"], outcome="done")
t("r6-impl-generic-needs-forall", "lexing-parsing", ["R6"], """
struct Box[T]
    v: T
impl [T] Box[T]
    fn get(self) -> T = self.v
""", "fit", diagnostic="W-SYN-1")
t("r7-map-literal-key", "lexing-parsing", ["R7"], """
fn main()
    let a: text = "k"
    let m: Map[text, i64] = {a: 1}
    emit m["k"]
""", "run", outputs=["emit 1"], outcome="done")
t("r8-unit-variant-pattern", "lexing-parsing", ["R8", "W-PAT-1"], """
kind Light
    Red
    Green
fn f(l: Light) -> i64
    match l
        Red -> return 1
        other -> return 2
fn main()
    emit f(Red)
    emit f(Green)
""", "run", outputs=["emit 1", "emit 2"], outcome="done")
t("r9-send-receive", "lexing-parsing", ["R9"], """
# rce-version: 0.6
fn main()
    let a = chan[i64](1)
    let b = chan[i64](1)
    b <- 4
    a <- <- b
    emit 0
""", "fit", diagnostic="W-TY-1", note="R9: `a <- <- b` parses as send(a, receive(b)); a receive gives i64? so sending it on chan i64 is a type error, which shows the parse")
t("s3-keyword-reserved", "lexing-parsing", ["S3"], """
fn f(from: i64) -> i64 = from
""", "fit", diagnostic="W-SYN-1")
t("s3-keyword-as-member", "lexing-parsing", ["S3"], """
struct E
    from: i64
fn main()
    let e = E(from: 3)
    emit e.from
""", "fit", diagnostic="W-SYN-1", note="S3: a keyword is a member name only after `.`, so a field cannot be declared with it")
t("s4-std-modules", "types-checking", ["S4", "CH04-M-c3c51cd1"], """
module m
use std.text
use std.collections
use std.math
fn main()
    emit 1
""", "run", outputs=["emit 1"], outcome="done")
t("s5-tail-value", "evaluation-fuel", ["S5"], """
fn f(a: i64) -> i64
    let b = a * 2
    b + 1
fn main()
    emit f(20)
""", "run", outputs=["emit 41"], outcome="done")
t("s1-guard-trap-reports", "forms-log", ["S1", "T8"], """
state S
    n: i64 = 1
event Div(d: i64)
error Bad
step S on Div(d: d) if n / d > 0 -> n + 1
""", "run", events="Div(d: 1), Div(d: 0)", outcome="trap:division by zero")
t("s2-task-fn-with-chan", "concurrency", ["S2"], """
# rce-version: 0.6
fn worker(c: chan i64)
    c <- 9
fn main()
    let c = chan[i64](1)
    spawn worker(c)
    emit <- c
""", "run", outputs=["emit some(9)"], outcome="done")

# --- numbers
t("n6-int-float-equality", "numbers", ["N6", "W-TY-3"], """
fn f() -> bool = 1 == 1.0
""", "fit", diagnostic="W-TY-3")
t("n6-signed-zero-equal", "numbers", ["N6", "F6"], """
fn main()
    let z: f64 = 0.0
    let n: f64 = z * -1.0
    emit n == z
""", "run", outputs=["emit true"], outcome="done")
t("w-ty-3-mixing", "numbers", ["W-TY-3", "CH05-M-2405dfe8", "N4"], """
fn f(a: i64, b: f64) -> f64 = a + b
""", "fit", diagnostic="W-TY-3")
t("w-ty-3-narrowing", "numbers", ["W-TY-3"], """
fn f(a: i64) -> i32
    let b: i32 = a
    return b
""", "fit", diagnostic="W-TY-3")
t("f5-nan-traps", "numbers", ["F5", "F4"], """
fn main()
    let z: f64 = 0.0
    emit z / z
""", "run", outcome="trap:non-finite float")
t("f5-inf-minus-inf", "numbers", ["F5", "F4"], """
fn main()
    let a: f64 = 1e308
    emit a * 10.0 - a * 10.0
""", "run", outcome="trap:non-finite float")
t("f6-signed-zero-state", "numbers", ["F6", "N1"], """
state S
    x: f64 = 0.0
event Neg
step S on Neg -> x * -1.0
""", "trace", events="Neg", trace_head_law="differs-from:pos-zero")
t("f6-f32-tostr-minus-zero", "numbers", ["F6", "T13"], """
fn main()
    let z = (0.0 * -1.0) as f32
    emit z.tostr()
""", "run", outputs=['emit "-0"'], outcome="done")
t("f6-f64-tostr-minus-zero", "numbers", ["F6", "F9"], """
fn main()
    let z: f64 = 0.0 * -1.0
    emit z.tostr()
""", "run", outputs=['emit "-0"'], outcome="done", note="law: canonical text is the shortest text that round-trips (F9); `0` reads back as +0.0, so -0.0 must print `-0`. See finding F6-TOSTR.")
t("t13-f32-single-rounding", "numbers", ["T13"], """
fn main()
    let a = 0.1 as f32
    let b = 0.2 as f32
    emit (a + b) as f64
""", "run", outputs=["emit 0.30000001192092896"], outcome="done")
t("t14-u64-edges", "numbers", ["T14", "N2"], """
fn main()
    let a: u64 = 18446744073709551615
    let b: u64 = 7
    emit a / b
    emit a % b
    emit a / b * b + a % b == a
""", "run", outputs=["emit 2635249153387078802", "emit 1", "emit true"], outcome="done")
t("t14-u64-overflow", "numbers", ["T14", "N2"], """
fn main()
    let a: u64 = 18446744073709551615
    emit a + 1
""", "run", outcome="trap:overflow: 18446744073709551616 is outside u64")
t("n9-size-limit", "evaluation-fuel", ["N9"], """
fn main()
    var s: text = "x"
    for _ in 0..21
        s = s + s
    emit s.len()
""", "run", outcome="trap:size limit")
t("n8-index-out-of-range", "evaluation-fuel", ["N8"], """
fn main()
    let xs: List[i64] = [1, 2, 3]
    let i: usize = 3
    emit xs[i]
""", "run", outcome="trap:index 3 out of range")
t("n8-slice-reversed", "evaluation-fuel", ["N8"], """
fn main()
    let xs: List[i64] = [1, 2, 3]
    let i: usize = 2
    let j: usize = 1
    emit xs[i:j].len()
""", "run", outcome="trap:slice bounds reversed")
t("n7-not-char-boundary", "text-unicode", ["N7", "CH05-M-84afe529"], """
fn main()
    let s: text = "é"
    let i: usize = 1
    emit s[0:i]
""", "run", outcome="trap:not a char boundary")
t("n7-byte-len", "text-unicode", ["N7", "CH05-M-84afe529"], """
fn main()
    let s: text = "aé😀"
    emit s.len()
    emit s.chars().len()
""", "run", outputs=["emit 7", "emit 3"], outcome="done")
t("n7-ascii-case", "text-unicode", ["N7"], """
fn main()
    emit "aé".upper()
""", "run", outputs=['emit "Aé"'], outcome="done")
t("n7-codepoint-order", "text-unicode", ["N7"], """
fn main()
    emit "Z" < "a"
    emit "é" > "z"
""", "run", outputs=["emit true", "emit true"], outcome="done")

# --- types and checking
t("w-ty-2-nested-option", "types-checking", ["W-TY-2"], "fn f(x: i64??) -> i64 = 0\n", "fit", diagnostic="any", note="see finding W-TY-2-UNREACHABLE: the grammar has no `τ??`, so W-TY-2 cannot be the rule that reports it")
t("w-ty-2-borrow-of-borrow", "types-checking", ["W-TY-2"], "fn f(x: &&i64) -> i64 = 0\n", "fit", diagnostic="W-TY-2")
t("w-pat-1-binding-is-variant", "types-checking", ["W-PAT-1"], """
kind Light
    Red
    Green
fn f(l: Light) -> i64
    match l
        Light.Red -> return 1
        x -> return 2
fn g(Red: i64) -> i64 = Red
""", "fit", diagnostic="W-PAT-1")
t("w-match-2-unreachable", "types-checking", ["W-MATCH-2"], """
fn f(x: i64) -> i64
    match x
        _ -> return 1
        0 -> return 2
""", "fit", diagnostic="W-MATCH-2")
t("w-state-1-no-fields", "types-checking", ["W-STATE-1"], "state S\nevent E\n", "fit", diagnostic="any", note="see finding W-STATE-1-UNREACHABLE")
t("w-state-3-borrow-field", "types-checking", ["W-STATE-3"], "state S\n    r: &i64 = 0\n", "fit", diagnostic="W-STATE-3")
t("w-step-1-undeclared-region", "types-checking", ["W-STEP-1"], """
state S
    n: i64 = 0
event E
step T on E -> 1
step S on E -> n + 1
""", "fit", diagnostic="W-STEP-1")
t("w-step-3-arrow-multi-field", "types-checking", ["W-STEP-3"], """
state S
    a: i64 = 0
    b: i64 = 0
event E
step S on E -> a + 1
""", "fit", diagnostic="W-STEP-3")
t("w-ref-1-float-refined", "types-checking", ["W-REF-1", "T35"], """
type P = f64 where self <= 1.0
""", "fit", diagnostic="W-REF-1")
t("t35-refined-input-reported", "types-checking", ["T35", "T8", "HC-3"], """
type Pct = u8 where self <= 100
state V
    level: Pct = 0
event Set(p: Pct)
step V on Set(p: p) -> p
""", "host", steps=[{"batch": "Set(p: 40)"}, {"offer": {"source": "ui", "seq": 1, "tick": 0, "event": "Set", "args": {"p": 101}}}, {"close": 0}], host_expect_law="replay")
t("w-trait-1-impl-mismatch", "types-checking", ["W-TRAIT-1", "CH06-M-8e9952a9"], """
trait D
    fn d(&self) -> text
struct P
    x: i64
impl D for P
    fn d(&self) -> i64 = self.x
""", "fit", diagnostic="W-TRAIT-1")
t("w-trait-3-eq-without-trait", "types-checking", ["W-TRAIT-3"], """
fn f(a: fn(i64) -> i64, b: fn(i64) -> i64) -> bool = a == b
""", "fit", diagnostic="W-TRAIT-3")
t("borrow-cannot-be-returned", "types-checking", ["CH05-M-c643f9c7", "T6"], """
struct H
    r: &i64
""", "fit", diagnostic="W-TY-2")

# --- modules (multi-file)
t("w-mod-1-private", "types-checking", ["W-MOD-1", "W-VIS-1"], """
module app
use lib.a
fn main()
    emit a.hidden()
""", "fit", diagnostic="W-VIS-1", fixtures={"lib-a_FIXTURE.rce": "module lib.a\nfn hidden() -> i64 = 1\n"})
t("w-mod-2-cycle", "types-checking", ["W-MOD-2"], """
module p.a
use p.b
pub fn fa() -> i64 = 1
""", "fit", diagnostic="W-MOD-2", fixtures={"p-b_FIXTURE.rce": "module p.b\nuse p.a\npub fn fb() -> i64 = 2\n"})
t("w-mod-3-clash", "types-checking", ["W-MOD-3"], """
module app
use lib.c
fn helper() -> i64 = 2
""", "fit", diagnostic="W-MOD-3", fixtures={"lib-c_FIXTURE.rce": "module lib.c\npub fn helper() -> i64 = 1\n"})

# --- effects and capabilities (0.8)
t("w-cap-1-cap-in-state", "effects-capabilities", ["W-CAP-1", "HC-5"], """
# rce-version: 0.8
struct Root
state S
    c: Cap[Root] = ()
""", "fit", diagnostic="W-CAP-1")
t("w-cap-2-write-without-cap", "effects-capabilities", ["W-CAP-2", "HC-5", "T36"], """
# rce-version: 0.8
struct Root
state Policy
    limit: Guarded[u32, Root] = 100
event Set(limit: u32)
step Policy on Set(limit: n) -> n
""", "fit", diagnostic="W-CAP-2")
t("t36-guarded-unchanged-without-cap", "effects-capabilities", ["T36", "HC-5"], """
# rce-version: 0.8
struct Root
state Policy
    limit: Guarded[u32, Root] = 100
state Ledger
    spent: u32 = 0
event SetLimit(admin: Cap[Root], limit: u32)
event Spend(amount: u32)
step Policy on SetLimit(admin: _, limit: n) -> n
step Ledger on Spend(amount: a) -> spent + a
""", "run", events="Spend(amount: 5), Spend(amount: 6)", outcome="done")
t("w-sec-1-emit-secret", "effects-capabilities", ["W-SEC-1", "T36"], """
# rce-version: 0.8
state V
    n: u32 = 0
event Login(pw: Secret[text])
step V on Login(pw: p)
    emit p
""", "fit", diagnostic="W-SEC-1")
t("w-sec-1-secret-method", "effects-capabilities", ["W-SEC-1"], """
# rce-version: 0.8
state V
    n: usize = 0
event Login(pw: Secret[text])
step V on Login(pw: p)
    n = p.len()
""", "fit", diagnostic="W-SEC-1")
t("w-ctx-1-plain-join", "effects-capabilities", ["W-CTX-1", "T36"], """
# rce-version: 0.8
state P
    html: Text[Html] = Html.lit("")
event C(body: text)
step P on C(body: b)
    html = Html.lit("<p>") + b
""", "fit", diagnostic="W-CTX-1")

# --- concurrency
t("w-bud-1-emit-in-budget-else", "concurrency", ["W-BUD-1"], """
# rce-version: 0.6
fn main()
    with budget 100
        emit 1
    else
        emit 2
""", "fit", diagnostic="W-BUD-1")
t("w-intr-1-priority-range", "concurrency", ["W-INTR-1", "HC-9"], """
# rce-version: 0.6
interrupt event A priority 300
state S
    n: i64 = 0
step S on A -> n + 1
""", "fit", diagnostic="W-INTR-1")
t("t18-tree-reduce-equals-fold", "concurrency", ["T18", "CH16-M-a84fabb6"], """
# rce-version: 0.6
fn main()
    let xs: List[i64] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    var s: i64 = 0
    for x in xs
        s += x
    emit s
""", "run", outputs=["emit 55"], outcome="done", schedule_seeds=[1, 2, 3])

# --- host contract
t("hc-7a-fuel-per-input", "host-contract", ["HC-7a", "HC-7"], """
state S
    n: i64 = 0
event Work
step S on Work
    var k: i64 = 0
    for _ in 0..20000
        k += 1
    n = n + k
""", "host", steps=[{"batch": "Work, Work, Work, Work, Work, Work, Work, Work"}],
  host_expect_law="all-applied")
t("hc-8-t32-t37-replay-equals-live", "host-contract", ["HC-8", "T32", "T37", "HC-6", "HC-3"], """
state S
    n: i64 = 0
event Inc
error Neg
event Dec
step S on Inc -> n + 1
step S on Dec -> if n > 0 then n - 1 else 0
""", "host", steps=[{"batch": "Inc, Inc"}, {"batch": "Dec"}, {"batch": "Dec, Dec"}], host_expect_law="replay")

# --- second batch
t("ch11-hash-chain-in-rce", "forms-log", ["CH11-M-ba4b6fd9", "T3"], """
fn main()
    let e: text = "{\\"event\\":{\\"payload\\":{},\\"type\\":\\"Inc\\"},\\"prev_hash\\":\\"0000000000000000000000000000000000000000000000000000000000000000\\",\\"seq\\":0,\\"state_hash\\":\\"a8d94f4388d06db57160fa4235a0c303c6da53e9dd48d7ff7b4a08c5689de184\\"}"
    emit e.sha256()
""", "run", outputs=['emit "f5b9e4f9376ffb85dcfaaf9ec0ba5e83a84e87e6d9a4920e99d3db076fa8fae6"'], outcome="done",
  note="the head of 01_counter after one Inc (golden trace), recomputed by an RCE program from the canonical JSON of the entry")
t("w-atom-2-two-nodes", "concurrency", ["W-ATOM-2", "T30"], """
# rce-version: 0.6
event Add(k: i64)
actor A
    n: i64 = 0
    on Add(k: k) -> n + k
actor B
    n: i64 = 0
    on Add(k: k) -> n + k
node east { A }
node west { B }
fn main()
    let a = A()
    let b = B()
    atomic
        a <- Add(k: 1)
        b <- Add(k: 2)
""", "fit", diagnostic="W-ATOM-2")
t("hc-12-float-reduction-schedule", "concurrency", ["HC-12", "CH16-M-a84fabb6", "T18", "T20"], """
# rce-version: 0.6
fn main()
    let xs: List[f64] = [0.1, 0.2, 0.3, 1e16, -1e16, 0.4, 0.5, 0.6]
    var s: f64 = 0.0
    var t: f64 = 0.0
    par
        branch left
            for x in xs
                s += x
        branch right
            for x in xs
                t += x * 2.0
    emit s
    emit t
""", "run", outputs=["emit 1.5", "emit 3"], outcome="done", schedule_seeds=[1, 2, 3, 4, 5, 6, 7, 8],
  note="each branch is a left fold in index order (N-9 / T18); no schedule may reassociate it")
t("hc-11-fabric-caps-logged", "host-contract", ["HC-11", "HC-13", "HC-14", "CH16-M-262166d9"], """
state S
    n: i64 = 0
event Inc
step S on Inc -> n + 1
""", "host", steps=[{"batch": "Inc"}], host_expect_law="fabric")
t("t38-torn-tail-keeps-prefix", "host-contract", ["T38", "T39", "HC-8"], """
state S
    n: i64 = 0
event Inc
step S on Inc -> n + 1
""", "host", steps=[{"batch": "Inc, Inc"}, {"batch": "Inc"}], tear=5, host_expect_law="torn")

t("lex-merge-int-key", "concurrency", ["CH14-M-9202f003", "W-CRDT-1", "T29s"], """
# rce-version: 0.6
struct V
    k: i64
    n: i64
impl Merge for V
    fn merge(self, o: V) -> V = if self.k > o.k then self else (if o.k > self.k then o else V(k: self.k, n: self.n.max(o.n)))
    fn bottom() -> V
        V(k: 0, n: 0)
""", "ok")
t("lex-merge-bitwise-reported", "concurrency", ["CH14-M-9202f003", "W-CRDT-1"], """
# rce-version: 0.6
struct V
    k: i64
impl Merge for V
    fn merge(self, o: V) -> V
        V(k: self.k + o.k)
    fn bottom() -> V
        V(k: 0)
""", "fit", diagnostic="W-CRDT-1", note="addition is not idempotent, so it is not a join")

for x in T:
    m = {"id": f"{x['area']}/targeted/{x['name']}", "area": x["area"], "clauses": x["clauses"], "expect": x["expect"],
         "provenance": "targeted.py: expected result read off the clause"}
    for k in ("diagnostic", "events", "outcome", "outputs", "schedule_seeds", "steps", "tear", "note"):
        if k in x: m[k] = x[k]
    fx = x.get("fixtures") or {}
    if fx: m["fixtures"] = list(fx)
    src = x["src"]
    if x["expect"] in ("ok", "fit"):
        r = ref.check(src, {"main.rce": src, **fx})
        got = (r.get("verdict"), [d.get("code") for d in r.get("diagnostics", [])])
        ok = got[0] == x["expect"] and (x["expect"] == "ok" or x.get("diagnostic") in got[1] + ["any"])
        if not ok: findings.append({"test": m["id"], "law": [x["expect"], x.get("diagnostic")], "reference": got,
                                    "messages": [d.get("message") for d in r.get("diagnostics", [])][:2]})
    elif x["expect"] == "run":
        r = ref.run(src, x.get("events"))
        got = (r.get("outputs"), r.get("outcome"), r.get("error"), r.get("diagnostics"))
        if r.get("outputs") != x.get("outputs", r.get("outputs")) or r.get("outcome") != x.get("outcome"):
            findings.append({"test": m["id"], "law": [x.get("outputs"), x.get("outcome")], "reference": got})
    elif x["expect"] == "trace":
        r = ref.trace(src, x["events"])
        pos = ref.trace(src.replace("x * -1.0", "x * 1.0"), x["events"])
        h, hp = (r.get("trace") or {}).get("head"), (pos.get("trace") or {}).get("head")
        if not h or h == hp:
            findings.append({"test": m["id"], "law": "state with -0.0 hashes differently from +0.0", "reference": [h, hp, r.get("error")]})
        m["trace_head"] = h
        m["note"] = "law F6: the head differs from the same program with x * 1.0 (" + str(hp) + ")"
    elif x["expect"] == "host":
        r = ref.call(op="host", source=src, steps=x["steps"], tear=x.get("tear"))
        h = r.get("host") or {}
        if x["host_expect_law"] == "all-applied":
            kept = [a for a in h.get("admitted", []) if "kept" in a[2]]
            if kept or r.get("error"): findings.append({"test": m["id"], "law": "every input applied (fresh fuel per input)", "reference": kept or r.get("error")})
            m["host_expect"] = {"admitted": [a for a in h.get("admitted", []) if "ok" in a[2]]}
        elif x["host_expect_law"] == "fabric":
            m["host_expect"] = {"fabric_caps_first": True}
        elif x["host_expect_law"] == "torn":
            # law: the tear is inside the last barrier entry, so the verified prefix holds every input
            m["host_expect"] = {"torn_reopen_state_hash": h.get("state_hash")}
            if (h.get("torn") or {}).get("reopen_state_hash") != h.get("state_hash"):
                findings.append({"test": m["id"], "law": "reopen keeps the verified prefix", "reference": h.get("torn")})
            if "error" in ((h.get("torn") or {}).get("replay") or {}):
                findings.append({"test": m["id"], "kind": "REPLAY-TORN", "law": "replay of D equals what reopening D derives",
                                 "reference": h["torn"]["replay"]})
        else:
            m["host_expect"] = {"state_hash": h.get("state_hash")}
    d = out / x["area"] / "targeted"; d.mkdir(parents=True, exist_ok=True)
    rcetest.write(d / f"{x['name']}.rce", src, m); n += 1
    for k, v in fx.items(): (d / k).write_text(v)
os.makedirs(out.parent / "findings", exist_ok=True)
json.dump(findings, open(out.parent / "findings" / "targeted.json", "w"), indent=1)
print(json.dumps({"written": n, "disagreements": len(findings)}))
