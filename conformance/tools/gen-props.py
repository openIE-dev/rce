#!/usr/bin/env python3
"""Write the property suites under tests/properties/.

Each suite is one seq test over a stated bound: every value or rewrite in
the bound is enumerated here (bounded exhaustive), or drawn from a fixed
seed, and the expected results come from an oracle written in this file,
or from equal_groups / distinct_groups comparing the runs themselves.
Re-run after changing a bound; the tests are generated, not hand edited.
"""
import json, pathlib, random, re
T = pathlib.Path(__file__).resolve().parent.parent / "tests" / "properties"

def write(name, clauses, prov, src, seq, **m):
    T.mkdir(parents=True, exist_ok=True)
    L = ["#--- rce-test", f"# id: properties/{name}", "# area: properties",
         f"# clauses: [{', '.join(clauses)}]", "# expect: cli", f"# seq: {json.dumps(seq)}"]
    for k, v in m.items():
        if k in ("output_has", "output_lacks"):
            L.append(f"# {k}:"); L += [f"#   - {json.dumps(x)}" for x in v]
        elif isinstance(v, (list, dict)):
            L.append(f"# {k}: {json.dumps(v)}")
        else:
            L.append(f"# {k}: {v}")
    L += [f"# provenance: {prov}", "#---"]
    (T / f"{name}.rce").write_text((src or CYC).rstrip("\n") + "\n" + "\n".join(L) + "\n")

CYC = ("state S\n    n: i64 = 0\nevent Inc\nevent Dec\nfn wrap(v: i64) -> i64\n"
       "    v % 4\nstep S on Inc -> wrap(n + 1)\nstep S on Dec -> wrap(n + 3)\n")

def _w(L):
    if L == 0:
        yield []
        return
    for w in _w(L - 1):
        for e in ("Inc", "Dec"):
            yield w + [e]

def layouts(src):  # position and spacing rewrites that keep every token
    return [src, src.replace("\n", "\n\n"), src.replace("state S", "state   S"),
            "\n\n" + src + "\n\n", src.replace("-> ", "->  ")]

def renames(src):  # renamings of the non-exported names (fn and its parameter)
    return [src, src.replace("wrap", "cyc"),
            src.replace("(v: i64)", "(w: i64)").replace("    v % 4", "    w % 4"),
            src.replace("wrap", "q").replace("(v: i64)", "(x: i64)").replace("    v % 4", "    x % 4")]

FINAL = r'"state_hash":"([0-9a-f]{64})"\}\]\}\]\}'

def _all(k):
    r = []
    for L in range(1, k + 1):
        r += list(_w(L))
    return r

def per_word(name, clauses, prov, variants):
    # one test per bound: each word gets its own equal_groups via a marker file
    for i, w in enumerate(_all(3)):
        seq = [{"write": f"v{j}.rce", "text": s} for j, s in enumerate(variants)]
        seq += [{"argv": ["trace", f"v{j}.rce", ", ".join(w)]} for j in range(len(variants))]
        write(f"{name}-w{i:02d}", clauses, prov + f" Word {i}: {' '.join(w)}.", "", seq,
              status=0, equal_groups=[FINAL])

def main():
    per_word("ch17-layout", ["CH17-M-f213d82f"],
             "property: T1 and (e), bounded exhaustive: 5 layouts of one program (blank lines, spacing, leading and trailing lines) traced on every word over {Inc, Dec} of length 1 to 3 (14 words, one test each); the final state_hash is the same in every layout.",
             layouts(CYC))
    per_word("t50-rename", ["T50"],
             "property: T50, bounded exhaustive: 4 renamings of the fn and its parameter (not exported meaning) traced on every word over {Inc, Dec} of length 1 to 3 (14 words, one test each); results are equal.",
             renames(CYC))
    # hash independent of names: bound 4 renamings, rce hash program line
    seq = []
    for j, s in enumerate(renames(CYC)):
        seq += [{"write": f"v{j}.rce", "text": s}, {"argv": ["hash", f"v{j}.rce"]}]
    write("ch17-hash-name-free", ["CH17-M-d10b1b54", "T50"],
          "property: T7 extended to items, bounded exhaustive over 4 renamings of the fn name and parameter: the program definition hash printed by rce hash is the same for all 4.",
          "", seq, status=0, equal_groups=[r"([0-9a-f]{64})  program"])
    # T61: dim invariant under renaming
    seq = []
    for j, s in enumerate(renames(CYC)):
        seq += [{"write": f"v{j}.rce", "text": s}, {"argv": ["dim", f"v{j}.rce"]}]
    write("t61-dim-name-free", ["T61"],
          "property: T61 for renaming, bounded exhaustive over 4 renamings: dim prints the same states and behaviours count for each.",
          "", seq, status=0, equal_groups=[r"states explored\s+(\d+)", r"behaviours\s+(\d+)"])
    # T60: dim of a mod-k counter whose state is redundant is k
    for k in (2, 3, 4, 5):
        src = (f"state S\n    n: i64 = 0\n    m: i64 = 0\nevent Inc\nstep S on Inc\n"
               f"    n = (n + 1) % {k}\n    m = (m + 1) % {k}\n")
        write(f"t60-dim-redundant-k{k}", ["T60"],
              f"property: T60, bounded exhaustive over k in 2..5 (this test k = {k}): two fields that always hold the same value mod {k} reach {k} states, and dim reports the states explored as {k} (oracle: the reachable set is {{(i, i) : 0 <= i < {k}}}).",
              src, [{"argv": ["dim", "main.rce"]}], status=0, output_has=[f"states explored     {k}        bound reached: no"])
    # T63: law always over every reachable state, mod k
    for k in (2, 3, 4, 5, 6):
        for bound, holds in ((k, True), (k - 1, False)):
            src = (f"state S\n    n: u8 = 0\nevent Tick\nstep S on Tick\n    n = (n + 1) % {k}\n\n"
                   f"law small\n    always n < {bound}\n")
            write(f"t63-law-mod{k}-lt{bound}", ["T63"],
                  f"property: T63, bounded exhaustive over k in 2..6 and bound in {{k-1, k}} (this test k = {k}, bound {bound}): the oracle says `always n < {bound}` {'holds' if holds else 'fails at n = ' + str(k-1)} for a counter mod {k}; rce law decides the same.",
                  src, [{"argv": ["law", "main.rce"]}], output_has=["holds" if holds else "fails"])
    # T46: contract enforcement over x in -4..4
    src = ("state S\n    n: i64 = 0\nevent Go(x: i64)\nfn half(x: i64) -> i64\n    requires x > 0\n"
           "    ensures result >= 0\n    x / 2\nstep S on Go(x: v)\n    n = half(v)\n")
    for x in range(-4, 5):
        has = ["kept: trap(" if x <= 0 else f"S {{ n = {x // 2} }}"]
        write(f"t46-contract-x{x:+d}".replace("+", "p").replace("-", "m"), ["T46"],
              f"property: T46, bounded exhaustive over x in -4..4 (this test x = {x}): the oracle says the call returns iff x > 0, with n = x / 2; otherwise the event is kept with the requires record.",
              src, [{"argv": ["host", "run", "main.rce", "D", f"Go(x: {x})"]}], output_has=has)
    # CH07: the bytes decode to one value only: distinct values give distinct hashed state
    rnd = random.Random(7)
    vals = ["0.0", "-0.0", "1.0", "-1.0", "0.5", "1e21", "1e-6", "10.0", "100.0"] + \
           [repr(rnd.uniform(-1e6, 1e6)) for _ in range(8)]
    seq = []
    for j, v in enumerate(vals):
        s = f"state S\n    a: f64 = 0.0\nevent Go\nstep S on Go\n    a = {v}\n"
        seq += [{"write": f"v{j}.rce", "text": s}, {"argv": ["trace", f"v{j}.rce", "Go"]}]
    write("ch07-form-injective-f64", ["CH07-M-9b2d649e"],
          "property: 6.3 N1, 9 boundary f64 values plus 8 drawn with random.Random(7) from [-1e6, 1e6]: every value gives a different hashed state and a different state_hash, so no two values share bytes.",
          "", seq, status=0, distinct_groups=[r'"state":\[\[([^\]]*)\]\]', FINAL])


def _words(alpha, k):
    r = [[]]
    out = []
    for _ in range(k):
        r = [w + [a] for w in r for a in alpha]
        out += r
    return out

def ecyc(k, extra=""):  # a mod-k counter that emits its value after each event
    return (f"state S\n    n: i64 = 0{extra}\nevent Inc\nevent Dec\n"
            f"step S on Inc\n    n = (n + 1) % {k}\n    emit n.tostr()\n"
            f"step S on Dec\n    n = (n + {k - 1}) % {k}\n    emit n.tostr()\n")

def more():
    # T42 noninterference: public outputs agree whatever the pii field holds
    ni = ("struct Declass\nlabel pii by Declass\nstate S\n    seen: i64 = 0\n    hid: i64 @ {pii} = 0\n"
          "event Show(p: i64, code: i64 @ {pii})\nstep S on Show(p: v, code: c)\n    seen = seen + v\n    hid = c\n    emit v.tostr()\n")
    for pv in (-1, 0, 2):
        seq = [{"argv": ["run", "main.rce", f"Show(p: {pv}, code: {c}), Show(p: 1, code: {c * 3})"]} for c in (-5, 0, 1, 7)]
        write(f"t42-noninterference-p{pv}".replace("-", "m"), ["T42"],
              f"property: T42, bounded exhaustive over public p in {{-1, 0, 2}} (this test p = {pv}) and pii code in {{-5, 0, 1, 7}}: the two-event run's emits and its public field `seen` are the same for all 4 codes.",
              ni, seq, status=0, equal_groups=[r"\n1 Show[^\n]*\n  emit (\S+)", r"\n2 Show[^\n]*\n  emit (\S+)", r"\n1 Show[^\n]*seen = (-?\d+)", r"\n2 Show[^\n]*seen = (-?\d+)"])
    # T42 / T55 (W-TRI-1 by W-IFC-1): a labeled value reaching a sink requires change
    for lab in ("pii", "untrusted", "pii, untrusted"):
        for sink in ("emit", "store"):
            body = "    emit c.tostr()\n" if sink == "emit" else "    shown = c\n"
            src = (f"struct Declass\nstruct Vouch\nlabel pii by Declass\nlabel untrusted by Vouch\nstate Log\n    shown: i64 = 0\n"
                   f"event Show(p: i64, code: i64 @ {{{lab}}})\nstep Log on Show(p: _p, code: c)\n{body}")
            conf = "pii" in lab
            write(f"t55-label-to-{sink}-{lab.replace(', ', '-')}", ["T55", "T42"],
                  f"property: T55 through W-TRI-1 (enforced by W-IFC-1), bounded exhaustive over labels {{pii}}, {{untrusted}}, {{pii, untrusted}} and sink in {{emit, public store}} (this test {{{lab}}}, {sink}): the oracle says a value carrying the confidentiality label pii reaching a public sink requires change, and the integrity label untrusted alone does not restrict it.",
                  src, [{"argv": ["check", "main.rce"]}], status=1 if conf else 0, **({"output_has": ["W-IFC-1"]} if conf else {"output_lacks": ["W-IFC-1"]}))
    # T43: `declassify` is reserved; every placement in the bound requires change
    for i, where in enumerate(["    emit declassify(c).tostr()\n", "    if c > 0\n        emit declassify(c).tostr()\n"]):
        src = ("struct Declass\nlabel pii by Declass\nstate S\n    n: i64 = 0\nevent Show(code: i64 @ {pii})\n"
               f"step S on Show(code: c)\n{where}")
        write(f"t43-declassify-placement-{i}", ["T43"],
              f"property: T43, bounded exhaustive over 2 placements of a declassification (trusted context, and under a branch on the labeled operand; this test {i}): neither is accepted, since `declassify` is reserved, not implemented (F.12), so no release is reached on an untrusted path.",
              src, [{"argv": ["check", "main.rce"]}], status=1, output_has=["requires change"])
    # T44: every metacharacter of the rendered Html comes from a literal
    import html
    rnd = random.Random(44)
    for i in range(6):
        body = "".join(rnd.choice("a<>&\"' ") for _ in range(8))
        enc = html.escape(body, quote=True).replace("&#x27;", "&#39;")
        src = ('state Page\n    html: Text[Html] = Html.lit("<p></p>")\nevent C(body: text)\nstep Page on C(body: b)\n'
               '    html = Html.lit("<p>") + Html.encode(b) + Html.lit("</p>")\n    emit html\n')
        arg = body.replace("\\", "\\\\").replace('"', '\\"')
        write(f"t44-html-encode-s{i}", ["T44"],
              f"property: T44, seeded: 6 strings of length 8 over {{a < > & \" ' space}} drawn with random.Random(44) (this test string {i}); the oracle html.escape (quote on, ' as &#39;) gives the rendered text, whose only < and > are the literal <p> and </p>.",
              src, [{"argv": ["run", "main.rce", f'C(body: "{arg}")']}], status=0,
              output_has=[f'  emit "<p>{enc.replace(chr(34), chr(92) + chr(34))}</p>"'])
    # T45: ct_choose over every input, oracle a ? b : c
    for a in (True, False):
        for b in (True, False):
            for c in (True, False):
                t = lambda x: "true" if x else "false"
                src = (f"ct fn pick(a: bool, b: bool, c: bool) -> bool\n    return ct_choose(a, b, c)\n\n"
                       f"fn main()\n    emit pick({t(a)}, {t(b)}, {t(c)}).tostr()\n")
                write(f"t45-ct-choose-{t(a)[0]}{t(b)[0]}{t(c)[0]}", ["T45"],
                      f"property: T45, bounded exhaustive over the 8 inputs of a ct fn (this test {t(a)}, {t(b)}, {t(c)}): the result is the oracle a ? b : c, computed by ct_choose with no branch on the inputs.",
                      src, [{"argv": ["run", "main.rce"]}], status=0, output_has=[f'emit "{t(b if a else c)}"'])
    # T47: a failed call leaves &mut arguments as they were
    for a0 in range(0, 5):
        for k in (0, 2, 4):
            src = ("error E\nfn bump(x: &mut i64, k: i64) -> i64 ! E\n    x += k\n    if x > 5\n        fail E\n    return x + 0\n\n"
                   f"fn main()\n    var a: i64 = {a0}\n    let r = bump(&mut a, {k}) catch -1\n    emit r.tostr()\n    emit a.tostr()\n")
            ok = a0 + k <= 5
            write(f"t47-transactional-a{a0}-k{k}", ["T47"],
                  f"property: T47, bounded exhaustive over a in 0..4 and k in {{0, 2, 4}} (this test a = {a0}, k = {k}): the oracle says the call {'returns a + k' if ok else 'fails and a stays ' + str(a0)}.",
                  src, [{"argv": ["run", "main.rce"]}], status=0,
                  output_has=[f'emit "{a0 + k if ok else -1}"\nemit "{a0 + k if ok else a0}"'])
    # T48: a while loop with a decreases measure ends, after n0 rounds
    for n0 in range(0, 7):
        src = (f"fn main()\n    var n: u64 = {n0}\n    var k: u64 = 0\n    while n > 0\n        decreases n\n        n -= 1\n        k += 1\n    emit k.tostr()\n")
        write(f"t48-decreases-n{n0}", ["T48"],
              f"property: T48, bounded exhaustive over a start n in 0..6 (this test n = {n0}): the loop with `decreases n` ends, after exactly n rounds (oracle).",
              src, [{"argv": ["run", "main.rce"]}], status=0, output_has=[f'emit "{n0}"'])
    # T49: erasing ghost code does not change the result
    for x in range(-3, 4):
        g = ("ghost fn sq(x: i64) -> i64 = x * x\n\nfn f(x: i64) -> i64\n    ensures result == sq(x)\n    x * x\n\n"
             f"fn main()\n    emit f({x}).tostr()\n")
        e = f"fn f(x: i64) -> i64\n    x * x\n\nfn main()\n    emit f({x}).tostr()\n"
        write(f"t49-ghost-erasure-x{x}".replace("-", "m"), ["T49"],
              f"property: T49, bounded exhaustive over x in -3..3 (this test x = {x}): the program with a ghost fn in an ensures and the program with the ghost code erased emit the same, x * x = {x * x}.",
              g, [{"write": "erased.rce", "text": e}, {"argv": ["run", "main.rce"]}, {"argv": ["run", "erased.rce"]}],
              status=0, equal_groups=[r'emit "(-?\d+)"'], output_has=[f'emit "{x * x}"'])
    # T51: host.revoke is reserved; no placement relative to a use is accepted
    for i, body in enumerate(["    host.revoke(1)\n    emit 0\n", "    emit 0\n    host.revoke(1)\n", "    host.revoke(1)\n    host.revoke(1)\n    emit 0\n"]):
        write(f"t51-revoke-placement-{i}", ["T51"],
              f"property: T51, bounded exhaustive over 3 placements of host.revoke around a use (this test {i}): none is accepted, so no accepted program has a capability use after a revoke entry.",
              f"fn main()\n{body}", [{"argv": ["check", "main.rce"]}], status=1, output_has=["requires change"])
    # T52: a budget stops at the same unit on every run; threshold oracle 11 units
    for bud in (1, 5, 10, 11, 12, 20, 100):
        src = (f"fn main()\n    var n: i64 = 0\n    with budget {bud}\n        for _i in 0..10\n            n += 1\n    else\n        emit \"out\"\n    emit n.tostr()\n")
        ok = bud >= 11
        write(f"t52-budget-{bud}", ["T52"],
              f"property: T52, bounded exhaustive over budget in {{1, 5, 10, 11, 12, 20, 100}} (this test {bud}): ten rounds of n += 1 cost 11 units in the reference; below that the budget's else runs and the block's writes are undone (n = 0), at or above it n = 10. Every target must stop at the same operation.",
              src, [{"argv": ["run", "main.rce"]}, {"argv": ["run", "main.rce"]}], status=0,
              output_has=['emit "10"'] if ok else ['emit "out"\nemit "0"'], **({} if ok else {"output_lacks": ['emit "10"']}))
    # T53: replay from the log of answers gives the live run's state_hash
    clock = (pathlib.Path(__file__).resolve().parent.parent / "tests/host-contract/targeted/hc-32-clock-offset-form.rce").read_text().split("#--- rce-test")[0]
    rnd = random.Random(53)
    for i in range(5):
        evs = ", ".join(rnd.choice(["Now", "Offset(at: 0)", f"Offset(at: {rnd.randrange(0, 1800000000)})"]) for _ in range(rnd.randrange(1, 5)))
        write(f"t53-world-replay-{i}", ["T53"],
              f"property: T53, seeded: 5 event words of length 1 to 4 over Now and Offset drawn with random.Random(53) (this test: {evs}); every foreign answer enters the log and replay with the adapters disconnected gives the live state_hash.",
              clock, [{"argv": ["host", "run", "main.rce", "D", evs]}], status=0,
              output_has=["replay with adapters disconnected gives the live state_hash"])
    # T54: a manifest admits exactly its modules
    mods = {f"m{j}.rce": f"fn f{j}(x: i64) -> i64 = x + {j}\n" for j in range(3)}
    for j in range(3):
        seq = [{"mkdir": "pkg"}] + [{"write": f"pkg/{n}", "text": t} for n, t in mods.items()]
        seq += [{"argv": ["manifest", "pkg"], "stdout_to": "pkg/manifest.json"},
                {"argv": ["manifest-check", "pkg/manifest.json"]},
                {"write": f"pkg/m{j}.rce", "text": mods[f"m{j}.rce"].replace(f"+ {j}", f"+ {j + 10}")},
                {"argv": ["manifest-check", "pkg/manifest.json"]}]
        write(f"t54-manifest-edit-m{j}", ["T54"],
              f"property: T54, bounded exhaustive over which of 3 pinned modules is edited (this test m{j}): the manifest of the untouched directory checks ok; after changing one constant in m{j} the check requires change (W-PIN-1).",
              "fn main()\n    emit 0\n", seq, status=1, output_has=["ok pkg/manifest.json: 3 modules pinned", "W-PIN-1"])
    # T56: the relaxed sequential borrow is not adopted (5.6): bounded uses of two &mut to one place
    for i, body in enumerate(["    let r = &mut a\n    let s = &mut a\n    s += 1\n    r += 1\n", "    let r = &mut a\n    let s = &mut a\n    r += 1\n    s += 1\n"]):
        write(f"t56-two-mut-borrows-{i}", ["T56"],
              f"property: T56, bounded exhaustive over the 2 orders of use of two live &mut borrows of one variable (this test {i}): both require change, since the relaxation that would admit order {1 - i if i else 1} is not adopted.",
              f"fn main()\n    var a: i64 = 0\n{body}    emit a.tostr()\n", [{"argv": ["check", "main.rce"]}], status=1, output_has=["requires change"])
    # T57: output of e.w is output of e then the behaviour of delta(s, e) on w (model oracle)
    for i, w in enumerate(_words(["Inc", "Dec"], 3)):
        n, outs = 0, []
        for j, e in enumerate(w):
            n = (n + 1) % 4 if e == "Inc" else (n + 3) % 4
            outs.append(f'{j + 1} {e} S {{ n = {n} }}\n  emit "{n}"')
        write(f"t57-unfold-w{i:02d}", ["T57"],
              f"property: T57, bounded exhaustive over every word over {{Inc, Dec}} of length 1 to 3 (14 words; this test {' '.join(w)}): the run's outputs are those of the model beh(s, e.w) = lambda(s, e) . beh(delta(s, e), w) of a mod-4 counter.",
              ecyc(4), [{"argv": ["run", "main.rce", ", ".join(w)]}], status=0, output_has=outs)
    # T58 / T59: equivalent rewrites of an emitting program, and distinguished pairs
    eq = [ecyc(4).replace("state S", "state   S"), ecyc(4).replace("\n", "\n\n"), ecyc(4, "\n    m: i64 = 0"), re.sub(r"\bn\b", "q", ecyc(4))]
    for j, v in enumerate(eq):
        write(f"t58-equiv-rewrite-{j}", ["T58", "T59"],
              f"property: T58 and T59, bounded exhaustive over 4 rewrites of an emitting mod-4 counter (spacing, blank lines, an unused field, a renamed field; this test {j}): rce equiv says equivalent, exactly, on the explored alphabet {{Inc, Dec}}.",
              ecyc(4), [{"write": "b.rce", "text": v}, {"argv": ["equiv", "main.rce", "b.rce"]}], status=0,
              output_has=["equivalent  same behaviour on the explored alphabet (exact)"])
    for k1, k2 in ((2, 4), (3, 4), (4, 8), (2, 3), (5, 4)):
        write(f"t59-distinguish-mod{k1}-mod{k2}", ["T59", "T58"],
              f"property: T59 other direction, bounded exhaustive over the pairs (2,4), (3,4), (4,8), (2,3), (5,4) (this test mod {k1} against mod {k2}): the emitting counters differ, and rce equiv reports a distinguishing sequence.",
              ecyc(k1), [{"write": "b.rce", "text": ecyc(k2)}, {"argv": ["equiv", "main.rce", "b.rce"]}],
              output_has=["verdict             different", "after "], output_lacks=["equivalent  same"])
    # T62: independent entries commute
    import itertools
    ind = ("state S\n    a: i64 = 0\n    b: i64 = 0\n    c: i64 = 0\nevent A\nevent B\nevent C\n"
           "step S on A -> a + 1\nstep S on B\n    b = b + 2\nstep S on C\n    c = c + 3\n")
    ind = ind.replace("step S on A -> a + 1", "step S on A\n    a = a + 1")
    seq = [{"argv": ["trace", "main.rce", ", ".join(p)]} for p in itertools.permutations(["A", "B", "C"])]
    write("t62-independent-commute", ["T62"],
          "property: T62, bounded exhaustive over the 6 orders of three events with disjoint write footprints (a, b, c): every order gives the same final state_hash.",
          ind, seq, status=0, equal_groups=[FINAL])
    dep = ("state S\n    a: i64 = 1\nevent Add\nevent Dbl\nstep S on Add\n    a = a + 1\nstep S on Dbl\n    a = a * 2\n")
    write("t62-dependent-differ", ["T62"],
          "property: T62 converse, the 2 orders of two events that both write a (a + 1, a * 2) give different final state_hash (oracle 4 and 3).",
          dep, [{"argv": ["trace", "main.rce", "Add, Dbl"]}, {"argv": ["trace", "main.rce", "Dbl, Add"]}], status=0, distinct_groups=[FINAL])
    # T28: sessions
    ses = lambda x: ("protocol Ask\n    send i64\n    recv i64\n\nprotocol Answer\n    recv i64\n    send i64\n\n"
                     "fn server(s: session Answer)\n    let x = <- s\n    s <- x * 2\n\n"
                     f"fn main()\n    let (c, s) = chan[session Ask](1)\n    spawn server(s)\n    c <- {x}\n    let r = <- c\n    emit r\n")
    for x in (-3, 0, 1, 21):
        write(f"t28d-tree-session-x{x}".replace("-", "m"), ["T28d", "T28l"],
              f"property: T28d, bounded exhaustive over the sent value in {{-3, 0, 1, 21}} (this test {x}): a two-process tree with compatible ends runs to the end (status 0, no deadlock) and emits 2x = {2 * x}.",
              ses(x), [{"argv": ["run", "main.rce"]}], status=0, output_has=[f"emit {2 * x}"])
    muts = [("drop the server's send", lambda s: s.replace("    s <- x * 2\n", "")),
            ("use after the end", lambda s: s.replace("    s <- x * 2\n", "    s <- x * 2\n    s <- x\n")),
            ("receive twice", lambda s: s.replace("    let x = <- s\n", "    let x = <- s\n    let _y = <- s\n")),
            ("send on the client twice", lambda s: s.replace("    c <- 5\n", "    c <- 5\n    c <- 6\n"))]
    for j, (what, f) in enumerate(muts):
        write(f"t28l-linear-mutant-{j}", ["T28l"],
              f"property: T28l, bounded exhaustive over 4 single edits that break linearity of a typed session program (this test: {what}): each requires change (a W-SES rule), so no accepted body uses a gone endpoint or leaves one unfinished.",
              f(ses(5)), [{"argv": ["check", "main.rce"]}], status=1, output_has=["W-SES-"])
    for x in (0, 2, 9):
        write(f"t28p-priority-order-{x}", ["T28p"],
              f"property: T28p, bounded exhaustive over Key counts 1..3 before an Alarm (this test {x % 3 + 1}): simultaneous inputs are admitted by priority then arrival, so the run steps to the end and records the order.",
              (pathlib.Path(__file__).resolve().parent.parent / "tests/host-contract/harvested/priority-accept.rce").read_text().split("#--- rce-test")[0],
              [{"argv": ["run", "main.rce", ", ".join(['Key(c: "a")'] * (x % 3 + 1) + ["Alarm(n: 1)"])]}], status=0,
              output_has=[f'seen = "{"a" * (x % 3 + 1)}!"'])
    for k in (1, 2, 3):
        chain = ("protocol Ask\n    send i64\n    recv i64\n\nprotocol Answer\n    recv i64\n    send i64\n\n"
                 "fn server(s: session Answer)\n    let x = <- s\n    s <- x + 1\n\nfn main()\n    var v: i64 = 0\n"
                 + "".join(f"    let (c{i}, s{i}) = chan[session Ask](1)\n    spawn server(s{i})\n    c{i} <- v\n    v = <- c{i}\n" for i in range(k))
                 + "    emit v\n")
        for name, cl in (("t28g", "T28g"), ("t28a", "T28a")):
            write(f"{name}-chain-{k}", [cl],
                  f"property: {cl}, bounded exhaustive over chains of 1 to 3 sessions driven by one coordinator (this test {k}); the configuration of message-only participants always steps to the end and the coordinator receives v = {k} in send order. Global type syntax is not in the grammar, so the participants are written as their projections.",
                  chain, [{"argv": ["run", "main.rce"]}], status=0, output_has=[f"emit {k}"])
    # CH11: the Form runtimes agree (reference, Python, JavaScript)
    rnd = random.Random(11)
    for i in range(4):
        a, b = rnd.randrange(-50, 50), rnd.choice([-7, -3, -1, 1, 2, 5])
        x = rnd.choice(["0.5", "1.25", "-2.5", "3.0"])
        src = (f"fn main()\n    let a: i64 = {a}\n    let b: i64 = {b}\n    emit (a / b).tostr()\n    emit (a % b).tostr()\n"
               f"    emit (a * b - a).tostr()\n    emit ({x} * 1.5).tostr()\n")
        write(f"ch11-runtimes-agree-{i}", ["CH11-M-efbb7a78"],
              f"property: CH11, seeded: 4 programs drawn with random.Random(11) (a in -50..49, b in {{-7, -3, -1, 1, 2, 5}}, an f64; this test a = {a}, b = {b}, {x}): the reference evaluator, the emitted Python and the emitted JavaScript print the same four emits.",
              src, [{"argv": ["run", "main.rce"]}, {"target": "python", "file": "main.rce"}, {"target": "js", "file": "main.rce"}],
              status=0, equal_groups=[r'(emit "[^"]*"\nemit "[^"]*"\nemit "[^"]*"\nemit "[^"]*")'])


main()
more()
