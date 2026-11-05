#!/usr/bin/env python3
"""Write the property suites under tests/properties/.

Each suite is one seq test over a stated bound: every value or rewrite in
the bound is enumerated here (bounded exhaustive), or drawn from a fixed
seed, and the expected results come from an oracle written in this file,
or from equal_groups / distinct_groups comparing the runs themselves.
Re-run after changing a bound; the tests are generated, not hand edited.
"""
import json, pathlib, random
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

main()
