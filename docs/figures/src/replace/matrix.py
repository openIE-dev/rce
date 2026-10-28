#!/usr/bin/env python3
"""Replacement matrix: failure classes and workload domains per language
family, mapped to the RCE Draft 0.7 rule or theorem that excludes them, or
GAP. Regenerate: python3 matrix.py  (writes CSV, MD tables, SVG figure).
Status values:
  EXCLUDED  the class cannot be written in a well-formed RCE program (a W-rule refuses it, or there is no construct for it)
  DEFINED   the class is a defined, deterministic outcome (trap, kept outcome, logged outcome) instead of undefined or silent behaviour
  GAP       no RCE rule or theorem excludes or defines it today
"""
import csv, json, os
H = os.path.dirname(os.path.abspath(__file__))

# Shared failure classes: (id, name, primary source, status, RCE rule/theorem)
C = {
 "OOB":   ("Out-of-bounds read/write (CWE-125, CWE-787, CWE-119)", "CWE Top 25 2024 (view 1430)", "DEFINED", "6.3 N8 index trap; no pointers or pointer arithmetic (W-SAFE-1); T6"),
 "UAF":   ("Use after free / dangling reference (CWE-416)", "CWE Top 25 2024", "EXCLUDED", "affine ownership W-OWN-1/2/3; second-class borrows (decision 9); arenas W-REG-1, 5.7; T6"),
 "NULL":  ("Null dereference (CWE-476)", "CWE Top 25 2024", "EXCLUDED", "no null (decision 28); option T? with exhaustive match W-MATCH-1; 6.3 N10"),
 "IOVF":  ("Integer overflow or wraparound (CWE-190)", "CWE Top 25 2024", "DEFINED", "6.3 N2 checked arithmetic traps in every build (decision 38); explicit wrapping_/saturating_ (R4); T14"),
 "RACE":  ("Data race", "language memory model", "EXCLUDED", "tasks share only channels W-SPAWN-1; par footprints W-PAR-1; T16, T17, T1 over 6.8 ticks"),
 "UNINIT":("Read of uninitialized memory", "language UB list", "EXCLUDED", "every binding has an initializer (section 3 LetStmt/VarStmt); W-NAME-1; T4"),
 "DIV0":  ("Division by zero", "language UB list / runtime error", "DEFINED", "trap (decision 7); the event has the kept outcome, state unchanged (T8)"),
 "DOS":   ("Uncontrolled resource consumption (CWE-400)", "CWE Top 25 2024", "EXCLUDED", "bounded loops 6.7 and LOOP_LIMIT W-LOOP-1; N9 size limits; budgets C.6; arena bound R3; T2"),
 "NONTERM":("Nontermination / stack overflow from recursion", "language docs", "EXCLUDED", "structural recursion only W-REC-1; T2 (struct_total, calls_eq_size)"),
 "CODEINJ":("Code injection / eval (CWE-94)", "CWE Top 25 2024", "EXCLUDED", "no eval, reflection or JIT (10.1, roadmap 'rejected by derivation'); closed library 5.15"),
 "OSCMD": ("OS command injection (CWE-78, CWE-77)", "CWE Top 25 2024", "EXCLUDED", "no process or shell primitive: no ambient authority (5.15)"),
 "SQLI":  ("SQL injection (CWE-89)", "CWE Top 25 2024", "EXCLUDED", "P1 v0.8 5.16 Text[Sql]: only Sql.lit(literal) and Sql.encode; W-CTX-1 refuses plain text in query text; T36 ctx_meta_from_literals; commit 9ab62d3c1"),
 "XSS":   ("Cross-site scripting (CWE-79)", "CWE Top 25 2024", "EXCLUDED", "P1 v0.8 5.16 Text[Html]/Text[Url]: markup only from literals, untrusted text only via encode (W-CTX-1); T36; commit 9ab62d3c1"),
 "DESER": ("Deserialization of untrusted data (CWE-502)", "CWE Top 25 2024", "EXCLUDED", "inputs are typed event payloads (6.1, W-TY-1); no code in data; canonical JSON (decision 26)"),
 "PATH":  ("Path traversal / unrestricted upload / SSRF (CWE-22, CWE-434, CWE-918)", "CWE Top 25 2024", "EXCLUDED", "no file or network primitive (5.15); I/O only through events and logged extern (5.11)"),
 "AUTHZ": ("Missing/incorrect authentication or authorization (CWE-287, 306, 862, 863, 269, 352)", "CWE Top 25 2024", "EXCLUDED", "P1 v0.8 5.16 Cap[A]/Guarded[τ,A]: guarded state written only on events carrying the unforgeable, unstorable capability (W-CAP-1/2, T36 guarded_needs_cap); policy is what the program declares; commit 9ab62d3c1"),
 "SECRET":("Hard-coded credentials, information exposure (CWE-798, CWE-200)", "CWE Top 25 2024", "EXCLUDED", "P1 v0.8 5.16 Secret[τ]: no literal/constructor (no hard-coded credential), never in emit/channel/error/foreign (W-SEC-1); T36 noninterference up to equality bits; commit 9ab62d3c1"),
 "INPUTV":("Improper input validation (CWE-20)", "CWE Top 25 2024", "DEFINED", "P2 refined types 5.1 W-REF-1: payload outside `type R = τ where p` is refused at admission, state write breaking p refuses the event (T35, RCE/Refine.lean); commit aa0abd3ee"),
 "FFI":   ("Foreign code memory errors", "language FFI docs", "GAP", "extern trust boundary named (5.11), not excluded: T6 assumes the callee touches only its buffers"),
 "NONDET":("Nondeterministic scheduling / iteration order", "language docs", "EXCLUDED", "deterministic ticks 6.8 (decision 21), select rotation T11, ordered maps (decision 26), nondet choices logged T21"),
 "BUILDMEAN":("Meaning changes with build mode / optimizer", "language docs", "EXCLUDED", "no build profile changes meaning (decision 38); T9-F/T9-L one reference trace"),
 "DEADLOCK":("Deadlock", "language docs", "DEFINED", "Deadlock is a defined, logged outcome (decision 21); not prevented statically"),
 "TYPEDYN":("Runtime type error (dynamic typing)", "language docs", "EXCLUDED", "static typing W-TY-1..3; T4 preservation"),
 "GCPAUSE":("Nondeterministic memory reclamation (GC)", "language docs", "EXCLUDED", "no GC (roadmap: rejected by derivation); arenas and affine drops"),
 "FLOATORD":("Float reduction order varies by schedule", "CUDA / parallel docs", "EXCLUDED", "fixed reduction tree T18; f32 rounded once N3 (T13)"),
 "UNIONTAG":("Wrong union field / invalid enum discriminant", "language UB list", "EXCLUDED", "kinds are tagged, exhaustive match W-MATCH-1; no unions or transmute W-SAFE-1"),
 "ALIAS": ("Aliasing violation (strict aliasing, &mut uniqueness)", "language UB list", "EXCLUDED", "no raw pointers; borrow paths T6 (RCE/Borrow.lean)"),
 "EVALORD":("Unsequenced side effects / evaluation order", "C11 6.5p2, J.2", "EXCLUDED", "fixed left-to-right order, ANF pass preserves it (T34); T1"),
 "ODR":   ("Multiple conflicting definitions (ODR)", "C++ [basic.def.odr]", "EXCLUDED", "W-NAME-2, W-MOD-3, W-TRAIT-2 coherence"),
 "ITERINV":("Iterator/reference invalidation", "C++ container docs", "EXCLUDED", "values never mutated through a live borrow; loop elements are borrows W-OWN-3"),
 "NUMPREC":("Silent precision loss of integers (Number > 2^53)", "ECMA-262 Number type", "EXCLUDED", "exact integer types (5.1), no implicit conversion W-TY-3"),
 "COERCE":("Implicit type coercion in equality/arithmetic", "ECMA-262 abstract equality", "EXCLUDED", "6.3 N4 no mixed arithmetic, N6 structural equality, W-TY-3"),
 "PROTO": ("Prototype pollution (CWE-1321)", "CWE", "EXCLUDED", "no prototypes, no dynamic fields; struct fields fixed (W-NAME-1)"),
 "UNSOUNDTS":("Type system unsoundness (any, bivariance, unchecked casts)", "TypeScript handbook (type compatibility)", "EXCLUDED", "one checker, no any or cast between unrelated types; T4"),
 "UNSAFEIO":("Escape hatches (unsafePerformIO, Obj.magic, unsafe, transmute)", "GHC base docs; OCaml Obj; Rust reference", "EXCLUDED", "W-SAFE-1 refuses escape hatches; effects by context W-EFF-1"),
 "PARTIAL":("Partial functions / non-exhaustive patterns", "Haskell 2010 report", "EXCLUDED", "W-MATCH-1 exhaustive match; Map get returns V ! Missing (decision 34)"),
 "LAZYLEAK":("Space leaks from lazy evaluation", "GHC docs", "EXCLUDED", "strict evaluation (section 6); N9; budgets C.6"),
 "STMIO": ("Side effects inside retried transactions", "Clojure refs docs", "EXCLUDED", "W-ATOM-1 forbids waits in atomic; atomic is all-or-none T30"),
 "MACRO": ("Unhygienic macros", "Lisp docs", "EXCLUDED", "no macros (decision 28)"),
 "ATOMS": ("Atom table / unbounded mailbox exhaustion", "Erlang system limits", "EXCLUDED", "no dynamic atoms; bounded channels with Backpressure (decision 21)"),
 "PARTITION":("Network partition and distributed consistency", "Erlang distribution docs", "GAP", "node placement and replicated state are models (T31 'close to trivial'); no network failure semantics"),
 "PARTIALDEF":("partial def / unchecked nontermination in proof languages", "Lean reference, recursive definitions", "EXCLUDED", "W-REC-1 refuses non-structural recursion; T2"),
 "DEPTYPE":("Proof obligations and dependent specifications not expressible", "Lean/Coq docs", "DEFINED", "P2 refined types 5.1 W-REF-1 (decidable fragment, T35); obligation discharged, or enforced at run time, undecidable dependent types excluded by derivation (soundness/determinism); commit aa0abd3ee"),
 "EXTRACT":("Trust in extraction to another language", "Coq extraction manual", "GAP", "T9-L proved for Rust on a core only; Python/JS emitters tested (roadmap W7)"),
 "SQLNULL":("Three-valued NULL logic", "SQLite quirks; SQL standard", "EXCLUDED", "no null; option explicit N10"),
 "ISOL":  ("Isolation anomalies (serialization anomaly)", "PostgreSQL transaction isolation docs", "EXCLUDED", "events applied atomically and serially by delta (T8, 6.4)"),
 "FLEXTYPE":("Flexible column typing", "SQLite quirks", "EXCLUDED", "static typing of state fields W-STATE-1..3"),
 "SHMRACE":("Shared-memory race between threads in a block", "CUDA programming guide", "EXCLUDED", "W-GRP-3 lanes may not read another lane's slot in a phase; T23"),
 "BARRIER":("Divergent barrier", "CUDA programming guide", "EXCLUDED", "W-GRP-2: barrier only at the top level of a group body"),
 "GPUREAL":("Actual accelerator execution", "CUDA programming guide", "GAP", "device regions run on CPU in emitters (9.4 open question)"),
 "MOJOPTR":("UnsafePointer misuse", "Mojo manual (ownership)", "EXCLUDED", "no raw pointers W-SAFE-1"),
 "GILFREE":("Races after free-threaded (no-GIL) Python", "Python glossary (GIL)", "EXCLUDED", "W-SPAWN-1 share nothing; T16"),
 "RELFAST":("ReleaseFast turns safety-checked illegal behaviour into UB", "Zig language reference (Illegal Behavior)", "EXCLUDED", "decision 38: no build mode changes meaning"),
 "UNREACH":("Reaching unreachable code", "Zig reference; C23 unreachable", "EXCLUDED", "no unreachable primitive; exhaustive match W-MATCH-1/2"),
 "NEGCAST":("Out-of-range conversion (negative to unsigned, float to int)", "Zig reference; C11 6.3.1.4", "DEFINED", "`as` conversions range-checked, trap (6.3 N2)"),
 "PANIC": ("Panic/exception unwinding through frames", "Rust reference; Java exceptions", "EXCLUDED", "no exceptions or unwinding (decision 28, 36); typed error sets with try/catch (decision 15)"),
 "LEAK":  ("Reference-count cycles / leaks", "Rust Rc docs", "EXCLUDED", "no Rc/Arc W-SAFE-1; value semantics (decision 10)"),
 "REFLECT":("Reflection, ClassCastException", "JLS 5.5 / 15.16", "EXCLUDED", "no reflection; no downcasts; T4"),
 "GOROUT":("Goroutine/task leak", "Go memory model", "EXCLUDED", "structured spawn W-SCOPE-1 (0.6); scope ends after every child T26 (model)"),
 "SELECTRAND":("Uniform random select choice", "Go spec, select statements", "EXCLUDED", "select rotation 6.8, T11; nondet only logged T21"),
 "MAPORDER":("Randomized map iteration / hash seed", "Go spec; Python PYTHONHASHSEED", "EXCLUDED", "maps ordered (decision 26)"),
}
L = {
 "Python": ["TYPEDYN","NONTERM","CODEINJ","OSCMD","DESER","SQLI","XSS","GILFREE","MAPORDER","FFI","IOVF","DOS","AUTHZ","INPUTV","GCPAUSE","PATH"],
 "C": ["OOB","UAF","NULL","IOVF","RACE","UNINIT","DIV0","ALIAS","EVALORD","UNIONTAG","NEGCAST","OSCMD","DOS","SQLI","AUTHZ","SECRET","INPUTV","BUILDMEAN","FFI"],
 "C++": ["OOB","UAF","NULL","IOVF","RACE","UNINIT","DIV0","ALIAS","EVALORD","ODR","ITERINV","PANIC","UNIONTAG","LEAK","DOS","AUTHZ","INPUTV","BUILDMEAN"],
 "Rust": ["RACE","UAF","ALIAS","UNIONTAG","UNSAFEIO","PANIC","LEAK","DEADLOCK","IOVF","BUILDMEAN","FFI","DOS","AUTHZ","INPUTV"],
 "Zig": ["OOB","NULL","IOVF","DIV0","UNREACH","NEGCAST","UNIONTAG","RELFAST","UAF","RACE","ALIAS","DOS","FFI","AUTHZ"],
 "Mojo": ["MOJOPTR","UAF","RACE","SHMRACE","IOVF","OOB","GPUREAL","FFI","DOS","AUTHZ"],
 "Go": ["RACE","NULL","IOVF","GOROUT","SELECTRAND","MAPORDER","DEADLOCK","OOB","DOS","SQLI","AUTHZ","INPUTV","GCPAUSE"],
 "Java/JVM": ["RACE","NULL","IOVF","REFLECT","DESER","GCPAUSE","DIV0","OOB","DOS","SQLI","XSS","AUTHZ","SECRET","INPUTV"],
 "JS/TS": ["NUMPREC","COERCE","PROTO","UNSOUNDTS","CODEINJ","XSS","DESER","TYPEDYN","DOS","AUTHZ","INPUTV","NONDET"],
 "Haskell": ["IOVF","PARTIAL","NONTERM","UNSAFEIO","LAZYLEAK","GCPAUSE","DEPTYPE","FFI"],
 "OCaml": ["UNSAFEIO","RACE","NONTERM","IOVF","PANIC","PARTIAL","GCPAUSE","FFI"],
 "Lisp/Clojure": ["TYPEDYN","NULL","MACRO","STMIO","CODEINJ","GCPAUSE","NONTERM","DOS"],
 "Erlang/Elixir": ["ATOMS","TYPEDYN","NONDET","PARTITION","DEADLOCK","DOS","CODEINJ","DESER"],
 "Lean/Coq (as program languages)": ["PARTIALDEF","DEPTYPE","EXTRACT","UNSAFEIO","GCPAUSE"],
 "SQL": ["SQLNULL","ISOL","FLEXTYPE","SQLI","AUTHZ","INPUTV","DOS"],
 "CUDA": ["SHMRACE","BARRIER","OOB","FLOATORD","IOVF","GPUREAL","UAF","NULL"],
}
SRC = {
 "Python":"docs.python.org (pickle, ctypes, glossary)","C":"ISO C11 N1570 Annex J.2","C++":"cppreference: undefined behavior","Rust":"Rust Reference: behavior considered undefined",
 "Zig":"Zig 0.14 language reference: Illegal Behavior","Mojo":"Mojo manual: ownership","Go":"Go memory model; Go spec","Java/JVM":"JLS SE21 ch.15, ch.17",
 "JS/TS":"ECMA-262 2024; TypeScript handbook","Haskell":"Haskell 2010 report ch.6; GHC System.IO.Unsafe","OCaml":"OCaml 5.2 manual: memory model, Obj",
 "Lisp/Clojure":"clojure.org reference: refs","Erlang/Elixir":"erlang.org system limits (fetch failed, see ledger)","Lean/Coq (as program languages)":"Lean reference: recursive definitions; Rocq extraction (fetch failed)",
 "SQL":"SQLite quirks; PostgreSQL transaction isolation","CUDA":"CUDA C programming guide"}
# Workload domains: (language, domain, example path or None, RCE construct / gap reason)
EX = "crates/rce/spec/examples/"
D = [
 ("Python","scripting and text processing",EX+"20_strings.rce","text methods 5.15"),
 ("Python","data pipelines (map/filter/fold)",EX+"24_lambda.rce","closures, list methods"),
 ("Python","numeric and scientific",EX+"19_floats.rce","f64 N3"),
 ("Python","machine learning (tensors, autodiff)",None,"no tensor type, no autodiff transform, no accelerator lowering"),
 ("Python","web services and HTTP",None,"no network I/O (5.15); only events and extern"),
 ("C","systems and device handles",EX+"48_linear_handle.rce","linear structs 5.12, extern 5.11"),
 ("C","C ABI interop",EX+"40_extern.rce","extern \"C\" (5.11)"),
 ("C","OS kernels, drivers, MMIO",None,"no raw memory, volatile or interrupt vectors beyond C.11.12"),
 ("C","embedded real-time control",EX+"17_traffic_light.rce","states and steps; deadlines C.11.12"),
 ("C++","generic programming",EX+"09_generics.rce","monomorphized generics, traits 5.9"),
 ("C++","game and simulation state",EX+"30_state_machine_guarded.rce","guarded steps"),
 ("C++","bit manipulation",EX+"27_bitops.rce","bitwise ops"),
 ("Rust","ownership-checked systems code",EX+"12_borrow.rce","second-class borrows"),
 ("Rust","arena allocation",EX+"41_arena_bound_wrapping.rce","arenas with bound R3"),
 ("Rust","compilers and parsers (self-hosting)",None,"W-REC-1 refuses mutual recursion; no file input as events; roadmap W1"),
 ("Rust","cryptography and hashing","rce-replace/examples/sha256_chain.rce","sha256 text method (this change)"),
 ("Zig","comptime metaprogramming",EX+"32_comptime_block.rce","comptime 5.10"),
 ("Zig","fixed-width unsigned arithmetic",EX+"46_u64.rce","full-range u64 N2"),
 ("Mojo","SIMD lanes",EX+"43_simd_lanes.rce","lane-wise arrays W-SIMD-1"),
 ("Mojo","accelerator kernels",EX+"44_device_region.rce","device regions (CPU meaning only)"),
 ("Mojo","GPU code generation",None,"emitters run device regions on CPU (9.4)"),
 ("Go","channels and tasks",EX+"13_spawn_chan.rce","spawn, chan 5.8"),
 ("Go","select over channels",EX+"47_select_fair.rce","select rotation 6.8"),
 ("Go","network servers",None,"no network I/O (5.15)"),
 ("Java/JVM","enterprise state and records",EX+"04_bank_account.rce","states, events, errors"),
 ("Java/JVM","inventory/business rules",EX+"31_inventory.rce","maps and steps"),
 ("JS/TS","UI state machines",EX+"17_traffic_light.rce","rce emit js (T9-L)"),
 ("JS/TS","browser DOM and events",None,"no DOM binding; events only through host"),
 ("Haskell","algebraic data and pattern matching",EX+"05_kind_match.rce","kinds, exhaustive match"),
 ("Haskell","structural recursion over trees",EX+"39_structural_recursion.rce","W-REC-1 structural"),
 ("Haskell","streams and generators",EX+"50_iterators.rce","push iterators 5.14"),
 ("OCaml","nested variants",EX+"29_nested_kind.rce","kinds"),
 ("OCaml","option and error handling",EX+"07_error_union.rce","error sets"),
 ("Lisp/Clojure","persistent maps",EX+"21_maps.rce","ordered maps"),
 ("Lisp/Clojure","transactional state",EX+"../conc06b/accept/04_scope_atomic.rce","atomic T30"),
 ("Erlang/Elixir","actors",EX+"15_actor.rce","actors with mailboxes"),
 ("Erlang/Elixir","supervision",EX+"../conc06b/accept/05_supervisor_lvar.rce","supervisor C.11.11 (model)"),
 ("Erlang/Elixir","distribution over nodes",EX+"../conc06b/accept/03_distributed.rce","node placement; partitions GAP"),
 ("Erlang/Elixir","real network distribution",None,"no network failure semantics"),
 ("Lean/Coq (as program languages)","total functional programs",EX+"39_structural_recursion.rce","structural recursion, T2"),
 ("Lean/Coq (as program languages)","proof-carrying programs",EX+"51_refined_input.rce","refined types 5.1 W-REF-1, T35 (decidable fragment)"),
 ("SQL","keyed state updates",EX+"21_maps.rce","Map methods"),
 ("SQL","declarative queries, joins, indexes",None,"no query language; only comprehensions over lists/maps"),
 ("CUDA","parallel reduction",EX+"../conc06/accept/02_par_for_reduce.rce","for ... par, fixed tree T18"),
 ("CUDA","workgroups with barriers",EX+"../conc06b/accept/01_workgroup.rce","group/barrier C.11.2"),
 ("CUDA","device kernel code generation",None,"no PTX/SPIR-V emitter"),
]

def run_results():
    p = os.path.join(H, "run-results.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def main():
    rr = run_results()
    rows = []
    for lang, ids in L.items():
        for i in ids:
            n, src, st, rule = C[i]
            rows.append((lang, i, n, src if src != "language UB list" and src != "language docs" else SRC[lang], st, rule))
    with open(os.path.join(H, "failure-classes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["language","class_id","failure_class","primary_source","status","rce_rule_or_theorem"]); w.writerows(rows)
    drows = []
    for lang, dom, ex, why in D:
        r = rr.get(ex, "") if ex else ""
        st = "GAP" if ex is None else {"ok": "RUNS", "check-only": "CHECKS"}.get(r, "NOT-RUN" if r == "" else "FAILS")
        drows.append((lang, dom, ex or "", st, why))
    with open(os.path.join(H, "domains.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["language","domain","rce_example","status","rce_construct_or_gap"]); w.writerows(drows)
    stats = {}
    for lang in L:
        s = {"EXCLUDED":0,"DEFINED":0,"GAP":0}
        for r in rows:
            if r[0]==lang: s[r[4]]+=1
        s["DOMAIN_RUNS"] = sum(1 for d in drows if d[0]==lang and d[3]=="RUNS")
        s["DOMAIN_GAP"] = sum(1 for d in drows if d[0]==lang and d[3]=="GAP")
        stats[lang]=s
    json.dump(stats, open(os.path.join(H,"summary.json"),"w"), indent=1)
    svg(stats)
    return rows, drows, stats

def svg(stats):
    W, rowh, x0 = 900, 26, 250
    hgt = 70 + rowh*len(stats) + 40
    mx = max(sum(v[k] for k in ("EXCLUDED","DEFINED","GAP")) for v in stats.values())
    sc = (W - x0 - 140) / mx
    col = {"EXCLUDED":"#2b7a3d","DEFINED":"#7fb069","GAP":"#c0392b"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{hgt}" font-family="Helvetica,Arial,sans-serif" font-size="12">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="10" y="22" font-size="15" font-weight="bold">RCE Draft 0.7 replacement matrix: failure classes per language family</text>',
         '<text x="10" y="42" fill="#444">bars: classes excluded / defined / GAP; right: domains with a running RCE example vs GAP</text>']
    y = 60
    for lang, v in stats.items():
        o.append(f'<text x="{x0-8}" y="{y+16}" text-anchor="end">{lang.replace("&","&amp;")}</text>')
        x = x0
        for k in ("EXCLUDED","DEFINED","GAP"):
            w = v[k]*sc
            if w: o.append(f'<rect x="{x:.1f}" y="{y+3}" width="{w:.1f}" height="{rowh-8}" fill="{col[k]}"/><text x="{x+w/2:.1f}" y="{y+16}" fill="white" text-anchor="middle" font-size="11">{v[k]}</text>')
            x += w
        o.append(f'<text x="{W-130}" y="{y+16}">domains {v["DOMAIN_RUNS"]} run / {v["DOMAIN_GAP"]} GAP</text>')
        y += rowh
    lx = x0
    for k in ("EXCLUDED","DEFINED","GAP"):
        o.append(f'<rect x="{lx}" y="{y+12}" width="12" height="12" fill="{col[k]}"/><text x="{lx+16}" y="{y+22}">{k}</text>'); lx += 110
    o.append('</svg>')
    open(os.path.join(H,"replacement-matrix.svg"),"w").write("\n".join(o)+"\n")
    import shutil, subprocess
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-z", "1.5", "-o", os.path.join(H, "replacement-matrix.png"), os.path.join(H, "replacement-matrix.svg")], check=True)

if __name__ == "__main__":
    rows, drows, stats = main()
    print(json.dumps(stats))
    import runpy; runpy.run_path(os.path.join(H, "make_md.py"), run_name="__main__")
