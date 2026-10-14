import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=["identity_vs_value","null","effects","determinism_time","totality","numerics_bitexact","concurrency","placement_fabric","capability_security","code_identity","eval_reflection","perf_by_default"]
# R right, W wrong, M mixed, N n.a.; source tag
L={
"Futhark":("R N R R M N R R N N R R","L1,P-FUT"),
"Dex":("R N R R R N R M N N R M","P-DEX"),
"Halide":("R N R R R M R R N N R M","L2,P-HAL"),
"Koka":("R R R M R N M N M N R R","L3,P-KOK,P-PER"),
"Unison":("R R R M N N M R M R R N","L4,D-UNA"),
"Elm":("R R R R M N N N M N R N","D-ELM(partial)"),
"Roc":("R R R M N N N N M N R M","D-ROC(partial)"),
"Lean 4":("R R R R R N N N N M R N","P-LEAN"),
"Idris 2":("R R R R R N N N N N R N","P-IDR"),
"Pony":("M R N M N N R N R N R N","L5,P-PONY"),
"E/Monte":("M N M M N N R M R N R N","P-MIL"),
"SPARK/Ada":("M R R R R M M N N N R N","D-SPK"),
"Esterel":("R M M R R M R N N N R M","P-EST5 (primer pages): id p25,p48 one status+value per instant; null p49,p52 undefined value until emitted, access is error; eff p47 functions side-effect free but host procs unchecked, p92 unordered concurrent calls; det p11-12,p83; tot p58 static non-instantaneous loops, p73 no recursive run, p87 constructive check; num p43,p46 float/double as C++/Java, no implicit conversion; conc p26,p73 synchronous parallel, no shared writes; place p106 code shape free but no distribution; eval p73 static instantiation; perf p4-5,p106 v3 automata blew up, v5 circuit/equation code"),
"Lustre":("R M R R R M R M N N R M","P-LUS: id p1307 substitution; null p1307-8 nil from pre banned in clocks/outputs; eff p1306 no side effects; det p1306; tot p1311 no recursive nodes+cycles need pre; num p1307 real type unspecified; conc p1306 data-dep parallel; place p1317 OC2REP; eval p1311 static networks; perf p1316 automaton blowup"),
"Erlang/Elixir":("R M W M N N R M N M W N","P-ARM(partial)"),
"Datalog/Souffle":("R R R R R N M N N N R M","P-SOU,D-SOUR"),
"Dhall":("R R R R R N N N R R R N","D-DHL"),
"Zig":("W M W M N R W N N N M R","D-ZIG"),
"WebAssembly (det. profile)":("R N N M N M M M R N R N","F-WASM"),
"Mojo":("M M N N N M M M N N N M","D-MOJO"),
"Hylo/Val":("R R N N N N R N N N R M","P-HYL"),
"Swift (value types)":("M R N N N N M N N N M N","L6,P-HYL"),
"Rust":("M R M M N M R N N N R R","D-RST(partial)"),
"APL/BQN":("R N M M N N M M N N W N","D-BQN"),
"Clojure":("R W M M N N R N N N W N","D-CLJ"),
"Prolog":("R M W M W W W N N N W M","D-SWI,P-PRO (FY=Fifty Years TPLP22(6); PB=ProB TPLP22(5); ISO=Szabo&Szeredi ICLP06; TAU=TPLP24(1)): id FY p823 immutable terms, vars bind once; null FY p823 no null/uninit terms but unbound vars; eff FY p781,p834 assert/retract change resolution, no declarative semantics; det FY p780-781 ISO depth-first deterministic, but ISO p262 error order unspecified, p265 7//-3 impl-specific, PB p763 hashes differ across systems, PB p765 false pos/neg on 2nd system; tot FY p828 easy to loop; num FY p810 float repr unspecified, NaN/Inf/rounding differ, ISO p265 1/0 gives inf in some impls, PB p764 CLP(FD) overflow vs bignum; conc FY p812 ISO single-threaded only, p814 threads per tech rec with differences; eval FY p780 call/1 required, p817 read(X),call(X); perf FY p813 only first-arg indexing portable, PB p767 3x-15x slower, one 186x"),
"Bend/HVM":("R N M M N N R M N N N M","P-HVM2,D-HVM"),
"Java (Valhalla)":("W W W M N M M N N N W M","S1,S1b,S1c,F-306"),
"Python (GIL, 2->3)":("W W W W N M W N N N W W","S2"),
"C/C++":("W W W W W W W N N N M W","S3,S3b,F-CUDA"),
"JavaScript":("W W W M N W M N N N W W","D-ECMA"),
"Go (generics)":("M W W M N M M N N N M M","S4,S4b"),
"C# (nullable)":("M W W M N M M N N N W M","S5"),
"Scala 2->3":("W W M M N N M N N N W N","D-SC3(partial)"),
"Perl 6/Raku":("W W W M N N M N N N W N","D-RAKU(partial)"),
"PHP":("W W W M N W N N N N W N","D-PHP"),
"Julia":("M N W W N M W N N N W M","P-JUL: id paper p15 immutable types opt-in; eff manual @assume_effects is an unchecked assertion; det manual muladd 'can be different on the same machine', reduce associativity implementation-dependent, @spawn schedule nondeterministic, Random streams may change in minor releases, PROBE BLAS dgemm 3 distinct results over 1/2/4 threads, muladd contracts on native target but not generic; num manual Int wraps (modular), Float64 always RoundNearest, PROBE strict emitted code 0/200000 mismatches; conc manual manual locking; eval world age, @eval, invalidations blog; perf paper p14 type inference, manual @fastmath/@simd/@inbounds opt-in"),
"MATLAB":("R N N W N W W N N N N N","D-ML: id 'Avoid Unnecessary Copies' pass-by-value copy-on-write, handle classes opt-in; det MathWorks Answer 1871377 results differ by CPU and thread count; num 'Integers' saturating int arithmetic, int64 op double 'as if 80-bit extended', NaN->int 0; FP page IEEE RNE; conc maxNumCompThreads implicit multithreading default"),
"GNU Octave":("N N N N N W N N N N N N","D-OCT: num 4.4.2 integer ops saturate to nearest, ./ on ints rounds to nearest (not truncate), complex narrowed to real automatically (4)"),
"R":("M W W N N W N N N N W N","D-R: id R-ints 1.1.2 call-by-value illusion via refcount, <<- superassignment (R-lang 3.4.4); null R-lang 2.1.6 NULL plus typed NA 3.3.4; eff R-lang 4.3.3 lazy args may never be evaluated; num R-FAQ 7.31 'typically' 53 bits, equal only if same algorithm 'and not always even then'; eval R-lang 2.1.8 promises, substitute, simulated dynamic scope"),
"Mathematica/Wolfram":("N N N W N M N N N N N W","D-WL: det Compile 'number of times and order ... may be different', machine numbers 'vary slightly from one computer system to another'; num exact Integer/Rational, precision tracking, overflow auto-converts to arbitrary precision, 1/0 ComplexInfinity; perf Compile RuntimeOptions->Speed disables overflow checks (opt-in wrong)"),
"Fortran 2018/2023":("N N M W N W N N N N N N","D-F23 (J3/23-007r1): eff 15.7 PURE and 15.8 SIMPLE checked at compile time but optional; det 10.1.5.2.4 processor may evaluate any mathematically equivalent expression unless parenthesized, 16.9 SUM processor-dependent approximation (p448); num 17.1 IEEE modules' provision processor dependent, 17.4 initial rounding modes processor dependent; gfortran doc: full compliance needs -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans; GCC -ffp-contract=fast default outside ISO C"),
"NumPy (baseline)":("W N N W N W N N N N N N","D-NP: id 'Copies and views' views share the buffer; det np.sum partial pairwise, precision depends on axis, PROBE 7-element np.sum != left fold; num integer arithmetic modular, no error on overflow; float errors default warn, global seterr mode"),
}
code={"R":1,"M":0.5,"W":0,"N":np.nan}
with open("MATRIX.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["language"]+D+["source"])
    for k,(v,s) in L.items(): w.writerow([k]+[{"R":"right","W":"wrong","M":"mixed","N":"n.a."}[x] for x in v.split()]+[s])
M=np.array([[code[x] for x in v.split()] for v,_ in L.values()])
fig,ax=plt.subplots(figsize=(12,14))
cm=matplotlib.colormaps["RdYlGn"].copy(); cm.set_bad("#dddddd")
ax.imshow(np.ma.masked_invalid(M),cmap=cm,vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(len(D)),D,rotation=45,ha="right"); ax.set_yticks(range(len(L)),[f"{k} [{s.split(':')[0].split(' (')[0]}]" for k,(v,s) in L.items()])
ax.set_title("Fig 1. Language x RCE dimension\ngreen right, yellow mixed, red wrong, grey n.a.; author judgement",loc="left")
plt.tight_layout(); plt.savefig("fig1_heatmap.png",dpi=120)
# timeline: (label, start, end, source)
T=[("Java strictfp split (Java 2/1.2 Dec 1998 -> JEP 306 JDK 17 2021)",1998,2021,"F-306"),
   ("PHP 0==\"foobar\" fix (RFC vote Jul 2020)",2020,2020.6,"D-PHP"),
   ("Python 2->3 (3.0 final Dec 2008 -> 2.7.18 Apr 2020)",2008,2020,"PEP 361, PEP 373"),
   ("Python GIL removal (PEP 703 acc. 2023; default-off roadmap to 2028-30)",2023,2030,"PEP 703"),
   ("Java Valhalla ('Value Types for Java' Apr 2014 -> JEP 401 preview JDK 28, 2026)",2014,2026,"S1b, S1c"),
   ("Go generics (open source Nov 2009 -> 1.18 Mar 2022)",2009,2022,"S4, S4b"),
   ("C# nullable refs (C# 8 Sep 2019; migration ongoing)",2019,2026,"S5"),
   ("Swift ABI stability (Swift 1.0 Sep 2014 -> Swift 5 Feb 2019)",2014,2019,"L6"),
   ("ISO Prolog core (work began 1984 -> Part 1 1995)",1984,1995,"P-PRO FY p802"),
   ("ISO Prolog Part 2 modules (2000), largely ignored",2000,2022,"P-PRO FY p803, PB p757"),
   ("ProB SICStus->SWI port (8.4 Sep 2021; 6.4% tests still fail 2022)",2021,2022.5,"P-PRO PB p759,p765"),
   ("Chromium memory safety (70%, bugs since 2015)",2015,2026,"S3")]
fig,ax=plt.subplots(figsize=(12,5.5))
for i,(l,a,b,s) in enumerate(T):
    ax.barh(i,b-a,left=a,color="#c0504d"); ax.text(a+0.1,i,f"{l}  [{s}]",va="center",fontsize=8,color="black")
ax.set_yticks([]); ax.set_xlim(1983,2048); ax.invert_yaxis()
ax.set_title("Fig 2. Retrofit cost timelines (years of effort; some start dates are background knowledge, see READ-LEDGER)",loc="left",fontsize=10)
plt.tight_layout(); plt.savefig("fig2_retrofit_timeline.png",dpi=120)
