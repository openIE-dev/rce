import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
# name, SO2025 used% (None = not a listed option), RedMonk Jan2026 rank (None = not top 20),
# TIOBE Oct2026 % (None = rank 51-100 or absent), PYPL Oct2026 % (None = not tracked),
# Innovation Graph 2026Q1 pushers (sum over economies), registry packages (source), latest release, date, releases in 12 mo,
# hard filter verdict, reason
R = [
 ("Haskell (GHC)",0.1,None,0.33,0.08,23803,19529,"Hackage index","GHC 9.14.1 (9.12.4 2025-12-27 newest on haskell.org news)","2025-12-27",None,"pass","Int64 + Integer checks; Double IEEE; NCG emits no implicit FMA; TCO; ordered Data.Map; GHC threads"),
 ("OCaml 5",1.2,None,0.61,None,12869,4682,"opam-repository packages/","5.5.1","2026-09-04",5,"pass","Int64 boxed + checks; float IEEE, no implicit FMA (Float.fma explicit); TCO; Map ordered; Domains"),
 ("F#",1.3,None,None,None,7936,None,"NuGet (not obtained)","F# 10 (with .NET 10)","2025-11",None,"pass","Checked ops; ECMA-335 permits wider intermediates (RyuJIT uses exact double); .tail calls; threads"),
 ("Scala 3",2.6,14,0.37,0.22,28457,9445,"Scaladex projects","3.10.0","2026-10-12",9,"pass","Long + Math.*Exact / BigInt checks; JVM strict IEEE (JEP 306, Java 17+); shortest Double.toString (JDK-4511638, JDK 19); no general TCO, mitigated: loops are emitted as while, recursion is measure-bounded, big-stack thread; JVM threads"),
 ("Elixir",2.7,None,None,None,20646,None,"Hex (not obtained)","v1.20.4","2026-08-28",14,"pass","Bignum ints + range checks; IEEE doubles, non-finite raises (RCE traps anyway); TCO; processes. Dynamic typing"),
 ("Gleam",1.1,None,None,None,618,1691,"packages.gleam.run","v1.19.2","2026-10-12",14,"pass","BEAM ints/floats as Elixir; TCO; no mutable vars, more lowering work"),
 ("Lean 4",None,None,None,None,4432,None,"Reservoir (not obtained)","v4.35.0","2026-10-12",21,"conditional","Float via C double compiled by clang; clang default -ffp-contract=on can fuse a*b+c; needs verified flags"),
 ("Clojure",0.0,None,None,None,16547,None,"Clojars (not obtained)","1.12.x stable (1.13.0-alpha9 2026-10-11)","2026-10-11",None,"pass","JVM: same float/int story as Scala; recur only, no TCO"),
 ("Elm",None,None,None,None,1675,None,"package.elm-lang.org (not obtained)","0.19.3","2026-10-02",2,"fail","Browser only: no stdout/process/threads to run observations or --run; Int is JS double"),
 ("PureScript",None,None,None,None,3590,None,"registry-index top-level 195 dirs","v0.15.16","2026-03-15",1,"conditional","JS backend: Int is 32-bit, needs BigInt FFI; single-threaded Aff"),
 ("Roc",None,None,None,None,None,None,"n/a","alpha4-rolling","2025-08-26",0,"fail","No stable release to pin as an oracle"),
 ("Unison",None,None,None,None,None,None,"Unison Share (not obtained)","1.5.0","2026-10-02",10,"conditional","Content-addressed codebase, no plain one-file program to emit and run"),
 ("Idris 2",None,None,None,None,126,None,"pack-db (not obtained)","v0.8.0","2025-10-31",1,"pass","Chez backend; Bits64 + Integer; Double IEEE; TCO"),
]
H=["language","so2025_used_pct","redmonk_jan2026_rank","tiobe_oct2026_pct","pypl_oct2026_pct","innovgraph_2026q1_pushers","registry_packages","registry_source","latest_release","latest_date","releases_last_12mo","hard_filter","feasibility_notes"]
# popularity: mean of per-signal ranks among filter survivors (missing value = worst rank)
surv=[r for r in R if r[11]=="pass"]
def rank(i, higher=True):
    vals=[(r[i] if r[i] is not None else (-1 if higher else 999)) for r in surv]
    order=sorted(range(len(surv)), key=lambda k: -vals[k] if higher else vals[k])
    rk=[0]*len(surv)
    for p,k in enumerate(order): rk[k]=p+1
    return rk
cols=[rank(1),rank(2,higher=False),rank(3),rank(4),rank(5)]
comp={surv[k][0]: sum(c[k] for c in cols)/len(cols) for k in range(len(surv))}
with open("candidates.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(H+["popularity_mean_rank"])
    for r in R: w.writerow(list(r)+[f"{comp[r[0]]:.2f}" if r[0] in comp else ""])
for k,v in sorted(comp.items(), key=lambda x:x[1]): print(f"{v:.2f} {k}")
names=[r[0] for r in surv]
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
s=sorted(names,key=lambda n:comp[n])
ax[0].barh(s[::-1],[comp[n] for n in s][::-1],color=["#c0392b" if n=="Scala 3" else "#7f8c8d" for n in s][::-1])
ax[0].set_xlabel("mean rank over 5 popularity signals (lower is more popular)"); ax[0].set_title("Popularity among candidates passing the hard filter")
pp=[r for r in R if r[5]]
ax[1].scatter([r[1] or 0 for r in pp],[r[5] for r in pp])
for r in pp: ax[1].annotate(r[0],(r[1] or 0,r[5]),fontsize=8)
ax[1].set_xlabel("Stack Overflow 2025: used extensively (%)"); ax[1].set_ylabel("GitHub Innovation Graph 2026Q1 pushers")
ax[1].set_title("Two independent usage signals")
plt.tight_layout(); plt.savefig("selection.png",dpi=130)
