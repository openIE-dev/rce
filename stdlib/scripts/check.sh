#!/bin/sh
# Run the `check` lines of every library. pq, hpke and mls build on
# crypto, so each is tested with a `#!lib crypto` line in front (the rce
# binary appends its embedded crypto, without crypto's own checks).
# Set RCE to the rce binary (default: rce on PATH).
# The reference checker embeds the library and honours the `#!lib` directive
# (specification 4.1). An older rce without it is detected by a probe, and
# scripts/libexpand.py then inlines lib/crypto.rce in place of the
# directive, so the same checks run.
set -e
RCE=${RCE:-rce}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
printf 'module lib_probe\n#!lib crypto\nfn f() -> List[i64]\n    return cat([1], [2])\n' > "$tmp/lib_probe.rce"
if "$RCE" check "$tmp/lib_probe.rce" > /dev/null 2>&1; then RCE_NO_LIB=0; else RCE_NO_LIB=1; echo "check.sh: this rce has no #!lib directive; crypto is inlined (scripts/libexpand.py)"; fi
export RCE_NO_LIB
rm -f "$tmp/lib_probe.rce"
# ONLY="col core ..." restricts the run to those lib/<area>/ directories
# ("crypto" names the top-level crypto, pq, hpke and mls modules), so CI can
# split the libraries over parallel jobs; unset, everything runs.
only() { [ -z "${ONLY:-}" ] || case " $ONLY " in *" $1 "*) ;; *) return 1 ;; esac; }
# The library is pinned by hash (specification F.5.3): lib/MANIFEST.json lists
# every module with the SHA-256 of its canonical text, and must be current.
# Regenerate with `rce manifest lib > lib/MANIFEST.json` after a change.
"$RCE" manifest lib > "$tmp/MANIFEST.json"
if ! cmp -s "$tmp/MANIFEST.json" lib/MANIFEST.json; then echo "check.sh: lib/MANIFEST.json is not current; run: rce manifest lib > lib/MANIFEST.json" >&2; exit 1; fi
"$RCE" manifest-check lib/MANIFEST.json
only crypto && "$RCE" test lib/crypto.rce
for n in pq hpke mls; do
  only crypto || continue
  { echo "module ${n}_checks"; echo "#!lib crypto"; grep -v '^module ' "lib/$n.rce"; } | python3 scripts/libexpand.py > "$tmp/$n.rce"
  "$RCE" test "$tmp/$n.rce"
done
# Foundation modules under lib/<area>/: each is tested alone (it must admit
# as a standalone module) and then with its checks file appended (vectors
# and property sweeps live in tests/<area>/<name>_checks[_N].rce so a library's
# own comptime fuel stays small for programs that use it).
# A module with a "# uses: a.b, c.d" line is checked as a package together
# with lib/a/b.rce and lib/c/d.rce. scripts/uses.py parses that marker
# strictly: a malformed or unknown entry stops the run with an error. A tests/<area>/<name>_host.rce file
# holds checks too heavy for comptime fuel: it is flattened with the module
# and its first checks file (scripts/flatten.py), its "# events:" are fed
# to `rce host run` one per input line, and the final state must read
# "good = N, bad = 0" for the N events.
python3 scripts/uses.py --selftest
python3 scripts/uses.py --all
for f in lib/*/*.rce; do
  only "$(basename "$(dirname "$f")")" || continue
  uses=$(python3 scripts/uses.py "$f") || { echo "check.sh: bad dependency marker in $f" >&2; exit 1; }
  pk="$tmp/pkg"; rm -rf "$pk"; mkdir "$pk"
  for u in $uses; do cp "lib/$u.rce" "$pk/"; done
  if [ -n "$uses" ]; then python3 scripts/libexpand.py < "$f" > "$pk/$(basename "$f")"; "$RCE" check "$pk"; else python3 scripts/libexpand.py < "$f" > "$tmp/one.rce"; "$RCE" check "$tmp/one.rce"; fi
  t="tests/${f#lib/}"; t="${t%.rce}_checks"
  # one comptime fuel budget per file: heavy vector sets are sharded; a
  # module without dependencies runs its shards in parallel (JOBS, default
  # the number of CPUs), each in its own file, and any failing shard fails
  for c in "$t".rce "$t"_*.rce; do
    [ -f "$c" ] || continue
    if [ -n "$uses" ]; then
      o="$pk/$(basename "$f")"
      { cat "$f"; echo; cat "$c"; } | python3 scripts/libexpand.py > "$o"
      "$RCE" test "$pk"
    else
      echo "$c"
    fi
  done > "$tmp/shards"
  if [ -z "$uses" ] && [ -s "$tmp/shards" ]; then
    mkdir -p "$tmp/sh"
    F="$f" T="$tmp/sh" xargs -P "${JOBS:-$(nproc)}" -I{} sh -c 'o="$T/$(basename "{}")"; { cat "$F"; echo; cat "{}"; } | python3 scripts/libexpand.py > "$o"; "$0" test "$o" > "$o.out" 2>&1 || { cat "$o.out"; exit 255; }; rm -f "$o" "$o.out"' "$RCE" < "$tmp/shards"
    echo "ok $f: $(wc -l < "$tmp/shards") check files"
  else
    cat "$tmp/shards"
  fi
  h="${t%_checks}_host.rce"
  # energy_host.rce seals receipts: Ed25519 signing in RCE source exceeds the
  # per-event fuel of the host (spec 6.7), so it needs an rce with the native
  # ed_* builtins (rce-ide 0.1.3 or later). Reported, never silently passed.
  if [ -f "$h" ] && [ "$RCE_NO_LIB" = 1 ] && [ "$h" = tests/energy/energy_host.rce ]; then
    echo "SKIPPED $h: requires an rce with native Ed25519 (rce-ide 0.1.3 or later); this rce refuses event Seal0 with 'comptime fuel exhausted'"
    continue
  fi
  if [ -f "$h" ]; then
    python3 scripts/flatten.py "$f" "$t.rce" "$h" | python3 scripts/libexpand.py > "$tmp/host.rce"
    # "# events: A, B" lists events without arguments; "# event: E" lines
    # give one event each (with arguments). "# host-flags: --fs" grants a
    # fresh empty file root.
    if grep -q "^# event: " "$h"; then ev=$(sed -n "s/^# event: //p" "$h"); else ev=$(sed -n "s/^# events: //p" "$h" | tr -d ' ' | tr ',' '\n'); fi
    hf=""; if grep -q "^# host-flags: --fs" "$h"; then rm -rf "$tmp/fs"; mkdir "$tmp/fs"; hf="--fs $tmp/fs"; fi
    n=$(printf '%s\n' "$ev" | grep -c .)
    res=$(printf '%s\n' "$ev" | "$RCE" host run "$tmp/host.rce" --log "$tmp/hostlog" $hf)
    printf '%s\n' "$res" | tail -3
    printf '%s\n' "$res" | grep -q "good = $n, bad = 0 }" || { echo "FAIL host checks $h"; exit 1; }
    rm -rf "$tmp/hostlog"*
  fi
done
# A tests/<area>/<name>_probes.py script checks the rce checker itself
# against a rule table of the spec (e.g. the effect contexts of 5.2).
for p in tests/*/*_probes.py; do
  [ -f "$p" ] || continue
  RCE="$RCE" python3 "$p" | tail -1
  RCE="$RCE" python3 "$p" > /dev/null || { echo "FAIL probes $p"; exit 1; }
done
