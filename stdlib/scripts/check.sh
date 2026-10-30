#!/bin/sh
# Run the `check` lines of every library. pq, hpke, mls and the MLS modules
# (treekem, framing, protect, welcome, grp) build on
# crypto and import it with `use crypto`; every module of lib/ is built
# into the rce binary (specification 4.1), so each is tested by itself.
# Set RCE to the rce binary (default: rce on PATH).
set -e
RCE=${RCE:-rce}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
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
for n in pq hpke mls treekem framing protect welcome grp; do
  only crypto || continue
  "$RCE" test "lib/$n.rce"
done
# Foundation modules under lib/<area>/: each is tested alone (it must admit
# as a standalone module) and then with its checks file appended (vectors
# and property sweeps live in tests/<area>/<name>_checks[_N].rce so a library's
# own comptime fuel stays small for programs that use it).
# A module that imports others (`use a.b`) is checked as a package together
# with lib/a/b.rce and every module that one imports in turn. scripts/uses.py
# reads the `use` items strictly: a malformed line or an unknown module stops
# the run with an error. A tests/<area>/<name>_host.rce file
# holds checks too heavy for comptime fuel: it is flattened with the module
# and its first checks file (scripts/flatten.py), its "# events:" are fed
# to `rce host run` one per input line, and the final state must read
# "good = N, bad = 0" for the N events.
python3 scripts/uses.py --selftest
python3 scripts/uses.py --all
for f in lib/*/*.rce; do
  only "$(basename "$(dirname "$f")")" || continue
  uses=$(python3 scripts/uses.py "$f") || { echo "check.sh: bad use item in $f" >&2; exit 1; }
  pk="$tmp/pkg"; rm -rf "$pk"; mkdir "$pk"
  for u in $uses; do cp "lib/$u.rce" "$pk/"; done
  if [ -n "$uses" ]; then cp "$f" "$pk/"; "$RCE" check "$pk"; else cp "$f" "$tmp/one.rce"; "$RCE" check "$tmp/one.rce"; fi
  t="tests/${f#lib/}"; t="${t%.rce}_checks"
  # one comptime fuel budget per file: heavy vector sets are sharded; a
  # module without dependencies runs its shards in parallel (JOBS, default
  # the number of CPUs), each in its own file, and any failing shard fails
  for c in "$t".rce "$t"_*.rce; do
    [ -f "$c" ] || continue
    if [ -n "$uses" ]; then
      o="$pk/$(basename "$f")"
      { cat "$f"; echo; cat "$c"; } > "$o"
      "$RCE" test "$pk"
    else
      echo "$c"
    fi
  done > "$tmp/shards"
  if [ -z "$uses" ] && [ -s "$tmp/shards" ]; then
    mkdir -p "$tmp/sh"
    F="$f" T="$tmp/sh" xargs -P "${JOBS:-$(nproc)}" -I{} sh -c 'o="$T/$(basename "{}")"; { cat "$F"; echo; cat "{}"; } > "$o"; "$0" test "$o" > "$o.out" 2>&1 || { cat "$o.out"; exit 255; }; rm -f "$o" "$o.out"' "$RCE" < "$tmp/shards"
    echo "ok $f: $(wc -l < "$tmp/shards") check files"
  else
    cat "$tmp/shards"
  fi
  h="${t%_checks}_host.rce"
  if [ -f "$h" ]; then
    python3 scripts/flatten.py "$f" "$t.rce" "$h" > "$tmp/host.rce"
    # "# events: A, B" lists events without arguments; "# event: E" lines
    # give one event each (with arguments). "# host-flags: --fs" grants a
    # fresh empty file root.
    if grep -q "^# event: " "$h"; then ev=$(sed -n "s/^# event: //p" "$h"); else ev=$(sed -n "s/^# events: //p" "$h" | tr -d ' ' | tr ',' '\n'); fi
    hf=""; if grep -q "^# host-flags: --fs" "$h"; then rm -rf "$tmp/fs"; mkdir "$tmp/fs"; hf="--fs $tmp/fs"; fi
    n=$(printf '%s\n' "$ev" | grep -c .)
    res=$(printf '%s\n' "$ev" | "$RCE" host run "$tmp/host.rce" --log "$tmp/hostlog" $hf 2> "$tmp/host.err") || true
    printf '%s\n' "$res" | tail -3
    grep "^rce host: validated " "$tmp/host.err" || { cat "$tmp/host.err"; echo "FAIL host checks $h: the run was not validated by replay"; exit 1; }
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
