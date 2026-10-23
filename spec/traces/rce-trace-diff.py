#!/usr/bin/env python3
"""Compare a projector's traces with the RCE reference traces (spec 10.9).

    rce-trace-diff.py REFERENCE OTHER

REFERENCE and OTHER are both trace files (rce-trace/2 JSON) or both
directories of them (files matched by name; a reference file missing
from OTHER is a divergence). For each trace, in order, each step's
required fields `seq`, `event` and `state_hash` must be equal; the
optional fields `prev_hash`, `hash` and `outcome` are compared when both
sides have them. Events compare as canonical JSON (sorted keys).
Prints `exact match` or the first divergence per file; exit 0 when all
match, 1 otherwise, 2 on unreadable input. Python 3 standard library only.
"""
import json
import os
import sys

REQUIRED = ("seq", "event", "state_hash")
OPTIONAL = ("prev_hash", "hash", "outcome")


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def first_divergence(ref, got):
    """None when `got` conforms to `ref`, else a one-line description."""
    if got.get("format") != ref.get("format"):
        return f"format: reference {ref.get('format')!r}, got {got.get('format')!r}"
    rt, gt = ref.get("traces", []), got.get("traces", [])
    for i, r in enumerate(rt):
        if i >= len(gt):
            return f"trace {i} ({r.get('source')}): missing"
        g = gt[i]
        if g.get("source", r.get("source")) != r.get("source"):
            return f"trace {i}: source reference {r.get('source')!r}, got {g.get('source')!r}"
        rs, gs = r.get("steps", []), g.get("steps", [])
        for k, a in enumerate(rs):
            if k >= len(gs):
                return f"trace {i} ({r.get('source')}) seq {a.get('seq')}: step missing (reference has {len(rs)}, got {len(gs)})"
            b = gs[k]
            for f in REQUIRED:
                if f not in b:
                    return f"trace {i} ({r.get('source')}) step {k}: required field {f!r} missing"
                if canon(a.get(f)) != canon(b.get(f)):
                    return f"trace {i} ({r.get('source')}) seq {a.get('seq')}: {f} reference {canon(a.get(f))}, got {canon(b.get(f))}"
            for f in OPTIONAL:
                if f in a and f in b and canon(a[f]) != canon(b[f]):
                    return f"trace {i} ({r.get('source')}) seq {a.get('seq')}: {f} reference {canon(a[f])}, got {canon(b[f])}"
        if len(gs) > len(rs):
            return f"trace {i} ({r.get('source')}): {len(gs) - len(rs)} extra steps after seq {len(rs) - 1}"
    if len(gt) > len(rt):
        return f"{len(gt) - len(rt)} extra traces"
    return None


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    ref, other = argv[1], argv[2]
    try:
        if os.path.isdir(ref):
            names = sorted(n for n in os.listdir(ref) if n.endswith(".json"))
            pairs = [(n, os.path.join(ref, n), os.path.join(other, n)) for n in names]
        else:
            pairs = [(os.path.basename(ref), ref, other)]
        bad = 0
        for n, a, b in pairs:
            if not os.path.exists(b):
                print(f"{n}: DIVERGES: no trace from the other projector")
                bad += 1
                continue
            d = first_divergence(load(a), load(b))
            if d is None:
                print(f"{n}: exact match")
            else:
                print(f"{n}: DIVERGES: {d}")
                bad += 1
        print(f"{len(pairs) - bad}/{len(pairs)} exact match")
        return 1 if bad else 0
    except (OSError, ValueError) as e:
        print(f"unreadable input: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
