"""Feed an emitted Python module numpy values: every integer of the stream
as numpy.int64 and every float as numpy.float64, as a host holding numpy
arrays would, with no JSON in between. Prints the transcript.

  python3 numpy_drive.py MODULE.py STREAM.json
"""
import json
import sys

import numpy as np

import rce_host


def as_numpy(v):
    if isinstance(v, bool) or isinstance(v, str):
        return v
    if isinstance(v, int):
        return np.int64(v) if -2**63 <= v < 2**63 else np.uint64(v)
    if isinstance(v, float):
        return np.float64(v)
    return v


p = rce_host.Program(rce_host.load(sys.argv[1]))
lines = ["0 (initial) " + p.show_state()]
for k, obj in enumerate(json.load(open(sys.argv[2])), 1):
    ev = p.event({f: as_numpy(v) for f, v in obj.items()})
    lines += p.lines(k, ev, p.step(ev))
sys.stdout.write("".join(x + "\n" for x in lines))
