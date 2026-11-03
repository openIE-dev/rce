#!/usr/bin/env python3
"""Unit tests of E4's harness: the ε count on tiny modules whose ε is worked out by hand here, the
RAPL, macmon and powermetrics readers on fake inputs, the statistics and the report.

    python3 test_e4.py
"""
import json
import math
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e4  # noqa: E402
import meters  # noqa: E402
import reference  # noqa: E402
import report  # noqa: E402
import stats  # noqa: E402
import wasm_eps  # noqa: E402

I32, I64 = 0x7F, 0x7E


def u(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def s(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        done = (n == 0 and not b & 0x40) or (n == -1 and b & 0x40)
        out.append(b | (0 if done else 0x80))
        if done:
            return bytes(out)


def vec(items):
    return u(len(items)) + b"".join(items)


def write(path, text, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def section(sid, body):
    return bytes([sid]) + u(len(body)) + body


# instruction helpers
def get(i): return b"\x20" + u(i)
def set_(i): return b"\x21" + u(i)
def c32(v): return b"\x41" + s(v)
def c64(v): return b"\x42" + s(v)
def call(f): return b"\x10" + u(f)
def br(d): return b"\x0c" + u(d)
def br_if(d): return b"\x0d" + u(d)
def store8(off=0): return b"\x3a\x00" + u(off)
def store32(off=0): return b"\x36\x02" + u(off)
def load32(off=0): return b"\x28\x02" + u(off)


ADD, ADD64, DROP, END, LT_S, GE_U, WRAP, SELECT = b"\x6a", b"\x7c", b"\x1a", b"\x0b", b"\x48", b"\x4f", b"\xa7", b"\x1b"
BLOCK, LOOP, BLOCK_I32 = b"\x02\x40", b"\x03\x40", b"\x02\x7f"


def module(funcs, pages=0, custom=False):
    """funcs: (params, results, locals, body); function 0 is exported as run."""
    types = vec([b"\x60" + vec([bytes([t]) for t in p]) + vec([bytes([t]) for t in r]) for p, r, _, _ in funcs])
    m = b"\0asm\x01\0\0\0" + section(1, types) + section(3, vec([u(i) for i in range(len(funcs))]))
    if pages:
        m += section(5, vec([b"\x00" + u(pages)]))
    m += section(7, vec([u(3) + b"run" + b"\x00" + u(0)]))
    bodies = []
    for _, _, locs, body in funcs:
        f = vec([u(1) + bytes([t]) for t in locs]) + body + END
        bodies.append(u(len(f)) + f)
    m += section(10, vec(bodies))
    if custom:
        m += section(0, u(4) + b"name" + b"\x00" * 10)
    return m


def eps(funcs, args=(), pages=0):
    return wasm_eps.count(module(funcs, pages), "run", list(args))


class Epsilon(unittest.TestCase):
    """Each expected ε is worked out in the comment above it from PROTOCOL.md's definition."""

    def test_copies_erase_nothing(self):
        # get0 copies the argument (permanent); const is permanent; add makes A (4 fresh bytes);
        # set1 moves A into local 1 over its permanent zeros; get1 copies A; the result is copied out,
        # so releasing local 1 at return leaves A with one copy: ε = 0.
        r = eps([([I32], [I32], [I32], get(0) + c32(1) + ADD + set_(1) + get(1))], [41])
        self.assertEqual((r["result"], r["eps_bits"]), (42, 0))

    def test_overwrite_and_release(self):
        # A is set into local 0, then B overwrites it: A's 4 bytes have no other copy, 32 bits.
        # At return local 0 (B, no other copy) is released: 32 bits. ε = 64.
        body = c32(5) + c32(3) + ADD + set_(0) + c32(7) + c32(1) + ADD + set_(0) + c32(0)
        self.assertEqual(eps([([], [I32], [I32], body)])["eps_bits"], 64)

    def test_drop(self):
        # add makes 4 fresh bytes; drop releases the only copy: ε = 32.
        self.assertEqual(eps([([], [I32], [], c32(1) + c32(2) + ADD + DROP + c32(0))])["eps_bits"], 32)

    def test_store8_keeps_one_byte(self):
        # store8 moves byte 0 of the fresh sum into memory and releases bytes 1..3: 24 bits.
        # Storing the constant 9 over it erases that byte: 8 bits. ε = 32.
        body = c32(0) + c32(1) + c32(2) + ADD + store8() + c32(0) + c32(9) + store8() + c32(0)
        self.assertEqual(eps([([], [I32], [], body)], pages=1)["eps_bits"], 32)

    def test_store_load_is_a_copy(self):
        # store32 moves the fresh sum into memory (nothing erased); load32 copies it and drop releases
        # that copy (the memory still holds it, nothing erased); storing a constant over it erases
        # the 4 bytes: ε = 32.
        body = (c32(0) + c32(1) + c32(2) + ADD + store32() + c32(0) + load32() + DROP
                + c32(0) + c32(7) + store32() + c32(0))
        self.assertEqual(eps([([], [I32], [], body)], pages=1)["eps_bits"], 32)

    def test_comparison_is_one_fresh_byte(self):
        # lt_s gives a fresh low byte and three permanent zero bytes; drop erases 8 bits.
        self.assertEqual(eps([([], [I32], [], c32(1) + c32(2) + LT_S + DROP + c32(0))])["eps_bits"], 8)

    def test_wrap_keeps_low_bytes(self):
        # i64.add makes 8 fresh bytes; wrap keeps the low 4 and releases the high 4 (32 bits);
        # drop releases the low 4 (32 bits). ε = 64.
        body = c64(1) + c64(2) + ADD64 + WRAP + DROP + c32(0)
        self.assertEqual(eps([([], [I32], [], body)])["eps_bits"], 64)

    def test_select_releases_the_other(self):
        # A = 1+2, B = 3+4, condition permanent; select keeps A and releases B (32); drop A (32). ε = 64.
        body = c32(1) + c32(2) + ADD + c32(3) + c32(4) + ADD + c32(1) + SELECT + DROP + c32(0)
        self.assertEqual(eps([([], [I32], [], body)])["eps_bits"], 64)

    def test_branch_discards(self):
        # In a block returning one i32, br 0 keeps B and discards A (32); drop B (32). ε = 64.
        body = BLOCK_I32 + c32(1) + c32(2) + ADD + c32(3) + c32(4) + ADD + br(0) + END + DROP + c32(0)
        self.assertEqual(eps([([], [I32], [], body)])["eps_bits"], 64)

    def test_call(self):
        # f1(a) = a + 1. The first call takes the permanent argument and returns A; nothing erased.
        # The second call moves A into f1's local 0, copies it, adds (fresh B) and returns B; at
        # return local 0 holds the last copy of A: 32 bits. run returns B, copied out. ε = 32.
        f0 = ([I32], [I32], [], get(0) + call(1) + call(1))
        f1 = ([I32], [I32], [], get(0) + c32(1) + ADD)
        r = eps([f0, f1], [5])
        self.assertEqual((r["result"], r["eps_bits"]), (7, 32))

    def test_counting_loop(self):
        # acc = sum of i for i < n, with n = 3. Per pass: ge_u's fresh byte is popped by br_if (8 bits,
        # 4 passes = 32); acc and i are each overwritten 3 times, the first time over permanent zeros
        # (2 x 2 x 32 = 128); at return local i is released (32) while acc is copied out.
        # ε = 32 + 128 + 32 = 192.
        body = (BLOCK + LOOP + get(1) + get(0) + GE_U + br_if(1)
                + get(2) + get(1) + ADD + set_(2)
                + get(1) + c32(1) + ADD + set_(1) + br(0) + END + END + get(2))
        r = eps([([I32], [I32], [I32, I32], body)], [3])
        self.assertEqual((r["result"], r["eps_bits"]), (3, 192))
        # block, loop, 3 instructions and br_if, 4 for acc, 4 for i, br, end, end, get: 18 nodes
        self.assertEqual(r["ir_nodes"], 18)

    def test_unsupported_is_named(self):
        body = b"\x43\x00\x00\x80\x3f" + DROP + c32(0)  # f32.const 1.0
        with self.assertRaises(wasm_eps.Unsupported) as cm:
            eps([([], [I32], [], body)])
        self.assertIn("f32.const", str(cm.exception))

    def test_strip_custom(self):
        m = module([([], [I32], [], c32(0))], custom=True)
        self.assertEqual(e4.strip_custom(m), module([([], [I32], [], c32(0))]))


def wasm_clang():
    if not shutil.which("clang"):
        return False
    d = tempfile.mkdtemp()
    try:
        write(os.path.join(d, "p.c"), e4.PROBE_C)
        return subprocess.run(["clang", "--target=wasm32", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=run",
                               "p.c", "-o", "p.wasm"], cwd=d, capture_output=True).returncode == 0
    finally:
        shutil.rmtree(d)


class Kernels(unittest.TestCase):
    @unittest.skipUnless(wasm_clang(), "no clang that builds wasm32")
    def test_c_kernels_match_reference(self):
        d = tempfile.mkdtemp()
        try:
            for k in ("sumsq", "sort", "fib"):
                out = os.path.join(d, k + ".wasm")
                subprocess.run(["clang", "--target=wasm32", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=run", "-O2",
                                os.path.join(HERE, "programs", "c", k + ".c"), "-o", out], check=True)
                a = wasm_eps.count(out, "run", [10])
                b = wasm_eps.count(out, "run", [10])
                self.assertEqual(a["result"], reference.KERNELS[k](10))
                self.assertEqual(a, b)
        finally:
            shutil.rmtree(d)


class FakeMeter:
    """A counter that reports a constant 3 W, so the expected joules follow from the clock."""
    kind = "reported"

    def mark(self):
        return time.perf_counter()

    def since(self, t0):
        return 3.0 * (time.perf_counter() - t0)


class EnergyPath(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "no C compiler")
    def test_measure_and_summarise(self):
        d = tempfile.mkdtemp()
        try:
            exe = os.path.join(d, "sumsq")
            subprocess.run(["cc", "-O2", os.path.join(HERE, "programs", "c", "sumsq.c"),
                            os.path.join(HERE, "programs", "c", "driver.c"), "-o", exe], check=True)
            row = {"program": "c/sumsq", "kernel": "sumsq", "level": "-O2", "eps_input": 2000, "exe": exe,
                   "equivalence": {"status": "equal to the reference"}}
            skipped = dict(row, program="c/other", equivalence={"status": "differs"})
            cfg = {"target_seconds": 0.3, "repeats": 2, "order_seed": 4}
            e4.measure([row, skipped], FakeMeter(), cfg, d)
            e4.summarise([row, skipped], 1.0)
            e = row["energy"]
            self.assertEqual((e["status"], e["runs_with_reference_output"]), ("reported", 2))
            # net = (3 W - 1 W idle) x seconds, per call
            self.assertAlmostEqual(e["joules_net_per_call"] / e["seconds_per_call"], 2.0, places=2)
            self.assertEqual(skipped["energy"]["status"], "not measured")
        finally:
            shutil.rmtree(d)


class Rapl(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        for zone, name, val in (("intel-rapl:0", "package-0", 100), ("intel-rapl:0:0", "core", 50),
                                ("intel-rapl:1", "psys", 900), ("intel-rapl:2", "package-1", 10)):
            d = os.path.join(self.root, zone)
            os.makedirs(d)
            write(os.path.join(d, "name"), name + "\n")
            write(os.path.join(d, "energy_uj"), "%d\n" % val)
            write(os.path.join(d, "max_energy_range_uj"), "1000\n")

    def tearDown(self):
        shutil.rmtree(self.root)

    def put(self, zone, v):
        write(os.path.join(self.root, zone, "energy_uj"), "%d\n" % v)

    def test_wrap_delta(self):
        self.assertEqual(meters.wrap_delta(100, 300, 1000), 200)
        self.assertEqual(meters.wrap_delta(900, 99, 1000), 200)  # 100 to the wrap, then 0..99

    def test_zones_and_wrap(self):
        m = meters.Rapl(self.root)
        self.assertEqual([z["zone"] for z in m.zones], ["intel-rapl:0", "intel-rapl:2"])
        mk = m.mark()
        self.put("intel-rapl:0", 600)
        self.put("intel-rapl:2", 5)       # wrapped: 990 to the top, then 0..5
        self.put("intel-rapl:0:0", 999)   # a sub-zone: never read
        self.assertAlmostEqual(m.since(mk), (500 + 996) * 1e-6)
        p = m.provenance()
        self.assertEqual([z["wraps"] for z in p["zones"]], [0, 1])
        self.assertEqual(p["kind"], "reported")

    def test_no_zones(self):
        with self.assertRaises(LookupError):
            meters.Rapl(os.path.join(self.root, "absent"))


class Powermetrics(unittest.TestCase):
    def test_sample_energy(self):
        self.assertEqual(meters.sample_energy({"elapsed_ns": 100_000_000, "processor": {"cpu_power": 2000.0}}),
                         (0.2, "processor.cpu_power", 0.1))
        self.assertEqual(meters.sample_energy({"elapsed_ns": 10, "processor": {"package_joules": 3.0}})[:2],
                         (3.0, "processor.package_joules"))
        self.assertIsNone(meters.sample_energy({"processor": {"cpu_power": 1}}))

    def test_split_plists(self):
        a = plistlib.dumps({"elapsed_ns": 1})
        docs, rest = meters.split_plists(a + b"\0" + a + b"\0" + a[:10])
        self.assertEqual(len(docs), 2)
        self.assertEqual(rest, a[:10])

    def test_integrate(self):
        samples = [(0.0, 1.0, 10.0), (1.0, 2.0, 20.0)]
        self.assertAlmostEqual(meters.integrate(samples, 0.5, 1.5), 5.0 + 10.0)
        self.assertAlmostEqual(meters.integrate(samples, 3.0, 4.0), 0.0)


class Macmon(unittest.TestCase):
    LINES = [
        '{"cpu_power":2.0,"gpu_power":0.5,"ane_power":0.0,"all_power":2.5,"timestamp":"2026-11-03T14:00:00.000000+00:00"}',
        'not json',
        '{"cpu_power":4.0,"gpu_power":0.0,"ane_power":1.0,"all_power":5.0,"timestamp":"2026-11-03T14:00:00.200000+00:00"}',
        '{"cpu_power":1.0}',
        '{"cpu_power":1.0,"gpu_power":1.0,"ane_power":1.0,"timestamp":"2026-11-03T14:00:00.300000+00:00"}',
    ]

    def test_sample(self):
        w, t = meters.macmon_sample(json.loads(self.LINES[0]))
        self.assertAlmostEqual(w, 2.5)
        self.assertAlmostEqual(t % 60, 0.0)
        self.assertIsNone(meters.macmon_sample({"cpu_power": 1.0}))
        self.assertEqual(meters.macmon_sample({"cpu_power": 1, "gpu_power": 1, "ane_power": 1})[1], None)

    def test_spans_use_measured_timestamps(self):
        sp = meters.macmon_spans(self.LINES, 0.1)
        self.assertEqual(len(sp), 3)
        (s0, e0, j0), (s1, e1, j1), (s2, e2, j2) = sp
        self.assertAlmostEqual(e0 - s0, 0.1, places=5)      # the first covers one interval
        self.assertAlmostEqual(j0, 2.5 * 0.1, places=5)
        self.assertAlmostEqual(s1, e0)              # the second covers the 200 ms since the first
        self.assertAlmostEqual(j1, 5.0 * 0.2, places=6)
        self.assertAlmostEqual(j2, 3.0 * 0.1, places=6)
        self.assertAlmostEqual(meters.integrate(sp, s0, e2), j0 + j1 + j2, places=6)

    def test_spans_without_timestamps(self):
        sp = meters.macmon_spans(['{"cpu_power":1,"gpu_power":0,"ane_power":0}'] * 3, 0.5)
        self.assertEqual([round(x[2], 6) for x in sp], [0.5, 0.5, 0.5])

    def test_meter_on_fake_macmon(self):
        d = tempfile.mkdtemp()
        try:
            fake = os.path.join(d, "macmon")
            with open(fake, "w") as f:
                f.write("#!%s\n" % sys.executable)
                f.write("import sys, time, json, datetime\n")
                f.write("if '--version' in sys.argv: print('macmon 9.9.9'); sys.exit()\n")
                f.write("while True:\n")
                f.write("    time.sleep(0.05)\n")
                f.write("    t = datetime.datetime.now(datetime.timezone.utc).isoformat()\n")
                f.write("    print(json.dumps({'cpu_power': 3.0, 'gpu_power': 1.0, 'ane_power': 0.0, 'timestamp': t}), flush=True)\n")
            os.chmod(fake, 0o755)
            m = meters.Macmon(d, 50, path=fake)
            try:
                t0 = m.mark()
                time.sleep(0.5)
                j = m.since(t0)
                self.assertAlmostEqual(j / 0.5, 4.0, delta=0.6)
                p = m.provenance()
                self.assertEqual(p["kind"], "reported")
                self.assertIn("macmon 9.9.9", p["sampler"])
                self.assertEqual(p["wraps"], 0)
                self.assertIn("cpu + gpu + ane", p["boundary"])
            finally:
                m.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Stats(unittest.TestCase):
    def test_ols_exact(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = stats.ols([x], [2 * v + 1 for v in x])
        self.assertAlmostEqual(r["r2"], 1.0)
        self.assertAlmostEqual(r["coef"][0], 2.0)
        self.assertAlmostEqual(r["intercept"], 1.0)

    def test_ols_r2(self):
        x, y = [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 6.0]
        mx, my = sum(x) / 5, sum(y) / 5
        corr = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
            sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
        r = stats.ols([x], y)
        self.assertAlmostEqual(r["r2"], corr ** 2)
        self.assertAlmostEqual(r["adj_r2"], 1 - (1 - corr ** 2) * 4 / 3)

    def test_mean_interval(self):
        m, iv = stats.mean_interval([1.0, 2.0, 3.0])
        self.assertEqual(m, 2.0)
        self.assertAlmostEqual(iv[1] - m, 4.303 / math.sqrt(3))

    def test_bootstrap_is_seeded(self):
        rows = synthetic_rows()
        a = stats.regime_b(rows, 200, 4)
        b = stats.regime_b(rows, 200, 4)
        self.assertEqual(a, b)
        self.assertGreater(a["point"]["baseline"], a["point"]["eps_lam"])


def synthetic_rows():
    """Joules exactly proportional to time; ε and λ only loosely related to it."""
    rows = []
    for p in range(6):
        for c in range(4):
            t = 1e-6 * (1 + p + 0.3 * c)
            rows.append({"program": "p%d" % p, "joules": 5.0 * t, "base": 5.0 * t,
                         "eps": 1000 + 37 * ((p * 7 + c * 3) % 11), "lam": 50 + 13 * ((p + 2 * c) % 5)})
    return rows


def synthetic_results():
    rows = []
    for r in synthetic_rows():
        rows.append({"program": r["program"], "language": "C", "kernel": "sumsq", "toolchain": "c", "config": "cc %s" % r["eps"],
                     "level": "-O2", "eps_input": 1,
                     "build_native": {"status": "built", "seconds": [0.1], "joules_gross": [1.0], "joules_net": [0.5]},
                     "build_wasm": {"status": "built"},
                     "equivalence": {"status": "equal to the reference"},
                     "eps": {"status": "counted", "bits": r["eps"], "ops": 10, "landauer_joules": r["eps"] * 2.87e-21},
                     "lambda": {"ir_nodes": r["lam"]},
                     "energy": {"status": "reported", "joules_net_per_call": r["joules"], "joules_net_per_call_interval": None,
                                "seconds_per_call": r["base"] / 5.0, "calls_per_run": 10, "runs_with_reference_output": 5}})
    return {"schema": e4.SCHEMA, "date_utc": "2026-11-02T00:00:00Z",
            "machine": {"system": "Linux", "machine": "x86_64", "cpu": "test", "cpus": 1, "os": "test", "python": "3", "zlib": "1"},
            "harness": {"e4.py": "0" * 64}, "toolchains": {"c": {"tool": "cc", "version": "cc 1", "wasm": True, "wasm_reason": None}},
            "absent": [], "landauer": {"temperature_k": 300}, "lambda": {"primary": "ir_nodes"},
            "energy": {"kind": "reported", "regime": "B", "sampler": "fake", "counter": "energy_uj", "interval": "window",
                       "boundary": "package", "zones": [{"zone": "intel-rapl:0", "domain": "package-0", "max_energy_range_uj": 1000, "wraps": 0}],
                       "idle_before": {"watts": 1.0, "seconds": 10.0}, "idle_after": {"watts": 1.0, "seconds": 10.0},
                       "idle_watts": 1.0, "net": "gross minus idle", "start_utc": "a", "end_utc": "b"},
            "rows": rows}


class Report(unittest.TestCase):
    def test_write_and_check(self):
        d = tempfile.mkdtemp()
        try:
            res = os.path.join(d, "fake.json")
            md = os.path.join(d, "R.md")
            write(res, json.dumps(synthetic_results()))
            py = sys.executable
            subprocess.run([py, os.path.join(HERE, "report.py"), "write", md, res], check=True, capture_output=True)
            text = read(md)
            self.assertIn("**Energy: reported**", text)
            self.assertIn("so the baseline explains more than ε and λ", text)
            ok = subprocess.run([py, os.path.join(HERE, "report.py"), "check", md, res], capture_output=True)
            self.assertEqual(ok.returncode, 0)
            write(md, "typed by hand\n", "a")
            bad = subprocess.run([py, os.path.join(HERE, "report.py"), "check", md, res], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
        finally:
            shutil.rmtree(d)

    def test_unmeasured_is_labelled(self):
        r = synthetic_results()
        r["energy"] = {"kind": "unmeasured", "reason": "no RAPL counter", "regime": None}
        for row in r["rows"]:
            row["energy"] = {"status": "unmeasured"}
        text = "\n".join(report.section("x", r, {"resamples": 10, "seed": 4, "level": 0.95}))
        self.assertIn("**Energy: unmeasured.**", text)
        self.assertIn("Not computed: this file has no energy figures.", text)
        self.assertNotIn("J, reported", text)


if __name__ == "__main__":
    unittest.main()
