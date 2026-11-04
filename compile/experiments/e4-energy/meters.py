#!/usr/bin/env python3
"""E4's energy counters: Linux RAPL through powercap, and macOS powermetrics.

Every figure these give is of kind "reported": an on-chip or operating-system
counter, not a meter on the supply ("measured"). A host with neither gives
the Unmeasured meter, and the harness records ε and λ without energy.

Each meter has `mark()`, which returns an opaque start mark, and
`since(mark)`, which returns the joules counted from the mark to now.
`provenance()` returns what the results file records about the counter.

RAPL follows the reading of rce's compute/crates/rce-energy (meter.rs),
reimplemented here because RCE Compile does not depend on RCE: every
top-level zone `intel-rapl:N` under /sys/class/powercap is read (the AMD
driver uses the same names), sub-zones are not read so nothing is counted
twice, and `psys` zones are skipped because they overlap the packages. The
counter is `energy_uj`; it wraps past `max_energy_range_uj`, and one
wraparound between two reads is added back.
"""
import os
import plistlib
import subprocess
import threading
import time

POWERCAP = "/sys/class/powercap"
POWERMETRICS = "/usr/bin/powermetrics"


class NeedsAccess(Exception):
    """The counter exists but this user cannot read it; the message is the one line that grants access."""


def wrap_delta(prev, now, max_uj):
    """Microjoules between two reads of a counter that wraps past max_uj (one wraparound at most)."""
    if now >= prev:
        return now - prev
    return max(max_uj - prev, 0) + now + 1


class Unmeasured:
    kind = "unmeasured"

    def __init__(self, reason):
        self.reason = reason

    def mark(self):
        return None

    def since(self, _mark):
        return None

    def provenance(self):
        return {"kind": "unmeasured", "reason": self.reason}

    def close(self):
        pass


class Rapl:
    kind = "reported"

    def __init__(self, root=POWERCAP):
        self.root = root
        try:
            names = sorted(n for n in os.listdir(root) if n.startswith("intel-rapl:") and n.count(":") == 1)
        except OSError as e:
            raise LookupError("%s: %s" % (root, e.strerror))
        self.zones = []
        for n in names:
            d = os.path.join(root, n)
            try:
                with open(os.path.join(d, "name"), encoding="utf-8") as f:
                    domain = f.read().strip()
            except OSError:
                domain = ""
            if domain == "psys":
                continue
            energy = os.path.join(d, "energy_uj")
            try:
                first = self._read(energy)
            except PermissionError:
                raise NeedsAccess("sudo chmod a+r %s/intel-rapl:*/energy_uj" % root)
            self.zones.append({"zone": n, "domain": domain or n, "energy_file": energy,
                               "max_energy_range_uj": self._read(os.path.join(d, "max_energy_range_uj")),
                               "last": first, "wraps": 0})
        if not self.zones:
            raise LookupError("no package zones under %s" % root)

    @staticmethod
    def _read(p):
        with open(p, encoding="utf-8") as f:
            return int(f.read().strip())

    def _poll(self):
        """Advance every zone to its current reading; return the cumulative microjoules per zone."""
        out = []
        for z in self.zones:
            now = self._read(z["energy_file"])
            if now < z["last"]:
                z["wraps"] += 1
            z["total"] = z.get("total", 0) + wrap_delta(z["last"], now, z["max_energy_range_uj"])
            z["last"] = now
            out.append(z["total"])
        return out

    def mark(self):
        return self._poll()

    def since(self, mark):
        now = self._poll()
        return sum(b - a for a, b in zip(mark, now)) * 1e-6

    def provenance(self):
        return {"kind": "reported", "sampler": "Linux powercap (RAPL)", "counter": "energy_uj",
                "interval": "read at the start and end of each window",
                "zones": [{k: z[k] for k in ("zone", "domain", "energy_file", "max_energy_range_uj", "wraps")}
                          for z in self.zones],
                "boundary": "sum of the top-level RAPL zones (" + ", ".join(z["domain"] for z in self.zones) + ")"}

    def close(self):
        pass


def sample_energy(d):
    """(joules, key used, elapsed seconds) of one powermetrics plist sample, or None.

    Apple silicon: `processor.cpu_power` (mW, the average over the sample)
    times `elapsed_ns`; else `processor.cpu_energy` (mJ). Intel Macs:
    `processor.package_joules`; else `processor.package_watts` times
    `elapsed_ns`.
    """
    el = d.get("elapsed_ns")
    if not el:
        return None
    sec = el * 1e-9
    p = d.get("processor") or {}
    if "cpu_power" in p:
        return p["cpu_power"] * 1e-3 * sec, "processor.cpu_power", sec
    if "cpu_energy" in p:
        return p["cpu_energy"] * 1e-3, "processor.cpu_energy", sec
    if "package_joules" in p:
        return p["package_joules"], "processor.package_joules", sec
    if "package_watts" in p:
        return p["package_watts"] * sec, "processor.package_watts", sec
    return None


def integrate(samples, t0, t1):
    """Joules in [t0, t1] from samples (start, end, joules), each spread evenly over its own span."""
    j = 0.0
    for s, e, x in samples:
        if e <= s:
            continue
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            j += x * ov / (e - s)
    return j


def split_plists(buf):
    """Complete NUL-separated plist documents in buf, and the incomplete rest."""
    parts = buf.split(b"\0")
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class Powermetrics:
    kind = "reported"

    def __init__(self, work, interval_ms):
        self.interval_ms = interval_ms
        self.path = os.path.join(work, "powermetrics.plist")
        self.argv = ["sudo", "-n", POWERMETRICS, "-i", str(interval_ms), "-f", "plist",
                     "--samplers", "cpu_power", "-o", self.path]
        self.samples = []
        self.keys = set()
        self.first_keys = None
        self.lock = threading.Lock()
        self.stop = False
        self.errpath = os.path.join(work, "powermetrics.stderr")
        with open(self.errpath, "wb") as err:
            self.child = subprocess.Popen(self.argv, stdout=subprocess.DEVNULL, stderr=err)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        deadline = time.monotonic() + 10 + 3 * interval_ms / 1e3
        while time.monotonic() < deadline:
            with self.lock:
                if len(self.samples) >= 2:
                    return
            if self.child.poll() is not None:
                break
            time.sleep(0.05)
        self.close()
        with open(self.errpath, "rb") as f:
            err = f.read()
        raise LookupError("powermetrics gave no samples: %s" % err.decode("utf-8", "replace").strip()[:200])

    def _read(self):
        buf, pos = b"", 0
        while not self.stop:
            try:
                with open(self.path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
            except OSError:
                chunk = b""
            if chunk:
                pos += len(chunk)
                docs, buf = split_plists(buf + chunk)
                t = time.monotonic()
                for doc in docs:
                    try:
                        d = plistlib.loads(doc)
                    except Exception:
                        continue
                    got = sample_energy(d)
                    if got is None:
                        continue
                    with self.lock:
                        if self.first_keys is None:
                            self.first_keys = sorted((d.get("processor") or {}).keys())
                        self.keys.add(got[1])
                        self.samples.append((t - got[2], t, got[0]))
            time.sleep(self.interval_ms / 4e3)

    def mark(self):
        return time.monotonic()

    def since(self, t0):
        t1 = time.monotonic()
        deadline = t1 + 5 * self.interval_ms / 1e3 + 2
        while time.monotonic() < deadline:
            with self.lock:
                if self.samples and self.samples[-1][1] >= t1:
                    return integrate(self.samples, t0, t1)
            time.sleep(self.interval_ms / 4e3)
        raise RuntimeError("powermetrics stopped sending samples")

    def provenance(self):
        with self.lock:
            keys, first = sorted(self.keys), self.first_keys
        return {"kind": "reported", "sampler": "powermetrics, sampler cpu_power", "argv": self.argv,
                "counter": " and ".join(keys) or "none read", "processor_keys_of_first_sample": first,
                "interval": "%d ms; each window takes the share of every sample that overlaps it" % self.interval_ms,
                "boundary": "CPU clusters (Apple silicon cpu_power) or the package (Intel Macs), as the counter names it"}

    def close(self):
        self.stop = True
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def sudo_cached():
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
