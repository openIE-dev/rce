"""A tiny client for an rce-adapter/1 process (used by the generators)."""
import json, os, subprocess

class Ref:
    def __init__(self, cmd=None):
        self.cmd = cmd or [os.environ.get("RCE_CONFORM", "rce-conform")]
        self.start()
    def start(self):
        cmd = self.cmd
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.n = 0
    def call(self, **req):
        self.n += 1
        req["id"] = self.n
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:   # the implementation died (a crash is a result, not a protocol error)
            self.p.kill(); self.start()
            return {"error": "implementation crashed", "verdict": "crash"}
        return json.loads(line)
    def check(self, src, files=None):
        return self.call(op="check", source=src, files=files or {"main.rce": src})
    def run(self, src, events=None, fuel=None, seed=None):
        return self.call(op="run", source=src, events=events, fuel=fuel, schedule_seed=seed)
    def trace(self, src, events):
        return self.call(op="trace", source=src, events=events)
