#!/usr/bin/env python3
"""Line-mode chat front end for apps/chat.rce on the RCE host.

Runs `rce host serve apps/chat.rce` as a child and translates commands into
events; every change is an event in the host's log D, so the session replays
with `rce host replay`. The UI holds no chat state of its own: it renders the
View query (thread with keys and flags) and the Unread query.

  python3 chat-ui.py [chan ADDR] [peer ADDR]... [poll S] [rce PATH] [app PATH] ME LOG

for example `python3 chat-ui.py chan 127.0.0.1:7001 peer 127.0.0.1:7002 ann /tmp/ann.log`
(spec/proposals/positional-cli.md: words, then the operands ME and LOG).

Commands:
  /open CONV TOPIC    switch to a thread (marks it read)
  TEXT                post to the current thread
  /edit N TEXT        edit your message N        /del N   delete your message N
  /react N EMOJI      toggle a reaction           /dm PEER TEXT   end-to-end encrypted DM
  /dms PEER           show the DM thread          /unread  unread counts per thread
  /read               mark the current thread read
  /quit
Lines starting with an event (e.g. `Poll`) are passed through.
"""
import os, queue, re, subprocess, sys, threading, time
from positional import read

US = "\x1f"


def q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace(US, "\\u{1f}") + '"'


class Host:
    def __init__(self, rce, grants, app, log):
        # positional (PC-1): the grant words, then FILE and the log D
        self.p = subprocess.Popen([rce, "host", "serve"] + grants + [app, log], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.out = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.out.put(line.rstrip("\n"))
        self.out.put(None)

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def do(self, op, conv="", topic="", body=""):
        self.send(f"Do(op: {q(op)}, conv: {q(conv)}, topic: {q(topic)}, body: {q(body)})")

    def collect(self, end, timeout=5.0):
        lines, t = [], time.time() + timeout
        while time.time() < t:
            try:
                l = self.out.get(timeout=0.1)
            except queue.Empty:
                continue
            if l is None or l == end:
                return lines
            lines.append(l)
        return lines


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # poll S: seconds between Poll inputs (0: off)
    w, (me, log) = read("scripts/chat-ui.py", {"chan": 1, "peer": 1, "poll": 1, "rce": 1, "app": 1}, "ME LOG", 2, 2,
                        removed={"--me": 0, "--log": 1}, many=("peer",))
    rce = w.get("rce", os.path.join(here, "spec/checker/target/release/rce"))
    app = w.get("app", os.path.join(here, "apps/chat.rce"))
    poll = float(w.get("poll", 1.0))
    peer = w.get("peer", [])
    grants = (["chan", w["chan"]] if "chan" in w else []) + [x for p in peer for x in ("peer", p)]
    h = Host(rce, grants, app, log)
    peers = ", ".join(q(p) for p in peer)
    h.send(f"Setup(me: {q(me)}, peers: [{peers}])")
    cur = {"conv": "general", "topic": "", "keys": []}
    lock = threading.Lock()

    def render():
        h.send(f"View(conv: {q(cur['conv'])}, topic: {q(cur['topic'])})")
        rows = [l.split("\t") for l in h.collect("view-end") if l.startswith("view\t")]
        cur["keys"] = [r[1] for r in rows]
        title = cur["conv"] + (" #" + cur["topic"] if cur["topic"] else "")
        print(f"\n\x1b[1m── {title} ──\x1b[0m")
        for i, (_, _, who, flags, text) in enumerate(rows, 1):
            fl = set(f for f in flags.split(",") if f)
            body = "\x1b[2m[deleted]\x1b[0m" if "deleted" in fl else text
            tag = (" \x1b[2m(edited)\x1b[0m" if "edited" in fl else "") + (" \x1b[32m🔒\x1b[0m" if "e2e" in fl else "")
            mark = "\x1b[33m●\x1b[0m" if "new" in fl else " "
            print(f"{mark}{i:3} {who}: {body}{tag}")
        sys.stdout.flush()

    def key(n):
        try:
            return cur["keys"][int(n) - 1]
        except (ValueError, IndexError):
            print("no such message"); return None

    def poller():
        while poll > 0:
            time.sleep(poll)
            with lock:
                if h.p.stdin.closed:
                    return
                h.send("Poll")
    threading.Thread(target=poller, daemon=True).start()
    print(__doc__.split("Commands:")[1])
    with lock:
        render()
    for line in sys.stdin:
        line = line.rstrip("\n")
        m = line.split(" ", 2)
        with lock:
            if not line:
                render(); continue
            if line == "/quit":
                break
            if m[0] == "/open" and len(m) >= 2:
                cur["conv"], cur["topic"] = m[1], m[2] if len(m) > 2 else ""
                render(); h.do("read", cur["conv"], cur["topic"]); continue
            if m[0] == "/dms" and len(m) == 2:
                cur["conv"], cur["topic"] = "dm:" + ",".join(sorted([me, m[1]])), ""
                render(); continue
            if m[0] == "/read":
                h.do("read", cur["conv"], cur["topic"]); continue
            if m[0] == "/unread":
                h.send("Unread"); h.send('View(conv: "", topic: "")')
                for l in h.collect("view-end"):
                    print(" ", l)
                continue
            if m[0] in ("/edit", "/del", "/react") and len(m) >= 2:
                k = key(m[1])
                if k:
                    if m[0] == "/edit":
                        h.do("edit", k, "", m[2] if len(m) > 2 else "")
                    elif m[0] == "/del":
                        h.do("delete", k)
                    else:
                        h.do("react", k, m[2] if len(m) > 2 else "+1")
                render(); continue
            if m[0] == "/dm" and len(m) == 3:
                h.do("dm", m[1], "", m[2]); continue
            if re.match(r"^[A-Z]\w*(\(|$)", line):
                h.send(line); continue
            h.do("post", cur["conv"], cur["topic"], line)
            render()
    with lock:
        h.p.stdin.close()
    h.p.wait()


if __name__ == "__main__":
    main()
