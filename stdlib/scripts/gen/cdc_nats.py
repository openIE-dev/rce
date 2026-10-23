"""JetStream entries from a live nats-server (-js): raw protocol frames
(INFO/MSG/HMSG) captured on a plain TCP connection while publishing with
Nats-Msg-Id (dedupe window), then pulling through a durable consumer.
nats-py's Msg.Metadata parse of each $JS.ACK reply subject is the
reference for ack metadata. Output: vectors/data/nats/jetstream.json."""
import json, socket, sys, time
from nats.aio.msg import Msg

PORT = 54222
OUT = sys.argv[1] if len(sys.argv) > 1 else "vectors/data/nats/jetstream.json"


class Conn:
    def __init__(self):
        self.s = socket.create_connection(("127.0.0.1", PORT))
        self.buf = b""
        self.frames = []
        self.frame()  # INFO
        self.send(b'CONNECT {"verbose":false,"pedantic":false,"headers":true,"no_responders":true,"protocol":1}\r\nPING\r\n')
        self.until(lambda f: f == b"PONG\r\n")

    def send(self, b):
        self.s.sendall(b)

    def frame(self):
        while True:
            j = self.buf.find(b"\r\n")
            if j >= 0:
                line = self.buf[:j]
                op = line.split(b" ")[0]
                if op in (b"MSG", b"HMSG"):
                    n = int(line.split(b" ")[-1])
                    end = j + 2 + n + 2
                    if len(self.buf) >= end:
                        f, self.buf = self.buf[:end], self.buf[end:]
                        return f
                else:
                    f, self.buf = self.buf[:j + 2], self.buf[j + 2:]
                    return f
            self.buf += self.s.recv(65536)

    def until(self, pred):
        while True:
            f = self.frame()
            if f == b"PING\r\n":
                self.send(b"PONG\r\n")
                continue
            if pred(f):
                return f

    def request(self, subj, payload=b"", hdrs=None, sid=[0]):
        sid[0] += 1
        inbox = b"_INBOX.r%d" % sid[0]
        self.send(b"SUB %s %d\r\n" % (inbox, sid[0]))
        if hdrs is None:
            self.send(b"PUB %s %s %d\r\n%s\r\n" % (subj, inbox, len(payload), payload))
        else:
            h = b"NATS/1.0\r\n" + b"".join(b"%s: %s\r\n" % kv for kv in hdrs) + b"\r\n"
            self.send(b"HPUB %s %s %d %d\r\n%s%s\r\n" % (subj, inbox, len(h), len(h) + len(payload), h, payload))
        f = self.until(lambda f: f.startswith(b"MSG ") or f.startswith(b"HMSG "))
        self.send(b"UNSUB %d\r\n" % sid[0])
        return f


def main():
    c = Conn()
    out = {"server": None, "pubacks": [], "pulls": [], "control": []}
    c.request(b"$JS.API.STREAM.DELETE.ORDERS")
    cfg = {"name": "ORDERS", "subjects": ["orders.>"], "duplicate_window": 120000000000, "storage": "memory"}
    out["stream_create"] = c.request(b"$JS.API.STREAM.CREATE.ORDERS", json.dumps(cfg).encode()).hex()
    pubs = [
        (b"orders.new", b'{"id":1}', [(b"Nats-Msg-Id", b"m1")]),
        (b"orders.new", b'{"id":1}', [(b"Nats-Msg-Id", b"m1")]),
        (b"orders.paid", b"", [(b"Nats-Msg-Id", b"m2"), (b"X-Multi", b"a"), (b"X-Multi", b"b"), (b"x-lower", b"  spaced value ")]),
        (b"orders.new", "é😀\r\n".encode(), None),
        (b"orders.new", b"x", [(b"Nats-Msg-Id", b"m3"), (b"Nats-Expected-Last-Sequence", b"99")]),
        (b"orders.new", b"y", [(b"Nats-Msg-Id", b"m4"), (b"Nats-Expected-Last-Sequence", b"4")]),
        (b"nostream.x", b"z", None),
    ]
    for subj, payload, h in pubs:
        f = c.request(subj, payload, h)
        out["pubacks"].append({"subject": subj.decode(), "headers": None if h is None else [[k.decode(), v.decode()] for k, v in h],
                               "payload": payload.hex(), "frame": f.hex()})
    cc = {"stream_name": "ORDERS", "config": {"durable_name": "C", "ack_policy": "explicit", "deliver_policy": "all"}}
    out["consumer_create"] = c.request(b"$JS.API.CONSUMER.CREATE.ORDERS.C", json.dumps(cc).encode()).hex()
    # pull: batch of 3, then the rest with no_wait, then an empty no_wait (404 status)
    for req in ({"batch": 3}, {"batch": 10, "no_wait": True}, {"batch": 1, "no_wait": True}):
        inbox = b"_INBOX.pull"
        c.send(b"SUB %s 900\r\n" % inbox)
        body = json.dumps(req).encode()
        c.send(b"PUB $JS.API.CONSUMER.MSG.NEXT.ORDERS.C %s %d\r\n%s\r\n" % (inbox, len(body), body))
        got = []
        deadline = time.time() + 3
        while time.time() < deadline:
            f = c.until(lambda f: f.startswith(b"MSG ") or f.startswith(b"HMSG "))
            got.append(f.hex())
            line = f.split(b"\r\n")[0].split(b" ")
            if b"$JS.ACK" not in f.split(b"\r\n")[0]:
                break  # status message ends the request
            if len(got) == req["batch"]:
                break
        c.send(b"UNSUB 900\r\n")
        out["pulls"].append({"request": req, "frames": got})
    # ack metadata reference (nats-py) for every delivered message
    refs = []
    for p in out["pulls"]:
        for fh in p["frames"]:
            line = bytes.fromhex(fh).split(b"\r\n")[0].split(b" ")
            reply = line[3].decode() if len(line) >= 5 else ""
            if reply.startswith("$JS.ACK."):
                md = Msg.Metadata._from_reply(reply)
                refs.append({"reply": reply, "stream": md.stream, "consumer": md.consumer,
                             "stream_seq": md.sequence.stream, "consumer_seq": md.sequence.consumer,
                             "delivered": md.num_delivered, "pending": md.num_pending,
                             "ts_ns": int(reply.split(".")[-2 if len(reply.split(".")) == 9 else -3]), "domain": md.domain})
    out["ack_metadata_ref"] = refs
    # v2 reply subjects (with domain and account hash), parsed by nats-py
    v2 = ["$JS.ACK.hub.ACCHASH.ORDERS.C.2.7.5.1700000000123456789.3.rnd",
          "$JS.ACK._.ACCHASH.ORDERS.C.1.1.1.1700000000000000000.0.rnd",
          "$JS.ACK.dom.ACCHASH.S.c.1.2.3.4.5"]
    for reply in v2:
        md = Msg.Metadata._from_reply(reply)
        out["control"].append({"reply": reply, "stream": md.stream, "consumer": md.consumer, "stream_seq": md.sequence.stream,
                               "consumer_seq": md.sequence.consumer, "delivered": md.num_delivered, "pending": md.num_pending,
                               "domain": md.domain})
    json.dump(out, open(OUT, "w"), indent=1)
    print(OUT, len(out["pubacks"]), [len(p["frames"]) for p in out["pulls"]], len(refs))


main()
