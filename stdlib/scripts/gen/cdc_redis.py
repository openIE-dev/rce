"""Redis Stream entries from a live redis-server: the raw RESP2 and RESP3
replies to XRANGE / XREAD / XREADGROUP / XAUTOCLAIM / XINFO-free reads, with
redis-py's parsed result as the reference. Output: vectors/data/redis/streams.json."""
import json, socket, sys, redis

PORT = 56379
OUT = sys.argv[1] if len(sys.argv) > 1 else "vectors/data/redis/streams.json"


def enc(*args):
    b = b"*%d\r\n" % len(args)
    for a in args:
        a = a if isinstance(a, bytes) else str(a).encode()
        b += b"$%d\r\n%s\r\n" % (len(a), a)
    return b


def raw(cmds, proto):
    s = socket.create_connection(("127.0.0.1", PORT))
    out = []
    if proto == 3:
        s.sendall(enc("HELLO", "3"))
        r = redis.connection.Connection(port=PORT, protocol=3)
    buf = b""
    def read_reply():
        nonlocal buf
        # read until redis-py's parser can parse one reply: use a fresh parser on the bytes
        while True:
            try:
                n = complete(buf)
            except IndexError:
                n = None
            if n is not None:
                rep, buf = buf[:n], buf[n:]
                return rep
            buf += s.recv(65536)
    if proto == 3:
        read_reply()
    for c in cmds:
        s.sendall(enc(*c))
        out.append(read_reply())
    s.close()
    return out


def complete(b, i=0):
    """offset just past one RESP2/RESP3 value starting at i, or None"""
    if i >= len(b):
        return None
    t = b[i:i + 1]
    j = b.find(b"\r\n", i)
    if j < 0:
        return None
    head = b[i + 1:j]
    k = j + 2
    if t in b"+-:_#,(":
        return k
    if t in b"$=!":
        n = int(head)
        if n < 0:
            return k
        return k + n + 2 if len(b) >= k + n + 2 else None
    if t in b"*~>%|":
        n = int(head)
        if n < 0:
            return k
        m = n * 2 if t in b"%|" else n
        for _ in range(m):
            k = complete(b, k)
            if k is None:
                return None
        return k
    raise ValueError(t)


def main():
    r = redis.Redis(port=PORT)
    r.flushall()
    r.xadd("s", {"name": "alice", "n": "1"}, id="1700000000000-0")
    r.xadd("s", {"name": "bob"}, id="1700000000000-1")
    r.xadd("s", {"k": "v", "k2": ""}, id="1700000000001-*")
    r.xadd("s", {"bin": b"\x00\xff\r\n", "utf": "é😀"}, id="1700000000002-5")
    r.xadd("s", [("dup", "1"), ("dup", "2")], id="1700000000003-0") if False else r.execute_command("XADD", "s", "1700000000003-0", "dup", "1", "dup", "2")
    r.xadd("t", {"a": "1"}, id="5-0")
    r.xgroup_create("s", "g", id="0")
    cmds = [
        ("XRANGE", "s", "-", "+"),
        ("XRANGE", "s", "-", "+", "COUNT", "2"),
        ("XREVRANGE", "s", "+", "-", "COUNT", "2"),
        ("XRANGE", "s", "(1700000000000-1", "+"),
        ("XRANGE", "nosuch", "-", "+"),
        ("XREAD", "COUNT", "10", "STREAMS", "s", "t", "1700000000001-0", "0"),
        ("XREAD", "STREAMS", "s", "$"),
        ("XREADGROUP", "GROUP", "g", "c1", "COUNT", "2", "STREAMS", "s", ">"),
        ("XREADGROUP", "GROUP", "g", "c1", "STREAMS", "s", "0"),
        ("XACK", "s", "g", "1700000000000-0"),
        ("XAUTOCLAIM", "s", "g", "c2", "0", "0-0"),
        ("XLEN", "s"),
        ("MEMORY", "DOCTOR"),
    ]
    out = {"server": r.info()["redis_version"], "cases": []}
    # redis-py parse of the same commands (on a fresh copy of the data, replies are deterministic
    # except for XAUTOCLAIM/XREADGROUP state, so raw runs replay against a re-made dataset)
    for proto in (2, 3):
        main_data(r)
        reps = raw(cmds, proto)
        for c, rep in zip(cmds, reps):
            out["cases"].append({"proto": proto, "cmd": list(c), "reply": rep.hex()})
    json.dump(out, open(OUT, "w"), indent=1)
    print(OUT, len(out["cases"]))


def main_data(r):
    r.flushall()
    r.xadd("s", {"name": "alice", "n": "1"}, id="1700000000000-0")
    r.xadd("s", {"name": "bob"}, id="1700000000000-1")
    r.xadd("s", {"k": "v", "k2": ""}, id="1700000000001-*")
    r.xadd("s", {"bin": b"\x00\xff\r\n", "utf": "é😀"}, id="1700000000002-5")
    r.execute_command("XADD", "s", "1700000000003-0", "dup", "1", "dup", "2")
    r.xadd("t", {"a": "1"}, id="5-0")
    r.xgroup_create("s", "g", id="0")


main()
