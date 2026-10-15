# Generates tests/os/ipc_checks*.rce for lib/os/ipc.rce.
#  - netstrings: the examples of vectors/os/netstrings.txt (D. J.
#    Bernstein), and random frame streams cut into random chunks checked
#    against a decoder written from that text (below);
#  - pipes: random non-blocking write/read/F_SETPIPE_SZ scripts run on a
#    real Linux pipe (os.pipe2 with O_NONBLOCK), counts recorded;
#  - Unix socket addresses: bind(2) and getsockname(2) on this kernel for
#    pathname and abstract names of every length 0 to 110.
# Usage: python3 scripts/gen/ipc_gen.py
import ctypes, fcntl, os, random, socket, struct, tempfile
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
def blit(b):
    return '[' + ', '.join(str(x) for x in b) + ']'
rng = random.Random(41)
rows = []
# netstrings.txt examples
nt = open(os.path.join(root, 'vectors/os/netstrings.txt')).read()
assert '"12:hello world!,"' in nt and '"0:,"' in nt
rows.append('check ns_enc("%s") == "%s"' % (b'hello world!'.hex(), b'12:hello world!,'.hex()))
rows.append('check ns_enc("") == "%s"' % b'0:,'.hex())
def ref_ns(stream):
    """Decoder from netstrings.txt: [len]":"[string]"," with len in ASCII
    decimal without leading zeros (the text: 'Extra zeros at the front
    of [len] are prohibited')."""
    out, i = [], 0
    while True:
        j = i
        while j < len(stream) and 48 <= stream[j] <= 57 and j - i < 10:
            if j == i + 1 and stream[i] == 48:
                return out, 'leading zero'
            j += 1
            if j - i > 9:
                return out, 'length too long'
        if j == len(stream):
            return out, ''
        if j == i or stream[j] != 58:
            return out, "expected digits and ':'"
        n = int(stream[i:j]); end = j + 1 + n
        if len(stream) <= end:
            return out, ''
        if stream[end] != 44:
            return out, "expected ','"
        out.append(stream[j + 1:end]); i = end + 1
for _ in range(300):
    fr = [bytes(rng.choice(b'ab:,0\x00\xff') for _ in range(rng.choice([0, 1, 3, 9, 12]))) for _ in range(rng.randint(0, 4))]
    st = b''.join(b'%d:%s,' % (len(f), f) for f in fr)
    k = rng.random()
    if k < 0.15:
        st = st + b'0' + bytes([rng.choice(b'0123:')]) + b':,'
    elif k < 0.3 and st:
        p = rng.randrange(len(st)); st = st[:p] + bytes([rng.choice(b'x:,9')]) + st[p + 1:]
    elif k < 0.35:
        st = st + b'1234567890:'
    out, bad = ref_ns(st)
    cuts = sorted(rng.sample(range(len(st) + 1), min(len(st) + 1, rng.randint(0, 4))))
    chunks = [st[a:b] for a, b in zip([0] + cuts, cuts + [len(st)])]
    rows.append('check ns_run("%s") == "%s|%s"' % (' '.join(c.hex() or '-' for c in chunks), ';'.join(f.hex() for f in out), bad))
for _ in range(150):
    fr = [bytes(rng.randrange(256) for _ in range(rng.choice([0, 1, 5, 300]))) for _ in range(rng.randint(0, 3))]
    st = b''.join(struct.pack('>I', len(f)) + f for f in fr)
    mx = rng.choice([1 << 20, 5, 300, 4, 299, 0])
    out, bad, i = [], '', 0
    while len(st) - i >= 4:
        n = struct.unpack('>I', st[i:i + 4])[0]
        if n > mx:
            bad = 'frame too large'; break
        if len(st) - i < 4 + n:
            break
        out.append(st[i + 4:i + 4 + n]); i += 4 + n
    cuts = sorted(rng.sample(range(len(st) + 1), min(len(st) + 1, rng.randint(0, 4))))
    chunks = [st[a:b] for a, b in zip([0] + cuts, cuts + [len(st)])]
    rows.append('check lp_run("%s", %d) == "%s|%s"' % (' '.join(c.hex() or '-' for c in chunks), mx, ';'.join(f.hex() for f in out), bad))
# real pipes
F_SETPIPE_SZ, F_GETPIPE_SZ = 1031, 1032
prows = []
for _ in range(500):
    r, w = os.pipe2(os.O_NONBLOCK)
    ops, outs = [], []
    for _ in range(rng.choice([4, 8, 16, 30])):
        k = rng.random()
        if k < 0.5:
            n = rng.choice([0, 1, 7, 100, 4095, 4096, 4097, 5000, 8192, 12000, 20000, 65536, rng.randint(1, 70000)])
            ops.append('w %d' % n)
            try:
                outs.append(str(os.write(w, b'x' * n)))
            except BlockingIOError:
                outs.append('EAGAIN')
        elif k < 0.93:
            n = rng.choice([0, 1, 10, 4096, 5000, 30000, 70000, rng.randint(1, 9000)])
            ops.append('r %d' % n)
            try:
                outs.append(str(len(os.read(r, n))))
            except BlockingIOError:
                outs.append('EAGAIN')
        else:
            n = rng.choice([4096, 8192, 10000, 32768, 65536, 131072])
            ops.append('sz %d' % n)
            try:
                outs.append(str(fcntl.fcntl(w, F_SETPIPE_SZ, n)))
            except OSError:
                outs.append('EBUSY')
    os.close(r); os.close(w)
    prows.append('check pipe_run("%s") == "%s"' % (';'.join(ops), ';'.join(outs)))
# Unix socket addresses
libc = ctypes.CDLL(None, use_errno=True)
class SA(ctypes.Structure):
    _fields_ = [('fam', ctypes.c_ushort), ('path', ctypes.c_char * 108)]
d = tempfile.mkdtemp(); cwd = os.getcwd(); os.chdir(d)
def bind(name):
    s = socket.socket(socket.AF_UNIX)
    sa = SA(); sa.fam = socket.AF_UNIX; ctypes.memmove(ctypes.addressof(sa) + 2, name, min(len(name), 108))
    r = libc.bind(s.fileno(), ctypes.byref(sa), 2 + len(name) if len(name) <= 108 else 2 + 108 + 1)
    if r < 0:
        s.close(); return -1
    out = SA(); ln = ctypes.c_uint(ctypes.sizeof(out)); libc.getsockname(s.fileno(), ctypes.byref(out), ctypes.byref(ln)); s.close()
    return ln.value
s = socket.socket(socket.AF_UNIX); out = SA(); ln = ctypes.c_uint(ctypes.sizeof(out)); libc.getsockname(s.fileno(), ctypes.byref(out), ctypes.byref(ln)); s.close()
rows.append('check UNNAMED_LEN == %d' % ln.value)
for n in range(0, 111):
    rows.append('check unix_addrlen(%d, false) == %d' % (n, bind(b'p%03d' % n + b'q' * (n - 4) if n >= 4 else b'pqrs'[:n])))
    rows.append('check unix_addrlen(%d, true) == %d' % (n, bind(b'\0' + b'a%03d' % n + b'b' * (n - 5) if n >= 5 else b'\0abcd'[:n])))
os.chdir(cwd)
hdr = '# Generated by scripts/gen/ipc_gen.py (kernel %s); do not edit.\n' % os.uname().release
helpers = open(os.path.join(here, 'ipc_helpers.rce')).read()
derived = open(os.path.join(here, 'ipc_derived.rce')).read()
shards = {'ipc_checks.rce': rows + derived.splitlines(), 'ipc_checks_1.rce': prows[:250], 'ipc_checks_2.rce': prows[250:]}
for name, rs in shards.items():
    with open(os.path.join(root, 'tests/os', name), 'w') as f:
        f.write(hdr + helpers + '\n'.join(rs) + '\n')
print(len(rows), len(prows))
