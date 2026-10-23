# Generates tests/os/fs_checks*.rce: random descriptor-relative operation
# scripts run against the real Linux filesystem (Python os.* with dir_fd on
# a fresh temporary directory; opens go through openat2(2) with
# RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS, so a path leaving the preopen
# fails with EXDEV as WASI's not-permitted). The recorded outcomes are the
# expected values for lib/os/fs.rce's in-memory model.
# Usage: python3 scripts/gen/fs_gen.py [seed] [n]   (writes into tests/os)
import ctypes, errno, os, random, shutil, sys, tempfile
libc = ctypes.CDLL(None, use_errno=True)
SYS_openat2 = 437
class OpenHow(ctypes.Structure):
    _fields_ = [('flags', ctypes.c_uint64), ('mode', ctypes.c_uint64), ('resolve', ctypes.c_uint64)]
RESOLVE_NO_SYMLINKS, RESOLVE_BENEATH = 0x04, 0x08
def openat2(dfd, path, flags):
    how = OpenHow(flags | os.O_CLOEXEC, 0o644 if flags & os.O_CREAT else 0, RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)
    r = libc.syscall(SYS_openat2, dfd, path.encode(), ctypes.byref(how), ctypes.sizeof(how))
    if r < 0:
        e = ctypes.get_errno(); raise OSError(e, os.strerror(e))
    return r
CODE = {errno.EEXIST: 'exist', errno.ENOENT: 'no-entry', errno.ENOTDIR: 'not-directory',
        errno.EISDIR: 'is-directory', errno.ENOTEMPTY: 'not-empty', errno.EINVAL: 'invalid',
        errno.EBADF: 'bad-descriptor', errno.EXDEV: 'not-permitted', errno.EBUSY: 'busy'}
NAMES = ['a', 'b', 'c']
def rpath(rng):
    if rng.random() < 0.03:
        return rng.choice(['.', '..', '/a', './', 'a/', 'a//b', '../a'])
    n = rng.choice([1, 1, 1, 1, 2, 2, 3])
    cs = [rng.choice(NAMES + NAMES + ['.', '..'] if rng.random() < 0.25 else NAMES) for _ in range(n)]
    p = '/'.join(cs)
    if rng.random() < 0.08:
        p += '/'
    return p
def lit(t):
    return t.replace('\0', '\\u{0}')
class Real:
    """Runs operations on a fresh real directory; outcome text per op."""
    def __init__(self):
        self.top = tempfile.mkdtemp(); root = os.path.join(self.top, 'r'); os.mkdir(root)
        self.rfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        self.hs = []  # [fd, live]
    def done(self):
        for fd, l in self.hs:
            if l:
                os.close(fd)
        os.close(self.rfd); shutil.rmtree(self.top)
    def escapes(self, p):
        try:
            d = os.path.dirname(p.rstrip('/')) or '.'
            fd = openat2(self.rfd, d, os.O_PATH); os.close(fd)
            if p.rstrip('/').split('/')[-1] == '..':
                fd = openat2(self.rfd, p, os.O_PATH); os.close(fd)
            return False
        except OSError as e:
            return e.errno == errno.EXDEV
    def size(self, i):
        st = os.fstat(self.hs[i][0])
        return None if os.path.stat.S_ISDIR(st.st_mode) else st.st_size
    def do(self, op):
        a = op.split(' '); k = a[0]; rfd = self.rfd
        try:
            if k == 'open':
                fl = a[2]
                f = (os.O_RDWR if 'r' in fl and 'w' in fl else os.O_WRONLY if 'w' in fl else os.O_RDONLY)
                f |= (os.O_CREAT if 'c' in fl else 0) | (os.O_EXCL if 'x' in fl else 0) | (os.O_TRUNC if 't' in fl else 0) | (os.O_DIRECTORY if 'd' in fl else 0) | (os.O_APPEND if 'a' in fl else 0)
                fd = openat2(rfd, a[1], f); self.hs.append([fd, True]); return 'h%d' % (len(self.hs) - 1)
            if k in ('read', 'write', 'seek', 'trunc', 'close'):
                i = int(a[1]); fd, live = self.hs[i]
                if not live:
                    assert fd not in [h[0] for h in self.hs if h[1]]
                if k == 'read':
                    return '=' + lit(os.read(fd, int(a[2])).decode())
                if k == 'write':
                    os.write(fd, a[2].encode()); return 'ok'
                if k == 'seek':
                    os.lseek(fd, int(a[2]), 0); return 'ok'
                if k == 'trunc':
                    os.ftruncate(fd, int(a[2])); return 'ok'
                os.close(fd); self.hs[i][1] = False; return 'ok'
            if k == 'mkdir':
                os.mkdir(a[1], dir_fd=rfd); return 'ok'
            if k == 'unlink':
                os.unlink(a[1], dir_fd=rfd); return 'ok'
            if k == 'rmdir':
                os.rmdir(a[1], dir_fd=rfd); return 'ok'
            if k == 'rename':
                os.rename(a[1], a[2], src_dir_fd=rfd, dst_dir_fd=rfd); return 'ok'
            if k == 'stat':
                # through openat2 too, so the lookup stays beneath the preopen
                fd = openat2(rfd, a[1], os.O_PATH)
                st = os.fstat(fd); os.close(fd)
                return 'd' if os.path.stat.S_ISDIR(st.st_mode) else 'f%d' % st.st_size
            if k == 'list':
                fd = openat2(rfd, a[1], os.O_RDONLY | os.O_DIRECTORY)
                try:
                    return ':' + ','.join(sorted(os.listdir(fd)))
                finally:
                    os.close(fd)
            raise ValueError(op)
        except OSError as e:
            if e.errno not in CODE:
                raise
            return CODE[e.errno]
def run(rng, nops):
    global NAMES
    NAMES = rng.choice([['a', 'b', 'c'], ['a', 'b']])
    r = Real(); ops, outs = [], []
    while len(ops) < nops:
        k = rng.random(); op = None
        if k < 0.2:
            fl = ''.join(c for c in 'rwcxtda' if rng.random() < {'r': .6, 'w': .5, 'c': .5, 'x': .15, 't': .15, 'd': .1, 'a': .1}[c])
            if 'r' not in fl and 'w' not in fl:
                fl = 'r' + fl
            op = 'open %s %s' % (rpath(rng), fl)
        elif k < 0.45 and r.hs:
            # a closed handle is used only while its fd number is not reused
            ok = [i for i, h in enumerate(r.hs) if h[1] or h[0] not in [g[0] for g in r.hs if g[1]]]
            i = rng.choice(ok); c = rng.random(); sz = r.size(i) if r.hs[i][1] else 0
            n = 0 if sz is None else rng.randint(0, sz + 2)
            if c < 0.3:
                op = 'read %d %d' % (i, rng.choice([1, 2, 4, 8]))
            elif c < 0.6:
                op = 'write %d %s' % (i, ''.join(rng.choice('xyz') for _ in range(rng.randint(1, 4))))
            elif c < 0.75:
                op = 'seek %d %d' % (i, n)
            elif c < 0.85:
                op = 'trunc %d %d' % (i, n)
            else:
                op = 'close %d' % i
        else:
            p = rpath(rng)
            if r.escapes(p):
                continue
            if k < 0.62:
                op = 'mkdir ' + p
            elif k < 0.68:
                op = 'unlink ' + p
            elif k < 0.76:
                op = 'rmdir ' + p
            elif k < 0.86:
                q = rpath(rng)
                if r.escapes(q):
                    continue
                op = 'rename %s %s' % (p, q)
            elif k < 0.94:
                op = 'stat ' + p
            else:
                op = 'list ' + p
        ops.append(op); outs.append(r.do(op))
    r.done()
    return ';'.join(ops), ';'.join(outs)
# Hand-written scripts for cases the random walk rarely reaches; their
# outcomes are recorded from the real filesystem like the others.
EDGE = [
    'stat ;mkdir ;open  r;unlink ;rmdir ;rename  a',
    'open f wc;write 0 abcdef;close 0;open f rt;stat f;read 1 4',
    'open f wc;write 0 abc;close 0;write 0 x;read 0 1;seek 0 0;trunc 0 1;close 0',
    'open f rwc;write 0 ab;trunc 0 5;stat f;seek 0 0;read 0 8;seek 0 7;write 0 z;stat f;seek 0 0;read 0 9',
    'open f wc;open f r;write 0 abc;read 1 2;unlink f;read 1 4;write 0 d;seek 1 0;read 1 9;stat f',
    'open f wac;write 0 ab;seek 0 0;write 0 cd;open f r;read 1 9',
    'mkdir a;mkdir a/b;mkdir a/b/c;rename a a/b/c/d;rename a/b/c a;rename a/b a/b/c/x;rename a/b/c a/b;list a/b',
    'mkdir a;mkdir . ;mkdir ..;mkdir a/.;mkdir a/..;mkdir ./;open . r;read 0 1;trunc 0 0;write 0 x',
    'open f wc;rename f f/;rename f g/;rename f f;stat f;mkdir d;rename d d;rename f d;rename d f;list .',
    'mkdir ..;rmdir ..;unlink ..;rename .. a;rename a ..;stat ..;list ..;open .. r;mkdir a;rmdir a/..;unlink a/..;stat a/..;list a/..;rmdir a/.;unlink a/.',
    'mkdir a;open a/f wc;rename a/f a;mkdir a/d;open a/d/g wc;rename a/d/g a;rename a/d/g a/d/..;stat a/d/g',
    'mkdir a;open a/f wc;mkdir b;rename b a;rename a b;list b;rmdir b',
]
def run_fixed(script):
    r = Real(); outs = [r.do(op) for op in script.split(';')]; r.done()
    return script, ';'.join(outs)
if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 38
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    rng = random.Random(seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tests/os')
    rows = [run_fixed(e) for e in EDGE] + [run(rng, rng.choice([6, 10, 16, 24, 40])) for _ in range(n)]
    n = len(rows)
    per = 150
    for s in range(0, n, per):
        name = 'fs_checks.rce' if s == 0 else 'fs_checks_%d.rce' % (s // per)
        with open(os.path.join(out, name), 'w') as f:
            f.write('# Generated by scripts/gen/fs_gen.py from the real Linux filesystem (kernel %s); do not edit.\n' % os.uname().release)
            for a, b in rows[s:s + per]:
                f.write('check script("%s") == "%s"\n' % (a, b))
