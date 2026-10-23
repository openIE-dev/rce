# Generates tests/os/devices_checks*.rce for lib/os/devices.rce.
#  - UEFI: GUID bytes against Python's uuid bytes_le (the EFI_GUID memory
#    layout); EFI_STATUS values and names parsed from EDK2 MdePkg Base.h;
#    memory and reset types from the enums of UefiMultiPhase.h;
#  - I2C: a C program built from the kernel's own functions, cut verbatim
#    out of vectors/os/linux_i2c-core-base.c, linux_i2c-core-smbus.c and
#    linux_i2c.h (address validity, strict probe check, address bytes,
#    10-bit address bytes, i2c_smbus_pec, i2c_smbus_msg_pec), run over
#    every address and random messages;
#  - SPI: the mode bit constants parsed from uapi spi.h, the CPOL/CPHA
#    table of the kernel's spi-summary, word bit orders;
#  - UART, register fields, transaction checks and grant tiers against
#    the Python references below.
# Usage: python3 scripts/gen/devices_gen.py
import os, random, re, subprocess, tempfile, uuid
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, '../..')
V = lambda n: open(os.path.join(root, 'vectors/os', n)).read()
rng = random.Random(42)
rows = []
def q(s): return '"' + s + '"'
def sp(l): return ' '.join(str(x) for x in l)
def b(x): return 'true' if x else 'false'

# ---- UEFI GUIDs
gs = ['8BE4DF61-93CA-11D2-AA0D-00E098032B8C'] + [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(400)]
for g in gs:
    t = ''.join(c.upper() if rng.random() < .5 else c for c in g)
    u = uuid.UUID(g)
    rows.append('check gb(%s) == %s' % (q(t), q(sp(u.bytes_le))))
    rows.append('check gt(%s) == %s' % (q(sp(u.bytes_le)), q(str(u).upper())))
def guid_ok(t):
    return re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', t) is not None
for _ in range(200):
    t = list(rng.choice(gs)); k = rng.randrange(4)
    if k == 0: del t[rng.randrange(len(t))]
    elif k == 1: t[rng.randrange(len(t))] = rng.choice('-gGxz {}')
    elif k == 2: t.insert(rng.randrange(len(t)), rng.choice('0-a'))
    else:
        i = rng.choice([8, 13, 18, 23]); j = rng.randrange(36); t[i], t[j] = t[j], t[i]
    t = ''.join(t)
    exp = sp(uuid.UUID(t).bytes_le) if guid_ok(t) else '-1'
    rows.append('check gb(%s) == %s' % (q(t), q(exp)))
for n in [0, 1, 15, 17, 32]:
    rows.append('check gt(%s) == "bad"' % q(sp([rng.randrange(256) for _ in range(n)])))
rows.append('check EFI_GLOBAL_VARIABLE == "8BE4DF61-93CA-11D2-AA0D-00E098032B8C"')

# ---- EFI_STATUS from Base.h
base = V('edk2_Base.h')
MAXBIT = 1 << 63
st = {}
for name, kind, n in re.findall(r'#define RETURN_(\w+)\s+ENCODE_(ERROR|WARNING)\s*\((\d+)\)', base):
    st[name] = (MAXBIT | int(n)) if kind == 'ERROR' else int(n)
assert re.search(r'#define RETURN_SUCCESS\s+\(RETURN_STATUS\)\(0\)', base); st['SUCCESS'] = 0
assert len(st) == 41
def s64(v): return v - (1 << 64) if v >= MAXBIT else v
for name, v in st.items():
    rows.append('check sv(%s) == %d' % (q(name), s64(v)))
    rows.append('check status_name(%d) == %s' % (s64(v), q(name)))
    rows.append('check status_is_error(%d) == %s' % (s64(v), b(v >= MAXBIT)))
known = set(st.values())
for v in [MAXBIT | k for k in range(0, 40)] + list(range(0, 12)) + [1 << 62, MAXBIT | (1 << 62)]:
    if v not in known:
        rows.append('check status_name(%d) == ""' % s64(v))
for name in ['WARN_HTTP_ERROR', 'success', 'NOT_FOUN', 'EFI_NOT_FOUND', '-', '']:
    rows.append('check sv(%s) == 1' % q(name))
rows.append('check MAX_BIT == %d' % s64(MAXBIT))

# ---- memory and reset types (enum order)
mp = V('edk2_UefiMultiPhase.h')
def enum(body_name):
    m = re.search(r'typedef enum \{([^{}]*)\}\s*' + body_name + ';', mp, re.S)
    body = re.sub(r'//.*', '', re.sub(r'/\*.*?\*/', '', m.group(1), flags=re.S))
    return [x.strip() for x in body.split(',') if x.strip() and '=' not in x]
mt = enum('EFI_MEMORY_TYPE'); rt = enum('EFI_RESET_TYPE')
assert mt[0] == 'EfiReservedMemoryType' and rt[0] == 'EfiResetCold'
for i, n in enumerate(mt): rows.append('check mty(%s) == %d' % (q(n), i))
for i, n in enumerate(rt): rows.append('check rty(%s) == %d' % (q(n), i))
for n in ['EfiLoadercode', 'EfiMaxMemoryTyp', 'EfiResetCold', 'x']: rows.append('check mty(%s) == -1' % q(n))
for n in ['EfiResetcold', 'EfiReservedMemoryType']: rows.append('check rty(%s) == -1' % q(n))

# ---- load options (UEFI 2.10 3.1.1: Boot#### with upper-case hex digits)
for _ in range(150):
    p = rng.choice(['Boot', 'Driver', 'SysPrep', 'PlatformRecovery', 'Key'])
    n = rng.choice([0, 65535, rng.randrange(65536), rng.randrange(16)])
    name = '%s%04X' % (p, n)
    rows.append('check lon(%s, %d) == %s' % (q(p), n, q(name)))
    rows.append('check lonum(%s, %s) == %d' % (q(name), q(p), n))
for n in [-1, 65536, 1 << 20]: rows.append('check lon("Boot", %d) == "bad"' % n)
for _ in range(150):
    p = 'Boot'
    s = rng.choice(['Boot', 'boot', 'Boo', 'Driver', 'Boot0']) + ''.join(rng.choice('0123456789ABCDEFabcdefG') for _ in range(rng.choice([3, 4, 4, 4, 5])))
    exp = int(s[4:], 16) if re.fullmatch(r'Boot[0-9A-F]{4}', s) else -1
    rows.append('check lonum(%s, "Boot") == %d' % (q(s), exp))

# ---- I2C through the kernel's code
def cut(src, sig):
    i = src.index(sig); j = src.index('\n}\n', i)
    return src[i:j + 3]
smb = V('linux_i2c-core-smbus.c'); cb = V('linux_i2c-core-base.c'); ih = V('linux_i2c.h'); uh = V('linux_uapi_i2c.h')
I2C_M_RD = int(re.search(r'#define I2C_M_RD\s+(0x\w+)', uh).group(1), 16)
I2C_M_TEN = int(re.search(r'#define I2C_M_TEN\s+(0x\w+)', uh).group(1), 16)
rows.append('check I2C_M_RD == %d' % I2C_M_RD); rows.append('check I2C_M_TEN == %d' % I2C_M_TEN)
csrc = '''#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t u8; typedef uint16_t u16;
#define EINVAL 22
#define EXPORT_SYMBOL(x)
#define EXPORT_SYMBOL_GPL(x)
#define GENMASK(h, l) (((~0U) << (l)) & (~0U >> (31 - (h))))
#define I2C_M_RD %d
#define I2C_CLIENT_TEN %d
struct i2c_msg { u16 addr; u16 flags; u16 len; u8 *buf; };
''' % (I2C_M_RD, I2C_M_TEN)
csrc += re.search(r'#define POLY.*\n', smb).group(0)
for f in [cut(smb, 'static u8 crc8(u16 data)'), cut(smb, 'u8 i2c_smbus_pec(u8 crc, u8 *p, size_t count)'),
          cut(ih, 'static inline u8 i2c_8bit_addr_from_msg'), cut(ih, 'static inline u8 i2c_10bit_addr_hi_from_msg'),
          cut(ih, 'static inline u8 i2c_10bit_addr_lo_from_msg'), cut(smb, 'static u8 i2c_smbus_msg_pec(u8 pec, struct i2c_msg *msg)'),
          cut(cb, 'static int i2c_check_addr_validity(unsigned int addr, unsigned short flags)'),
          cut(cb, 'int i2c_check_7bit_addr_validity_strict(unsigned short addr)')]:
    csrc += f.replace('EXPORT_SYMBOL(i2c_smbus_pec);', '') + '\n'
csrc += r'''
int main(void) {
    char op[8];
    while (scanf("%7s", op) == 1) {
        if (op[0] == 'v') { unsigned a, t; scanf("%u %u", &a, &t); printf("%d\n", i2c_check_addr_validity(a, t ? I2C_CLIENT_TEN : 0)); }
        else if (op[0] == 's') { unsigned a; scanf("%u", &a); printf("%d\n", i2c_check_7bit_addr_validity_strict(a)); }
        else if (op[0] == 'a') { unsigned a, f; scanf("%u %u", &a, &f); struct i2c_msg m = {a, f, 0, 0};
            printf("%u %u %u\n", i2c_8bit_addr_from_msg(&m), i2c_10bit_addr_hi_from_msg(&m), i2c_10bit_addr_lo_from_msg(&m)); }
        else if (op[0] == 'm') { int n; scanf("%d", &n); u8 pec = 0;
            for (int k = 0; k < n; k++) { unsigned a, f, l; u8 buf[64]; scanf("%u %u %u", &a, &f, &l);
                for (unsigned j = 0; j < l; j++) { unsigned x; scanf("%u", &x); buf[j] = x; }
                struct i2c_msg m = {a, f, l, buf}; pec = i2c_smbus_msg_pec(pec, &m); }
            printf("%u\n", pec); }
        else if (op[0] == 'p') { unsigned l; u8 buf[256]; scanf("%u", &l);
            for (unsigned j = 0; j < l; j++) { unsigned x; scanf("%u", &x); buf[j] = x; }
            printf("%u\n", i2c_smbus_pec(0, buf, l)); }
    }
    return 0;
}
'''
os.makedirs(os.path.join(root, 'scripts/ref'), exist_ok=True)
open(os.path.join(root, 'scripts/ref/i2c_ref.c'), 'w').write('/* Generated by scripts/gen/devices_gen.py from the vendored kernel sources. */\n' + csrc)
d = tempfile.mkdtemp(); exe = os.path.join(d, 'i2c')
subprocess.check_call(['cc', '-O1', '-o', exe, os.path.join(root, 'scripts/ref/i2c_ref.c')])
cmds, kinds = [], []
for a in range(0, 1100): cmds.append('v %d 0' % a); kinds.append(('v', a, False)); cmds.append('v %d 1' % a); kinds.append(('v', a, True))
for a in range(0, 256): cmds.append('s %d' % a); kinds.append(('s', a))
for _ in range(400):
    a = rng.randrange(1024); f = rng.choice([0, 1, I2C_M_TEN, I2C_M_TEN | 1, rng.randrange(65536)])
    cmds.append('a %d %d' % (a, f)); kinds.append(('a', a, f))
for _ in range(300):
    l = rng.choice([0, 1, 2, 3, 8, 32, rng.randrange(64)]); bs = [rng.randrange(256) for _ in range(l)]
    cmds.append('p %d %s' % (l, sp(bs))); kinds.append(('p', bs))
for _ in range(300):
    ms = [(rng.randrange(1, 128), rng.choice([0, 1]), [rng.randrange(256) for _ in range(rng.randrange(0, 34))]) for _ in range(rng.choice([1, 2, 2, 3]))]
    cmds.append('m %d ' % len(ms) + ' '.join('%d %d %d %s' % (a, f, len(bs), sp(bs)) for a, f, bs in ms)); kinds.append(('m', ms))
out = subprocess.run([exe], input='\n'.join(cmds) + '\n', capture_output=True, text=True, check=True).stdout.split('\n')
for k, o in zip(kinds, out):
    if k[0] == 'v':
        rows.append('check i2c_addr_valid(%d, %s) == %s' % (k[1], b(k[2]), b(int(o) == 0)))
    elif k[0] == 's':
        rows.append('check i2c_addr_valid_strict(%d) == %s' % (k[1], b(int(o) == 0)))
    elif k[0] == 'a':
        a8, hi, lo = map(int, o.split())
        rows.append('check i2c_addr_byte(%d, %d) == %d' % (k[1], k[2], a8))
        rows.append('check tbb(%d, %s) == "%d %d"' % (k[1], b(k[2] & 1), hi, lo))
    elif k[0] == 'p':
        rows.append('check pecof(%s) == %s' % (q(sp(k[1])), o))
    else:
        rows.append('check mpec(%s) == %s' % (q(';'.join(sp([a, f] + bs) for a, f, bs in k[1])), o))

# ---- transactions (embedded-hal mock semantics)
def tref(e, a):
    for i, x in enumerate(a):
        if i >= len(e) or e[i] != x: return i
    return len(a) if len(a) < len(e) else -1
def tspec(l): return ';'.join(sp([op, ad] + bs) for op, ad, bs in l)
for _ in range(300):
    e = [(rng.randrange(2), rng.choice([0x48, 0x50, 0x68]), [rng.randrange(4) for _ in range(rng.randrange(3))]) for _ in range(rng.randrange(0, 5))]
    a = [list(x) for x in e]
    k = rng.randrange(5)
    if k == 1 and a: del a[rng.randrange(len(a))]
    elif k == 2: a.append([rng.randrange(2), 0x48, [1]])
    elif k == 3 and a:
        x = a[rng.randrange(len(a))]; f = rng.randrange(3)
        if f == 0: x[0] ^= 1
        elif f == 1: x[1] += 1
        else: x[2] = x[2] + [7]
    a = [tuple(x) for x in a]
    rows.append('check tc(%s, %s) == %d' % (q(tspec(e)), q(tspec(a)), tref(e, a)))

# ---- SPI
sh = V('linux_spi.h')
defs = {n: int(v) for n, v in re.findall(r'#define\s+(SPI_\w+)\s+_BITUL\((\d+)\)', sh)}
for n in ['SPI_CPHA', 'SPI_CPOL', 'SPI_CS_HIGH', 'SPI_LSB_FIRST', 'SPI_3WIRE', 'SPI_LOOP', 'SPI_NO_CS', 'SPI_READY']:
    rows.append('check %s == %d' % (n, 1 << defs[n]))
assert re.search(r'#define SPI_MODE_USER_MASK\s+\(_BITUL\(19\) - 1\)', sh)
rows.append('check SPI_MODE_USER_MASK == %d' % ((1 << 19) - 1))
# Documentation/spi/spi-summary.rst: CPOL is the idle clock level; CPHA 0
# samples on the leading edge, 1 on the trailing edge.
for m in range(0, 1 << 9):
    cpol, cpha = (m >> defs['SPI_CPOL']) & 1, (m >> defs['SPI_CPHA']) & 1
    leading_rises = cpol == 0
    rows.append('check spi_mode_number(%d) == %d' % (m, cpol * 2 + cpha))
    rows.append('check spi_idle_high(%d) == %s' % (m, b(cpol)))
    rows.append('check spi_sample_rising(%d) == %s' % (m, b(leading_rises if cpha == 0 else not leading_rises)))
for _ in range(300):
    bits = rng.choice([4, 8, 9, 12, 16, 24, 32]); w = rng.getrandbits(bits); m = rng.randrange(16)
    bs = [(w >> k) & 1 for k in (range(bits) if m & 8 else range(bits - 1, -1, -1))]
    rows.append('check swb(%d, %d, %d) == %s' % (w, bits, m, q(sp(bs))))
    rows.append('check sbw(%s, %d) == %d' % (q(sp(bs)), m, w))

# ---- UART
def par(v, d, p):
    o = bin(v & ((1 << d) - 1)).count('1') & 1
    return {'E': o, 'O': 1 - o, 'M': 1, 'S': 0}[p]
def frame(v, d, p, s):
    f = [0] + [(v >> i) & 1 for i in range(d)]
    if p != 'N': f.append(par(v, d, p))
    return f + [1] * s
def dec(line, d, p, s):
    out, i, n = [], 0, 1 + d + (p != 'N') + s
    while i < len(line):
        if line[i] == 1: i += 1; continue
        if i + n > len(line): break
        v = sum(line[i + 1 + k] << k for k in range(d)); r = v
        if p != 'N' and line[i + 1 + d] != par(v, d, p): r = 'P'
        if any(line[i + n - s + k] != 1 for k in range(s)): r = 'F'
        out.append(str(r)); i += n
    return ' '.join(out)
for _ in range(400):
    d = rng.choice([5, 6, 7, 8]); p = rng.choice('NEOMS'); s = rng.choice([1, 2])
    vs = [rng.getrandbits(d) for _ in range(rng.randrange(1, 5))]
    rows.append('check uf(%d, %d, %s, %d) == %s' % (vs[0], d, q(p), s, q(sp(frame(vs[0], d, p, s)))))
    line = [1] * rng.randrange(3)
    for v in vs: line += frame(v, d, p, s) + [1] * rng.randrange(3)
    for _ in range(rng.choice([0, 0, 1, 2])): line[rng.randrange(len(line))] ^= 1
    if rng.random() < .2: line = line[:rng.randrange(len(line))]
    rows.append('check ud(%s, %d, %s, %d) == %s' % (q(sp(line)), d, q(p), s, q(dec(line, d, p, s))))
for baud in [50, 75, 110, 300, 1200, 2400, 9600, 19200, 38400, 57600, 115200, 7, 230400, 0, -9600, 14400, 28800, 1000]:
    rows.append('check uart_divisor(%d) == %d' % (baud, 115200 // baud if baud > 0 and 115200 % baud == 0 else -1))

# ---- register fields
for _ in range(400):
    w = rng.randrange(1, 17); l = rng.randrange(0, 33 - w); r = rng.getrandbits(32)
    m = ((1 << w) - 1) << l
    rows.append('check fg(%d, %d, %d) == %d' % (r, l, w, (r & m) >> l))
    a = rng.choice(['rw', 'ro', 'w1c', 'wo']); v = rng.choice([rng.getrandbits(w), 1 << w, -1, (1 << w) - 1])
    if v < 0 or v >= 1 << w or a == 'ro': e = -1
    elif a == 'w1c': e = r & ~((v << l) & m)
    else: e = (r & ~m) | (v << l)
    rows.append('check fset(%d, %d, %d, %s, %d) == %d' % (r, l, w, q(a), v, e))
for _ in range(200):
    W = rng.choice([8, 16, 32]); fs = [(rng.randrange(0, W + 2), rng.randrange(1, 9)) for _ in range(rng.randrange(1, 5))]
    used, ok = 0, True
    for l, w in fs:
        m = ((1 << w) - 1) << l
        if l + w > W or used & m: ok = False; break
        used |= m
    rows.append('check fok(%s, %d) == %s' % (q(';'.join(sp(f) for f in fs)), W, b(ok)))

# ---- Sandbox device classes (Sandbox v0.1 5.1)
cls = ['Cpu', 'Simd', 'Gpu', 'Npu', 'Tpu', 'Lpu', 'Photonic', 'Quantum']
rows.append('check DEVICE_CLASSES == %s' % q(' '.join(cls)))
for c in cls + ['cpu', 'Fpga', '']:
    for pt in [False, True]:
        for cf in [False, True]:
            if c not in cls: t = -1
            else:
                t = 0 if c in ('Cpu', 'Simd') else 2
                if pt: t = max(t, 5)
                if cf: t = 6
            rows.append('check gtier(%s, %s, %s) == %d' % (q(c), b(pt), b(cf), t))

helpers = open(os.path.join(here, 'devices_helpers.rce')).read()
hdr = '# Generated by scripts/gen/devices_gen.py; do not edit.\n'
N = 4; per = (len(rows) + N - 1) // N
for k in range(N):
    name = 'devices_checks.rce' if k == 0 else 'devices_checks_%d.rce' % k
    with open(os.path.join(root, 'tests/os', name), 'w') as f:
        f.write(hdr + helpers + '\n'.join(rows[k * per:(k + 1) * per]) + '\n')
print(len(rows), 'checks')
