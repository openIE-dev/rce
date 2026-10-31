# Mutation test for lib/energy/energy.rce: each mutant must fail the
# comptime checks (package with enc and blake3; crypto, which energy also
# imports, is built into rce) or, for mutants marked
# host, the Ed25519 host checks. Usage: RCE=path/to/rce python3 scripts/mut/energy.py
import os, re, shutil, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
L = lambda p: open(os.path.join(root, p)).read()
src = L('lib/energy/energy.rce'); chk = L('tests/energy/energy_checks.rce'); host = L('tests/energy/energy_host.rce')
events = re.search(r'^# events: (.*)$', host, re.M).group(1).replace(' ', '').split(',')
M = [
    ('vals: [jn(-8), enc.CBytes(b: utf8_of(kid))]', 'vals: [jn(-7), enc.CBytes(b: utf8_of(kid))]', 0),
    ('    if vs[0] != jn(-8)\n', '    if false\n', 0),
    ('    if uk.len() != 0\n', '    if false\n', 0),
    ('let unprot = enc.CMap(keys: [], vals: [])', 'let unprot = enc.CMap(keys: [jn(4)], vals: [enc.CBytes(b: utf8_of(kid))])', 0),
    ('[jt("Signature1"),', '[jt("Signature"),', 0),
    ('let b = en_cat([237, 1], pk)', 'let b = en_cat([237, 0], pk)', 0),
    ('const B58: text = "123456789ABC', 'const B58: text = "123456789ACB', 0),
    ('    if r.verdict == "deny"\n        ks.push', '    if true\n        ks.push', 0),
    ('    if r.partial\n', '    if not r.partial\n', 0),
    ('    if r.chain_ids.len() > 0\n', '    if r.chain_ids.len() >= 0\n', 0),
    ('        if r.verdict != "allow"\n            continue\n', '', 0),
    ('        if names.len() == 0\n', '        if names.len() == 1\n', 0),
    ('            out[i].joules_uj += r.cost_uj', '            out[i].joules_uj += 0', 0),
    ('        if r.budget_remaining_uj != r.budget_uj - s', '        if r.budget_remaining_uj != r.budget_uj', 0),
    ('        if not fields.contains(name)\n', '        if false\n', 0),
    ('    if (r.profile == "jcp/2") != has_payload\n', '    if false\n', 0),
    ('    if r.budget_remaining_uj > r.budget_uj\n', '    if false\n', 0),
    ('    if tag_num(&c) != 18\n', '    if tag_num(&c) != 98\n', 0),
    ('    if sig.len() != 64\n', '    if sig.len() != 65\n', 0),
    ('    if utf8_of(&kt) != kid or kt != r.steward\n', '    if utf8_of(&kt) != kid\n', 0),
    ('    if r.profile != "jcp/2"\n        fail BadReceipt(why: "profile")\n    let bad = check_fields(r, true)', '    if false\n        fail BadReceipt(why: "profile")\n    let bad = check_fields(r, true)', 0),
    ('            if t != "Partial"\n', '            if false\n', 0),
    ('    if r.trust != "control" and r.trust != "data"\n', '    if false\n', 0),
    ('    if r.verdict == "allow" and r.deny_reason != ""\n', '    if false\n', 0),
    ('    if ks.len() != 2 or ks[0] != jn(1) or ks[1] != jn(4)\n', '    if ks.len() != 2\n', 0),
    ('    return crypto.ed_verify(hex_of(pk), sig_structure(&o.prot, &o.payload), hex_of(&o.sig))', '    return true', 1),
    ('enc.CBytes(b: copy(prot)), enc.CBytes(b: [])', 'enc.CBytes(b: []), enc.CBytes(b: [])', 1),
    ('    if hex_of(&pkb) != pk\n        fail BadReceipt(why: "steward")\n', '', 1),
]
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (a, b, need_host) in enumerate(M):
        assert src.count(a) == 1, a
        ms = src.replace(a, b)
        pk = os.path.join(d, 'pk%d' % i); os.mkdir(pk)
        for dep in ['enc/enc', 'hash/blake3']: shutil.copy(os.path.join(root, 'lib', dep + '.rce'), pk)
        open(os.path.join(pk, 'energy.rce'), 'w').write(ms + '\n' + chk)
        dead = subprocess.run([RCE, 'test', pk], capture_output=True).returncode != 0
        if not dead and need_host:
            mf = os.path.join(d, 'm%d.rce' % i); open(mf, 'w').write(ms)
            flat = subprocess.run([sys.executable, os.path.join(root, 'scripts/flatten.py'), mf, os.path.join(root, 'tests/energy/energy_checks.rce'), os.path.join(root, 'tests/energy/energy_host.rce')], capture_output=True, text=True).stdout
            ff = os.path.join(d, 'f%d.rce' % i); open(ff, 'w').write(flat)
            r = subprocess.run([RCE, 'host', 'run', ff, os.path.join(d, 'log%d' % i)], input='\n'.join(events) + '\n', capture_output=True, text=True)
            dead = ('{ good = %d, bad = 0 }' % len(events)) not in r.stdout
        killed += dead
        print('%s %2d %s' % ('killed' if dead else 'SURVIVED', i, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
