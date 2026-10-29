# Usage: /path/to/python scripts/gen/energy_gen.py (needs cbor2, blake3,
# cryptography, base58), then replace the lines after "# ---- generated"
# in tests/energy/energy_checks.rce and tests/energy/energy_host.rce.
# Builds sample jcp/2 receipts (JCP v0.3 section 8) with cbor2 (keys
# sorted by encoded bytes, RFC 8949 4.2.1), blake3 ids, COSE_Sign1
# envelopes (RFC 9052 Sig_structure, alg -8) signed by `cryptography`
# Ed25519 with the RFC 8032 test 1 and 2 seeds, and did:key ids. Base58
# vectors are from draft-msporny-base58-03 section 5. Derived vectors.
import sys, cbor2, blake3, base58
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
out = sys.argv[1] if len(sys.argv) > 1 else 'checks'
def canon(v):
    if isinstance(v, dict):
        items = sorted(((canon(k), canon(x)) for k, x in v.items()), key=lambda kv: kv[0])
        n = len(items)
        head = bytes([0xa0 | n]) if n < 24 else bytes([0xb8, n])
        return head + b''.join(k + x for k, x in items)
    if isinstance(v, list):
        n = len(v); head = bytes([0x80 | n]) if n < 24 else bytes([0x98, n])
        return head + b''.join(canon(x) for x in v)
    if isinstance(v, cbor2.CBORTag): return cbor2.dumps(cbor2.CBORTag(v.tag, None))[:1] + canon(v.value)
    return cbor2.dumps(v)
SEEDS = ['9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60', '4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb']
def pk(seed): return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed)).public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
def did(p): return 'did:key:z' + base58.b58encode(b'\xed\x01' + p).decode()
D0, D1 = did(pk(SEEDS[0])), did(pk(SEEDS[1]))
G1 = blake3.blake3(b'grant one').hexdigest(); G0 = blake3.blake3(b'root grant').hexdigest()
RES = blake3.blake3(b'tool:search').hexdigest()
P = [blake3.blake3(b'payload %d' % i).hexdigest() for i in range(6)]
# (profile, grant, issuer, subject, resource, action, payload, frame, trust, verdict, reason, cost, prov, partial, remaining, budget, steward, chain)
S = [
 ('jcp/2', G1, D1, D0, RES, 'search', P[0], 'tool-call', 'control', 'allow', '', 1200, 'HwShunt', False, 8800, 10000, D1, []),
 ('jcp/2', G1, D1, D0, RES, 'search', P[1], 'tool-call', 'data', 'deny', 'data-frame', 0, 'Estimator', False, 8800, 10000, D1, []),
 ('jcp/2', G1, D1, D0, RES, 'search', P[2], 'tool-call', 'control', 'allow', '', 3000, 'ModelBased', True, 5800, 10000, D1, []),
 ('jcp/2', G1, D1, D0, RES, 'search', P[3], 'tool-call', 'control', 'deny', 'budget-exceeded', 6000, 'HwShunt', False, 5800, 10000, D1, []),
 ('jcp/2', G1, D1, D1, RES, 'fetch', P[4], 'agent-call', 'control', 'allow', '', 800, 'Estimator', False, 5000, 10000, D1, [G0, G1]),
 ('jcp/2', G1, D1, D1, RES, 'fetch', P[5], 'agent-call', 'control', 'allow', '', 5000, 'HwShunt', False, 0, 10000, D1, [G0, G1]),
]
def core(r, profile=None):
    m = {'jcp_receipt': profile or r[0], 'grant_id': r[1], 'issuer': r[2], 'subject': r[3], 'resource': r[4], 'action': r[5]}
    if (profile or r[0]) == 'jcp/2': m['payload_hash'] = r[6]
    m.update({'frame_kind': r[7], 'trust': r[8], 'verdict': r[9]})
    if r[9] == 'deny': m['deny_reason'] = r[10]
    m.update({'cost_uj': r[11], 'energy_provenance': r[12]})
    if r[13]: m['energy_coverage'] = 'Partial'
    m.update({'budget_remaining_uj': r[14], 'budget_uj': r[15], 'steward': r[16]})
    if r[17]: m['chain_ids'] = list(r[17])
    return m
def prot(kid): return canon({1: -8, 4: kid.encode()})
def sigs(payload, kid): return canon(['Signature1', prot(kid), b'', payload])
def env(payload, kid, sig): return canon(cbor2.CBORTag(18, [prot(kid), {}, payload, sig]))
def sign(seed, payload, kid): return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed)).sign(sigs(payload, kid))
import json, os
CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../vectors/energy/jcp-conformance-v1')
def conf():
    for f in ['vectors.json', 'legacy_jcp1.json', 'dual_ceilings.json']:
        for r in json.load(open(os.path.join(CONF, f)))['receipts']:
            yield f, r
def wire_tuple(w):
    return (w['jcp_receipt'], w['grant_id'], w['issuer'], w['subject'], w['resource'], w['action'], w.get('payload_hash', ''), w['frame_kind'], w['trust'], w['verdict'], w.get('deny_reason', ''), w['cost_uj'], w['energy_provenance'], w.get('energy_coverage') == 'Partial', w['budget_remaining_uj'], w['budget_uj'], w['steward'], w.get('chain_ids', []))
def q(s): return '"' + s + '"'
def rce_receipt(r):
    return 'Receipt(profile: %s, grant_id: %s, issuer: %s, subject: %s, resource: %s, action: %s, payload_hash: %s, frame_kind: %s, trust: %s, verdict: %s, deny_reason: %s, cost_uj: %d, provenance: %s, partial: %s, budget_remaining_uj: %d, budget_uj: %d, steward: %s, chain_ids: [%s])' % (
        q(r[0]), q(r[1]), q(r[2]), q(r[3]), q(r[4]), q(r[5]), q(r[6]), q(r[7]), q(r[8]), q(r[9]), q(r[10]), r[11], q(r[12]), 'true' if r[13] else 'false', r[14], r[15], q(r[16]), ', '.join(q(g) for g in r[17]))
if out == 'checks':
    print('fn sample(n: i64) -> Receipt')
    for i, r in enumerate(S):
        print('    if n == %d\n        return %s' % (i, rce_receipt(r)))
    print('    return %s' % rce_receipt(S[0]))
    print()
    print('# draft-msporny-base58-03 section 5')
    for data, enc in [(b'Hello World!', '2NEpo7TZRRrLZSi2U'), (b'The quick brown fox jumps over the lazy dog.', 'USm3fpXnKG5EUBx2ndxBDMPVciP5hGey2Jh4NDv6gmeo1LkMeiKrLJUUBk6Z'), (bytes.fromhex('0000287fb4cd'), '11233QC4')]:
        assert base58.b58encode(data).decode() == enc
        print('check b58_ok("%s", "%s")' % (data.hex(), enc))
    print('# derived: did:key of the RFC 8032 test 1 and 2 public keys')
    for s in SEEDS: print('check did_ok("%s", "%s")' % (pk(s).hex(), did(pk(s))))
    print('# derived: cores, ids, Sig_structures and envelopes (cbor2, blake3, cryptography)')
    for i, r in enumerate(S):
        c = canon(core(r)); sg = sign(SEEDS[1], c, r[16])
        print('check core_ok(%d, "%s", "%s")' % (i, c.hex(), blake3.blake3(c).hexdigest()))
        print('check sigs_ok(%d, "%s")' % (i, sigs(c, r[16]).hex()))
        print('check env_ok(%d, "%s", "%s")' % (i, sg.hex(), env(c, r[16], sg).hex()))
    tally = {}
    spent = {}
    for r in S:
        k = (r[1], r[3]); t = tally.setdefault(k, [0, 0, 0])
        if r[9] == 'allow':
            t[0] += 1; t[2] += r[11]
            for g in (r[17] or [r[1]]): spent[g] = spent.get(g, 0) + r[11]
        else: t[1] += 1
    print('# derived: replay (8.1) and group accounting (8.2) of the stream')
    print('check spent_ok("%s", %d) and spent_ok("%s", %d)' % (G1, spent[G1], G0, spent[G0]))
    for (g, sub), t in tally.items(): print('check tally_ok("%s", "%s", %d, %d, %d)' % (g, sub, t[0], t[1], t[2]))
    print('# derived: a jcp/1 core (no payload_hash) opens; malformed envelopes do not open')
    c1 = canon(core(S[0], 'jcp/1')); print('check opens_v1("%s")' % env(c1, S[0][16], bytes(64)).hex())
    bad = []
    m = core(S[0]); del m['payload_hash']; bad.append(('jcp/2 without payload_hash', canon(m)))
    m = core(S[0], 'jcp/1'); m['payload_hash'] = P[0]; bad.append(('jcp/1 with payload_hash', canon(m)))
    m = core(S[0]); m['jcp_receipt'] = 'jcp/3'; bad.append(('unknown profile', canon(m)))
    m = core(S[0]); m['extra'] = 'x'; bad.append(('unknown field', canon(m)))
    m = core(S[0]); m['cost_uj'] = -5; bad.append(('negative cost', canon(m)))
    m = core(S[0]); m['cost_uj'] = '5'; bad.append(('text cost', canon(m)))
    m = core(S[0]); m['energy_coverage'] = 'Whole'; bad.append(('Whole written out', canon(m)))
    m = core(S[0]); m['deny_reason'] = 'x'; bad.append(('allow with reason', canon(m)))
    m = core(S[1]); del m['deny_reason']; bad.append(('deny without reason', canon(m)))
    m = core(S[0]); del m['steward']; bad.append(('no steward', canon(m)))
    m = core(S[0]); m['trust'] = 'admin'; bad.append(('bad trust', canon(m)))
    for why, c in bad: print('check rejects(%s, "%s")' % (q(why), env(c, S[0][16], bytes(64)).hex()))
    c0 = canon(core(S[0]))
    print('check rejects("non-canonical text head", "%s")' % env(c0[:1] + bytes([0x78, c0[1] & 31]) + c0[2:], S[0][16], bytes(64)).hex())
    print('check rejects("wrong kid", "%s")' % env(c0, D0, bytes(64)).hex())
    print('check rejects("tag 98", "%s")' % canon(cbor2.CBORTag(98, [prot(S[0][16]), {}, c0, bytes(64)])).hex())
    print('check rejects("kid label 5", "%s")' % canon(cbor2.CBORTag(18, [canon({1: -8, 5: S[0][16].encode()}), {}, c0, bytes(64)])).hex())
    print('check rejects("kid unprotected", "%s")' % canon(cbor2.CBORTag(18, [canon({1: -8}), {4: S[0][16].encode()}, c0, bytes(64)])).hex())
    print('check rejects("unprotected not empty", "%s")' % canon(cbor2.CBORTag(18, [prot(S[0][16]), {3: 0}, c0, bytes(64)])).hex())
    print('check rejects("alg -7", "%s")' % canon(cbor2.CBORTag(18, [canon({1: -7, 4: S[0][16].encode()}), {}, c0, bytes(64)])).hex())
    print('# JCP conformance/v1 (official): open, published core, wire fields, re-mint')
    for f, r in conf():
        core = cbor2.loads(cbor2.loads(bytes.fromhex(r['envelope_hex'])).value[2])
        assert core == r['wire']
        c = cbor2.loads(bytes.fromhex(r['envelope_hex'])).value[2]
        print('check conf_ok("%s", "%s", %s)' % (r['envelope_hex'], c.hex(), rce_receipt(wire_tuple(r['wire']))))
else:
    # host checks: one Ed25519 operation per event
    c0 = canon(core(S[0])); sg0 = sign(SEEDS[1], c0, S[0][16])
    e0 = env(c0, S[0][16], sg0)
    c4 = canon(core(S[4])); e4 = env(c4, S[4][16], sign(SEEDS[1], c4, S[4][16]))
    tampered = canon(core(S[0][:6] + (P[5],) + S[0][7:]))
    def count(cond): return '    if %s\n        good += 1\n    else\n        bad += 1' % cond
    def vstep(name, e, want):
        return 'step H on %s\n    let e = hexb("%s")\n' % (name, e.hex()) + count(('' if want else 'not ') + 'verify(&e)')
    # sealing and decoding the expected envelope together use more than one
    # input's fuel (6.7), so the seal and the comparison are two events
    print('step H on Seal0\n    let r = sample(0)\n    sealed = seal_receipt(&r, "%s") catch []\n' % SEEDS[1] + count('sealed.len() > 0'))
    print('step H on Seal0Check\n' + count('sealed == hexb("%s")' % e0.hex()))
    print(vstep('Verify0', e0, True))
    print(vstep('Verify4', e4, True))
    print('# payload substitution (C23): same signature over a different payload_hash')
    print(vstep('Substituted', env(tampered, S[0][16], sg0), False))
    print('# wrong steward (C5): signed by the RFC 8032 test 1 key, steward is test 2')
    print(vstep('WrongSigner', env(c0, S[0][16], sign(SEEDS[0], c0, S[0][16])), False))
    mpk = json.load(open(os.path.join(CONF, 'meta.json')))['pubkey_hex']
    print('# JCP conformance/v1 (official): every published envelope verifies under meta.json pubkey_hex')
    for i, (f, r) in enumerate(conf()):
        print('step H on Conf%d\n    let e = hexb("%s")\n    let k = hexb("%s")\n' % (i, r['envelope_hex'], mpk) + count('verify_with(&e, &k)'))
    e = bytearray(bytes.fromhex(next(conf())[1]['envelope_hex'])); e[-1] ^= 1
    print('step H on ConfFlipped\n    let e = hexb("%s")\n    let k = hexb("%s")\n' % (e.hex(), mpk) + count('not verify_with(&e, &k)'))
    print('step H on SealWrongSeed\n    let r = sample(0)\n' + count('(seal_receipt(&r, "%s") catch [7]) == [7]' % SEEDS[0]))
