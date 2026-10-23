# Regenerates the vector lines of tests/hash/blake3_checks*.rce from
# vectors/hash/blake3/test_vectors.json: one check per case and mode, input
# i % 251 of the case length, 131-byte output. Shards: lengths <= 7169;
# 8192..16384; 31744; then 102400 for hash, keyed_hash, derive_key apart.
import json, sys
d = json.load(open(sys.argv[1]))
for c in d['cases']:
    n = c['input_len']
    print('check v_hash(%d) == "%s"' % (n, c['hash']))
    print('check v_keyed(%d) == "%s"' % (n, c['keyed_hash']))
    print('check v_derive(%d) == "%s"' % (n, c['derive_key']))
