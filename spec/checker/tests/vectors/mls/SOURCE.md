Official RFC 9420 test vectors, copied unchanged from
https://github.com/mlswg/mls-implementations (test-vectors/tree-math.json and
test-vectors/key-schedule.json, branch main, fetched 2026-10-15). Used by
tests/mls_rce.rs; key-schedule.json is checked for cipher suite 1
(MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519).

crypto-basics-suite1.json is the cipher suite 1 entry of
test-vectors/crypto-basics.json from the same repository, unchanged.
../hpke-base-x25519-sha256-aes128gcm.json holds the entries of the official
RFC 9180 vectors (github.com/cfrg/draft-irtf-cfrg-hpke, test-vectors.json)
for mode 0 (base), KEM 0x0020, KDF 0x0001, AEAD 0x0001, unchanged.

tree-validation-suite1.json is the cipher suite 1 entries of
test-vectors/tree-validation.json from the same repository, unchanged.

treekem-suite1.json is the cipher suite 1 entries of test-vectors/treekem.json
from the same repository, unchanged.

welcome-suite1.json and passive-client-welcome-suite1.json are the cipher
suite 1 entries of test-vectors/welcome.json and
test-vectors/passive-client-welcome.json from the same repository, unchanged
(the suite 1 passive-client-welcome entries have no later epochs).

messages.json is test-vectors/messages.json from the same repository,
unchanged (it has no cipher suite field).

secret-tree-suite1.json and message-protection-suite1.json are the cipher
suite 1 entries of test-vectors/secret-tree.json and
test-vectors/message-protection.json from the same repository, unchanged.

passive-client-handling-commit-suite1.json is the cipher suite 1 entries of
test-vectors/passive-client-handling-commit.json from the same repository,
unchanged.
