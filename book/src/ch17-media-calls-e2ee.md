# Media, calls and end-to-end encryption

This chapter describes the programs and libraries that give RCE
programs media, calls and end-to-end encryption (E2EE): encryption in
which only the communicating peers hold the keys, so relays and storage
see ciphertext only. Everything here is ordinary RCE code run on the
host contract of Appendix E of [the RCE specification](https://github.com/openIE-dev/rce/blob/main/spec/RCE-SPEC.md). Nothing in it adds a primitive to that
contract or changes the meaning of any program. The rules it relies on
are HC-3 (one result event per output request), HC-7a (fuel per input),
HC-8 (replay with adapters disconnected) and HC-10 (every input is an
event in the log D).

## Principles

- **Randomness is an input.** No library here draws random bytes. Every
  private key, ephemeral key and seed comes from a logged `Random` result
  (HC-10), so a session replays bit for bit and a test can fix every byte.
- **One public-key operation per input.** A step that runs X25519 (the
  Diffie-Hellman function on Curve25519 of RFC 7748, a Request for
  Comments of the Internet Engineering Task Force) or a post-quantum key
  encapsulation does one such operation per input, so each input stays
  within the fuel of 6.7 that HC-7a grants it. Programs queue the work
  and run one job per `Poll` input.
- **Not constant time.** The libraries are reference code for
  correctness and replay. Their running time depends on secret data, so
  a host that faces timing attackers runs them where timing is not
  observable or uses a vetted adapter through 5.11.
- **Secrets stay in D, and D stays on the device.** Private keys and
  ratchet state are program state, and the `Random` results that made
  them are entries of D. D is device-local; `rce host replay --redact`
  (chapter 16) shows such fields as digests.
- **Checked against official vectors.** Each library carries `check`
  lines for its primitives (run with `rce test stdlib/lib/FILE.rce`), and the
  test suite runs the official vectors of each standard as host steps
  and then replays the log.

## Media and the selective forwarding unit (SFU)

The media adapters (chapter 16, enabled with `--media DIR`) turn capture and
playback into events: D holds (track, sequence number, digest) per frame
and the frame bytes are blobs addressed by their Secure Hash Algorithm
256 (SHA-256) digest. A call between more than two peers needs a relay
so that each publisher sends a frame once. `apps/sfu.rce` is that relay,
a selective forwarding unit (SFU), written in RCE.

Peers reach the SFU over remote channels. Each wire body is fields
separated by U+001F:

| Body | Meaning |
|---|---|
| `sub ROOM TRACK` | subscribe the sender to a track of a room |
| `unsub ROOM TRACK` | drop that subscription |
| `frame ROOM TRACK SEQ DIGEST PAYLOAD` | a frame from its publisher; SEQ orders it, DIGEST names its blob, PAYLOAD is opaque |

The SFU forwards a frame, naming its publisher, to every other
subscriber of the track, as an unreliable send: never retried, so a lost
frame is one `ChanSendDone` result that is not `delivered` (HC-3). A
frame whose SEQ is not newer than the last one forwarded for that
(room, track, publisher) is dropped as stale, so the latest frame wins.
A SEQ that is not 1 to 18 decimal digits, or a body of another shape, is
counted as malformed: no input from a peer makes a step of the SFU keep
the state. The
SFU folds only (track, sequence number, digest) into its state, never a
payload, and payloads are end-to-end ciphertext it cannot read. A runnable example is `run.sh sfu` in the examples of the host runtime.

## Direct messages: X3DH and the Double Ratchet

`stdlib/lib/crypto.rce` provides SHA-256, the keyed hash HMAC (Hash-based
Message Authentication Code), HKDF (the HMAC-based Key Derivation
Function of RFC 5869), X25519, the Ed25519 signature scheme (RFC 8032)
and an authenticated encryption with associated data (AEAD) built from
the ChaCha20 stream cipher with an HMAC-SHA-256 tag. The chat program of chapter 16 uses them for direct messages. Each
replica has an X25519 identity key, an Ed25519 signing key and an X25519
signed prekey. It publishes a bundle of the public keys with an Ed25519
signature over the prekey; a received bundle is used only when its
signature verifies. For each pair of peers, the one whose name sorts
first runs X3DH (the Extended Triple Diffie-Hellman key agreement) as
initiator, and both sides then keep one Double Ratchet per peer, so
every message has its own key. A message whose plaintext is not
well-formed UTF-8 is dropped and counted.

## Post-quantum key agreement: ML-KEM-768 and PQXDH

`stdlib/lib/pq.rce` adds post-quantum key agreement, used together with
`stdlib/lib/crypto.rce`:

- Keccak and its functions SHA3-256, SHA3-512, SHAKE128 and SHAKE256 of
  the Federal Information Processing Standard (FIPS) 202.
- ML-KEM-768, the Module-Lattice-based Key Encapsulation Mechanism (KEM)
  of FIPS 203 at its middle security level: `mlkem_keygen(d, z)` gives
  an encapsulation key and a decapsulation key, `mlkem_encaps(ek, m)` a
  ciphertext and a shared secret, `mlkem_decaps(dk, c)` the shared
  secret. A forged ciphertext yields an unrelated secret (implicit
  rejection) rather than an error.
- PQXDH, Signal's post-quantum extension of X3DH. The responder
  publishes an identity key, a signed prekey and an ML-KEM prekey signed
  with its Ed25519 identity. The initiator runs three X25519 exchanges
  (`pqxdh_dh_init`) and encapsulates to the ML-KEM prekey; the responder
  runs the same exchanges from its side (`pqxdh_dh_resp`) and
  decapsulates. Both derive the session key
  SK = HKDF(F ‖ DH1 ‖ DH2 ‖ DH3 ‖ SS) with `pqxdh_sk`, where F is 32
  bytes of 0xFF and SS the KEM shared secret. An attacker must break
  both X25519 and ML-KEM to learn SK.

Each of these is several inputs, one operation each (see Principles). A runnable example is `run.sh pqxdh` in the examples of the host runtime, which checks the ML-KEM
shared secret against the RustCrypto `ml-kem` crate, that both sides
derive the same SK, and that the log replays with secrets redacted.

## HPKE and AES-128-GCM

`stdlib/lib/hpke.rce` provides AES-128-GCM, the Advanced Encryption Standard of
FIPS 197 with a 128-bit key in Galois/Counter Mode (GCM) of the National
Institute of Standards and Technology (NIST) Special Publication
800-38D, an AEAD. On top of it is HPKE, Hybrid Public Key Encryption of
RFC 9180, in base mode with the suite DHKEM(X25519, HKDF-SHA256),
HKDF-SHA256 and AES-128-GCM:

- `hp_derive_sk(ikm)` derives a private key from input keying material
  (DeriveKeyPair).
- `hpke_seal(pkr, ske, info, aad, pt, seq)` encrypts to the recipient's
  public key with the ephemeral private key `ske` (from logged random
  bytes) and gives `enc.ciphertext` in hexadecimal; `seq` is the message
  number of the context.
- `hpke_open(skr, enc, info, aad, ct, seq)` gives the plaintext, or the
  empty text when the ciphertext, the additional authenticated data
  (AAD) or the key does not match.
- `encrypt_with_label` and `decrypt_with_label` are the
  EncryptWithLabel and DecryptWithLabel operations of MLS (below).

The test suite checks the RFC 9180 base-mode vector at sequence numbers
0, 1, 2, 255 and 256 and the MLS EncryptWithLabel vector. A runnable example is `run.sh hpke` in the examples of the host runtime.

## Group keys: MLS

`stdlib/lib/mls.rce` is the base of Messaging Layer Security (MLS), the
group key agreement of RFC 9420, for cipher suite 1
(MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519). It provides:

- The ratchet tree math of Appendix C of RFC 9420: the nodes of a tree
  of n leaves are 0 to 2n-2 with leaves at even indices, and
  `tm_root`, `tm_parent`, `tm_sibling`, `tm_direct_path` and
  `tm_copath` give the structure.
- The key schedule of section 8: `mls_epoch(init, commit, psk, ctx)`
  takes the previous epoch's init secret, this commit's commit secret,
  the pre-shared key (PSK) secret and the serialized GroupContext, and
  gives the joiner, welcome and epoch secrets, every secret derived from
  the epoch and the next init secret.
- `mls_export`, the MLS exporter, and `mls_external_pub`, the external
  HPKE public key of an epoch.

The test suite checks the official `tree-math.json` and
`key-schedule.json` vectors (all five epochs of suite 1). A runnable
example is `run.sh mls` in the examples of the host runtime, whose examples
hold the epoch 0 vector.

Five more modules of `stdlib/lib`, each pure RCE on the ones before it,
carry MLS further, each checked against the official vectors of suite 1 by
a test of the checker:

- `treekem.rce`: ratchet trees (RFC 9420 section 7): decoding a serialized
  tree, tree hashes, resolutions, parent-hash validity, and processing and
  generating update paths (`tree-validation.json`, `treekem.json`).
- `framing.rce`: the wire structures of section 6 and a decoder and encoder
  for MLSMessage (`messages.json`).
- `protect.rce`: message protection: the secret tree, the application and
  handshake ratchets, and PrivateMessage and PublicMessage
  (`secret-tree.json`, `message-protection.json`).
- `welcome.rce`: joining a group from a Welcome message
  (`welcome.json`, `passive-client-welcome.json`).
- `grp.rce`: the group state of a passive member across epochs, applying
  proposals and commits (`passive-client-handling-commit.json`).

## The sandbox under these programs

The sandbox of HC-5 is one contract, independent of any operating system
(OS): the checker admits only programs whose effects go through granted
capabilities, and the host does not perform any effect without one, before
any OS layer is involved. The media, channel and storage adapters above are
reached only through such capabilities. The same contract has
enforcement targets underneath the host: WebAssembly (WASM), bare metal,
Linux, macOS, Windows and other OSes, each entered as strictly as the
target allows (HC-5 lists the mechanisms). `rce host` prints the target
it entered at start, in a line made from the rules it installed (HC-5a):
on Linux, Landlock restricts TCP bind and connect by port and not by
address, so the line says "by port" and names the addresses the net
adapter allows, and it says when the User Datagram Protocol (UDP) is not
restricted; with `--http` alone the listener is opened before the target
is entered and the line says that no new Internet Protocol (IP) socket
can be opened. The contract conforms to OpenIE Sandbox v0.1
(https://sandbox.transaction.science) and to the Joule Context Protocol
(JCP) v0.3 (https://jcp.transaction.science).
