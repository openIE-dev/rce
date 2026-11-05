# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `c7fd2060f48c6da332c48fbdcb5fe22dd969eff13bc12d3f7829cc79abee82e3` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `a521503afcfbe33debdf95f67ba7d2b35759e82e0c4f7ec2a5157fa2585874d1` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `95ca24c86bef9bf604a6d38f5c4fb6c263cac070bbd3c9d0a7bb88b47dc9c08b` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `310b9dcdde1089d3f6c550f8bb7c5fe29e8e631d6897e12e614f3a7359b48450` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `b3b7015d34a76f1338720bd84070b978e43da537c795e53a8594f10c8bd9a47a` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `857c1b43e187f03f4ad2efc72660d85d8627a71a67ebec9aa7f63636561fdb14` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `7373242d19488234fe6a9c6ff8969ebdc543c84917ecece876b8fcd7026cfbfb` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `94b2b681e25dff8783d24e1b5ea37a0a8d10998a68d39ff39aebc1ca5035966c` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `001390114b17e3b711db808f185e9de73e1b7c5167abcbc1bcded716e76a669e` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `32d6996acd42399995cd76082e4c7d9b6bd5fc0bfdcb298c3624a6282123e130` | Under which rule does a region with two unguarded steps whose patterns overlap require change? |
| `10-theorems.md` | `204c290b5f6600d0db4c39088894fe0193aa3c73c262c12f1bd7d0d23c693cf8` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `ebbf77ef787a4497ed66e4fca238dfc4accd3e99a178ee78c8831a08bb730045` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `48c4127b23fd0eb459c38d11f8ca05d983c43926a1822806e8865de109154348` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `cbf255393d0ebb75e6883a0dcba027fa9680e9a25235d3cffa785e9b1bfb27d9` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `6cc74754c95fff25ccdc4b65a5443f23ae8465df6684fb1a32ccc89095adca35` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `b0e1e03eae1c13162da0184c173fa81d1510c95dd9c684ade2fd225a2dcda84a` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `c27e706210376d17948c166a34a4756ca035a91227d5d415ceb87bc29a2073ce` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 37437c9eba02a3402e2e1dc1bbe2aee5ac1be686eb49e687337616f5f12542b3 -->
