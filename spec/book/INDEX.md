# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `ebccae41a8e68e329210c1334e50703923037434dd0352d5607fa704f099fbe4` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `a521503afcfbe33debdf95f67ba7d2b35759e82e0c4f7ec2a5157fa2585874d1` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `95ca24c86bef9bf604a6d38f5c4fb6c263cac070bbd3c9d0a7bb88b47dc9c08b` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `310b9dcdde1089d3f6c550f8bb7c5fe29e8e631d6897e12e614f3a7359b48450` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `f176cdbdab83223bd6e746a60ee394053648bbe43bb5f1afe1fec87c22fbb3ff` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `f3e970007bb2faf57da1674ca03cb6cc103a163c3fcbad7da4add6bc741e1baf` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `7373242d19488234fe6a9c6ff8969ebdc543c84917ecece876b8fcd7026cfbfb` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `2b3344a046f82366a7cdd55da681cdbb746739153fe1250057430391c5de8397` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `a8873dad6351fd96dba942fa8c55c0ac04c7a0e914e5666ebc5ad552ddbd6153` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `bcd9aa8480eedf926c955c50278a1ea6fcbdaf5610859356a24afa973ca726e1` | Under which rule does a region with two unguarded steps whose patterns overlap require change? |
| `10-theorems.md` | `204c290b5f6600d0db4c39088894fe0193aa3c73c262c12f1bd7d0d23c693cf8` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `ebbf77ef787a4497ed66e4fca238dfc4accd3e99a178ee78c8831a08bb730045` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `4b827e1518e2ea13a01f3594d536245fb32ab8a5c67bef50f7fb4a493ddc5b33` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `233ca44eac98e6de27f662deccfe20bff582d401c481b767d3eda1559ce9a557` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `fad7eeeb89f4f794212ec8699e52d99e0f345f5dff6edfb02af7c3dc3dbbec30` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `657e63da98492403c4e783b7e0067c2a56016c8d5f70609c43035cef877441d7` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `c27e706210376d17948c166a34a4756ca035a91227d5d415ceb87bc29a2073ce` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 2f8827c77be58a60a190a34d2b4b2364089b668e49d0c1285b49375882c8133e -->
