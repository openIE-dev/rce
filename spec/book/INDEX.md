# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `8964928ff50dae3e6973daf085b0dbf7fa17ca001d9453226c984fba4a5681d3` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `a521503afcfbe33debdf95f67ba7d2b35759e82e0c4f7ec2a5157fa2585874d1` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `b48794be4b4fba7520cbc7764d17fc9bf7ce3a2979375ed708f18ef63dd502fe` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `341fe96ee16fb89234a62e7a4375f3b1530503840641f456230253255a6b6d4c` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `f176cdbdab83223bd6e746a60ee394053648bbe43bb5f1afe1fec87c22fbb3ff` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `9bf4da7ffa7afa6304e34a4b9a523bad2c22f667eabfe0dcc63d66a6514199aa` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `7373242d19488234fe6a9c6ff8969ebdc543c84917ecece876b8fcd7026cfbfb` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `0dc0c2af67c908c2000eb760a3f14aa6dabf408b6729723fe42b4fcd39f788a8` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `6e403282d3fe412ba0ac3a00a518a340e3b1337b0978399aeaba3b2571c162ad` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `d3f925d7b7230629754f2574572ec5c4c10b0d7d655060f420d6d0b5ee9f5182` | Under which rule does a region with two unguarded steps whose patterns overlap require change? |
| `10-theorems.md` | `86c527f15354073e564232027b6775535873033fe643ebc8f86c43e51f4fc626` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `f59a404300eae78da54846bbfeb67fd4129498b1e1043eb7a88f7b7e9caf0831` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `422df7a55c421b0839307be62f3344dcfc49233b40ae9ba27ae92da962dc1a90` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `49e483f0feaec3fa0e01ead23f929e43e21bd2a464145581692bce29366aac48` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `aae630976deb0a9785fa66664bfaa6b662998b4d70aa6cb978085e45c087dc5f` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `bd336e3d6d4ffa571835374688a503290e3685466643d1c4b2eb2d7aa80032cd` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `d758122e6b3bc467c6c7cc553e72dc63cb7c3a6ae67ecb968045950837b14b53` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 078013f559a9ece7b6c24ae147451ed2e847809e465d9176314bd1dce33f37bb -->
