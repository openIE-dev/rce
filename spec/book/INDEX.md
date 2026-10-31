# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `8964928ff50dae3e6973daf085b0dbf7fa17ca001d9453226c984fba4a5681d3` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `a521503afcfbe33debdf95f67ba7d2b35759e82e0c4f7ec2a5157fa2585874d1` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `b48794be4b4fba7520cbc7764d17fc9bf7ce3a2979375ed708f18ef63dd502fe` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `90fe10b3aa097327e2661efbb342052335cc4b1507154dcde90fd2f3f857f23a` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `f176cdbdab83223bd6e746a60ee394053648bbe43bb5f1afe1fec87c22fbb3ff` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `f3e970007bb2faf57da1674ca03cb6cc103a163c3fcbad7da4add6bc741e1baf` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `7373242d19488234fe6a9c6ff8969ebdc543c84917ecece876b8fcd7026cfbfb` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `0dc0c2af67c908c2000eb760a3f14aa6dabf408b6729723fe42b4fcd39f788a8` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `6589124d862f2d00926c070cf9ca1e1b5966945b0a21d5914af00b7f75aed583` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `515463b2081ca4bbf0514f709572c0de1e3ddfa3a5e2c92a952035a4fe9058ff` | Under which rule does a region with two unguarded steps whose patterns overlap require change? |
| `10-theorems.md` | `8c2c35f781366ce0c47415700a619ff11f18ce6af1cadbe4dea5128f90468218` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `f59a404300eae78da54846bbfeb67fd4129498b1e1043eb7a88f7b7e9caf0831` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `5bdae8a5e7f82293cadcf9c788f894641af38be20f173bf270cbfa3e22d82037` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `c31f7a72b58278b52c6ffa13d2984b06563e73d695e5db7c670c9b4da5e2060f` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `aae630976deb0a9785fa66664bfaa6b662998b4d70aa6cb978085e45c087dc5f` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `853b3e84375cb238c7b1c2e095bd2c5d95096ac2f329aaaa8dc19b02c5689f35` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `d758122e6b3bc467c6c7cc553e72dc63cb7c3a6ae67ecb968045950837b14b53` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: fdcd0439f7c41972535f1e6f85f824e7a89ac3079b4144905d10736590802550 -->
