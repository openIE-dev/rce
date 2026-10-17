# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `a206308d0719d6a8f643f7d815fb54a2f010ac99ff3cf5338fed09303aece2e7` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `19f3c9be912b1b572c040f42ade9fa62fe20f75a6a571aae9219ac6a89090316` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `4f9772b399f34c966fc5da1df89d62269093dd926a1c33c5349447c5b52df47c` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `d74545f9b1aab9a1393a8d026a43591cbb186b12df7c8e7e43038c66238119bb` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `350558d934b1128b283bfd3348c032658783e93b9d4242165fbfcd2dd77041ae` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `ee87ffe62ac9f08ed53358a73eb30374b1730a6aedcb02300f9d38bc71bbe374` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `829fc50b7504db98a35c3a0382beda7c371537723726a18f8538afb7720f1f64` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `7ff6ba12fd3a4ee1bcae602f3d561bf0c5c851611f563b9d233f6ad8e1f227fb` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `925406c4a664c8898fe2833acc52c72402a8bfa3aeb2567f2037dbd85f7396c3` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `b0d7cb2e7884f1c40bfc7c0f68e4458e4e5803de07bdfe849fb109f6f19f2fc7` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `8788bdc65ab982bcce09aab56bed504fcbee2ef245f727cf86837b09db2a3fc3` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `93c398eee22b1538960aca4057d67e98d642d4cbd5e999a47bffe93af21b0529` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `f31927494ae82d1f36aa8d69d4e8a40800913fdc0591783876bd2faf8753c0f4` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `c437da9b8325fb3713c7d007b61d6b06a8797a785f1b7217adab8e122b81993b` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `41797bb8bddf6eae3f21b767ba1af2abc894dfc7967df7434ea8c0c00deaa862` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 285ba0c2d265e5f645eda615d6054b94960a8f36218fabe78994859fb45c7125 -->
