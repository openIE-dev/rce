# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `7c7905698ac336da05bd41b7544bf2a580c65806f22dac65639c5640cbe63d82` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `19f3c9be912b1b572c040f42ade9fa62fe20f75a6a571aae9219ac6a89090316` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `86baf828c7f79a62c2158222a7559b8ba950fa8f9e9be5d0facd6d65019d3e2a` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `d74545f9b1aab9a1393a8d026a43591cbb186b12df7c8e7e43038c66238119bb` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `21077fbff8c5f2130986fe87680644c391a6c39997502e3e2f486e76e988fe32` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `8caa95f44e22a22df26601dd74e4845a9468250fb801422f7dd50199537ad996` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `829fc50b7504db98a35c3a0382beda7c371537723726a18f8538afb7720f1f64` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `2e6c0e0ff7bbee51116558a003506f9d6976ce57eb922edfd73a5e021b1ec50d` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `925406c4a664c8898fe2833acc52c72402a8bfa3aeb2567f2037dbd85f7396c3` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `1855f984ba5ac9080eafaf9ef3173df0cc97c536d2029815c9fb0197fa925276` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `9e9a299d131246b173c6baf45c989a4e9c3506009466e1cebd8ea227167ed695` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `e9c4b051453346ea2d579b1d749789977198d7484a8ba65350d7517c136e6321` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `b6cc0af8b9dfcc62c00f083f0dcaeabb797846c3b59826b0986696c71ef6c217` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `77247f238663dcfbf3dd4ce08bc2e54ed7ea6131ae4284439540c6592ab305e7` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `41797bb8bddf6eae3f21b767ba1af2abc894dfc7967df7434ea8c0c00deaa862` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: adb713e951cb930120da7e8833f99b987d51bb181b181edfe86c3ef313cff050 -->
