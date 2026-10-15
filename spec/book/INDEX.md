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
| `07-form-evaluation.md` | `4ac2cd808a0f75a796fa26b86f7bc2a7fe42fa05528f94bd5e8e24dfe571ee0a` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `829fc50b7504db98a35c3a0382beda7c371537723726a18f8538afb7720f1f64` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `2e6c0e0ff7bbee51116558a003506f9d6976ce57eb922edfd73a5e021b1ec50d` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `5bfae39f553379d66b8f3e4ef214de386e855cff0e5d4d518ab103e4bde86650` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `5ec607113e53f30a7430c572baef72d864bb3d508b0b414a90f03be6131f94a5` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `c2639c1cdf4890c67abf706c388af5d62050b62e187629828bc83efe09a65a11` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `003af0204609ec8dcf5a0fdb7b9b3d5aacf5b596976fa870cd56d6a6a3a7c5e6` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `d6a1a6894f575dd98ceb07c7ce0115eb8ffda51623cc70a9e520e36522db35f3` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `24c5bed573317d35a073aac5ed9418a622bab356787c089febeae0a4a3cdf02a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `233eecaf3a038d636feeb220386e2329b357287cfcb579881d4e455eebded79d` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 2e842c2e990ea73dba8385d56f1f4c67bddd16d1c6af7020949ce0ce5fcc9e4f -->
