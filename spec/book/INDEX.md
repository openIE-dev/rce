# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `b43df751959d71cebc1ccb3581f92588878dac9712150d342ccc20902377baa6` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `19f3c9be912b1b572c040f42ade9fa62fe20f75a6a571aae9219ac6a89090316` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `86baf828c7f79a62c2158222a7559b8ba950fa8f9e9be5d0facd6d65019d3e2a` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `d74545f9b1aab9a1393a8d026a43591cbb186b12df7c8e7e43038c66238119bb` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `32c25dd3faf5a8cb1f2f2e594db3b24adaa291248e2c971be6d11f7344f59c7e` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `9716fd2250d26108818fed2ac577808f800249e52ec7085e642c01bb8dd6f399` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `59c642644ddf6e2559938f49dd7ad89185b6af717a8825cdad2a108db7b4c42c` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `2e6c0e0ff7bbee51116558a003506f9d6976ce57eb922edfd73a5e021b1ec50d` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `5bfae39f553379d66b8f3e4ef214de386e855cff0e5d4d518ab103e4bde86650` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `5ec607113e53f30a7430c572baef72d864bb3d508b0b414a90f03be6131f94a5` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `1a429e5dd8e4f771e09dd71e01e782578168ebfb9b0b174ff9041bdb6acd8e49` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `003af0204609ec8dcf5a0fdb7b9b3d5aacf5b596976fa870cd56d6a6a3a7c5e6` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `78ecf181df4042a8dd09f4635de1e2ebda703cede45176a2e35007290b34a21a` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `78c5611767eb59366d670b4d9b81c357f6347df0f30663d46d98643491e9ea8a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `e7177eb45e23c4e7c520c6b06164bfefe2a0991c1d6a57ee600ce63ff519ea03` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 51c2a286f8459e52233de60ff6d0afe4761f7da14cc608420e040f07eca47f9b -->
