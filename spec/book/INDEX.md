# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `46ea14d3ed337ef5da978600614fbe3339179e6cbf1f4e3ccf4557efa8898659` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `1ddd28e32ada7648b2261ba0addc0899841eb277b307fec08f826bdd32153041` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `6ae5395feaedeeaca66120e545c63a5cbebfc4bc70423d7745eb524484e0925d` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `b2fc1cbff2e0fc74726cbfe875bd8ac2e642c27db75dc5399d11898d6a3e6706` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `f76dd473cd84a9d2911ba3b1fb03f61e89abfbbb9d63c964b876a7de38ae7dca` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `e3fd75261cf645a472c609e85dee7d5c6f309cd89eb732cb2973025eb4fe1760` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `22b852f3d01aa50cf11433f9c5206c856ceb0205e9991b8c4f276b6ceb772fbe` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `f97d0e776a90e124af05ca02979bb6b84c81409ee381abd5351bae86dcb87118` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `0c2359d99dc3e253e8816443f1d87cc9b5fb1318cc452de9b999c0c1a4819b7d` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `ff45ae94e1c1a3b84f3306b673307ff11857d3b8bb97053640d2e415c947e50c` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `5c2b8f2dfd9dea59d60dce9af7edd5aeb9b5eed1aa745d3c4ccdec4cd5941263` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ac72852386a36ef0d84e868bf2f2c771b662a7699b29363418f1f659ad730068` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `71be1c1d289be03ce5bf0bb48c8cbba73fb238c6f59d6759578a4a21c615b54a` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `1874e9e31a2d9052aa44b11e865e0a8bb2977c7c3bd072bdb8a9dd40b79d205a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `52ca914e4ad21dce546e4215776e6d6f4f990d48a7d28911b3fea83b3ae9ca14` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `71ebf70733ae8a588014ddeb5e6c3a234707001685afd9675faf8ce912b05abb` | How many words does a version 0.9 module reserve in addition to the 0.8 keywords? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 121cf8e89a2ae668f29e4db02f72cd181224f3fa22fee53a65ba46513b266d45 -->
