# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `c2e9145f50a8a9ffa3f30fe5e2b9fed41067ec95c2630b66681bafe43ecf72ac` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `a521503afcfbe33debdf95f67ba7d2b35759e82e0c4f7ec2a5157fa2585874d1` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `b48794be4b4fba7520cbc7764d17fc9bf7ce3a2979375ed708f18ef63dd502fe` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `17203e1adc917bd4a8f5bf953ff8ed993671c39176c5bd064772c6c73ac8703a` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `f176cdbdab83223bd6e746a60ee394053648bbe43bb5f1afe1fec87c22fbb3ff` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `f3e970007bb2faf57da1674ca03cb6cc103a163c3fcbad7da4add6bc741e1baf` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `7373242d19488234fe6a9c6ff8969ebdc543c84917ecece876b8fcd7026cfbfb` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `2b3344a046f82366a7cdd55da681cdbb746739153fe1250057430391c5de8397` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `37a6b2eb111a41d9e711ee5c4d22e4afac865b2c68e2d3f8f9bc7624e95ce697` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `f3a0cdbb5432c7fe84f9406ca809f796ab258130766e69d844563b0ce6a1c60a` | Under which rule does a region with two unguarded steps whose patterns overlap require change? |
| `10-theorems.md` | `6d6be4c963419a95f1e9f8abc03f57ba39611012fa203e691e5d8f10dba7fe18` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `ebbf77ef787a4497ed66e4fca238dfc4accd3e99a178ee78c8831a08bb730045` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `09bd6caa0cd27429d5b7d46a745093654a640e7d259582d7db4fa30eb5119fdf` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `62e6ddd483f6f4259708c203b44642455cde6fcfe34c3abc75cf28dc658bd730` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `fad7eeeb89f4f794212ec8699e52d99e0f345f5dff6edfb02af7c3dc3dbbec30` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `57e86d79cf7737fe809910f5d72642f21f1549a093674786e46ff51afa213ed1` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `4870b5eb4c4da34715387cdbf76f6d4729decc3890cb2735267a32ddbfd64dc1` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 623518cfaa2a0b6506be31df544b905fbc59cf2235477861b39a535cb3ab6450 -->
