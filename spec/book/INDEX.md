# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `7e384e15ee67f436c7a8cd88928b40f57ac7dab844897bd7848df1c8fa8f484b` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `a521503afcfbe33debdf95f67ba7d2b35759e82e0c4f7ec2a5157fa2585874d1` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `b48794be4b4fba7520cbc7764d17fc9bf7ce3a2979375ed708f18ef63dd502fe` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `90fe10b3aa097327e2661efbb342052335cc4b1507154dcde90fd2f3f857f23a` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `f176cdbdab83223bd6e746a60ee394053648bbe43bb5f1afe1fec87c22fbb3ff` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `f3e970007bb2faf57da1674ca03cb6cc103a163c3fcbad7da4add6bc741e1baf` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `7373242d19488234fe6a9c6ff8969ebdc543c84917ecece876b8fcd7026cfbfb` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `c4765869e945dca69fceef55a29bff6571f4905257bc63107ec7dfa0edc40e66` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `2d204e20fd1fe4c1ea64067c6c18e26830c00933c994eff77fe907c541871a3a` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `515463b2081ca4bbf0514f709572c0de1e3ddfa3a5e2c92a952035a4fe9058ff` | Under which rule does a region with two unguarded steps whose patterns overlap require change? |
| `10-theorems.md` | `0827a0cecea9f014e17e0fd134d7bc2009c6a0979a5f3eeff4123228d4cbc962` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `f59a404300eae78da54846bbfeb67fd4129498b1e1043eb7a88f7b7e9caf0831` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `58f388ddfbe9129c5b462befa3b30d3f919479f8c88dd2dbdde4aabb81428ff0` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `49e483f0feaec3fa0e01ead23f929e43e21bd2a464145581692bce29366aac48` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `aae630976deb0a9785fa66664bfaa6b662998b4d70aa6cb978085e45c087dc5f` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `853b3e84375cb238c7b1c2e095bd2c5d95096ac2f329aaaa8dc19b02c5689f35` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `d758122e6b3bc467c6c7cc553e72dc63cb7c3a6ae67ecb968045950837b14b53` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: dedd07c02321ceff217884d6e0e370a883addfe22150a960796a589e58c8569d -->
