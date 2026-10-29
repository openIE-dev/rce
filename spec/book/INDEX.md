# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `9b5d00a9125a58ae37ba6cd42fc74a8c18a657c45e6968df672118433c277fe3` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `26ba6fca7955f9b0b6cf03553dcb7125d5622f3317faa1e4372581f48d773936` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `dec736c456a93745d0fa9fce74425689c6ed362e39b78bd4ebb8d7bb6ce1d72f` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `c7a94eb7c4ad0ad0643dd4815a99397eae5b442b81d8e726a0a58e74ab21285b` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `60fbd30d15e72a6d3919864f7259ba7ccb7f16f4b309d6367c2cb479e7498d07` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `94ed5923623f65f5b78682ecd4a03a433fe1000598dc036f40ffc187a1892918` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `d2c8c424a4caf7d186e74c94911e47b77554d7c7224a186a9c64d36914fd988a` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `9a7958f8ccdeb615373dbb7f21a8d51f89c9fcb9e3fdd7363ad3b47ab6dbb05e` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `8bb53d8444bbf4edf7382eafe256d25442a976309d92ad5a00ec36bfe5a67a33` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `3c4b45822990cb3d6235a553c79ab4c18048d833709a51ee548de68f55ffd6ae` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `99c630ddf821f23275ba918b18626438421b6a37cf3049e6da9aec6f58bdb456` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `8f77ea09fc5d4d7468ed3d850c24759790db6d3f7216c2fd07586c09ec0c248c` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `811e62db5099b0132ece333c593c223dc53146aab32822a6738a679921988358` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `375c3aa0d3316d41bda6c6f4b3b948717cd752f06a5a96302f4e15d45b00038f` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `1f5d65c77edae33585d11b8873dc7ca714eb78ae0aafc37a895c26db3a45cbeb` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `8bd5fe456449680c69da527d048b20482d76c77fd57e38dbb85ced12065223c5` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 06b6abdcb8f836a3a4814e0a1cd398836c7052759619ae6a4f93def167c3be15 -->
