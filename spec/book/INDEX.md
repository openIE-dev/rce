# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `92fe7e6226d3c9478ed1d35acb37c0901d42c0cb4f08b2ccc340165abaf1a78f` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `0eab4fa5900a167b5f8470385d89c71b6c6cfd720ccd25c7aefe4a54a4b581fb` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `33c18e881e88f99afaba8b4797da5e26db03da9fae7aa1d4798491ff79cc5a81` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `54d6978f571923b458fab4e0502b32d58ee3dfcf59a31f5b5c5a664f6cf026e2` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `ef0ed9aa6d755e5a1ec767da9ac734086f9f2dccbeb86a218c19be73c5b23270` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `bf6c8917142ebcd54bfd4b3a40788ad0bc8b01456033e77eb5ae9142e45deb35` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `4d90e5d4d01a0c866448c178c9c6fc93ea320e987af09bb5f66a527cbc41184d` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `1ce3b9d19e6e413684b1abf4d63dd945b6d947007ba000471e1e1b6b87fc8e6f` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `10b3788613a7d0ad4d60fd481b2e04ed2b5c947ded211dd7a44ef9f97fc0dcc6` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `3a73540af023371c2804f7abe735990214fa4b613410a2134d891c14559d1862` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `e7acda0628191b779c72cd4708e431c882c350364606fa49c55d571769f74bf2` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `003af0204609ec8dcf5a0fdb7b9b3d5aacf5b596976fa870cd56d6a6a3a7c5e6` | Who wrote "Communicating Sequential Processes"? (surname) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 9d5824a11ec42b99f1b820488914a4769b3917e94a3fe4401bc6c093eb09d521 -->
