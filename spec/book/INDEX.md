# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `8a4ec3110284660c9e026f4b35fe729d75a3ae1d8482e6880b7ff593de1660fd` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `dd32cf81b6db8c0c1fde74f2660ddf8d6d5616eb6450422dca4c1eb272aa41f5` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `33c18e881e88f99afaba8b4797da5e26db03da9fae7aa1d4798491ff79cc5a81` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `54d6978f571923b458fab4e0502b32d58ee3dfcf59a31f5b5c5a664f6cf026e2` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `b83bfe6846b399f4757df883b39da654eca2443adfb0b4518e72515b5225cb73` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `826f06907c3a4d84629e15a469b75d5277675645ddc325db5ed225ad66e59378` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `4d90e5d4d01a0c866448c178c9c6fc93ea320e987af09bb5f66a527cbc41184d` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `866fa41fbeab9642ea390cd560ac26f7bc72ca9075c26a31514965edd6e19b4c` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `182f9afb88e9a12e15e7e7ccc0a31cc1e895905a3e60aeb72b668b2fafea9921` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `fe6770a3a4f699c13462389094e5d2e83aab723583c7d2cd65005041f76eb780` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `e7acda0628191b779c72cd4708e431c882c350364606fa49c55d571769f74bf2` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `003af0204609ec8dcf5a0fdb7b9b3d5aacf5b596976fa870cd56d6a6a3a7c5e6` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `2e7604ee602db61fd17615ca0bde1454756accc45249b689328b5d5c15d1281d` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `697ddf5c7edf2a2bedb5043b8e8ab42f80169543c6486b36a94e1cde3e5070a7` | Which compiled node id does `deliverable D1` become? (for example role_R1) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: c2f76d96e3c83c4b4734cb2bb064d0400b567d24f5c3a2193eb27885a5b5a099 -->
