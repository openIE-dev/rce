# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `b43df751959d71cebc1ccb3581f92588878dac9712150d342ccc20902377baa6` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `921ab947ddc8d7add3bfb8ef045d934ab2e4efa1dc53a24496ba4badbbdfe7b2` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `33c18e881e88f99afaba8b4797da5e26db03da9fae7aa1d4798491ff79cc5a81` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `54d6978f571923b458fab4e0502b32d58ee3dfcf59a31f5b5c5a664f6cf026e2` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `0f0c4a2a2457da4a9dcedda9d940d7375ed0318aa79b79ec9ab38d8a37a6640a` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `826f06907c3a4d84629e15a469b75d5277675645ddc325db5ed225ad66e59378` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `4d90e5d4d01a0c866448c178c9c6fc93ea320e987af09bb5f66a527cbc41184d` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `eb687dbb49ffa4e868bae0ac0c55e8d755a304fa97560c784a7b389d540e87b7` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `182f9afb88e9a12e15e7e7ccc0a31cc1e895905a3e60aeb72b668b2fafea9921` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `fc357fd314cc892a9aafd29438a1fed1a9649512de33953ee1b8b8e8a4c0ff7b` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `8dffc76f2a0428e554e3667e1d2307672e21ff71d4a028d6514b55fa3b7ce7f0` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `003af0204609ec8dcf5a0fdb7b9b3d5aacf5b596976fa870cd56d6a6a3a7c5e6` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `abdefff233178ef31f6bf646a2a702939839e6d35b48de4b53a2b4ac28815509` | How many words does a version 0.6 module reserve in addition to the 0.5 keywords? (a number) |
| `15-handler-spec-profile.md` | `c553c04b34372a4bba8b83d92d2a05c0860733fdd82e8515d7d96164edaca8e8` | Which compiled node id does `deliverable D1` become? (for example role_R1) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 23ace10e9d1d07d0126fe3b4d689f972327d08d49304e2e6fac9525e84367bd2 -->
