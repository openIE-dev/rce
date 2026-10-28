# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `fda0152e7116bacf059e6e350c3dee069cb183d82b3ea21dd913fd317b193658` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `4eeab6a05c17dd46d9db462dc4c801fcde4ded882648bafd843fde43777a4dd4` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `9bff3a57294945da586ae15052a6cbce53ea166720571859eb2b9f878c8ef6f2` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `c7a94eb7c4ad0ad0643dd4815a99397eae5b442b81d8e726a0a58e74ab21285b` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `02ac9d12d4e0e96656f200f8328965028a511328a7affa24d233988588eef3b4` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `94ed5923623f65f5b78682ecd4a03a433fe1000598dc036f40ffc187a1892918` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `ae975bf1f62fc55bb3be460a2aeb925f42095687b06fbb19908386d724977d24` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `075fa38adc30b3add818021a4bb6df49e95aa6a25addafc0223ea201eeccb208` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `8bb53d8444bbf4edf7382eafe256d25442a976309d92ad5a00ec36bfe5a67a33` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `cd5569c71a7b7d3190501d6db96b1e06364798b8bae4d5c89ba86389e0a94e18` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `fb7ee5625e3f9dff21379fba6af986ab2cbeb65cf5d577637c07ba477c4a45dd` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `2329b271394658198556ea9212b768a29e9262303f060d485b904051b71305f8` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `731558e20966c95c1c100bd469f1587cd5ee034ff64d8ee8836d68f217782aba` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `375c3aa0d3316d41bda6c6f4b3b948717cd752f06a5a96302f4e15d45b00038f` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `63f298c4a2a6a2ae68fe4952a7291ca6967060775df100848b8ff0d487170c5e` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `8b35e0fcce44bf4929bfc6736dc04859fe42b50516a69a6e14ca308c87a53fa5` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 1d1b22b469d775cf5d710b9aac1800b929f2a9a145961b59a07ea26bb82b2a60 -->
