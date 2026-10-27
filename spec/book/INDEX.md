# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `fda0152e7116bacf059e6e350c3dee069cb183d82b3ea21dd913fd317b193658` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `4eeab6a05c17dd46d9db462dc4c801fcde4ded882648bafd843fde43777a4dd4` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `9bff3a57294945da586ae15052a6cbce53ea166720571859eb2b9f878c8ef6f2` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `5ca7d8a15cdf8973b8489ee52c0c6f634b5b029aaf692148c113fd02b0064a8d` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `02ac9d12d4e0e96656f200f8328965028a511328a7affa24d233988588eef3b4` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `94ed5923623f65f5b78682ecd4a03a433fe1000598dc036f40ffc187a1892918` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `44118c5facf9237f3d0de65dedf3eea86fc64fd352ad9c3907e642ddcffed259` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `075fa38adc30b3add818021a4bb6df49e95aa6a25addafc0223ea201eeccb208` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `8bb53d8444bbf4edf7382eafe256d25442a976309d92ad5a00ec36bfe5a67a33` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `cd5569c71a7b7d3190501d6db96b1e06364798b8bae4d5c89ba86389e0a94e18` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `2346d053d92fe62068dcec8afd8e1b3608ddfbbe6249ad2cc3ea79da9634ee90` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `28c8d5a61d12bf7334a77c577728789a5c2687fa7a108568fa32e95f66e85198` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `731558e20966c95c1c100bd469f1587cd5ee034ff64d8ee8836d68f217782aba` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `375c3aa0d3316d41bda6c6f4b3b948717cd752f06a5a96302f4e15d45b00038f` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `094ded0b9b9561a8e8ca69c1a6d895e3670ca664ea90bc7a7565fbabc0aeb2f7` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `8b35e0fcce44bf4929bfc6736dc04859fe42b50516a69a6e14ca308c87a53fa5` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: adaa4900cf949c2fb75fe3eddf68171131571cf7510434eefc78d2e60d8bfe2d -->
