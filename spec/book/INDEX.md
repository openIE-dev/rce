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
| `09-wellformedness.md` | `7041f47ee2a8fce5fbabdcebf3b6088c4e8d36d9cdd59d4488c8ae2e445c9b90` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `b6ef7d1bb4301247e1f415b7e7135fe128ac2cd1d6dfe53188cf43760d68fbc0` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `bd0bb229a63243931045b661da84e5778b9b128e3e1e6476c308f52bb859a695` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `28c8d5a61d12bf7334a77c577728789a5c2687fa7a108568fa32e95f66e85198` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `d3bca8a016b4baf5afb72e69dd707e000fe63e2782bd54e3f45acb68c10068eb` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `1874e9e31a2d9052aa44b11e865e0a8bb2977c7c3bd072bdb8a9dd40b79d205a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `e63ff63cda5de3a82c7cf715302f2f005431bf28c0bf92e1d05d017b83031244` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `d11cb7d202e8202fbfe636cd5b119574c4a1c246b8e272d925857f77ff09c804` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: ababb75cc9cc540628c7156011fd0dc29b639923378c5e05b8a4cb913f3d55b8 -->
