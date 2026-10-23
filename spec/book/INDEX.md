# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `ebfd49270144e26d93e86d9eca9458ce30b0f5042a8d962958144ade91f111a0` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `4eeab6a05c17dd46d9db462dc4c801fcde4ded882648bafd843fde43777a4dd4` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `9bff3a57294945da586ae15052a6cbce53ea166720571859eb2b9f878c8ef6f2` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `02ac9d12d4e0e96656f200f8328965028a511328a7affa24d233988588eef3b4` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `94ed5923623f65f5b78682ecd4a03a433fe1000598dc036f40ffc187a1892918` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `a8690bf5ebef41737c815bd294626929e6d8513911d41a02e7e7bbce5eabfb6f` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `075fa38adc30b3add818021a4bb6df49e95aa6a25addafc0223ea201eeccb208` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `46e73d53d86f66272f6a543db803efeb9538bd8737ba7c81ceaa8b4b78767bd7` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `057b95ec391227d6f11bb538364c7de2a3c88c2d413d9910790f5785b103e0b3` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `6230862e98dd205fd73e56e722ef8dc823d2f999bf7628fcbbc872b75269bbab` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `1d5900f63c0ba912e90222aa4e2d5bf83be424def7e5b62d928b3b0c08c4b70e` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `d3bca8a016b4baf5afb72e69dd707e000fe63e2782bd54e3f45acb68c10068eb` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `1874e9e31a2d9052aa44b11e865e0a8bb2977c7c3bd072bdb8a9dd40b79d205a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `63f298c4a2a6a2ae68fe4952a7291ca6967060775df100848b8ff0d487170c5e` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `33ad5d7a99ee6b49f0d4150ebee3c41a7bef17a49d35377e6cdc9c25a68ee293` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: ac7c97f9a67d0265bd54bf6e2d682b24f9c3e677c7f8ec54e84596c30873bc94 -->
