# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `094aa8d8115d833fd355a6d6d7e4737d3bcd8cdb13d2fd4ae2b0700243f3b463` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `4eeab6a05c17dd46d9db462dc4c801fcde4ded882648bafd843fde43777a4dd4` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `98186dd03ae4fb20812b3b9772ea0a8e54d8afbc9e6ecc930a9e847ecd2b1395` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `02ac9d12d4e0e96656f200f8328965028a511328a7affa24d233988588eef3b4` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `94ed5923623f65f5b78682ecd4a03a433fe1000598dc036f40ffc187a1892918` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `c6ddcd5b3a07f34bb7e5aaaaabf42ff1310957fae62c7391855c042f8c0939c9` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `ffc10ff1391aeba8d2bc8cb2bdad7e727f270df9294c09ee5da25869c4596c49` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `46e73d53d86f66272f6a543db803efeb9538bd8737ba7c81ceaa8b4b78767bd7` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `ce4aa2f16025ab52182438e8c52a4bdc3575dbcb25363f22097e11561bd6e91e` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `6230862e98dd205fd73e56e722ef8dc823d2f999bf7628fcbbc872b75269bbab` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `adc8feee84b2083743eeb5846cb03a86c019a8f7631182c9029af1dbe98abbba` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ed1215a6287919d452ba2ff9b15b321bde445a7a3cd139470c48db7d43fbe913` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `c47ff1e49f4202be941f9a9e58f84cf2e92b5c7d39bdffd92b64e745790a1bf4` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `1874e9e31a2d9052aa44b11e865e0a8bb2977c7c3bd072bdb8a9dd40b79d205a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `af42acba4808eee58f23829fc4016b031c93bd74808c6341bcd0aae014eac337` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `076a0176f0a848259d7ba7171335109affcc607f50e781c745b6aa378f785a10` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: c9c373af3258ee224a4124980822ed644cba3db7a8ed4748c349b97eab896c87 -->
