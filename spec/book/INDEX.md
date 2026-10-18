# RCE spec book: index

Read the chapters in this order. Each ends with the SHA-256 of its body;
an agent proves it read them by following `READ-PROTOCOL.md`.

| File | Body SHA-256 | Read-receipt question |
|---|---|---|
| `00-overview.md` | `3c2dfb5798710e0d9dd38677d2148e4131a7c70ebfe736a673b23d2450613bc5` | In the overview's tiny example, what is the name of the event? |
| `01-notation.md` | `c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4` | In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words? |
| `02-lexical.md` | `2969f1420272ac651ff7991aa16714f7ed9285441021160f137f33441c871bf3` | Which code point is the horizontal tab, as written in this section? |
| `03-syntax.md` | `6ae5395feaedeeaca66120e545c63a5cbebfc4bc70423d7745eb524484e0925d` | Which numbered resolution rule in 3.11 covers `send`? (for example R1) |
| `04-programs.md` | `36aa75c5ccf91fc1e58278eac48086a4b441d0175ea55753e3ad5b3e3788f878` | What is the outermost scope called? (one word) |
| `05-types-ownership.md` | `02ac9d12d4e0e96656f200f8328965028a511328a7affa24d233988588eef3b4` | What is the name of the example kind used to illustrate error sets? |
| `06-regions-effects-extensions.md` | `94ed5923623f65f5b78682ecd4a03a433fe1000598dc036f40ffc187a1892918` | What kind of channel does a capacity of 0 give? (one word) |
| `07-form-evaluation.md` | `e2af17e430e29c18185be20a37e52a7399287867b874f975b7888189f6c506a7` | Which JavaScript function is named for rounding to binary32? |
| `08-steps-loops-concurrency.md` | `22b852f3d01aa50cf11433f9c5206c856ceb0205e9991b8c4f276b6ceb772fbe` | Name-free hashing uses indices named after which logician? (two words) |
| `09-wellformedness.md` | `46e73d53d86f66272f6a543db803efeb9538bd8737ba7c81ceaa8b4b78767bd7` | Which rule refuses two unguarded steps in one region whose patterns overlap? |
| `10-theorems.md` | `d9ae93a59fa8b931bc10795c61b40d29543f8ebf826334ece944569efefe7830` | What is the deep embedding of the emitted Rust subset called? |
| `11-decisions.md` | `de04c4ad325a113a32bdde1a093d0eaf81b9ec5b2072856072aa012efa8bf641` | The decisions list says it decides gaps in which file? |
| `12-tooling.md` | `5c2b8f2dfd9dea59d60dce9af7edd5aeb9b5eed1aa745d3c4ccdec4cd5941263` | Which JSON-RPC error code does the language server return for a request it does not know? (a number) |
| `13-references.md` | `ac72852386a36ef0d84e868bf2f2c771b662a7699b29363418f1f659ad730068` | Who wrote "Communicating Sequential Processes"? (surname) |
| `14-draft06-concurrency.md` | `f3932e9234b5c563e3d751652691dd70f2d9e8ce32fc6c7cd6a36c0922bf5c4b` | How many concurrency words does C.1 add to the keywords of 2.4? (a number) |
| `15-handler-spec-profile.md` | `1874e9e31a2d9052aa44b11e865e0a8bb2977c7c3bd072bdb8a9dd40b79d205a` | Which compiled node id does `deliverable D1` become? (for example role_R1) |
| `16-host-contract.md` | `52ca914e4ad21dce546e4215776e6d6f4f990d48a7d28911b3fea83b3ae9ca14` | Which fcntl command is the append barrier on macOS? (for example F_SETFL) |
| `17-draft09.md` | `3d98b3edaaee7de5fb9d734ec4169242854cc8a55fcfb6de6626f19daab2cf0a` | How many words does F, "Reserved words", add to the reserved list of 2.4? (a number) |

The single-file spec `spec/RCE-SPEC.md` is generated from these files.

<!-- body-sha256: 27b95c1c8bde297c06d3284ba8eea6520a0120be44090e9893950be888271b01 -->
