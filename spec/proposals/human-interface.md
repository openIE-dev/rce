# Proposal: human-interface

Status: open. The owner decided its first two questions on 2026-10-31
(4.6: Q1 full WCAG 2.2 AA, Q2 WAI-ARIA names). Its first slice (4.2,
`ui.rules`) is implemented by the change on branch `claude/ui-rules`
(ledger row of 2026-10-31). H1 (4.3), the frame sink and replay of
frames, is implemented by the change on branch `claude/frame-replay`
as HC-29, which also decides Q4 (ledger row of 2026-11-01); every other
later item is a plan. Sections 1
to 3 describe `main` at a5df5be7, before that change.

The RCE owner wrote, after reading the Raven GUI library: "this is the
lens RCE must also start looking at: UI/UX, HMI, HCI." This proposal
reads that lens in two directions and asks which parts of it a machine
can check exactly.

## 0. Terms

- **UI** (user interface): the part of a system a person perceives and
  acts on. **UX** (user experience): the person's perceptions and
  responses from using it. **HMI** (human-machine interface): the term
  used for operator interfaces of machines and plants (panels, SCADA
  screens, cockpit displays). **HCI** (human-computer interaction): the
  field that studies all three.
- **D**: the durable log the host keeps (Appendix E, E.3). The state of a
  program is always the replay of D (HC-8).
- **View tree**: a value of `ui.view.View` (`stdlib/lib/ui/view.rce`): a
  flat arena of nodes, each with a tag, a key, attributes sorted by name,
  text, a parent and ordered children.
- **Frame**: one view tree that a renderer shows at one log position.
  `ui.view` defines a frame record `["rce.ui.frame/1", seq, view_digest]`
  and its BLAKE3 digest.
- **Renderer**: code outside the program that turns a frame into pixels,
  sound or a platform accessibility tree.
- **Adapter**: host code that connects a program to the outside world
  under the host contract (Appendix E); a renderer is reached through an
  adapter.
- **Accessibility tree**: the tree of roles, names, values and states a
  platform exposes to assistive technology (UI Automation on Windows,
  NSAccessibility on macOS, AT-SPI on Linux, the browser's tree built from
  the DOM and WAI-ARIA). **Assistive technology**: software such as a
  screen reader that reads that tree.
- **Exact requirement**: one that a program can decide from a view tree,
  an event log or a trace with no measurement and no judgement, given the
  attributes the tree declares. **Human requirement**: one that needs
  people (a test with users, an expert review, a rationalisation
  meeting).
- **Status words** used in the mapping tables: **exists now** (the
  mechanism is in `main` and can express the check), **small addition** (a
  library function or one host rule), **new work** (a new adapter, a new
  kind of check, or a spec change).
- **WCAG**: the W3C Web Content Accessibility Guidelines; this proposal
  uses version 2.2. **SC** (success criterion): one testable requirement
  of WCAG, numbered like 1.4.3. **Level**: each SC has level A, AA or AAA;
  **full AA** means every SC of level A and AA. WCAG 2.2 has 55 of them:
  31 of level A and 24 of level AA (4.1.1 Parsing is obsolete in 2.2 and
  has no level).
- **Other abbreviations**: **DOM**, the browser's document object
  model; **WAI-ARIA** (ARIA), the W3C vocabulary of roles, states and
  properties for the DOM; **Core-AAM**, the W3C mapping of ARIA to
  platform accessibility interfaces; **UIA**, Microsoft UI Automation;
  **AT-SPI**, the Linux Assistive Technology Service Provider Interface;
  **LSP**, the Language Server Protocol; **PTY**, a pseudo-terminal;
  **IME**, an input method editor (the window that composes text in
  scripts such as Chinese or Japanese); **CSI** and **SGR**, terminal
  control sequences; **CBOR**, the binary encoding of RFC 8949; **IRP**,
  the SCXML implementation report test suite; **est_j**, RCE's modelled
  energy figure in joules (C.5), never a meter reading; **EN 301 549**,
  the European accessibility requirements for ICT products; **BCP 47**,
  the IETF format of language tags (`en`, `en-GB`); **CSS px**, the CSS
  reference pixel, 1/96 inch.

In plain words:

1. RCE already holds most of the parts a checked human interface needs: a
   view tree with a digest, decoded input events, statecharts, a frame
   tick, a log of every input, and replay. Before this change nothing
   joined them, and no adapter draws a view tree on a screen.
2. A large share of the published requirements for human interfaces are
   exact propositions over a view tree or over a log of events: a
   contrast ratio, a target size, whether every action has a name and a
   key, how many alarms an operator received in ten minutes. RCE checks
   these the way it checks everything else: as examples and checks that
   a program must meet to be admitted.
3. The rest (whether a label means the right thing to its reader, whether
   a task is easy, whether an alarm is worth raising) stays with people,
   and this proposal says so per requirement.
4. The first step, now built, is a pure library of exact rules over view
   trees for every SC of WCAG 2.2 level A and AA, `ui.rules`. Where a part
   of an SC needs a person, the program declares who decided it and when,
   and the library reports the SC as undecided until it does. It touches
   only `stdlib/` and this proposal.

## 1. Inventory: what RCE had before this change

### 1.1 Programs with a human interface (face A)

| Path | What it is |
|---|---|
| `stdlib/lib/ui/view.rce` (625 lines) | `ui.view`: view tree as a flat arena; `view_ok` (well-formed); `diff`/`apply` with the fewest removals, creations and moves; `diff_vue` matching Vue 3; deterministic CBOR, `view_digest`, `frame_record`, `frame_digest`. Pure. Attributes are free text: the module defines no roles, names, colours or sizes, and no layout. |
| `stdlib/lib/ui/input.rce` (3,627 lines) | `ui.input`: key, mouse, focus, paste, resize, composition and clipboard records; the crossterm 0.29 terminal byte decoder; W3C UI Events `code` and `key` tables and the mapping to DOM KeyboardEvent, PointerEvent and WheelEvent. |
| `stdlib/lib/ui/states.rce` (2,570 lines) | `ui.states`: the SCXML 1.0 algorithm (W3C Recommendation 2015-09-01, Appendix D) with virtual time; passes the W3C SCXML IRP suite (200 tests) and an XState 5 mode that matches XState 5.33.2 on 300 charts. |
| `stdlib/lib/ui/text.rce` (37,122 lines, mostly tables) | `ui.text`: Unicode 16.0 grapheme, word and sentence boundaries (UAX #29), line breaks (UAX #14), bidi (UAX #9), terminal cell width. |
| `stdlib/lib/ui/ecma.rce` (1,756 lines) | `ui.ecma`: the ECMAScript subset of the SCXML datamodel. |
| `stdlib/tests/ui/view_checks*.rce` | the only users of `ui.view` in the repository: tests of diff, apply and digests against Vue vectors and breadth-first search. |
| Appendix E, HC-2 and HD-4 (`spec/book/16-host-contract.md`) | the frame tick: wall time enters only as `Tick(n)` (and `host.WallTime`). `rce host serve` delivers `Tick` every `FRAME_MS` = 16 ms (`spec/checker/src/cli.rs`), chosen by the engine, never by a setting (HC-14). |
| Appendix E, HC-10 | key, pointer, resize and focus input are events in D like any other input. |
| Appendix E, HC-24 to HC-26 | the HTTP event host: each request is an `HttpRequest` input and every response byte is a function of D. A web page served by a program is therefore a function of D. |
| `spec/checker/src/host.rs` (`Admitted.display`) | the only screen sink today: the `data` of each emitted `Display(data: text)` is written to stdout. |
| `apps/terminal.rce` (308 lines) | a terminal: PTY as events, a screen model (rows × cols, cursor, CSI moves, SGR, scrollback) queried by `Screen` and `History`; output through `Display`. It does not use `ui.view` or `ui.input`. |
| `apps/chat-app.rce` → `apps/chat.rce` | chat state (workspaces, threads, reactions, E2EE DMs) with no view; `scripts/chat-ui.py` is a line-mode Python front end that sends events and prints the program's `View` query. |
| `programs/network/notes_http.rce` | a JSON service over the HTTP host; no HTML. |

### 1.2 RCE's own interface to its users (face B)

| Path | What it is |
|---|---|
| `spec/book/12-tooling.md` 10.1, 10.2 | exit statuses (0, 1 requires change or a kept step, 2 usage, 3 undecided) and the report form `requires change FILE: line N: RULE message`; a `recommended` line after `ok` (W-INT-2). |
| 7.1 (merged in #85) | the three methods of guidance: each static rule is *altered* (`rce fmt` writes the rewrite, printed `altered FILE: line N: RULE what`), *recommended* (admitted, with a `recommended` line) or *requires change* (the report ends `[requires change]` or `[altered by rce fmt]`). |
| `spec/proposals/positional-cli.md` (accepted and implemented 2026-10-31, #91) | one positional grammar (PC-1 to PC-8) for every binary; a removed flag prints its new spelling and exits 2; usage text printed from one table per binary. |
| HC-28 (merged in #88) | HC-28: `rce host run`/`serve` report `idle`, `working`, `done`, `error` to a terminal as OSC 7501 records; not in D; never `blocked`, "since it asks no person anything". |
| 10.13 (merged in #89) | `rce repair`: a ladder of exact proposers judged by the checker, 0 model calls; the candidate a person then reviews. |
| `spec/checker/src/lsp.rs`, 10.7 | `rce lsp`: diagnostics equal to the reports of `rce check`; hover, definition, completion, symbols. No code actions yet. |
| `tree-sitter/` | grammar plus `highlights.scm`, `folds.scm`, `indents.scm`; CI runs a three-parser agreement check. |
| `playground/` | a web page: the checker compiled to WebAssembly, an editor `<textarea>`, verdict, rule reports and example traces. |
| `ide/crates/rce-ide` | the desktop and web IDE: a photon view tree on wgpu + winit + Parley; editor with live `rce check`, proposals accepted by the checker or requiring change, receipts, energy columns, network permission prompts. |
| `ide/crates/rce-mcp`, `acp-client` | the interface for agents: `read_form`, `propose_edit`, `rce_check`, `rce_test`, `emit`; every proposal goes through `rce check`. |
| `book/` (`book/src/ch16-host-tools.md` and others) | the RCE Book; chapter 16 names `Tick` and `Display`. No chapter on interfaces. |
| `spec/proposals/compliance.md` COMP-19, `scripts/vocab-allow.tsv` | one vocabulary for outcomes and reports across code, output and documents. |
| `docs/figures/src/host-survey/QUALITY-METRICS.csv` | the host survey's row for accessibility reads "no metric found … opinion only". |

### 1.3 What is missing

1. **Nothing draws a `ui.view` tree.** No program, app or adapter uses
   `ui.view` outside its own tests; the host has no frame sink, only
   `Display(data: text)` to stdout; no adapter exposes an accessibility
   tree. A program can compute frames and their digests today, but no
   person can see one.
2. **`ui.view` has no semantics for people.** Attributes are free text.
   There is no agreed attribute for a role, an accessible name, a key
   binding, focus order, colour, size or alarm priority, and no layout,
   so no rule can be stated over a tree yet.
3. **No rule of RCE speaks about people.** No check, example convention or
   host rule covers contrast, keyboard access, names, timing or alarm
   load. The host survey records accessibility as "opinion only".
4. **No link from input events to views.** `ui.input` decodes keys and
   `ui.states` runs charts, but no module says which node an event
   targets (hit testing, focus) or what focus does.
5. **The terminal app is the one complete interface, and it is text.**
   Its screen model is not a `ui.view` tree, so none of the view
   machinery applies to it.
6. **Face B has exact pieces but no shared rule set.** Report forms,
   exit statuses and the positional grammar are specified; the IDE and
   the playground are not checked against any accessibility rule.

This change addresses items 2 and 3 for WCAG 2.2 AA (section 4.2): the
vocabulary `rce.ui.attrs/1` gives `ui.view` attributes a meaning, and
`ui.rules` states a rule for every exact part. Items 1, 4, 5 and 6 remain
later items (4.3).

### 1.4 Exact findings in RCE's own interfaces

Computed for this proposal with the WCAG 2.2 relative-luminance formula
from the colours in the source (no rendering was measured):

| Where | Pair | Ratio | Requirement | Result |
|---|---|---:|---|---|
| `ide/crates/rce-ide/src/view.rs`, panel title (15 px, `ACCENT` on `PANEL`) | rgb(0.33,0.55,0.95) on rgb(0.15,0.16,0.19) | 4.46 : 1 | WCAG 1.4.3, 4.5 : 1 for text under 24 px | fails |
| same file, `button()` label (13 px, white on `ACCENT`) | white on rgb(0.33,0.55,0.95) | 3.27 : 1 | WCAG 1.4.3, 4.5 : 1 | fails |
| same file, `DIM` and `ERR` text on `PANEL` | | 4.64 and 4.55 : 1 | 4.5 : 1 | meets, by less than 0.2 |
| `playground/style.css`, `<textarea>` border `#ccc` on `#fff` | | 1.61 : 1 | WCAG 1.4.11, 3 : 1 for the boundary that identifies a control | fails if the border is the only cue |
| `playground/app.js`, `#verdict` rewritten on each check | no `role="status"` or `aria-live` | n/a | WCAG 4.1.3 status messages | fails |
| `rce-ide` | photon's `accessibility_label` annotations are used to place the IME window and to find elements in tests; no platform accessibility bridge (such as AccessKit) is a dependency; the web build draws to a canvas | n/a | WCAG 4.1.2, EN 301 549 11.5.2 | no accessibility tree is exposed |
| `rce-ide` editor | Tab is an edit operation (`EditOp::Tab`); the code read for this proposal did not show a key that moves focus out of the editor | n/a | WCAG 2.1.2 no keyboard trap | to confirm by hand |

These are the kind of results the first slice produces for any
program, and the reason face B is in scope. At 8 bits per channel (the
IDE colours rounded to `#548cf2` and `#262930`) the first two ratios are
4.44 : 1 and 3.28 : 1; both pairs, and the playground pair, are contrast
vectors of `ui.rules` (`stdlib/vectors/ui/contrast-axe.tsv`). Fixing them
is a separate change (4.5).

## 2. The lens in two faces

Each table lists the requirement, its source clause, whether it is exact
(and over what), the RCE mechanism that would check it, and that
mechanism's status before this change. For WCAG 2.2, section 4.2 replaces
these rows with one row per success criterion and says what this change
implements. "View" means one view tree; "log" means D or an
event list of an example; "trace" means the output of `rce trace`.

The mechanisms are:

- **check over a view**: a `check` (6.9) or an `example` whose final
  expression evaluates a pure rule function on the view the program
  computes from its state;
- **example over events**: an `example` whose event list drives the
  interaction (keys, focus, ticks) and whose expression states the
  outcome;
- **bounded exploration**: the explored alphabet of `rce dim` and
  `rce fingerprint` (10.11, T57 to T59), used to apply a view rule to
  every state reachable within a fuel bound rather than only to the
  states the examples reach;
- **host rule**: a numbered rule of Appendix E;
- **fuel per input** (HC-7a) as a work bound per event; a modelled
  joule figure where `est_j` is available; neither is a measured time;
- **replay of D** (HC-8) for incident review.

### 2.1 Face A: programs with a human interface

#### Exact on a view (given declared attributes)

| Requirement | Source | Exact over | RCE mechanism | Status |
|---|---|---|---|---|
| Text contrast ≥ 4.5 : 1, or ≥ 3 : 1 for large text (≥ 18 pt, or ≥ 14 pt bold) | WCAG 2.2 SC 1.4.3 (AA); EN 301 549 V3.2.1 9.1.4.3 (web) and 11.1.4.3 (software) | view, when every text node's foreground and the nearest declared background are known | check over a view | small addition (rule library) |
| Contrast ≥ 3 : 1 for control boundaries, states and graphics needed to understand them | WCAG 1.4.11 (AA) | view, with declared boundary and adjacent colours | check over a view | small addition |
| Pointer target ≥ 24 × 24 CSS px, or the spacing exception | WCAG 2.5.8 (AA) | view with declared sizes; the spacing exception needs positions | check over a view (size part); layout needed for spacing | small addition (size); new work (layout) |
| Every user interface component has a name and a role, and its states and values are exposed | WCAG 4.1.2 (A); EN 301 549 11.4.1.2; ARIA required states per role | view | check over a view | small addition |
| All functionality is operable by keyboard | WCAG 2.1.1 (A) | view (every actionable node is focusable or has a key) plus log (the key reaches the action) | check over a view and example over events | small addition |
| Focus order preserves meaning | WCAG 2.4.3 (A) | exact part: one total order, every focusable node once, tree order unless declared; whether the order "preserves meaning" is human | check over a view | small addition (exact part) |
| Focus visible | WCAG 2.4.7 (AA); stricter form 2.4.13 (AAA: area and 3 : 1 change) | view: every focusable node declares a focus style that differs from its unfocused style; 2.4.13's ratio is exact with declared colours | check over a view | small addition |
| Focus not entirely hidden by other content | WCAG 2.4.11 (AA) | needs geometry and stacking | renderer report | new work |
| Status messages are exposed without moving focus | WCAG 4.1.3 (AA) | view diff between two frames: a node whose text changed, outside the focused subtree, is in a live region | check over two views (`diff`) | small addition |
| Reflow at 320 CSS px without two-dimensional scrolling | WCAG 1.4.10 (AA) | needs layout | renderer at a fixed width, compared to a reference | new work |
| HMI style: colour reserved for abnormal and alarm states; a consistent style guide | ANSI/ISA-101.01-2015 (style guide, colour use) | view: a declared palette where alarm colours appear only on nodes with an alarm state | check over a view against the program's own style table | small addition |
| Display hierarchy depth and navigation (overview to detail) | ANSI/ISA-101.01-2015 (display hierarchy) | the navigation graph between displays: depth and reachability are exact; the right number of levels is a design choice | example over events, bounded exploration | small addition |

#### Exact on a log or a trace

| Requirement | Source | Exact over | RCE mechanism | Status |
|---|---|---|---|---|
| No change of context on focus; none on input unless announced | WCAG 3.2.1 (A), 3.2.2 (A) | log: a declared context attribute of the root is unchanged by `Focus` and by value input events | example over events; bounded exploration | small addition |
| Time limits: the user can turn off, adjust or extend (warning, ≥ 20 s to extend, up to ten times) | WCAG 2.2.1 (A) | log in `Tick` counts: every timeout step is preceded by a warning step and an extend action exists; ticks convert to seconds through `FRAME_MS` | example over events | small addition |
| Pause, stop, hide moving content that lasts more than 5 s | WCAG 2.2.2 (A) | log: an auto-advancing view has a pause action that stops view change under further `Tick` | example over events | small addition |
| No keyboard trap | WCAG 2.1.2 (A) | log over the focus chart: from every focusable node a key sequence leaves it; a graph property of the focus statechart | bounded exploration over `ui.states` charts | small addition |
| Alarm rates per operator: about 6 per hour "very likely acceptable", 12 "maximum manageable"; floods (more than 10 alarms in 10 minutes), their share of time; chattering and fleeting alarms; stale alarms; priority distribution | ANSI/ISA-18.2-2016 section 16 (Table 5); IEC 62682:2022; EEMUA 191 | log: every one is a count over alarm events in windows of D | a pure metric library over D, checked by examples; replay of D | small addition (library); time base is an open question for the owner, listed at the end |
| Shelving: authorised, logged, time limited, automatically unshelved | ISA-18.2-2016 and IEC 62682 (shelving is required functionality) | log: every `Shelve` names an operator capability and an expiry; an unshelve step occurs at expiry | host rule HC-5 (capabilities) plus examples | exists now (capabilities, D); small addition (library) |
| Alarm priority and acknowledgement state machine | ISA-18.2 alarm states | chart: the state set and transitions are exact | `ui.states` chart plus examples | exists now |
| Interaction is bounded per event | ISO 9241-110 controllability; response-time goals of ISA-101 | fuel per input (HC-7a) bounds work exactly; milliseconds are not exact | fuel per input; `est_j` modelled | exists now (fuel); time stays measured outside D |
| Use errors in a medical device are traceable to scenarios | IEC 62366-1:2015+A1:2020, 5.4/5.5 (hazard-related use scenarios), 5.6 (user interface specification), 5.9 (summative evaluation) | log: a use scenario is an event list; the UI specification's testable requirements are examples | examples; replay of D for observed use errors | exists now (examples); the scenarios themselves are human |
| Incident review: what the operator was shown and did | ISA-18.2 monitoring and assessment; IEC 62366-1 post-production; ISO 26262 and IEC 61508 incident analysis | log: every input is in D (HC-10); each frame is a function of state, so its digest is a function of D | replay of D; a frame sink so that replay can print frame digests per seq | exists now (inputs, replay); small addition (frame sink) |

#### Human (no check covers them)

| Requirement | Source | Why it is human |
|---|---|---|
| Effectiveness, efficiency and satisfaction for specified users, goals and context of use | ISO 9241-11:2018 | defined through measurement with users |
| Suitability for the user's tasks, self-descriptiveness, conformity with user expectations, ease of discovery, exploration and retention (clause 5.4), controllability, use error robustness, user engagement | ISO 9241-110:2020, clause 5 (5.4 discovery, exploration and retention; 5.6 use error robustness; 5.7 user engagement) | principles about people; individual recommendations can be made exact only after a team chooses a concrete rule |
| Use specification, user groups, formative and summative evaluation with representative users | IEC 62366-1:2015+A1:2020, 5.1, 5.7, 5.9 | needs people; RCE can hold the scenarios as examples and the observed errors as D |
| Controllability classes C0 to C3 in the hazard analysis and risk assessment | ISO 26262-3:2018, clause 6 | a judgement about an average driver |
| Reasonably foreseeable misuse and human error in hazard analysis | IEC 61508-1:2010, 7.4 (hazard and risk analysis; text not fetched for this proposal) | a judgement |
| Alarm philosophy and rationalisation: whether an alarm should exist, its priority and its response | ISA-18.2, IEC 62682 | a review by people; RCE checks only that the result is consistent and measures the load |
| Meaningful sequence, purpose of a link, quality of a label or of alternative text | WCAG 1.3.2, 2.4.4, 2.4.6, 1.1.1 | presence is exact, meaning is not |
| Nielsen's ten usability heuristics (visibility of system status; match between system and the real world; user control and freedom; consistency and standards; error prevention; recognition rather than recall; flexibility and efficiency; aesthetic and minimalist design; help users recognise, diagnose and recover from errors; help and documentation) | Nielsen Norman Group | an expert review method; some have exact parts (a status indicator exists, an undo action exists), the heuristic itself is a judgement |

### 2.2 Face B: RCE's own interface to its users

#### Exact

| Requirement | Source | Exact over | RCE mechanism | Status |
|---|---|---|---|---|
| Every report names its rule and its line | ISO 9241-110 self-descriptiveness; Nielsen heuristic 9 | output text | 10.2 report form; tests in `spec/checker/tests/cli.rs` | exists now |
| Every report says what kind of guidance it is: altered, recommended, requires change | ISO 9241-110 controllability | output text | 7.1 table and report tags | exists now |
| A fixed set of exit statuses | ISO 9241-110 conformity with user expectations | process status | 10.1 | exists now |
| The usage text and the parser cannot disagree; a removed flag prints its new spelling | ISO 9241-110 use error robustness | usage table | PC-8 and decision 2 of `positional-cli.md` | exists now |
| A preview before any write | ISO 9241-110 controllability | command behaviour | `rce fmt` with `check` (10.6) | exists now |
| The editor's diagnostics are the checker's reports | consistency (Nielsen 4) | LSP messages | 10.7 | exists now |
| One vocabulary across code, output and documents | ISO 9241-110 conformity with user expectations | text | COMP-19, `scripts/vocab-allow.tsv` | exists now |
| The host's status is visible to a person watching a terminal, and never alters D | Nielsen 1 (visibility of system status) | stderr records; D byte-identical with and without | HC-28 | exists now |
| IDE and playground meet the face A rules (contrast, names, keyboard, status messages) | WCAG 2.2 AA; EN 301 549 clause 9 (web) and clause 11 (software, including 11.5.2 platform accessibility services) | the IDE's photon tree and the playground DOM | the same rule set, applied to a projection of each tree; axe-core on the playground | new work (IDE projection); small addition (playground) |
| Documents define a term before first use and number labels in reading order | AGENTS.md "Writing"; the research site's `scripts/lint-order.mjs` | built text | no such lint in this repository | small addition |

#### Human

| Requirement | Why it is human |
|---|---|
| Whether a report's message helps a reader fix the program | needs readers; RCE can count how often a rule is followed by a passing edit, not whether its text is clear |
| The book's order and examples | a reading by people |
| The human checkpoint when agents write RCE: whether the intent says what the owner wants | the checker decides that a program meets its examples and checks; W-INT-2 states what the examples leave unconstrained; only a person can say the intent is the right one. AGENTS.md makes the reviewer's approval part of "done". |
| Whether `rce repair`'s first admitted candidate is the intended fix | `rce repair` reports restored, behaviour-equal and examples-only cases separately; choosing among them is a review |

## 3. Prior art for deterministic or verified UI

| Work | What it checks exactly | Where RCE would differ |
|---|---|---|
| The Elm Architecture (Model, `update`, `view`) — <https://guide.elm-lang.org/architecture/> | Purity of `update` and `view` by the type system; no runtime exceptions in Elm code. | RCE adds a durable log D with a hash chain, examples and checks as admission, fuel per input, and replay by any host. Elm has no admission rule over the view. |
| Redux and Redux DevTools time travel — <https://redux.js.org/understanding/thinking-in-redux/three-principles>, <https://github.com/reduxjs/redux-devtools> | Nothing is checked; reducers are pure by convention, and the action log is replayed in a development tool. | In RCE, replay is the definition of state (HC-8), in production, with the clock as an input (HC-2). |
| SCADE Display and its KCG code generator — <https://www.ansys.com/products/embedded-software/ansys-scade-display> | The generator is qualifiable as a DO-330 TQL-1 tool for DO-178C Level A (vendor material): the code matches the display model. It does not decide usability. | RCE's admission is per program, with examples; a qualified generator path for RCE would be the emit targets plus a tool-confidence argument (ISO 26262-8 clause 11), which RCE does not have. |
| ARINC 661 (cockpit display system interfaces) — <https://en.wikipedia.org/wiki/ARINC_661>; specification from SAE ITC | A fixed widget set and a Definition File loaded at start; user applications change only runtime parameters. The structure is exact, the behaviour is in the user applications. | `ui.view` is a plain tree like a Definition File. A fixed role vocabulary and "no widget created after the definition phase" are design choices RCE could state as rules. |
| Qt Safe Renderer — <https://doc.qt.io/QtSafeRenderer/>; TÜV NORD certificate SEBS-A.113256/17 V3.0 (<https://doc.qt.io/QtSafeRenderer/Qt_SafeRenderer_certificate.pdf>) | A separate, certified renderer (IEC 61508 SIL 3, ISO 26262 ASIL D, EN 50128 SIL 4) for safety-critical indicators, isolated from the main UI. | The pattern of a small checked renderer for a small set of safety frames beside a general one maps onto RCE as a second adapter that draws only frames whose digests the program names. |
| PVSio-web (Oladimeji, Masci, Curzon, Thimbleby; ECEASST, 2013) — <https://github.com/pvsioweb/pvsio-web> | Device UI behaviour modelled in PVS and proved (for example, infusion pump number entry); prototypes driven by the model. | RCE runs the program itself rather than a model of it; a proof would come from RCE's Lean model and its theorems, not from PVS. |
| IVY workbench (Campos, Harrison; University of Minho) — <https://ivy.di.uminho.pt/> | Interactor models checked with NuSMV against properties such as "every mode can be left" or "an action is always available". | The same properties are bounded exploration over a `ui.states` chart in RCE; RCE's bound is fuel, not a model checker's full state space. |
| Statecharts: Harel (1987); SCXML (W3C 2015) — <https://www.w3.org/TR/scxml/>; XState — <https://stately.ai/docs/xstate> | The execution algorithm is exact; tools visualise and test charts. | RCE has this now: `ui.states` passes the W3C IRP suite and matches XState 5 on 300 charts. |
| axe-core (Deque) — <https://github.com/dequelabs/axe-core>; W3C ACT Rules — <https://www.w3.org/WAI/standards-guidelines/act/rules/> | Rules over the browser's DOM and accessibility tree with a stated aim of no false positives; cases it cannot decide are reported "incomplete" for people. Deque reports that its rules find on average 57 % of issues (by count, over about 2,000 audits; a vendor study, 2021). | RCE's rules would run over `ui.view` before any renderer exists, inside admission, with "undecided" (exit 3) for what axe-core calls incomplete. axe-core stays useful as an independent reference on the DOM projection. |


## 4. The proposal

### 4.1 Goals

1. **G1.** Every requirement in section 2 that is exact becomes a rule a
   program can be held to, by the same examples and checks that admit it
   today. No new admission gate.
2. **G2.** What a rule cannot decide is reported as undecided, never as
   met. What no rule covers is listed as human, per program, and stays
   undecided until a person attests it.
3. **G3.** A person can see a `ui.view` frame on a real screen and reach
   it with a screen reader, through an adapter that obeys Appendix E.
4. **G4.** Replay of D can show, per log position, which frame the
   program produced and which input came next, for incident review.
5. **G5.** RCE's own interfaces (the IDE, the playground, the command
   line) meet the same rules.

### 4.2 First slice: `ui.rules`, full WCAG 2.2 AA (implemented)

A pure module `stdlib/lib/ui/rules.rce` (`ui.rules`), with vectors and
tests in `stdlib/`, and nothing else: no host rule, no command word, no
chapter of the specification changes.

#### Scope and classes

The scope is every SC of level A and AA of WCAG 2.2 (decision Q1, 4.6):
55 criteria, read from the W3C source (section 5). Each has exactly one
class:

- **E**: exact on one view tree, given the attributes the tree declares;
- **L**: exact on a frame trace (below) or on the event log D;
- **P**: partly exact: a rule decides the exact part, and the table states
  the part a person decides;
- **H**: human only.

A **frame trace** is a list of steps `Step(ev, ms, v)`: the view `v` a
program shows after event `ev`, `ms` milliseconds after the previous step.
The event words are `start`, `tick`, `focus K`, `input K`, `activate K`,
`key NAME`, `hover K`, `unhover` and `navigate`, where K is a node's
`ui.view` key and NAME a UI Events key value as `ui.input` uses them. A
rule over one view applies to every step's view. A program builds a frame
trace from its own state function and the events of an example, so an L
rule needs nothing from the host.

Counts: **E 16, L 10, P 22, H 7** (55).

- E: 1.4.2, 1.4.3, 1.4.4, 1.4.10, 1.4.12, 2.1.1, 2.1.2, 2.1.4, 2.4.7,
  2.4.11, 2.5.1, 2.5.2, 2.5.4, 2.5.7, 2.5.8, 4.1.2.
- L: 1.4.13, 2.2.1, 2.2.2, 3.2.1, 3.2.2, 3.2.3, 3.2.4, 3.2.6, 3.3.7, 4.1.3.
- P: 1.1.1, 1.2.1, 1.2.2, 1.2.3, 1.2.4, 1.2.5, 1.3.1, 1.3.4, 1.3.5, 1.4.11,
  2.3.1, 2.4.1, 2.4.2, 2.4.3, 2.4.4, 2.5.3, 3.1.1, 3.1.2, 3.3.1, 3.3.2,
  3.3.4, 3.3.8.
- H: 1.3.2, 1.3.3, 1.4.1, 1.4.5, 2.4.5, 2.4.6, 3.3.3.

Five E criteria need layout geometry, which RCE does not have: 1.4.4
(resize text), 1.4.10 (reflow), 1.4.12 (text spacing), 2.4.11 (focus not
obscured) and the spacing exception of 2.5.8 (target size). For each,
`ui.rules` implements the part that is exact on declared attributes (a
refused zoom, a declared width over 320 CSS px, refused text spacing, a
malformed layer, a target without a declared size) and reports the part
that needs geometry like a human part: undecided until attested, until a
renderer reports geometry (4.3, H1 and H4). For 2.4.11 and 2.5.8 that
part applies only when a view declares a layer above the flow, or a target
under 24 by 24.

#### The table

One row per SC, in WCAG order. "Rule" is the function of `ui.rules` that
decides the exact part (its name in `rule_names()`; the module defines it
as `r_RULE`). "What a person decides" is the human part, which the program
attests; "nothing" means the rule decides the whole SC.
`stdlib/tests/ui/rules_table_probes.py` checks that this table,
`criteria()` in the module and the W3C list agree.

| SC | Title | Level | Class | Rule | What a person decides |
|---|---|---|---|---|---|
| 1.1.1 | Non-text Content | A | P | `text_alternative` | whether each text alternative serves the purpose of its image |
| 1.2.1 | Audio-only and Video-only (Prerecorded) | A | P | `media_alternative` | whether each transcript or audio track presents the same information as the media |
| 1.2.2 | Captions (Prerecorded) | A | P | `captions_prerecorded` | whether the captions are accurate and synchronised |
| 1.2.3 | Audio Description or Media Alternative (Prerecorded) | A | P | `description_or_alternative` | whether the audio description or alternative is accurate |
| 1.2.4 | Captions (Live) | AA | P | `captions_live` | whether the live captions are accurate and synchronised |
| 1.2.5 | Audio Description (Prerecorded) | AA | P | `audio_description` | whether the audio description is accurate |
| 1.3.1 | Info and Relationships | A | P | `info_relationships` | whether every structure and relationship shown is expressed by roles and states |
| 1.3.2 | Meaningful Sequence | A | H | - | whether the reading order (tree order) preserves meaning |
| 1.3.3 | Sensory Characteristics | A | H | - | whether instructions rely on shape, colour, size, location, orientation or sound alone |
| 1.3.4 | Orientation | AA | P | `orientation` | whether a declared orientation lock is essential |
| 1.3.5 | Identify Input Purpose | AA | P | `input_purpose` | whether every field that collects information about the user declares its purpose |
| 1.4.1 | Use of Color | A | H | - | whether colour is the only visual means of conveying information |
| 1.4.2 | Audio Control | A | E | `audio_control` | nothing |
| 1.4.3 | Contrast (Minimum) | AA | E | `contrast_minimum` | nothing |
| 1.4.4 | Resize Text | AA | E | `resize_text` | layout geometry, which RCE does not have yet: text resized to 200 percent without loss of content |
| 1.4.5 | Images of Text | AA | H | - | whether images of text are essential or customisable |
| 1.4.10 | Reflow | AA | E | `reflow` | layout geometry, which RCE does not have yet: content at 320 CSS px without two-dimensional scrolling |
| 1.4.11 | Non-text Contrast | AA | P | `non_text_contrast` | which boundaries and graphics are required to identify components and understand content |
| 1.4.12 | Text Spacing | AA | E | `text_spacing` | layout geometry, which RCE does not have yet: no loss of content under the stated text spacing |
| 1.4.13 | Content on Hover or Focus | AA | L | `hover_focus_content` | nothing |
| 2.1.1 | Keyboard | A | E | `keyboard` | nothing |
| 2.1.2 | No Keyboard Trap | A | E | `no_keyboard_trap` | nothing |
| 2.1.4 | Character Key Shortcuts | A | E | `character_key_shortcuts` | nothing |
| 2.2.1 | Timing Adjustable | A | L | `timing_adjustable` | nothing |
| 2.2.2 | Pause, Stop, Hide | A | L | `pause_stop_hide` | nothing |
| 2.3.1 | Three Flashes or Below Threshold | A | P | `three_flashes` | flashing inside media and images, which the tree does not describe |
| 2.4.1 | Bypass Blocks | A | P | `bypass_blocks` | whether blocks repeat across views when no main landmark, heading or skip link is declared |
| 2.4.2 | Page Titled | A | P | `page_titled` | whether the title describes the topic or purpose |
| 2.4.3 | Focus Order | A | P | `focus_order` | whether the focus order preserves meaning and operability |
| 2.4.4 | Link Purpose (In Context) | A | P | `link_purpose` | whether each link's purpose follows from its name and context |
| 2.4.5 | Multiple Ways | AA | H | - | whether more than one way locates this view within its set |
| 2.4.6 | Headings and Labels | AA | H | - | whether headings and labels describe topic or purpose |
| 2.4.7 | Focus Visible | AA | E | `focus_visible` | nothing |
| 2.4.11 | Focus Not Obscured (Minimum) | AA | E | `focus_not_obscured` | layout geometry, which RCE does not have yet: whether a declared layer hides the focused component |
| 2.5.1 | Pointer Gestures | A | E | `pointer_gestures` | nothing |
| 2.5.2 | Pointer Cancellation | A | E | `pointer_cancellation` | nothing |
| 2.5.3 | Label in Name | A | P | `label_in_name` | labels shown outside the component or as images of text |
| 2.5.4 | Motion Actuation | A | E | `motion_actuation` | nothing |
| 2.5.7 | Dragging Movements | AA | E | `dragging` | nothing |
| 2.5.8 | Target Size (Minimum) | AA | E | `target_size` | layout geometry, which RCE does not have yet: the spacing exception for targets under 24 by 24 |
| 3.1.1 | Language of Page | A | P | `language_of_page` | whether lang names the language of the content |
| 3.1.2 | Language of Parts | AA | P | `language_of_parts` | whether every passage in another language declares lang |
| 3.2.1 | On Focus | A | L | `on_focus` | nothing |
| 3.2.2 | On Input | A | L | `on_input` | nothing |
| 3.2.3 | Consistent Navigation | AA | L | `consistent_navigation` | nothing |
| 3.2.4 | Consistent Identification | AA | L | `consistent_identification` | nothing |
| 3.2.6 | Consistent Help | A | L | `consistent_help` | nothing |
| 3.3.1 | Error Identification | A | P | `error_identification` | whether every detected error sets aria-invalid and its message describes the error |
| 3.3.2 | Labels or Instructions | A | P | `labels_or_instructions` | whether the labels and instructions are sufficient |
| 3.3.3 | Error Suggestion | AA | H | - | whether a known correction is suggested |
| 3.3.4 | Error Prevention (Legal, Financial, Data) | AA | P | `error_prevention` | whether every submission with legal, financial or data commitments declares commits |
| 3.3.7 | Redundant Entry | A | L | `redundant_entry` | nothing |
| 3.3.8 | Accessible Authentication (Minimum) | AA | P | `accessible_authentication` | whether any other cognitive function test is used, and the object and personal-content exceptions |
| 4.1.2 | Name, Role, Value | A | E | `name_role_value` | nothing |
| 4.1.3 | Status Messages | AA | L | `status_messages` | nothing |

#### Findings, verdicts and attestations

A rule returns **findings**: one per node and frame, `fails` or
`undecided` (a needed value is missing, or the case is within the error
bound of a threshold), each with the SC, the rule, the frame, the node and
the reason. A rule never meets on missing data: text without a declared
foreground and background is undecided under 1.4.3, a target without a
declared size undecided under 2.5.8, a focusable node no step focuses
undecided under 3.2.1.

`report(trace)` gives one **verdict** per SC:

1. **not applicable**, with the reason, when the SC does not apply to
   any frame (no media for 1.2.x, no input for 3.3.2, one context for
   3.2.3);
2. **fails** when an exact finding fails;
3. **undecided** when an exact finding is undecided;
4. otherwise, when a human part applies (class H, the human part of a P,
   the geometry part of the five E above, or a node that claims an
   exception), **attested** if every frame's root carries the attribute
   `attest-ID` (ID the SC number) with the value `WHO;YYYY-MM-DD`, naming
   who decided and when, and **undecided** ("needs human attestation:
   ...") if not; a malformed attestation is undecided;
5. otherwise **meets**.

An attestation never covers an exact failure: 2.4.2 with an empty title
fails whatever the root attests. An exception that only a person can
judge (essential, logotype, real-time, the media alternative of 1.2.x) is
claimed on a node with `exempt` (a list of SC numbers): the rule skips the
node and its subtree, and the SC then needs an attestation. Full AA for a
trace is `unmet(trace)` empty: every SC meets, is attested or does not
apply, and nothing meets silently.

#### The attribute vocabulary `rce.ui.attrs/1`

All values are text, as `ui.view` requires: booleans `true` or `false`,
numbers decimal integers, colours `#rrggbb`. Role, state and property
names are WAI-ARIA's (decision Q2); names without an ARIA equivalent are
HTML's where HTML has one, and RCE's own otherwise.

| Attribute | Value | Read by |
|---|---|---|
| `role` | a WAI-ARIA role (the 87 of `stdlib/vectors/ui/aria-roles.tsv`) | 1.1.1, 1.3.1, 2.4.4, 3.3.2, 4.1.2 and most others |
| `aria-label`, `aria-labelledby` | the accessible name; labelledby names node keys. Without either, roles whose name comes from content (and elements with `action` and no role) take the visible text below them | 1.1.1, 2.4.4, 2.5.3, 3.2.4, 3.3.2, 4.1.2 |
| `aria-describedby`, `aria-errormessage`, `aria-controls` | node keys | 3.3.1, 4.1.2 |
| `aria-hidden`, `aria-disabled` | `true` on a node hides it from assistive technology, or makes it inactive, with its subtree | every rule |
| `aria-checked`, `aria-pressed`, `aria-expanded`, `aria-selected`, `aria-required`, `aria-invalid`, `aria-modal`, `aria-level`, `aria-valuenow`, `aria-valuemin`, `aria-valuemax`, `aria-live`, `aria-keyshortcuts` | WAI-ARIA values (integers for the value range and level) | 2.1.1, 2.1.2, 2.1.4, 3.3.1, 4.1.2, 4.1.3 |
| `tabindex` | integer; 0 or more is focusable, positive values first in ascending order, then 0 in tree order (`focus_order`) | 2.1.1, 2.4.3, 2.4.7, 3.2.1 |
| `lang`, `title` | BCP 47 tag; the root's title | 2.4.2, 3.1.1, 3.1.2 |
| `autocomplete`, `value`, `href` | an input purpose of WCAG 2.2 section 7; the current value; `#K` for the node with key K | 1.3.5, 2.4.1, 3.3.7, 3.3.8 |
| `fg`, `bg`, `border`, `bg-image`, `graphic` | colours, inherited from the nearest ancestor for `fg` and `bg`; `bg-image` makes text contrast undecided; `graphic` marks a graphic needed to understand content | 1.4.3, 1.4.11, 2.3.1 |
| `font-px`, `font-weight` | integers, inherited; large text is 24 px, or 19 px at weight 700 (18 pt and 14 pt bold at 96 px per inch, rounded as axe-core rounds) | 1.4.3 |
| `focus-fg`, `focus-bg`, `focus-border`, `focus-outline` | the style when focused | 2.4.7 |
| `w`, `h`, `inline`, `layer` | declared minimum size in CSS px; a target inside a sentence; `layer` above 0 draws a node above the flow | 1.4.10, 2.3.1, 2.4.11, 2.5.8 |
| `orientation`, `user-scale`, `max-scale`, `user-spacing` | root: `any`, `portrait` or `landscape`; `no` refuses zoom; percent; `no` refuses user text spacing | 1.3.4, 1.4.4, 1.4.12 |
| `action`, `focused`, `context` | the event a node raises when activated (a node with `action` is interactive); the node with focus; the root's context id (a change of it is a change of context) | 2.1.1, 3.2.x, 4.1.3 and others |
| `dismiss-key`, `dismisses`, `captures-tab` | how focus leaves a modal or Tab-capturing node | 2.1.2 |
| `shortcut-scope`, `shortcut-off-action`, `shortcut-remap-action` | `focus` limits a shortcut to the focused node; root actions that turn shortcuts off or remap them | 2.1.4 |
| `gesture`, `drag`, `alt-action`, `activate-on`, `undo-action`, `motion-action`, `motion-off-action` | `multipoint` or `path`; `true`; a single-pointer alternative; `up` (default) or `down`; an undo; a motion trigger; the root action that turns motion off | 2.5.1, 2.5.2, 2.5.4, 2.5.7 |
| `popup` | `hover` or `focus`: content shown on hover or focus | 1.4.13 |
| `time-limit-ms`, `time-left-ms`, `time-off-action`, `time-adjust-action`, `time-adjust-max-ms`, `time-extend-action`, `time-extend-max`, `time-warning` | root time limit and its mechanisms; a node that shows the warning | 2.2.1 |
| `moving`, `duration-ms`, `pause-action`, `volume-action` | moving content and its length; the action that pauses it (or that controls volume) | 1.4.2, 2.2.2 |
| `media`, `media-live`, `autoplay`, `captions`, `audio-description`, `transcript`, `audio-track` | `audio`, `video` or `audio-video`; live; plays by itself; the declared alternatives | 1.2.1 to 1.2.5, 1.4.2 |
| `status-msg` | `true` marks a status message; `false` declares a changing text not to be one | 4.1.3 |
| `help`, `entry`, `entry-offer` | the kind of a help mechanism; the id of the information an input asks for (or `none`); a node offering it | 3.2.6, 3.3.7 |
| `commits`, `confirm-action`, `review-action`, `auth`, `paste` | `legal`, `financial` or `data`; the mechanisms; `password`, `cognitive`, `object` or `personal`; `false` refuses paste | 3.3.4, 3.3.8 |
| `exempt`, `attest-ID` | claimed exceptions; attestations (above) | every SC |

**Mapping to platform interfaces** (not in code; from Core-AAM, section
5). A renderer (H4, H5) exposes each role and state through the platform
interface:

| WAI-ARIA | UIA | AT-SPI |
|---|---|---|
| `button` | Control Type Button | ROLE_PUSH_BUTTON |
| `checkbox` | CheckBox | ROLE_CHECK_BOX |
| `radio` | RadioButton (Toggle, SelectionItem) | ROLE_RADIO_BUTTON |
| `switch` | Button, localized "toggleswitch" (Toggle) | ROLE_TOGGLE_BUTTON, xml-roles:switch |
| `textbox` | Edit | ROLE_ENTRY, STATE_SINGLE_LINE |
| `searchbox` | Edit, localized "search box" | ROLE_ENTRY, xml-roles:searchbox |
| `combobox` | ComboBox | ROLE_COMBO_BOX |
| `slider` | Slider (RangeValue) | ROLE_SLIDER, Value |
| `spinbutton` | Spinner (RangeValue) | ROLE_SPIN_BUTTON, Value |
| `link` | HyperLink (Value) | ROLE_LINK |
| `tab`, `tablist`, `tabpanel` | TabItem, Tab, Pane | ROLE_PAGE_TAB, ROLE_PAGE_TAB_LIST, ROLE_SCROLL_PANE |
| `menu`, `menubar`, `menuitem` | Menu, MenuBar, MenuItem | ROLE_MENU, ROLE_MENU_BAR, ROLE_MENU_ITEM |
| `menuitemcheckbox`, `menuitemradio` | MenuItem (Toggle) | ROLE_CHECK_MENU_ITEM, ROLE_RADIO_MENU_ITEM |
| `listbox`, `option` | List (Selection), ListItem | ROLE_LIST_BOX, ROLE_LIST_ITEM |
| `list`, `listitem` | List, ListItem | ROLE_LIST, ROLE_LIST_ITEM |
| `heading` | Text, localized "heading" | ROLE_HEADING |
| `img` | Image | ROLE_IMAGE |
| `dialog`, `alertdialog` | Pane | ROLE_DIALOG, ROLE_ALERT |
| `alert` | Group, LiveSetting Assertive | ROLE_NOTIFICATION |
| `status`, `log` | Group, LiveSetting Polite | ROLE_STATUS_BAR, ROLE_LOG (container-live:polite) |
| `timer`, `progressbar` | Group "timer", ProgressBar | ROLE_TIMER, ROLE_PROGRESS_BAR |
| `table`, `grid`, `row`, `cell`, `gridcell`, `columnheader`, `rowheader` | Table, DataGrid, DataItem, DataItem, DataItem, DataItem, HeaderItem | ROLE_TABLE, ROLE_TABLE (xml-roles:grid), ROLE_TABLE_ROW, ROLE_TABLE_CELL, ROLE_TABLE_CELL, ROLE_COLUMN_HEADER, ROLE_ROW_HEADER |
| `tree`, `treeitem` | Tree, TreeItem | ROLE_TREE, ROLE_TREE_ITEM |
| `navigation`, `main`, `region` | Group, Landmark Type Navigation, Main, Custom | ROLE_LANDMARK (xml-roles:navigation, main, region) |
| `group`, `toolbar`, `tooltip`, `separator`, `scrollbar` | Group, ToolBar, ToolTip, Separator, ScrollBar | ROLE_PANEL, ROLE_TOOL_BAR, ROLE_TOOL_TIP, ROLE_SEPARATOR, ROLE_SCROLL_BAR |
| `aria-label` / `aria-labelledby` | Name / LabeledBy | accessible-name |
| `aria-describedby` | DescribedBy | accessible-description |
| `aria-checked`, `aria-pressed` | ToggleState (Toggle pattern) | checked, pressed |
| `aria-expanded` | ExpandCollapseState | expanded |
| `aria-disabled` | IsEnabled | enabled, sensitive |
| `aria-invalid`, `aria-required` | IsDataValidForForm, IsRequiredForForm | invalid_entry, required |
| `aria-valuenow` | RangeValue Value | accessible-value |

#### Exact arithmetic

WCAG's relative luminance raises each linearised sRGB channel to the
power 2.4, so it is irrational for most colours. `ui.rules` holds the 256
linearised channel values times 10^12 in a table `LIN`, computed by the
generator `stdlib/scripts/gen/rules_gen.py` with 60-digit decimal
arithmetic and asserted within 1/2 of the exact value. A luminance times
10^16 is then within 5,000 of its exact value, and the test (L1 + 0.05)
≥ k (L2 + 0.05) is an integer comparison on both ends of that interval:
it meets, fails, or is undecided when the exact value could lie on either
side of the threshold. Expected values come from outside the module:
319 colour pairs (the IDE and playground pairs of 1.4, the gray pairs
nearest each threshold, 300 seeded pairs) with axe-core 4.14.0's
`getContrast` and the WCAG formula in 60-digit decimals. For every pair
axe-core's ratio lies in the interval `ui.rules` proves, and the 4.5 : 1
and 3 : 1 decisions agree; no pair is undecided (957 checks). axe-core
decides with `>`, WCAG with `>=`; no pair lies within 10^-9 of a
threshold, so the two never differ on these vectors.

#### How a program uses it

Through ordinary `example` and `check` lines, so a view that fails a rule
makes the program require change by W-INT-1 (6.9). No admission rule is
added (Q3 stays open). This program was checked with `rce test` during
this change (2 examples, 2 checks); with `tabindex` removed from the
button, both checks are false and the program requires change (2.1.1,
and 3.2.1 since no step can focus it):

```
module door
use ui.view
use ui.rules

intent "a door panel: one button opens and closes the door, and its views meet WCAG 2.2 AA"
    example Open -> true
    example Open, Close -> false
    check frames_ok(false)
    check frames_ok(true)

state Door
    opened: bool = false

event Open
event Close

step Door on Open
    opened = true
step Door on Close
    opened = false

# the parts no rule decides, attested by the person who reviewed them
fn attest(v: &mut View, ids: text)
    for i in ids.words()
        view.set_attr(v, 0, "attest-" + i, "A. Reviewer;2026-10-31")

# the view of a state: a heading and one button that opens or closes the door
fn panel(is_open: bool, focused: bool) -> View
    var v = view.view_new("div", "")
    for kv in [["lang", "en"], ["title", "Door"], ["fg", "#000000"], ["bg", "#ffffff"], ["font-px", "16"]]
        view.set_attr(&mut v, 0, copy(kv[0]), copy(kv[1]))
    attest(&mut v, "1.3.1 1.3.2 1.3.3 1.4.1 1.4.4 1.4.10 1.4.11 1.4.12 2.4.2 2.4.5 2.4.6 3.1.1 3.1.2 3.3.4")
    let m = view.add_el(&mut v, 0, "main", "")
    view.set_attr(&mut v, m, "role", "main")
    let h = view.add_el(&mut v, m, "h1", "")
    view.set_attr(&mut v, h, "role", "heading")
    view.set_attr(&mut v, h, "aria-level", "1")
    let _ = view.add_text(&mut v, h, "Door")
    let b = view.add_el(&mut v, m, "button", "toggle")
    for kv in [["role", "button"], ["tabindex", "0"], ["w", "44"], ["h", "44"], ["border", "#000000"], ["focus-outline", "#005fcc"]]
        view.set_attr(&mut v, b, copy(kv[0]), copy(kv[1]))
    view.set_attr(&mut v, b, "action", if is_open then "Close" else "Open")
    if focused
        view.set_attr(&mut v, b, "focused", "true")
    let _ = view.add_text(&mut v, b, if is_open then "Close" else "Open")
    return v

# the view before and after the button receives focus has no unmet criterion
fn frames_ok(is_open: bool) -> bool
    let t = [rules.Step(ev: "start", ms: 0, v: panel(is_open, false)), rules.Step(ev: "focus toggle", ms: 16, v: panel(is_open, true))]
    return rules.unmet(&t).len() == 0
```

Of the 55 criteria, this panel's trace meets 13 by rule alone (1.3.4,
1.4.3, 2.1.1, 2.1.2, 2.3.1, 2.4.1, 2.4.3, 2.4.7, 2.4.11, 2.5.2, 2.5.8,
3.2.1, 4.1.2), 28 do not apply, and 14 are attested (after their exact
parts meet); the same panel is a fixture of
`stdlib/tests/ui/rules_checks.rce`. Applying the rules to every state
reachable within a fuel bound, rather than to the states examples reach,
is H3.

#### Work bound

Facts every rule reads (role, hidden, inactive, interactive, focusable)
are computed once per frame, and nodes are matched across frames through a
map of their key paths, so a report grows with the number of nodes. The
report over three frames of an 801-node view runs within one file's
comptime fuel (`stdlib/tests/ui/rules_checks_fuel.rce`); no limit was
raised.

#### Not in this slice

- Layout geometry for 1.4.4, 1.4.10, 1.4.12, 2.4.11 and the 2.5.8 spacing
  exception (above): attested until a renderer reports positions (H1,
  H4).
- The red flash of 2.3.1: exact (a transition to or from a state with
  R/(R+G+B) of 0.8 or more and a CIE 1976 u'v' distance over 0.2) but not
  written yet; a node whose colour changes involving a saturated red
  reach 8 transitions in one second is undecided. The general flash is
  computed from declared backgrounds; flashing inside media and images is
  the human part.
- Fractional ARIA values (`aria-valuenow="0.5"`) are undecided.
- AAA criteria (for example 2.4.13 focus appearance) are not covered; each
  can be added as a rule.
- The IDE and playground fixes of 1.4, host changes (a frame sink,
  replay) and any renderer.

### 4.3 Later items

- **H1 Frame sink and replay of interaction.** A program emits
  `Frame(view)` as it emits `Display(data)` today. The host passes it to
  a renderer adapter if one is granted; nothing is appended to D for it,
  because a frame is a function of the state and so of D. `rce host
  replay` gains a mode word that prints, per seq, the input, the frame
  digest and the `ui.rules` findings. One new host rule after HC-28.
  Touches `host.rs`, `cli.rs` and Appendix E. *Small addition.*
  *Implemented* (Appendix E, HC-29): a module declares `struct Frame` with
  `view: View` (and, if it chooses, `ev: text` and `ms: i64`, the
  `ui.rules` step word and time, which the host otherwise derives from
  D); the grant `frames FILE` of `rce host run` and `serve` appends one
  line per frame (seq, both digests, the tree and the `view.diff`
  operations from the previous frame); `rce host replay frames FILE D`
  prints the same lines from D, each with its `ui.rules` findings, and the
  report of the session; `rce host serve presented FILE` reads a
  renderer's confirmations into D (Q4). The digests are computed by the
  host and checked against `ui.view` itself; the operations and the rules
  are `ui.view` and `ui.rules`, run by the interpreter. `ui.rules` gained
  `findings` (every rule over a trace), and its per-frame struct `Frame`
  became `Facts`, so a module that declares the output `Frame` can still
  use `ui.rules` (W-MOD-3).
- **H2 Log rules and alarms.** The WCAG rules over frame traces are part
  of 4.2 (the ten L criteria). Left: rules over the event log D itself
  once H1 lets replay produce frames; 2.1.2 checked over a `ui.states`
  focus chart rather than over the declared `dismiss-key`; and a
  separate module `ui.alarm` with the ISA-18.2 metrics over alarm events
  (rates per window, floods, chattering, stale, priority distribution,
  shelving with expiry). Pure library. *Small addition.*
- **H3 Rules over reachable views.** Apply a view rule to every state
  that bounded exploration (10.11) reaches, and report the first event
  list that reaches a failing view. *New work.*
- **H4 DOM reference renderer.** A small renderer from `ui.view` to the
  browser DOM, with `role`, `aria-*` and `tabindex` from the vocabulary,
  in the playground (which already runs the checker as WebAssembly). The
  browser then supplies a platform accessibility tree, and axe-core run
  over that DOM in CI is an independent cross-check of the E rules of 4.2:
  a disagreement is a defect in one of them. The renderer also supplies
  the layout geometry that 1.4.4, 1.4.10, 1.4.12, 2.4.11 and 2.5.8 need,
  so those parts become exact. *Small addition.*
  *Implemented in part* (`ui/web/`, branch `claude/dom-renderer`): a
  renderer of HC-29's sink lines, with no framework, that applies the
  `view.diff` operations in order, passes every attribute through, gives
  each keyed node its key as DOM id and turns the presentation attributes
  into inline style; `ui/web/README.md` says what it guarantees and what it
  does not (the view tree and its digest stay the reference). Its test in
  headless Chromium (`.github/workflows/ui-web.yml`) requires the DOM to
  equal each frame's tree and axe-core 4.14.0 (WCAG 2.x A and AA rules) to
  report no violation on the 13 frames of `ui/web/test/shop.rce` on which
  `ui.rules` finds nothing, and exactly `color-contrast`, `button-name`
  and `aria-hidden-focus` on the frames that fail 1.4.3, 4.1.2 (no name)
  and 4.1.2 (focusable under `aria-hidden`); for a widget role with no
  `tabindex` (2.1.1) axe-core has no rule, and the test records that. Not
  yet: layout geometry for 1.4.4, 1.4.10, 1.4.12, 2.4.11 and 2.5.8, and
  the playground integration.
- **H5 Native reference adapter.** A host adapter that draws `ui.view`
  frames in a desktop window with a native accessibility tree, feeds
  keys, pointer, focus and resize back as inputs (HC-10), and reports
  nothing into D of its own. Candidates:
  - **Raven** (<https://github.com/oscar-soirey/Raven>, Apache-2.0, C11).
    For: native screen-reader bridges (UI Automation, NSAccessibility,
    AT-SPI), high contrast, text scale, reduced motion, colour filters; a
    host-platform mode (`raven_host.h`) in which the application feeds
    every input through `Rhost_inject_*` and Raven reads no OS event
    queue; a null backend for headless tests; one frame per call
    (`Rdraw_frame`, `Rwait_frame`). Against: the clone holds one commit
    (f459f71, 2026-10-31) by one author, version 0.2.0 in
    `CMakeLists.txt`, and its README states the public API is subject to
    change; about 89,000 lines of C to audit; a render thread and a job
    system, and spring animations driven by its own clock (all outside D,
    but they make two renderings of the same frame differ in pixels);
    Vulkan only, through MoltenVK on macOS; and it is not known whether
    the screen-reader bridges work in host-platform mode.
  - **AccessKit with the IDE's winit stack** (Rust, used by egui):
    the same toolkit RCE's IDE already uses, so one accessibility bridge
    would serve both H5 and the IDE finding of 1.4.
  - **The DOM of H4**, as the reference, with a native adapter later.
  A spike outside this repository should answer the host-platform
  question for Raven before any choice. *New work.*
- **H6 RCE's own interfaces.** Fix the 1.4 findings (IDE colours,
  playground border and live region); project the IDE's photon tree into
  `ui.view` attributes in a test so `ui.rules` checks it; run axe-core on
  the playground in CI. *Small addition* (playground), *new work* (IDE
  accessibility bridge).
- **H7 Writing.** A define-before-use and label-order check for
  `spec/book` and `book/src`, like the research site's
  `scripts/lint-order.mjs`. *Small addition.*
- **H8 Book and spec.** A section in the specification for
  `rce.ui.attrs/1` and the rules, and a book chapter, once 4.2, H1 and H2
  are accepted.

### 4.4 What stays outside D

Inside D, as today (HC-10): every key, pointer, focus and resize input,
every `Tick`, every adapter result. Derived from D and not stored: every
frame and its digest, every `ui.rules` finding, every alarm metric.

Outside D, and never read by a program:

- pixels, rasterisation, glyph shaping and font fallback done by the
  renderer, GPU state;
- animation in between frames (Raven's springs, transitions), cursor
  blink, hover effects the program does not model;
- the timing of presentation (vsync, frame drops) and measured latency;
- what the platform accessibility API and the screen reader say;
- the HC-28 status records;
- meter readings, unless a later calibration rule records them;
- everything people decide: rationalisation, usability tests,
  hazard judgements, review approvals. These belong in documents that
  cite D, not in D.

If a renderer's confirmation that it presented frame digest X matters to
a program (for example, an alarm counted as annunciated only when shown),
that confirmation must enter as an input under HC-10. Q4 below is decided
that way: the confirmation enters D as `host.Presented` (HC-29); what the
renderer drew, and when, stays outside.

### 4.5 Ordering

The positional-CLI implementation (#91) and the merge batch (#85 to #89,
merged by #90) are in `main`, so nothing in flight collides with this
proposal.

1. **Done:** 4.2 (`ui.rules`), this change. Touches only `stdlib/` and
   this proposal.
2. **Next, in parallel:** the rest of H2 (`ui.alarm`). Only `stdlib/`.
3. **Next, small:** H6's fixes, as a separate change: the playground
   (`playground/style.css`, `app.js`; pull request #25 also edits the
   playground, so merge it first or coordinate) and the IDE colours in
   `ide/crates/rce-ide/src/view.rs`.
4. **Done:** H1 (frame sink, replay mode word `frames`, HC-29). It
   edits `host.rs`, `cli.rs` and Appendix E, and its words follow PC-1 to
   PC-8.
5. **After H1:** H4 (DOM renderer and axe-core in CI; the renderer and
   its CI job are built, layout geometry is not), then H3.
6. **Spike, outside the repository, any time:** the Raven host-platform
   and accessibility question for H5. The choice of H5 goes to the
   owner.
7. **Last:** H7 and H8.

### 4.6 Open questions for the owner

1. **Q1. Scope of the first face.** *Decided 2026-10-31: full WCAG 2.2
   AA* (every level A and AA criterion; the EN 301 549 V4.1.1 baseline),
   each classed E, L, P or H as in 4.2. AAA items (2.4.13) can be added
   as separate rules.
2. **Q2. Vocabulary.** *Decided 2026-10-31: WAI-ARIA role, state and
   property names*, with the mapping to UI Automation and AT-SPI as a
   table in 4.2, not in code.
3. **Q3. Where findings live.** Are the rules a library that programs
   opt into (as built), or should a module that emits `Frame` be held to
   `ui.rules` by the checker without an example line? The second is a new
   admission rule and moves the spec.
4. **Q4. Renderer confirmations.** Does a renderer's "frame X presented"
   enter D as an input (so alarm annunciation can be counted from D), or
   stay outside as HC-28's status does? *Decided 2026-11-01, under the
   owner's instruction to proceed with the later items: it enters D, as
   the host-named input `host.Presented` {digest, seq} followed by the
   bound `Presented(seq, digest)` (HC-29).* Being shown a frame is a fact
   from outside the program that a program may act on, and HC-10 gives
   such a fact no other way in; a status report changes nothing a program
   reads. Only which frame was presented enters, not how it looked.
5. **Q5. `blocked`.** HC-28 never reports `blocked` because the host asks
   no person anything. The IDE does ask (network permission, accepting a
   proposal). Should the human checkpoint of agent-written RCE become a
   named state with its own record?
6. **Q6. Time base for HMI rules.** ISA-18.2 rates are per hour and per
   10 minutes. Should `ui.alarm` count `Tick` (16 ms each, engine-chosen
   under HC-14) or require `host.WallTime` entries?
7. **Q7. Native renderer.** Raven, AccessKit, or the DOM first?
8. **Q8. Safety domains.** Does RCE intend to make claims under IEC 62366-1,
   ISO 26262 or IEC 61508? If so, the tool-confidence clauses (for
   example ISO 26262-8 clause 11) apply to `rce` itself, and that is a
   project of its own.

### 4.7 Ledger row

The row of 2026-10-31 in `LEDGER.md` records this proposal: open, with
Q1 and Q2 decided and the first slice (4.2) implemented. The row of
2026-11-01 records H1 (HC-29) and the decision on Q4.


## 5. Sources

Fetched or confirmed for this proposal (2026-10-31). `www.w3.org` and
`www.iso.org` could not be reached; the W3C texts were read from their
source repositories on GitHub at the commits below, and the ISO texts were
confirmed through search results and secondary pages only.

- W3C, Web Content Accessibility Guidelines (WCAG) 2.2, <https://www.w3.org/TR/WCAG22/>, read from <https://github.com/w3c/wcag> branch main, commit 9b94e18a0738a2b5eeca6895cc211553bf06e77a (2026-10-29): `guidelines/index.html` (the editor's draft of WCAG 2.2) and `guidelines/sc/**` for every success criterion's number, title, level and text; `guidelines/terms/20/relative-luminance.html`, `contrast-ratio.html`, `general-flash-and-red-flash-thresholds.html` and `guidelines/input-purposes.html` for the definitions 4.2 uses. The list of level A and AA criteria is `stdlib/vectors/ui/wcag22-sc.tsv`.
- W3C, WAI-ARIA 1.2, <https://www.w3.org/TR/wai-aria-1.2/>; the role table used is axe-core's (below). W3C, Core Accessibility API Mappings (Core-AAM), read from <https://github.com/w3c/aria> (`core-aam/index.html`), commit 3bee34f667e5b88945e9855d4bc8d1ab7cc37cdd (2026-10-30): the UIA and AT-SPI columns of 4.2. ACT Rules, <https://www.w3.org/WAI/standards-guidelines/act/rules/>.
- ETSI EN 301 549 V3.2.1 (2021-03), <https://www.etsi.org/deliver/etsi_en/301500_301599/301549/03.02.01_60/en_301549v030201p.pdf>: clause 9 (web, WCAG 2.1 AA), clause 11 (software; 11.5.2 accessibility services). V4.1.1 (published 2026, aligned with WCAG 2.2 AA; its citation in the Official Journal was pending at the time of the sources found).
- ISO 9241-110:2020, Interaction principles, <https://www.iso.org/standard/75258.html>: clause 5 (5.4 on discovery, exploration and retention, 5.6 use error robustness and 5.7 user engagement, confirmed from the table of contents).
- ISO 9241-11:2018, Usability: definitions and concepts, <https://www.iso.org/standard/63500.html>.
- ANSI/ISA-101.01-2015, Human Machine Interfaces for Process Automation Systems, <https://www.isa.org/standards-and-publications/isa-standards/isa-101-standards>. Clause numbers were not confirmed; the colour and hierarchy items above are as secondary sources describe them.
- ANSI/ISA-18.2-2016, Management of Alarm Systems for the Process Industries, <https://www.isa.org/products/ansi-isa-18-2-2016-management-of-alarm-systems-for>: metric targets of Table 5 as quoted by exida, <https://exida.com/Blog/evaluating-alarm-system-performance-in-a-multi-operator-control-room>. IEC 62682:2022, edition 2.0, <https://webstore.iec.ch/publication/65543>. Shelving as required functionality: exida, <https://www.exida.com/Resources/Whitepapers/the-alarm-shelving-ebook>.
- IEC 62366-1:2015+AMD1:2020, Usability engineering for medical devices, <https://webstore.iec.ch/publication/67220>: clauses 5.1, 5.4 to 5.7, 5.9 as summarised by secondary sources (Johner Institute, Medical Device HQ).
- ISO 26262-3:2018 clause 6 (controllability C0 to C3) and ISO 26262-8:2018 clause 11 (confidence in software tools), <https://www.iso.org/standard/68383.html>; IEC 61508-1:2010 clause 7.4, <https://webstore.iec.ch/publication/5515> (not fetched).
- J. Nielsen, 10 Usability Heuristics for User Interface Design, <https://www.nngroup.com/articles/ten-usability-heuristics/>.
- Deque, axe-core, <https://github.com/dequelabs/axe-core> (fetched: "on average 57% of WCAG issues", "zero false positives (bugs notwithstanding)", undecided cases as "incomplete"); study <https://www.deque.com/blog/automated-testing-study-identifies-57-percent-of-digital-accessibility-issues/>. Used as the independent reference of 4.2: the npm package 4.14.0 (`axe.commons.color.getContrast`, `axe.utils.getStandards().ariaRoles`, run in Node 22), whose source is the git repository at commit 6b52b841461bd23345970b91ec75fdaff6bad31a (`lib/commons/color/color.js`, `get-contrast.js`, `lib/checks/color/color-contrast-evaluate.js`).
- Prior art links as given in section 3. Qt Safe Renderer certificate fetched through search: TÜV NORD SEBS-A.113256/17 V3.0, 2023-02-01 to 2028-02-01.
- Raven, <https://github.com/oscar-soirey/Raven>, read from a local clone: `README.md`, `include/raven/raven_accessibility.h`, `raven_host.h`, `raven_core.h`, `CMakeLists.txt`.
