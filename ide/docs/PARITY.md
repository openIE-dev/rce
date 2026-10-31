# Parity matrix (updated 2026-10-12, second pass)

Sources: docs/LANDSCAPE-VIEWS.md (see its "Second reading pass" for what was read in full and what is still search-only).

Any joules shown on this box are synthetic: the box has no hardware energy meters, and the static cost table uses placeholder prices until `telemetry calibrate` fits it on a real meter. Screens label them SYNTHETIC or demo.

Status words: Done = implemented and tested here. Library = implemented and tested, not yet connected to every surface. Scaffold = shape and math tested, real data source missing. Missing = not started.

| Feature | Who has it | Our status | Commit | Remaining |
|---|---|---|---|---|
| Split a shared device's energy among processes | Ampere | Done (energy conserved, idle kept apart) | 5c6e461 | Feed per-process busy time from the live telemetry loop |
| Perfetto trace export | CodeGreen | Done: `rce-ide telemetry export perfetto out OUT.json FILE...`, `synthetic_energy` in metadata | 5c6e461, 9449b4f | Export measured receipts too, not only static E_sim |
| Energy budget gate in tests / CI | CodeGreen | Done: `rce-ide test energy-budget J [measured J] [strict] FILE...`; no measurement is its own verdict | 9449b4f | Also in the shipped `rce` binary: `rce test [energy-budget J] [measured J | measure] [strict] FILE...`; IDE test panel button not added |
| Per-function energy attribution | CodeGreen, Oaklean, JoularJX | Partial: static per-function E_sim; measured attribution by time window | 6a5e18f | Evaluator spans to map measured windows to functions |
| GPU kernel launch attribution | CodeGreen, Nsight | Scaffold: attribution math tested with synthetic samples | c016591 | No launch source (CUPTI, Metal, ROCm); never run on a GPU |
| Static energy model feeding E_sim | SPEAR-Viewer | Done: per-construct op counts over the Form AST, loop trips as a stated assumption, callee roll-up without recursion | 6a5e18f | Calibrated prices; per-line overlay in the editor |
| E_sim vs E_meas divergence per node | none found | Done: paired actions, log2 ratio, diverging colour, labels | 8ff9e7a, 00a3a76 | Real receipts on a metered host |
| hwloc / lstopo XML import | hwloc | Done: `RCE_HWLOC_XML=topo.xml`, tested on hwloc's own test topology | a13556f | Distances matrix and memory attributes not imported |
| Semantic zoom | ExplorViz, NicoV7 | Done: definitions fold into files when the camera is far | 0886c55, 00a3a76 | Clicking a folded file does not expand it in place; picking still uses the unfolded graph |
| Mini-map | ExplorViz (claimed in its 2025 paper, unread) | Done | 0886c55, 00a3a76 | Click to move the camera |
| User-mapped metric to size / colour / glow | CodeCharta, CodeCity | Done: rank, joules, measured, simulated, divergence, actions, watts, degree | 0886c55, 00a3a76 | UI picker (today set by `g3.set_style` JSON) |
| Graph and text as equal editable views | Enso, Luna | Library: rename, delete, add stub, move become text edits that must pass rce check | 29090f2 | Graph panel gestures that call it; edges as edits (add a call) |
| Review marks expiring on change, tied to receipts | NicoV7 codebase-visualizer | Library: marks pin a BLAKE3 item hash, written as signed receipts, expire on change | c722111 | Panel badge and a "% reviewed" header |
| Git history replay | Code-XR, code-nexus, Gource | Library: one Form graph per commit into the existing scrubber | 3f08ce2 | IDE flag to load it instead of the receipt timeline |
| Code property graph queries | Joern, CodeQL | Done (small): callers, callees, reach, path, kind with energy filters, change (files that require change), uses-hw | e81f837 | No AST/CFG/data-flow layers like Joern; no query language beyond these forms |
| SCIP export | SCIP | Done: definitions and call references, hand-encoded protobuf | e81f837 | Uses the deprecated `range` field; not yet validated with the `scip` CLI |
| MCP graph tools for agents | Graphify, GitNexus, codebase-memory-mcp | Done: `graph_query`, `graph_scip`, each writes a receipt | b2156f3 | Architecture summary and diff-overlay tools |
| Live while incomplete | Hazel | Partial: item-level holes keep the rest live (graph, E_sim, check) | 26747a9 | Expression-level typed holes; dependents of a hole are holed too |
| 3D graph over time with hardware layer | ours | Done before this pass | bb0b584 | |

Tests: `cargo test --workspace --offline` passes (54 test binaries reporting ok). New tests cover apportioning, budgets, Perfetto, cost model, test gate and export, divergence, lens, queries, SCIP, MCP graph tools, hwloc import, graph edits, review marks, git replay, holes, and GPU launch attribution. No cc or cmake build dependency in the tree.

Screenshots (browser WebGPU, headless Chrome, llvmpipe Vulkan, synthetic energy labelled): screens/web-divergence.png, screens/web-mapping.png, screens/web-zoom-files.png.
