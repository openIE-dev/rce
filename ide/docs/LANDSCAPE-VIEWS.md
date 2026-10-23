# Landscape: code and program views (searched 2026-10-12)

Scope: who has built views of code as a graph, city, database, timeline or hardware map, and what we can learn from each. The reference is our RCE IDE graph view: a code graph built from a formal program model (P = (S, E, delta, D, lambda)), a time scrubber, an overlay of measured vs simulated joules (E_meas vs E_sim), a discovered hardware layer with edges from code to hardware, and a linked editor.

How sources were read: pages marked [read] were fetched in full. Pages marked [search] are cited from search-result extracts only and should be opened before anyone quotes them. Searches covered English, Chinese, Japanese, German and Russian queries. The non-English results mostly pointed back to the same tools (CodeCity, Getaviz, Graphify, GitNexus, CodeAtlas). Status is as the sources described it on the search date.

## 1. Code cities and 3D software visualization

| Who / what | Year | Status | Does well | Missing next to our view |
|---|---|---|---|---|
| CodeCity, Richard Wettel and Michele Lanza (USI Lugano). https://www.inf.usi.ch/lanza/PUBS/P/Wett2008a.pdf [read: buildings, districts, metric mapping confirmed], ICSE 2011 experiment https://si.usi.ch/assets/publications/conf/icse/icse2011/WettelLR11.pdf [search: the 24.26% and 12.01% figures are from this unread paper] | 2007 to 2011 | Research, no longer developed | Started the city metaphor: classes are buildings, packages are districts, and metrics set height, footprint and color. In a controlled experiment with 41 people, correctness went up 24.26% and time went down 12.01% compared with Eclipse plus Excel | Static metrics only. No runtime data, no energy, no hardware, no editor link |
| CodeCharta (MaibornWolff, open source). https://codecharta.com/ , https://codecharta.com/docs/visualization/user-controls/map/ [read: map page] | 2017 to now | Active | Browser 3D treemap that runs locally. Users pick which metric drives area, height and color. Edges show dependencies or co-change. Five layouts, saved scenarios, and STL/3MF export for 3D printing | Built from metrics files, not a program model. No time scrubber in the map itself, no energy, no hardware |
| ExplorViz (Kiel University). https://explorviz.dev/1-about/ [read: static plus OpenTelemetry traces, 3D city, WebXR, controlled experiments confirmed; semantic zoom and mini-maps are not on this page], VISSOFT 2025 paper https://zenodo.org/records/16965816 [search: record page returned no text] | 2012 to now | Active research | Combines static analysis with live OpenTelemetry traces in a 3D city. Collaborative WebXR/VR. Semantic zoom and mini-maps (2025). Backed by controlled experiments | Traces carry time but not energy. No hardware layer |
| Getaviz (Leipzig). https://www.researchgate.net/publication/320083290 [search] | 2017 or so | Research | Model-driven generator for structural, behavioral and evolutionary views in 2D, 3D and VR. Built for controlled comparisons between metaphors | No energy or hardware. The model-driven approach is the closest relative of building views from a formal model |
| Code-XR, Adrian Montes Linares. https://code-xr.adrianmonteslinares.com/ [read] | 2025 | Active | Live 3D city inside VS Code, XR, and commit-by-commit replay of git history | Time comes from git, not from execution. No energy |
| SecCityVR. https://arxiv.org/html/2504.18238 [read] | 2025 | Research | VR city with call-graph edges and vulnerability overlays, explored collaboratively | Security overlay only |
| Gource, Andrew Caudwell. https://gource.io/ [read] | 2009 to now | Maintained | Animated tree of repository history with contributors and a time axis. Exports video | History of files only. No semantics or runtime |

What we can learn: the city metaphor has the strongest user-study evidence. Letting users choose which channel shows which metric (CodeCharta) and semantic zoom (ExplorViz) are proven ways to keep large graphs readable. ExplorViz shows that runtime traces fit into a spatial view.

## 2. Node and graph programming environments (the graph is the program)

| Who / what | Year | Status | Does well | Missing next to our view |
|---|---|---|---|---|
| Enso (formerly Luna). https://github.com/enso-org/enso/blob/develop/docs/enso-philosophy.md [read: philosophy intro] | 2015 to now | Active, 2025 releases | Graph and text are two equal views of one program and stay in sync. Shows evaluation progress on nodes. Parser written in Rust | Data workflows only. No energy or hardware |
| Luna (the language). https://docs.luna-lang.org/luna-user-guide/visual_representation/visual_representation [read] | 2016 to 2019 | Became Enso | Visual and text editing kept in sync, with typed ports | Same as Enso |
| Unit, Samuel Timbo. https://github.com/samuelmtimbo/unit/ [read: README] | 2019 to now | Active | Programs are graphs of MIMO finite-state units. Live and reactive | No telemetry |
| Unreal Blueprints, Houdini, LabVIEW, Max/MSP, Node-RED | 1986 to now | Active products | Node graphs are mainstream in games, VFX, instruments, audio and IoT. LabVIEW in particular ties graphs to hardware I/O | Not verified in this pass. None is known to show measured energy per node. Listed here from general knowledge and should be checked before citing |
| Hazel (Cyrus Omar et al.). https://hazel.org/ [read], POPL 2019 https://dl.acm.org/doi/10.1145/3290327 [search] | 2017 to now | Active research | Typed holes keep incomplete programs running, so feedback never stops while editing | Text and structure editor, no spatial graph |
| Glamorous Toolkit / moldable development (feenk). https://gtoolkit.com/ [read], https://book.gtoolkit.com/moldable-development-patterns-vuflnrgp5r5o4m1szatoo4e2 [search] | 2018 to now | Active | Many small custom views for each object and question, built on Smalltalk liveness | Views are made by hand. No energy or hardware meters |
| Dynamicland / Realtalk, Bret Victor. https://dynamicland.org/2024/Intro/ [read] | 2017 to now | Active | Spatial, physical, communal programming where programs react to what is near them | Not about measurement or hardware |
| Code Bubbles, Steven Reiss (Brown). https://cs.brown.edu/people/spr/codebubbles/about.html [read] | 2010 to now | Maintained | Working-set editor made of function-level bubbles on a 2D canvas | No runtime overlay |
| Light Table. https://github.com/lighttable/lighttable [read: README; it does not say archived, so the 2022 end date and Archived status are unverified] | 2012 to 2022 | Archived | Inline evaluation results next to code | Dead |
| Eve. https://groups.google.com/g/eve-talk/c/YFguOGkNrBo/m/EozaCfheAQAJ [search] | 2014 to 2018 | Wound down | Relational, database-style programming | Dead |
| Darklang. https://blog.darklang.com/winding-down-darklang-classic/ [read: confirms Classic winding down; the post does not mention live traces] | 2019 to now | Classic version in maintenance, rewrite under way | Editor, runtime and deployment as one thing, with live traces | No energy |

What we can learn: Enso's rule that graph and text are equal views of one program matches our editor-graph link. Hazel's total liveness (incomplete code still runs) would let the E_sim overlay stay live while someone is typing. Glamorous Toolkit argues that views should be cheap to build for each question, which suggests our graph needs a way to add custom lenses.

## 3. Code as a graph database

| Who / what | Year | Status | Does well | Missing |
|---|---|---|---|---|
| Joern code property graph. https://docs.joern.io/code-property-graph/ [read], https://cpg.joern.io/ [search] | 2014 to now | Active | One multigraph that merges AST, CFG and data dependence, with an extensible schema and a query language (CPGQL) | Built for security analysis. No time or energy |
| CodeQL (GitHub) | 2019 to now | Active | Code stored as a relational database and queried with Datalog | Not fetched in this pass |
| SCIP (Sourcegraph). https://github.com/scip-code/scip [read: README and scip.proto] | 2022 to now | Active | Precise index format that works across repositories | Index only, not a view |
| Kythe (Google). https://github.com/kythe/kythe/blob/master/kythe/proto/explore.proto [read] | 2014 to now | Active | Semantic graph with call graphs and type hierarchies | Index only |
| GitHub stack graphs. https://github.blog/changelog/2022-11-09-cross-repo-code-navigation-for-python-projects/ [search] | 2021 to 2025 | Repo archived 2025-09-09 per search result | Incremental name resolution | Archived |

What we can learn: a CPG layer stack (syntax, control, data) is a proven schema. Because our graph comes from P = (S, E, delta, D, lambda), it already has states and transitions, which a CPG has to recover by analysis.

## 4. Code graphs for AI agents (2024 to 2026)

| Who / what | Year | Status | Does well | Missing |
|---|---|---|---|---|
| Graphify. https://www.infoq.com/news/2026/09/graphify-codebase-exploration/ [read: community detection, tree-sitter, MCP confirmed; graph.html not named in the article] | 2026-04 to now | Active, MIT/Apache | Tree-sitter graph of code and docs with community detection, graph.html, GRAPH_REPORT.md, graph.json and MCP | 2D HTML view. No runtime, energy or hardware |
| codebase-memory-mcp (DeusData). https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/README.md [read] | 2025 to 2026 | Active | Native binary, 162 languages, 17 MCP tools, a 3D "Galaxy" view and a read-only Monaco reader that follows the graph | Static only. The reader is read-only |
| code-nexus. https://github.com/snagrecha/code-nexus [read: README] | 2025 to 2026 | Small project | 3D WebGL graph, time travel through git, blast radius, churn and ownership overlays, MCP | Time comes from git. No runtime energy |
| codegraph / codebase-visualizer (NicoV7). https://github.com/NicoV7/codebase-visualizer [read: README; confirms hash-bound "Got it" review marks that expire when code changes, a % understood header, PR diff badges, expand-in-place city, MCP tools] | 2026 | Small project | 3D city that expands in place, PR diff badges, call traces, records why the agent did something, and a heatmap of code no human has reviewed yet | No telemetry |
| GitNexus. https://github.com/AbhigyanPatwari/gitnexus [read: README]. Chinese coverage: https://www.cnblogs.com/xiaohuatongxueai/p/19660201 [search] | 2025 to 2026 | Active | Runs in the browser with no server. Tree-sitter WASM, Cypher queries, Graph RAG chat | Static only |
| CodeAtlas 3D (VS Code). https://marketplace.visualstudio.com/items?itemName=codeatlas.0nu-codeatlas [search, found through a Russian-language query] | 2025 to 2026 | Active | 3D call and import graph that jumps to source | Static only |
| Understand-Anything, Mimir, CodeGraph-Explorer (found through Chinese-language results). https://github.com/Egonex-AI/Understand-Anything/blob/main/READMEs/README.zh-CN.md [search] | 2026 | Small projects | Architecture layers and change-impact analysis | Static only |

What we can learn: this group moves fastest, and almost everyone ships MCP plus a graph. NicoV7's idea of tracking which code humans have actually understood, and expiring that mark when the code's hash changes, fits naturally with our receipts. None of these tools uses runtime or energy data.

## 5. Runtime, energy and hardware overlays on code

| Who / what | Year | Status | Does well | Missing next to our view |
|---|---|---|---|---|
| JoularJX and JoularJX Reader, Adel Noureddine. https://github.com/joular/joularjx/ [read: README], https://github.com/joular/joularjx-reader , https://github.com/joular/joularcode-java-viewer [search] | 2021 to now | Active | Power per Java method and per call-tree branch, with a timestamped history for each method. Uses RAPL, powermetrics, and regression models on Raspberry Pi | Tables and line charts, no graph view. Measured only, nothing simulated. No hardware topology |
| CodeGreen (SMART-Dal, Rajput and Sharma 2026, arXiv 2603.17924). https://smart-dal.github.io/codegreen/ [read] | 2025 to 2026 | Active, v0.4.9 | Energy per function from RAPL, IOReport plus kpc on Apple Silicon (CPU, GPU, ANE, DRAM), EMI on Windows, NVML and ROCm. GPU work is attributed to the function that launched it. A unified timeline (power, stacks, GC, GPU kernels) exports to Perfetto. `--budget` gates CI on energy. Reports 0.03% error against perf RAPL | Plotly timeline, no code graph. No simulated energy, no hardware graph, no editor link. The closest match to our measurement side, and worth comparing against |
| SPEAR and SPEAR-Viewer (sse-labs). https://github.com/sse-labs/spear-viewer [read] | 2023 or so | Side project, 0 stars | Static, LLVM-based energy prediction from a RAPL profile of the machine. In VS Code it shows energy highlights per line, a call graph with energy per function, and a colored program graph | Predicted only, never compared with measurement. Intel RAPL only. The graph opens as a PDF made with DOT. This is the closest existing thing to our E_sim |
| Oaklean (Hitabis GmbH). https://github.com/hitabisgmbh/oaklean [read] | 2023 to now | Active, MIT | Energy profiler for Node.js. Highlights energy per source line in VS Code. Jest integration tracks energy across releases | Measured only, no graph view, no hardware |
| Ampere (Adam McDaniel). https://github.com/adam-mcdaniel/ampere [read] | 2025 to 2026 | Research | Attributes energy from OTF2 exascale traces to functions, using a topology resolver that splits a shared GPU's energy between the MPI ranks using it. Flamegraphs, rank-by-function heatmaps, power lines next to function execution | Topology is a function you write by hand, not discovered. No code graph from a program model, no editor, no simulated energy |
| PowerAPI / SmartWatts (Inria Spirals). https://powerapi.org/ [read] | 2013 to now | Active | Software-defined power meters (sensor plus formula) for process, container and VM | A meter, not a view |
| Arm Streamline with Energy Probe. https://support.arm.com/documentation/101816/9-9/Analyze-your-capture/Analyze-your-code/Viewing-your-code [search] | 2010s to now | Active | Correlates measured energy from a probe with PC samples down to functions, source lines and disassembly | Timeline plus code view, no graph, no simulated energy |
| Apple Instruments Power Profiler (WWDC25). https://developer.apple.com/videos/play/wwdc2025/226/ [read: transcript page] | 2025 | Active | Power-impact lanes for CPU, GPU, display and network, linked to call stacks | Impact scores rather than joules. Apple only. No graph |
| NVIDIA Nsight Systems and Compute. https://docs.nvidia.com/nsight-systems/2026.1/UserGuide/index.html , https://github.com/NVIDIA/nsight-training/blob/master/cuda/nsight_compute/python_report_interface/Source_correlated_metrics.ipynb [search] | 2018 to now | Active | Timeline lining up CPU ranges (NVTX) with GPU activity. Compute maps metrics to SASS, PTX and source lines | NVIDIA only. Timeline, not a graph |
| hwloc lstopo (Open MPI project). https://hwloc.readthedocs.io/en/latest/doxygen/html/tools.html [read]; XML format checked against hwloc's own test topology | 2009 to now | Active | Discovers and draws the topology: packages, NUMA, caches, cores, PCI. Exports XML | Hardware only. Code is not shown |
| Intel VTune, Linux perf, flame graphs (Brendan Gregg) | 2000s to now | Active | Standard ways to attribute samples to code | Not fetched in this pass. VTune has had energy analysis on some platforms, which should be verified before citing |

## 6. Does anyone combine energy, hardware, code graph and time?

Not in what this search found. Here is what each tool combines:

- CodeGreen: measured energy + time + functions (timeline), across many hardware vendors. It has no graph, no hardware layer and no simulated energy.
- Ampere: measured energy + topology (written by hand) + call graph (flamegraph) + time. It is the closest to all four, but it is a Python notebook library for HPC traces with no program model and no editor.
- SPEAR-Viewer: simulated energy + call graph + editor. It has no measurement next to the prediction, no time and no hardware graph.
- ExplorViz: 3D graph + runtime traces + time. It has no energy.
- Nsight, Streamline, Instruments: hardware + time + source lines, each tied to one vendor. They have no code graph.
- lstopo: discovered hardware only.

No source we found shows measured and simulated joules side by side on one code graph, linked to a discovered hardware graph, with a time scrubber and a linked editor. This is a negative search result, not proof that nothing like it exists. Commercial internal tools and closed HPC portals may do parts of it.

## 7. Things to adopt

1. Have CodeGreen's measurement approach on Apple Silicon (IOReport plus kpc) and its GPU-launch attribution checked against our ide-energy meters. Export our timeline to Perfetto the way it does.
2. Follow Ampere's rule that shared-device energy is split between its users rather than counted for each of them, and apply it in our attribution from code to hardware edges.
3. Learn from SPEAR's static energy model (loop bounds, worst-case strategy, calibration profile) when building E_sim, and show where E_sim and E_meas disagree, which SPEAR never does.
4. Read lstopo XML as one discovery input for the hardware layer.
5. Borrow semantic zoom and mini-maps (ExplorViz) and user-chosen metric channels (CodeCharta).
6. Keep graph and text equal, as Enso does. Keep the graph live while code is incomplete, as Hazel does.
7. Use MCP export of the graph, as the agent-era tools all do. Consider a mark for code humans have understood that expires on hash change (NicoV7), tied to receipts.

## Gaps in this survey

Unreal Blueprints, Houdini, LabVIEW, Max/MSP, Node-RED, CodeQL, VTune, perf and flame graphs were not fetched in this pass, and their table rows rely on general knowledge. Most rows are cited from search extracts ([search]). Only CodeGreen, SPEAR-Viewer, Oaklean and Ampere were read in full.


## Second reading pass (2026-10-12)

Pages fetched in full in this pass are now tagged [read]; where a page did not support a claim in its row, the tag says so and the claim stays unverified. Still [search] only: the ICSE 2011 CodeCity experiment, the ExplorViz VISSOFT 2025 paper (record page returned no text), Getaviz (ResearchGate blocks fetches), Eve, cpg.joern.io, Glamorous Toolkit book, GitHub stack graphs, the Chinese-language GitNexus and Understand-Anything pages, the JoularJX reader and viewer repos, Arm Streamline, the Nsight docs, and the Hazel POPL 2019 paper. Unreal Blueprints, Houdini, LabVIEW, Max/MSP, Node-RED, CodeQL, VTune, perf and flame graphs are still general knowledge only.
