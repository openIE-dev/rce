# Proposal: the library covers all code

Status: open. Source: the coverage survey of 24 and 25 October 2026 (seven surveys merged into one matrix) and a completeness critique of that matrix, read against 4.1, 4.2, 5.9, 5.10, 5.14, 5.15, 6.3 N9, 6.7, C.6.4, C.15, F.2, F.8, 9.8 and Appendix E, and against the proposals `one-library-every-fabric.md` (rules XF-1 to XF-14), `one-commit-change.md` (rules CC-1 to CC-6), `closing-the-loop.md` and `intent-holes.md`. The matrix is committed as `spec/data/coverage-map.json`, and `spec/scripts/coverage.py` prints its totals.

## The premise

The Form P = (S, E, δ, D, λ) is the shape of all code: the state a program keeps, the events it answers, its step, its log and its outputs. A program in any language has these parts whether or not its text names them (XF premise; section 3.6 of the RCE paper, `research-openie-web/src/content/papers/rce.md`). The Form is rigid at each commit and grows across commits: when code needs a form the shape does not yet have, the shape gains it in the text, and the change moves together with every program, example and library module of the repository in one commit (CONTRIBUTING rule 3).

This proposal states what that premise requires of the library. Three bodies of RCE text hold code for programs to reuse:

- the **standard library**, the modules of `stdlib/lib` at the commit, built into every tool and pinned by hash (XF-1, 4.1, F.5.3);
- the **example Forms**, the programs of `programs/`, `apps/` and `rcedb/`, each of which shows a shape in use;
- the **Form templates**, modules that fix the shape of a common kind of program and leave named places, called holes, for the code that differs between programs (rules LC-12 to LC-14 below).

Together these hold every known pattern, paradigm and piece of commonly used code, of the present and of the future, without restriction, because the world is not perfect. The phrase "without restriction" has three consequences. First, the library accepts the inputs real code meets: malformed, partial and larger than one input can hold. A function returns a typed error or a partial result for ordinary bad input, and where a size or fuel limit binds it offers a form that resumes over chunks. Second, no capability is placed out of scope. A pattern that the three bodies do not yet hold is recorded with the status *missing* and a planned place, and a pattern that needs the shape itself to grow is recorded as growth of the Form, with the specification text the growth amends. Third, the record is open: a capability that no record names gains one when a program needs it, which is how code of the future enters.

## Terms

The rules below use these terms.

- A **capability** is one thing that code in use needs, such as "find a substring", "parse CSV" or "a circuit breaker".
- A **row** is the record of one capability in the matrix. A row that groups closely related capabilities (for example the CRC variants) lists its members in its name.
- A row's **status** is *have* (RCE holds the capability), *partial* (RCE holds part of it, or holds it only through a form the row names) or *missing*. There is no fourth status.
- A row's **home** is where the capability belongs: a **stdlib module**; a **Form template**; an **example Form**; a **host adapter**, the host component that performs a request a program emits and logs its result (HC-3 of Appendix E); **language growth**, a change to the Form itself; or **tooling**, a command of `rce` that reads or writes programs without changing their meaning.
- **Prevalence** is a count of how widely code in use relies on a capability, defined in the method below.
- **Priority** is 1, 2 or 3, and a **wave** is one of nine ordered groups of work, both defined below.
- A **reference** is an implementation or a published test suite outside RCE that a module's results are compared with, and a **vector** is one input and its expected output taken from a reference.
- A **brick** is a library function as the brick catalog of `rce index` records it (`closing-the-loop.md` 5.2).

## The method and the totals

**Inputs.** Seven surveys were made on 24 and 25 October 2026:

1. *stdlibs*: the standard libraries of seven languages (from the Go package list, the Node.js core module list, the Rust `std` module list, `java.base`, the CPython 3.13 library tree, the .NET runtime libraries and Swift Foundation).
2. *packages*: the most-used packages of five registries (crates.io top 500, PyPI top 1,000, npm top 2,000 by downloads or top 1,000 by dependents, Go modules with 1,000 or more importers in a sample of 399, and Maven).
3. *patterns*: 353 rows from the pattern catalogs of Gamma et al. (1994), Buschmann et al. (1996), Hohpe and Woolf (2003), Fowler (2002), Joshi (2023), the Azure cloud design patterns, microservices.io and the resilience libraries.
4. *paradigms*: 149 rows of paradigms and idioms over twelve languages.
5. *algorithms*: about 240 rows from CLRS (Cormen et al. 2022), KACTL, the Rosetta Code task list and the algorithm modules of the standard libraries.
6. *formats*: 239 rows of data formats, encodings, compression, archives, checksums, protocols and media over eight ecosystems.
7. *rce-inventory*: about 380 rows of what the repository holds, read from the library, the specification, the host code and the programs.

Several catalog pages could not be fetched through the network proxy, so some counts come from the surveyors' knowledge; each survey records which.

**Merging.** The seven surveys were merged into a working table of 505 rows (80 have, 186 partial, 239 missing). Rows naming the same capability were merged under the most specific name, and closely related rows were grouped, giving the 292 rows of the survey matrix (41 have, 125 partial, 126 missing).

**Status.** Where the inventory survey gave a status, it was kept. Doubtful statuses were rechecked by running the reference checker on probe programs and by searching `stdlib/lib`. For example, the methods `replace` on text, `sort_by` on lists and `div_floor` on `i64` are W-NAME-1, `<` on a struct is W-TRAIT-3, and the slice `xs[1:3]` works.

**Prevalence** is L + P + C. L is the largest language count any language-based survey gives the row (a maximum, since those surveys draw on overlapping languages). P is the number of registries in which a widely used package provides the capability. C is the number of pattern or algorithm catalogs that name it. A grouped row takes the highest prevalence among its members.

**Priority 1** is a missing or partial row that either has prevalence 8 or more and is everyday code (strings, sorting, regular expressions, JSON, CSV, dates and times, compression, the HTTP server shape, UUIDs, logging, test helpers, collections, caches), or is a prerequisite of several priority-1 rows, or is the template mechanism with its first two templates. **Priority 2** is a row with prevalence 4 or more, or a dependency of a priority-2 row. **Priority 3** is the rest, including *have* rows with nothing left to build.

**The critique.** A completeness critique then read the matrix against the specification, the checker and the library. It found whole domains missing, because L + P + C counts general-purpose sources and so scores domain code near zero by construction; about eighteen statuses to correct, several of them where the language already does what the library comments said it could not; priority changes; and plan steps that conflict with the specification or with the size and fuel limits. The proposal applies the critique in full. The ledger therefore holds the 292 survey rows, less the one geospatial row that the critique's four geospatial rows replace, plus 72 rows the critique adds.

**Totals.** These figures, and the row counts of the wave table below, are those `spec/scripts/coverage.py` printed from the ledger when this proposal was written (commit ea37873b). A row's status changes in the commit that lands its module, so the script prints the current figures.

| | Rows | Have | Partial | Missing |
|---|---|---|---|---|
| All rows | 363 | 39 | 149 | 175 |
| from the surveys | 291 | 39 | 130 | 122 |
| added by the critique | 72 | 0 | 19 | 53 |
| Priority 1 | 43 | 2 | 23 | 18 |
| Priority 2 | 171 | 1 | 79 | 91 |
| Priority 3 | 149 | 36 | 47 | 66 |

| Home | Rows | Have | Partial | Missing |
|---|---|---|---|---|
| stdlib module | 225 | 19 | 80 | 126 |
| language growth | 47 | 8 | 23 | 16 |
| Form template | 42 | 4 | 24 | 14 |
| host adapter | 26 | 3 | 12 | 11 |
| tooling | 19 | 1 | 10 | 8 |
| example Form | 4 | 4 | 0 | 0 |

The stdlib rows name 171 target modules, of which 22 are modules of the library today that the rows extend and 149 are new, and the template rows name 41 templates. The two priority-1 *have* rows are the first two templates: the Form already expresses state machines and supervised actors, and the template that packages each shape is still to be written.

## The ledger

1. **LC-1 One ledger.** `spec/data/coverage-map.json` is the one list of what code needs and where RCE holds it. Each row has the fields `name`, `area`, `sources`, `prevalence`, `rce_status`, `rce_where` (the evidence in the repository), `home`, `target_module` (a module or template name), `priority`, `wave`, `depends_on`, `note`, `origin` (`survey` or `critique`) and `correction` (what the critique changed in a survey row). Rows are sorted by area, then name. `spec/scripts/coverage.py` prints the totals by status, priority, wave, home and area, and with `--check` it verifies the fields, the values, unique names and the order. Every figure this proposal quotes from the ledger is that script's output.
2. **LC-2 The ledger moves with the code.** A row's status changes in the commit that lands its module, template, adapter, tool or growth item, together with `stdlib/lib/MANIFEST.json` where a module changes (CONTRIBUTING rule 3). The rows of wave 1 are marked *have* as their modules land. A capability that no row names receives a row in the commit of the first program that needs it. No status, field or note places a capability out of scope.
3. **LC-3 Domain rows are scored singly.** The surveys did not keep, for each condensed row, which survey contributed which count, so the `sources` field of a survey row says so; a critique row names its own sources, and those marked "from knowledge, not fetched" carry that uncertainty. The prevalence of the 72 critique rows is `null` until a domain survey scores each row on its own, never by the highest member of a group.

## Library modules

4. **LC-4 One file, pinned and built in.** A module is one file `stdlib/lib/<area>/<name>.rce`, named `<area>.<name>`, pinned by SHA-256 in the manifest and built into `rce`, and a program brings it in with `use` (XF-1, XF-2). A module that needs others names them on one `# uses:` line. An imported name that clashes with a name of the importer or of another import is refused (W-MOD-3: imported names are not renamed), so public names carry a prefix that no other module uses, checked with `rce lib` and with a probe program that uses the module together with `net.http`, `enc` and `data.events`.
5. **LC-5 An independent reference.** Each module is checked against a reference outside RCE: CPython 3.13, Go 1.24, Node 22, or a published test suite. The generator script is in `stdlib/scripts/gen/`, its output in `stdlib/tests/<area>/<name>_checks.rce` (sharded as `_checks_2.rce` and onward when one file's comptime fuel is not enough) or in `stdlib/vectors/` with an entry in `vectors/SOURCE.md`.
6. **LC-6 Options are values.** A choice between behaviours is a value passed to a function, such as a `Dialect` for CSV or a gap-and-fold policy for local times, never a flag, directive or selected version. A module is pure: its effects are requests to host adapters (XF-4).
7. **LC-7 Ordinary bad input is an answer.** A function given malformed or partial input returns a typed error naming what was wrong and where, or a partial result with the position where reading stopped, as `net.http` already reports "incomplete" when more bytes are needed. A trap is reserved for what the specification defines as one (6.3, 6.5).
8. **LC-8 Chunks where the limits bind.** A list or text holds at most 1,000,000 elements and a state value at most 1,048,576 bytes of canonical JSON (6.3 N9); a loop runs at most LOOP_LIMIT = 1,000,000 times (6.7); and each input has the fuel of 6.7. A function of the form `inflate(bytes) -> bytes` therefore cannot decode a large file in one input, and neither can a PNG, zip or Parquet reader, an Argon2id hash with its 19 MiB of memory, or a reader of a 1 MB CSV file. Every codec over bytes is written as a state value that is fed chunks and returns what it has decoded so far, in the style of the line reader of `io` and the frame decoders of `os.ipc`, so a program spreads the work over inputs. Data larger than a state value stays in Blob storage, named by digest. Each module states its chunk size and the size of its state. A native implementation of a module, where one is added for speed, gives the same results and charges the same fuel (XF-6).
9. **LC-9 Fast-changing data as hash-pinned inputs.** Data that the world changes several times a year enters a program as input, not as library text: the IANA time zone database, the public suffix list, the root certificate set, the media type table, the Unicode Common Locale Data Repository (CLDR), the EPSG coordinate system registry and holiday tables. The library holds the reader and the format; the data set is a Blob read by its SHA-256 digest and admitted by a logged event, so a program's log names the exact data it used, and an update of the data is a new input whose effect on a program `rce change` reports, not a new library hash for every program. CLDR enters with a fixed locale scope. Data that changes with a published standard and that the library's results are defined by, such as the Unicode Character Database tables generated into `ui.text` and `net.idna`, stays generated into the library as it is today. The survey's plan generated every such set into the library hash; this rule replaces that.
10. **LC-10 Emitter parity for every module.** Each module's checks also run through `rce trace --json` on every emit target (Rust, Python, JavaScript and Scala today), and the traces must agree byte for byte (T9), as `scripts/math_parity.sh` already checks for `math`. A target that cannot implement a rule refuses the module with a fixed message (F.10); the module then states that target's refusal. `ct fn` is refused by the Python and JavaScript emitters (F.12) and the library's cryptography is not constant time, so RSA and elliptic-curve signing and password hashing are stated to run at reference quality, or are offered as host adapters whose results are logged.
11. **LC-11 Moves keep the old export.** `pub use` is W-SYN-1 today, so a module cannot re-export another module's items, and moving an item changes the type that other modules name (`os/pkg.rce` names `events.Jv`; `energy` holds `base58`). A new module is added beside the old one and the old export stays until import renaming and re-export exist (a language-growth row of wave 3). Every library commit runs `rce api-diff` and `rce change` over the repository (CC-2).

### The module plan

The module plan follows the existing tree; new top-level areas are added only where nothing fits. The table gives the build order of the survey plan, with each module's main references; the full list of modules per wave is in the ledger.

| Order | Modules | Contents | References |
|---|---|---|---|
| 1 | `text.str` (new) | find, rfind, count, replace, repeat, split_once, slices by character, case converters, slugs, quoting per syntax, natural sort keys | CPython `str`, Go `strings`; two-way search (Crochemore and Perrin 1991) keeps find linear |
| 2 | `col.seq` (new), `col` additions | zip, enumerate, any, all, find, min_by, flat_map, unique, chunks, windows, group_by, partition, scan, sort_by_key, top_k, merge_k, permutations, combinations; set algebra, bisection, heaps, `Ring`, `Counter`, `Multimap`, `Arena` | CPython builtins, `itertools`, `bisect`, `heapq`, `collections` |
| 3 | `fmt` additions | format specifications, exactly rounded float text and correctly rounded float parsing, regular expression captures, find-all, replace and split | CPython `format` and `float`, Go `strconv` tests, RE2 tests; Lemire (2021), Adams (2018) |
| 4 | `data.json`, `data.csv`, `id` (new) | a JSON module with numeric accessors, pretty printing, typed getters and JSON Lines; CSV with a `Dialect` and a chunked reader; UUID (RFC 9562), ULID, nanoid, Snowflake | JSONTestSuite, CPython `json`, `csv`, `uuid`, Go `encoding/csv` |
| 5 | `obs.log`, `test.prop`, `test.fake`, `col.cache` (new) | log records as outputs routed by the host; generators and shrinking seeded from `rand.det` (Claessen and Hughes 2000); LRU, LFU and TTL caches on ticks | Go `slog` JSON output, CPython `OrderedDict` traces |
| 6 | `core.bits`, `hash.crc`, `hash.legacy`, `hash.fast`, `id.check` (new) | bit counts and rotations, bit readers and writers; a parametric CRC; MD5 and SHA-1 named as non-security digests; FNV, xxHash, SipHash, Murmur3; Luhn and ISO 7064 check digits | Go `math/bits`, the CRC catalogue, RFC 1321, FIPS 180-4, python-stdnum |
| 7 | `compress.deflate`, `archive.tar`, `archive.zip` (new) | inflate and deflate as chunked state values, zlib and gzip, tar and zip with sorted entries | zlib, CPython `gzip`, `tarfile`, `zipfile` |
| 8 | `time.tz`, `time.fmt`, `time.cron` (new) | a TZif reader over hash-pinned tzdata (LC-9) with an explicit gap and fold policy; strftime, HTTP dates, RFC 2822, ISO 8601 durations; cron and recurrence rules | CPython `zoneinfo`, Go `time` tests, robfig/cron |
| 9 | `enc`, `num.special` additions | base32, base85, quoted-printable, PEM, COBS; CBOR and JCS floats; f32 and binary16 bits | RFC 4648, RFC 8949, RFC 8785 |
| 10 | `net.url`, `net.route`, `net.cookie`, `net.mime`, `net.ip`, `net.http` | query strings, URI templates, the public suffix list as input, routing, cookies, media types and multipart, IP addresses, HTTP semantics | WPT, RFC 6265bis tests, Go `mime/multipart`, RFC 9110 |
| 11 | `effect`, `io`, `core.err`, `core.fun`, `os.args`, `os.env`, `os.dirs`, `os.glob`, `data.toml`, `data.ini`, `obs.metrics`, `obs.trace` | breakers, jittered backoff, token buckets, bulkheads; reader and writer traits, cancellation scopes, pools; error chains; argument parsing; `.env` files; glob; TOML and INI; OpenMetrics; W3C Trace Context | resilience4j, `golang.org/x/time/rate`, CPython `argparse`, `fnmatch`, `configparser`, toml-test |
| 12 | `col.tree`, `col.bits`, `col.graph`, `text.diff`, `text.similar`, `text.search`, `text.parse`, `text.pretty` | trees, bitsets, graph algorithms, diff (Myers 1986), edit distance, string search, parser combinators, a pretty printer (Wadler 2003) | CPython `graphlib`, `difflib`, GNU diff |
| 13 | `num.*`, `math`, `core.exact`, `rand.det`, `num.stats` additions | number theory, complex numbers, FFT, numerical methods, geometry, IEEE helpers, money and big floats, sampling, online statistics | CPython `math`, `cmath`, `random`, GMP |
| 14 | `crypto` and its new submodules, `enc.der`, `auth.*`, `net.ws`, `net.sse` | standard AEADs, password hashing, DER, RSA, elliptic curves, X.509, JOSE, sessions, policies, WebSocket, server-sent events | RFC 8439, NIST CAVP, Wycheproof, x509-limbo, the Autobahn suite |
| 15 | data formats and messaging | YAML, JSON Schema, validation, Protocol Buffers, binary formats, XML extensions, HTTP/2, RPC, GraphQL, message brokers, database wire protocols, mail, relational operations, SQL, tables, streams | yaml-test-suite, JSON-Schema-Test-Suite, protobuf conformance, h2spec, SQLite and DuckDB outputs |
| 16 | text, internationalization and terminal | Unicode data, character sets, HTML, Markdown, templates, locales, collation, ANSI styling, line editing, terminal UI, charts, layout, colour, SVG, accessibility | UCD tests, WHATWG encoding tests, html5lib-tests, the CommonMark spec, CLDR outputs, UCA CollationTest |
| 17 | media, documents, storage, geospatial | images, PNG, rasterization, audio, PDF and XLSX, Zstandard, LZ codecs, Arrow, search indexes, geodesy and GeoJSON, git objects; MLS inside the manifest; ML-DSA | PngSuite, the zstd and LZ4 reference decoders, PROJ, Turf |
| 18 | paradigm libraries, distribution and domains | logic and constraints, arrays, probabilistic programming, machine learning, sketches, sharding, Merkle proofs, incremental computation, CRDT additions, NLP, real-time media, fieldbus; the critique's finance, science, robotics, bioinformatics and health modules | per module, as the ledger's `sources` field names |

## Form templates

12. **LC-12 A template is an ordinary module.** A template lives at `stdlib/templates/<name>/<name>.rce` and is pinned in the same manifest as the library, which requires `rce manifest` and `rce lib` to cover that directory, so there is still one library hash. A template holds the fixed shape (`intent`, `state`, `event`, the step and its `emit`s); its holes; its obligations (`ensures`, `invariant`, `uses`, `budget`, as `intent-holes.md` defines them); examples and checks that hold for every valid fill, so `rce test` passes on a new instance; and the `use` lines of the modules it composes. Until the checker has typed holes (`closing-the-loop.md`, increment 2), each hole is a named function with a default body, such as a handler that answers 404, and the template's catalog record lists those names as its holes. A template's vectors are traces from a reference replayed in `stdlib/tests/templates/<name>_checks.rce`: h11 and Go `net/http` for the HTTP templates, resilience4j and `x/time/rate` for breaker and limiter stages, robfig/cron for job triggers, OTP restart tables for the actor service, and the SCXML IRP suite for the state machine.
13. **LC-13 Instantiation by copy, first.** `rce new TEMPLATE NAME` copies the one template pinned in the binary's library, renames the module to NAME (the only substitution; no text is expanded), writes `NAME.rce` with a first line naming the template's hash, and prints the holes. `rce test NAME.rce` passes at once, and filling a hole is an ordinary edit. Composition of templates through `use` waits for typed holes, for three reasons the critique found: an imported module's `check`, `example` and `intent` items are not part of the importing program (4.1), so "checks hold for every fill" would never run; a hole written as a default-bodied function in a pinned module cannot be replaced, since a scope has one function per name (F.8 Calls); and the states and events of every reachable Form module merge into the program's Form (4.2). Where composition is needed before typed holes, a hole is a trait method that the user implements on a type of their own module, which the orphan rule (W-TRAIT-2) allows.
14. **LC-14 Catalog and composer.** `rce index` records each template beside the bricks: its hash, name, intent, holes with their types and refinements, the bricks it uses, its events, outputs and examples, and its completeness score (W-HOLE-2). `rce search` matches templates by hole type, event shape or name. `rce fit` fills holes from the catalog and logs each derivation. Its budget is the intent's `budget` obligation (`intent-holes.md` section 1; `closing-the-loop.md` 5.1), not a command-line flag, because no flag changes meaning (F.8 Tooling).

### The template plan

| Order | Template | Shape | Holes | Built from |
|---|---|---|---|---|
| 1 | state-machine | states as a kind, guarded steps, refusals, checked by `rce dim` | the state kind, transitions, outputs | the Form, `ui.states` |
| 2 | actor-service | an actor under a supervisor with a restart bound | message handlers, restart policy | actors, `os.svc` |
| 3 | http-service | request events, response outputs, connection table, keep-alive, limits, health and metrics endpoints | routes, handlers, middleware, limits | `net.http`, `net.route`, `net.url`, `data.json`, `obs.log`, `obs.metrics` |
| 4 | http-client | planned requests, redirect policy, jittered retries, cookie jar, decompression | retry and redirect policy, authentication | `net.http`, `net.cookie`, `effect`, `compress.deflate` |
| 5 | config-loader | defaults, files, environment and arguments merged in a fixed order, with the source of each value recorded | the configuration record, precedence | `data.toml`, `data.ini`, `os.env`, `os.args` |
| 6 | cli-app | the argument snapshot, subcommands, help text and exit status | commands, handlers | `os.args`, config-loader, `obs.log` |
| 7 | rest-resource | create, read, update and delete over state, problem details, ETags from the state hash, pagination | the record type, validation | http-service, `data.json`, `data.valid`, `crypto.kdf` |
| 8 | job-queue | enqueue, lease, retry, dead letters, cron triggers | the job kind, handler, retry policy | `time.cron`, `effect`, `data.feed` |
| 9 | lock-manager | acquire and release events, FIFO grants, deadlock reported as an outcome | the resource key | actors |
| 10 | pubsub, eip-routing | topics as state; aggregator and scatter-gather | filters, completion rules | `data.feed`, `effect` |
| 11 | workflow | activities as requests, timers, signals, compensations | steps and compensations | `effect`, `time.cron` |
| 12 | pipeline | sources, windowed stages, sinks | stage functions, windows | `data.stream`, `data.rel` |

These follow, in order: oauth-client, step-middleware, api-gateway, grpc-service, dns-resolver, entity-store and outbox; tui-app, web-app, ui-app, game and undo-stack; and registry, reconcile, rules, simulation, dp-table, gof-catalog, smtp-client and ssh-client. The critique adds ledger, payment, order-book, pricing, rollback, control-loop, sync-node, agent and vm.

## Host adapters

15. **LC-15 Every input is logged.** A host adapter row is admitted only in a form where everything the program reads arrives as a logged event or a logged result (T3). Platform directories and `PATH` lookup are computed from the admitted environment snapshot. Memory mapping becomes logged ranged reads, file watching becomes watch events, and file locks become grant events. DNS answers, which are resolved inside `NetConnect` and not logged today, are logged by the resolver. TLS uses one root store, either the host's with the presented chain logged or the pinned set of `crypto.x509` read under LC-9, not both. At this commit `FileRead` and `BlobGet` read text and `NetRecv` decodes received bytes lossily, so binary-safe payloads come first: no PNG, zip, gzip, TZif or binary protocol arrives intact without them.

The 26 host-adapter rows, by wave:

| Wave | Rows |
|---|---|
| 1 | the argument and environment snapshot admitted by the host |
| 2 | binary-safe payloads; file delete, rename, mkdir, stat and ranged reads; temporary files and directories; hash-pinned data inputs (LC-9) |
| 3 | TLS with WebPKI, SNI and ALPN; UDP datagrams; Unix domain sockets and named pipes; process spawn with pipes; host snapshots; a database access adapter that takes `Text[Sql]` and logs rows |
| 5 | key stores, the operating-system keyring and capability revocation |
| 6 | a model-answer adapter and LLM API clients; cloud provider APIs |
| 7 | browser automation and container-backed tests; accessibility, clipboard and desktop integration |
| 8 | file locking, memory mapping and change notifications; media containers and capture devices |
| 9 | a segmented log with key compaction; HTTP/3 and QUIC; plugins loaded by hash; cloud hosting patterns carried by the host; GPIO, I2C, SPI, PWM and ADC as request events |
| have | text file read, atomic write, append and list; TCP sockets; the replicated log (Raft, leader terms, write-ahead log, snapshots) |

## Tooling

16. **LC-16 Tools read programs and never change their meaning.** Each tooling row is a command of `rce` that obeys F.8 Tooling: no flag changes meaning, build artifacts are caches, and tests run in canonical order. The 19 tooling rows are: `rce new` with a hash-pinned catalog and emitter parity for every module (wave 1); `rce sbom`, which writes CycloneDX and SPDX documents from the F.5.3 manifest (wave 3); diagnostics (trap call paths, fuel profiles, `rce bench`, a step debugger), test tooling (fixtures, snapshots, fuzzing, coverage, contract tests) and a documentation generator with Markdown doctests (wave 4); `rce fit` with logged derivations, brick catalog coverage (339 of 1,230 refused today), Protocol Buffers and OpenAPI generation, schema migrations with event upcasting, and hot upgrade of a running host as a logged migration entry in D through `rce change` (wave 6); a REPL and notebooks whose cells are inputs of a replayable session log (wave 7); and a hardware description emitter (XF-12), schedulability and worst-case execution time read from the fuel cost model, the language server, native runtime and emitter coverage, and a package fetcher by hash (wave 9). The testing row (examples, checks, recorded handlers, replay, exploration and equivalence) is *have*.

## Growth of the Form

Language growth is the part of the plan that changes the Form itself. Each item below says what the Form does today, how it grows and what the growth serves. Every growth keeps δ total, keeps effects logged and keeps one meaning on every target, and none adds a macro, flag or directive.

17. **LC-17 Growth passes the gate.** A growth item enters the normative text only through CONTRIBUTING rule 1 (an accepted and a refused example, a Lean statement of the theorem it relies on linked to the checker, and 0 LALR(1) conflicts) and rule 2 (the same program written with existing constructs, or a showing that none exists). For every item the library form that expresses it today, listed in the tables below, is that rule-2 evidence; it stays in the library as the way to write the pattern until the item lands, and moves to the new form in the item's commit (rule 3).
18. **LC-18 Conformance before growth.** An item the text already states is a checker conformance fix, an erratum in the terms of C.15, and comes first in wave 1: the structural derive of `Ord`, `Hash` and `Show` that 5.9 states (today struct `<` is W-TRAIT-3, and `List[P].sort()` on a struct `P` passes the checker and then traps with "values are not ordered", so an accepted program traps); the division methods `div_floor`, `div_trunc`, `div_euclid` and `div_exact` that F.3.4 names; the `char` type that F.8 Literals names; and generic kinds and generic struct construction, which 5.1 admits but the checker fails to use (`kind Tree[T]` is accepted and a use of it fails; a user `struct Box[T]` is accepted and its construction runs the built-in `Box`).
19. **LC-19 F.2 becomes a list of growth items.** F.2's heading "Non-goals (excluded because each breaks a theorem)" is restated as "Forms not yet in the text, and the theorem each must keep". Each item keeps its derivation, now read as the obligation a growth must discharge, and gains the library form that expresses it today. The F.8 rows that exclude a form ("no default parameters", "no virtual dispatch", "no weak references", "Reflection and derive: none", "types are not values", "no coroutine iterators except `iter`") are restated the same way. A proposal that adds one of these forms carries five things: (1) the theorem argument, naming each theorem at stake and why the growth keeps it; (2) the Lean update, with `lake build` passing with no `sorry` and only the standard axioms (F.8 Release gate); (3) a 9.8 decision row that restates the earlier decision; (4) a C.15 row that classes the change as an extension (it admits only programs refused before), an erratum or a meaning change, with `rce change` listing the repository programs it moves; and (5) a row in `spec/proposals/LEDGER.md`.

### Items that need no amendment of a non-goal

| Wave | Item | Today | Growth | Serves |
|---|---|---|---|---|
| 1 | structural derive; division methods; generic kinds and struct fields | stated by 5.9, F.3.4 and 5.1; refused or trapping in the checker | conformance (LC-18) | records as sortable map keys and log lines; floor and Euclidean arithmetic; trees, `Result` and `Either` helpers |
| 1 | a `hash` method from the `Hash` trait | `Hash` is derived and exposes nothing; built-in maps take struct keys in canonical order | a 5.15 method | generic hash tables, caches, sketches |
| 3 | a closed built-in `Codec` derive to canonical JSON and CBOR | the built-in derive set of 5.9 | the closed set gains `Codec`; 9.8 Q20 (no user derive) is untouched | typed serialization, as serde, Codable and `encoding/json` give |
| 3 | trait default methods, supertraits under bounds, calls through a type parameter | every member is defined in each impl (W-TRAIT-1) | W-TRAIT-1 admits an impl that omits a member with a default body | Template Method, mixins, type classes such as `Monoid` and `Default` |
| 3 | import renaming and re-export | W-MOD-3 (imported names are not renamed); `pub use` is W-SYN-1 | `use p.q.{f as g}` and `pub use`, resolved before hashing, so T7 holds | library moves without breaking importers (LC-11); XF-2 |
| 4 | numeric traits with arithmetic operators | named functions | an arithmetic operator on a type with a numeric trait impl calls that impl (5.3) | generic numerics, dual-number differentiation |
| 4 | struct and slice patterns | field access after `match` | patterns over records and lists in `match` | functional matching |
| 4 | the `char` type | code points as `i64` | conformance to F.8 Literals (LC-18) | text processing |
| 5 | constant-time lowering of `Secret` operations | `ct fn` refused by the Python and JavaScript emitters (F.12) | each emitter lowers `ct fn` and T45 is tested per target | secure coding |
| 5 | per-field privacy and opaque types with validating constructors | refined types are opaque outside their module (9.8 Q15) | the same for fields of a struct | encapsulation |
| 6 | typed holes with `ensures`, `invariant`, `uses` and `budget` | holes as default-bodied functions (LC-12) | `intent-holes.md` | specification-first development, templates and `rce fit` |
| 9 | 128-bit integers; f16, bf16 and f128 | `core.exact` big integers; `num.special` bit access | 6.3 classes with stated rounding, as XF-11 states binary32 | numeric, ML and scientific work |
| 9 | SIMD masks and shuffles | lane-wise fixed arrays (decision 43) | masks and shuffles over lane-wise arrays; layout stays unpromised (9.8 Q29) | data-parallel kernels |
| 9 | units of measure | values with unit fields | unit types checked by the type system | science and engineering |
| 9 | multi-writer logs merged by a certified merge, cross-node commit as a logged two-phase commit, choreographies | one writer per log; `partial-order-log.md` | D as a partial order (T62) | distributed programming |
| 9 | persistent collections, lenses, string builders, interning, enum flag sets | `col` structures, content hashes (6.6) | library first; language forms only where the library shows a cost | functional data, large renders under fuel |

### Items where the text lists the form as a non-goal, and the amendment proposed

| Wave | Item | Today (the rule-2 form) | Spec text at this commit | Amendment this proposal makes | Theorems and Lean |
|---|---|---|---|---|---|
| 1 | a user `impl Ord` honoured by `<` and `sort` | `sort_with` and `sort_by_key` in `col.seq` | F.2 item 5 (overloading of order operators); F.8 Order operators | `<`, `<=`, `>`, `>=` and `sort` on a type with a user `impl Ord` call that impl; the impl must be a total order, checked as a trait law by examples | T1, with the impl pure and total; `ValOrder.lean`. C.15: a meaning change for accepted programs that trapped |
| 3 | function values in data that never reaches state, events, channels or outputs | closed kinds of strategies with `match`; `fn` parameters | W-TY-2 (a `fn` type inside a data type); F.8 Function values; 9.8 Q23 | W-TY-2 admits a `fn` type in a data type that is never stored, sent or emitted; a recursive `kind` still may not hold one | T3 (the log holds data only); Q23's positivity kept by the `kind` exception; `Typing.lean`, `Struct.lean` |
| 4 | pull iterators (stored steppers) | stepper structs with a `next` function in `col.seq` | F.8 Iteration; 5.14 (an `iter` value cannot be stored, passed or returned) | a stepper is a value whose `next` is a pure total function; each `next` is bounded and counts against LOOP_LIMIT | T15 extended to steppers, T2; `Iter.lean` |
| 4 | files embedded by hash | a generator writes the data into a module | 5.10 W-CT-1 (comptime reads no files) | a file named with its SHA-256 in the program's manifest is part of the program's text (F.5.3), so comptime may read it | T54; T1, since the bytes are fixed by the hash |
| 7 | compile-time reflection and user derive whose output is ordinary hashed source | the `rce.syntax` library row: a tool step writes source that is then checked | F.2 item 4; F.8 Reflection and derive, and Compile-time code ("types are not values"); 9.8 Q20 | a derive is a comptime function from a declaration, as `rce.syntax` data, to source text; the output is checked and hashed as source; a derive reads a field name only as a declared label, which the definition hash covers, as argument labels are checked (9.8 Q16) | T7 restated for derived source (renaming a name no derive reads leaves meaning unchanged); `NameFree.lean`, `Positional.lean`; Sheard and Peyton Jones (2002) |
| 7 | `async` functions and coroutines compiled into state regions | request and `Done` events with `io.Pending`, packaged as a template | F.2 item 5 (`await`, coroutine iterators); F.8 Iteration | an `async fn` is compiled into a state region; each suspension point is ordinary state written to D, and each resumption is one step on one event, as C# compiles `async` methods to state machines (formalized by Bierman et al. 2012) | T2 (each segment between suspensions is bounded), T3 (the suspended state is in S), T15; `Bound.lean`, `Replay.lean`, `Iter.lean` |
| 9 | resumable effect handlers with logged continuations | `Plan[C]` built by pure code, with the recorded handlers of D.4 | F.2 item 1 (T2, T3); the ledger row of 2026-10-22, declined on T2 and T6 | one-shot handlers whose continuation is a state region, as in `async`, so captured state is never cloned (the T6 ground); multi-shot resumption remains a growth item whose obligation is a bound on clones; OCaml 5 ships one-shot continuations (Sivaramakrishnan et al. 2021) | T2 counts each resume as a step; T3 logs the continuation as state; T6 holds for one-shot; `Bound.lean`, `Ownership.lean` |
| 9 | `dyn Tr` as a whole-program generated kind; implementation reuse by delegation | a closed kind of the implementing types with `match`; forwarding functions | 5.9 (no dynamic dispatch, no vtables); F.8 Classes; F.2 item 5 (inheritance and runtime casts) | `dyn Tr` denotes the closed kind of the program's impls of `Tr`, generated at check time, and a call on it is an exhaustive `match`, which is defunctionalization (Reynolds 1972); `impl Tr for T by field` generates forwarding functions as source; subtyping and runtime casts are not part of either | T1, T7 (variants ordered by definition hash, not by name); `Typing.lean`, `NameFree.lean` |
| 9 | weak handles with region-scoped liveness | generation-checked `Arena[T]` indices in `col` | F.8 Containers ("no weak references") | a weak handle is a generation-checked index into a region (5.7); reading it gives an option, and liveness is decided by the region's own writes, never by a collector's timing | T1, T6; `Ownership.lean`, `Borrow.lean` |
| 9 | atomic cells with a per-tick order | actors and channels; `par` with disjoint writes and reductions | F.2 item 3 (relaxing W-PAR-1) | W-PAR-1 stays for plain places; an atomic cell's operations within one tick are applied in task-id order, so every interleaving gives one result | T16 extended to atomic cells; `ParSem.lean`, `Par.lean` |
| 9 | default parameters | an options record with a constructor that fills the defaults | F.8 Calls ("no default parameters") | a default is a comptime constant written in the signature, and a call that omits the argument is the call with that constant; W-NAME-2 (one function per name) is unchanged | T1; `Typing.lean`; a grammar change with 0 LALR(1) conflicts |
| 9 | map and set comprehensions, a pipe operator, variadic parameters, `defer` | `set_of([e for x in xs])`, function composition in `core.fun`, a `List` parameter, a linear handle consumed on every path (W-LIN-1) | F.2 item 5, principle (e) of Appendix F (one spelling per construct) | none to principle (e): the list of F.2 item 5 records the present single spelling; a new spelling that replaces the present one is a rule-3 change that moves the repository in one commit | T1; T12 for scope-end code |
| 9 | scoped context values and lazy constants | explicit parameters; `const` | principle (e) (no meaning that depends on position) | a context value is a parameter named in the signature and bound by an enclosing block, so the value a call reads is visible in its text | T1 |
| 9 | higher-kinded types, trait laws (W-LAW-1), dependent and shape-indexed types (tensors with shapes and differentiation) | generic functions per kind; examples as laws; lengths as comptime constants | 9.8 Q10 (value-indexed types), Q25 | Q10 restated as open, with the T35 predicate fragment as the candidate decidable fragment for indices | T35; `Refine.lean` |

The five things of LC-19 are required for each row of the second table before its form is admitted; until then the row's status in the ledger is *partial* or *missing* and its rule-2 form is the way to write the pattern.

## The nine waves

20. **LC-20 Nine waves in dependency order.** The work is ordered in nine waves, each with library modules, templates, host adapters and tools, and with language items as a parallel track. The critique's order corrections are applied: the `pub` items of `crypto`, `pq`, `hpke` and `mls` move to wave 1, because name-based UUIDs, HMAC variants, PBKDF2, TOTP, signed cookies and the WebSocket accept key all need them, and `use crypto` is what XF-2 needs to retire the `#!lib` directive; PBKDF2 and the database adapter move from waves 5 and 6 to wave 3, because the rest-resource template of wave 3 needs them; WebSocket moves from wave 5 to wave 3, since its dependencies land by then; and the conformance fixes of LC-18 come first in wave 1.

| Wave | Theme | Rows (have, partial, missing) | Main modules and templates |
|---|---|---|---|
| 1 | everyday code | 31 (2, 18, 11) | `text.str`, `col.seq`, `col`, `fmt`, `data.json`, `data.csv`, `id`, `obs.log`, `test.prop`, `test.fake`, `col.cache`, `crypto` `pub` items, the text-from-bytes refresh of `data.events`, `enc` and `net.http`; state-machine, actor-service; `rce new`; emitter parity |
| 2 | bytes, checksums, compression, time | 19 (0, 9, 10) | `core.bits`, `hash.crc`, `hash.legacy`, `hash.fast`, `id.check`, `compress.deflate`, `archive.tar`, `archive.zip`, `time.tz`, `time.fmt`, `enc`, `num.special`, `os.fs`, `os.glob`; binary payloads; hash-pinned inputs |
| 3 | services | 44 (0, 24, 20) | `net.*`, `effect`, `io`, `core.err`, `core.fun`, `os.args`, `os.dirs`, `data.toml`, `data.ini`, `obs.metrics`, `obs.trace`, `crypto.kdf`, `net.ws`, `time.cron`, `app.flags`, `test.http`; http-service, http-client, rest-resource, config-loader, job-queue, step-middleware; TLS, UDP, process spawn, database adapter |
| 4 | algorithms and numbers | 44 (0, 23, 21) | `col.tree`, `col.bits`, `col.graph`, `col.fixed`, `text.diff`, `text.similar`, `text.search`, `text.parse`, `text.pretty`, `num.*`, `math`, `core.exact`, `rand.det`, `fin.money`, `fin.rate`, `time.bizcal`, `ctl.pid`, `ctl.filter`; dp-table, lock-manager, control-loop |
| 5 | security and real-time web | 19 (0, 9, 10) | `crypto.aead`, `crypto.rsa`, `crypto.ec`, `crypto.x509`, `enc.der`, `auth.jose`, `auth.session`, `auth.policy`, `auth.webauthn`, `net.sse`; oauth-client, api-gateway, dns-resolver |
| 6 | data formats, messaging, data processing | 40 (0, 19, 21) | `data.yaml`, `data.jsonschema`, `data.valid`, `enc.proto`, `enc.bin`, `text.xml`, `net.h2`, `net.rpc`, `net.graphql`, `net.mq`, `net.db`, `net.mail`, `data.rel`, `data.sql`, `data.table`, `data.stream`, `data.vector`, `fin.msg`; pubsub, eip-routing, workflow, pipeline, grpc-service, entity-store, outbox, ledger, payment, agent; `rce fit` |
| 7 | text, internationalization, terminal and UI | 31 (1, 12, 18) | `text.unicode`, `text.charset`, `text.html`, `text.markdown`, `text.template`, `text.locale`, `text.collate`, `ui.*`, `ui.a11y`, `media.color`, `media.svg`, `rce.syntax`; tui-app, web-app, ui-app; REPL and notebooks |
| 8 | media, documents, storage, geospatial, games | 31 (0, 7, 24) | `media.*`, `doc.pdf`, `compress.zstd`, `compress.lz`, `data.arrow`, `data.npy`, `data.search`, `geo`, `geo.crs`, `geo.cell`, `game.physics`, `num.gf`, `vcs.git`, `mls`, `pq`; game, rollback, undo-stack, smtp-client, ssh-client |
| 9 | paradigm libraries, distribution and domains | 69 (1, 28, 40) | `logic`, `num.array`, `num.prob`, `num.ml`, `num.sparse`, `col.sketch`, `data.shard`, `data.merkle`, `data.incr`, `data.sync`, `text.nlp`, `media.rtc`, `os.devices`, the bioinformatics, health, robotics and identity modules; registry, reconcile, rules, simulation, gof-catalog, order-book, pricing, sync-node, vm; the later language items |

The other 35 rows are *have* with nothing left to build, and have no wave.

## The critique's corrections

**Missing domains, as added rows.** The critique's rows are in the ledger with `origin` set to `critique`. Twelve new areas hold 48 of them:

| Area | Rows | Modules and templates |
|---|---|---|
| finance | 8 | `time.bizcal` (business-day calendars, holiday rules as pinned inputs), `fin.rate` (day counts, NPV, IRR), `fin.money` (amounts with a currency, splitting without losing minor units), `id.check` (Luhn, ISO 7064, IBAN, ISIN, LEI), `fin.fix`; ledger, order-book and pricing templates |
| payments | 2 | `fin.msg` (ISO 8583, ISO 20022, NACHA); the payment template (authorize, capture, refund, idempotency keys) |
| science | 5 | `data.npy`, `data.hdf5`, `num.const` (CODATA), `num.interval` (IEEE 1788), Bessel and related functions in `num.special` |
| games | 5 | `game.physics` (fixed-step, deterministic), `num.noise`, `game.anim`, `media.mesh` (glTF, OBJ, STL, PLY); the rollback template, whose desync check compares state hashes |
| embedded | 5 | `col.fixed` (capacity in the type), `os.fw` (signed firmware manifests); device requests as host events; worst-case execution time from the fuel model; a hardware description emitter |
| robotics | 5 | `ctl.pid`, `ctl.filter` (Kalman and particle filters), `ctl.kin`, `net.mavlink`; the control-loop template |
| geo | 4 | `geo` (geodesy, GeoJSON, WKT and WKB, predicates), `geo.crs` over a pinned EPSG subset, `geo.cell` (H3, S2, geohash, tiles) |
| bio | 2 | `bio.seq` (FASTA, FASTQ, GenBank, Newick), `bio.align` (Needleman-Wunsch, Smith-Waterman, SAM, VCF) |
| health | 1 | `health.fhir` with HL7 version 2 messages and DICOM |
| ml | 5 | `data.vector` (exact top-k with a fixed tie order, HNSW), `num.ml` preprocessing and metrics, `ml.format`; the agent template (a model call as a logged request); notebooks |
| accessibility | 4 | `ui.a11y` (the accessibility tree and accessible names for `ui.view`), WCAG contrast in `media.color`, `media.caption`; focus order and preferences in ui-app |
| identity | 2 | `auth.webauthn` (passkeys), `auth.saml` with LDAP and SCIM |

The other 24 added rows extend existing areas: constant-time comparison, privacy (redaction, keyed pseudonyms, differential privacy with logged noise) and blockchain primitives in security; Reed-Solomon codes; the RCE syntax as a library (`rce.syntax`), a pretty printer, feature flags evaluated as runtime values, SBOM output, HTTP test helpers, OCI images, a WebAssembly codec and source maps, DOT writers, an interpreter template, a spreadsheet engine, spell checking and MIDI; synchronous dataflow nodes in the style of Lustre (Halbwachs et al. 1991); hot upgrade; one row for barcodes, receipt printing, mDNS, SNTP, SNMP, Web Push, CGI, image resizing and mutation testing; and the rows from the critique's priority changes and plan problems: `crypto` `pub` items, the text-from-bytes refresh, hash-pinned inputs, emitter parity, and import renaming.

**Status corrections.** Seven statuses change. Text from code points and bytes becomes *have*: `List.utf8` is normative in 5.15, and `[195, 169, 240, 159, 152, 128].utf8() == "é😀"` passes on the reference checker. Canonical CBOR and JCS become *partial* (CBOR text is ASCII-only and neither has floats); semver, lockfiles and SRI become *partial* (nothing in the library computes SHA-384); resilience becomes *partial* (no fallback). Floor and Euclidean division, query strings and rate limiting become *partial*, because `time.fdiv` and `time.fmod`, `net.url.pct_encode` and the `data.feed` inbox budget already exist. Twenty-four more rows gain a note or a priority change, recorded in each row's `correction` field: among them, `use crypto` gives W-VIS-1 so the crypto rows are reached only through `#!lib`; the end-to-end encryption protocols are not interoperable with Signal (X3DH has no one-time prekeys); `core.exact.Dec` is further along than the money row said; case mapping is ASCII-only; and `fmt.search` returns no match position.

**Fast-changing data** is LC-9; **chunked and resumable APIs** are LC-8; **emitter parity** is LC-10.

**Stale documents.** A document that says RCE lacks what it has is corrected in the commit that finds it. This proposal's commit corrects the comments of `enc`, `net.http`, `net.idna`, `data.events` and `text.utf8`, a comment of the `enc` checks, five passages of `stdlib/vectors/SOURCE.md`, the README and `account.rce` of `programs/money` (which said there is no decimal type, where `core.exact.Dec` exists) and the README of `programs/body` (which said there are no duration or date types, where `time` and `os.clock` exist). `programs/money/price.rce` shows prices as exact decimals. The comment edits change no module hash.

## What changes in the normative text

- **F.2:** the heading and introduction restated as growth items with obligations (LC-19); item 1 restated for one-shot handlers, item 3 for atomic cells with a per-tick order, item 4 for derives whose output is source, and item 5 for the order-operator, inheritance, `await` and coroutine entries.
- **F.8:** the Calls, Classes, Containers, Reflection and derive, Compile-time code, Function values, Iteration and Order operators rows restated as in the second growth table.
- **9.8:** new decision rows restating Q10, Q20, Q23 and Q25 as each item is admitted.
- **5.9, 5.10, 5.14, 5.15, W-TY-2, W-TRAIT-1, W-MOD-3:** amended per item, each in its own commit.
- **C.15:** one row per admitted item, classed as extension, erratum or meaning change.
- **4.1 and F.5.3:** the manifest covers `stdlib/templates` (LC-12).
- **Appendix E:** binary-safe payloads, hash-pinned data inputs by digest, and the logged forms of LC-15.

## Order of work

The conformance fixes of LC-18 and the wave 1 library work come first, because every later wave uses them. Binary-safe payloads and hash-pinned inputs (wave 2) precede every codec, time zone and certificate module. Emitter parity runs from wave 1 on every module that lands. The growth items of the second table follow their waves and are each a proposal of their own under LC-19.

## What stays open

- Whether one-shot handlers and `async` compiled into state regions are one construct or two. Both store a continuation as state; one construct would serve both.
- Whether multi-shot effect handlers can be bounded. The obligation is a bound on clones of captured state that T2 can compute from the program's text.
- The domain survey of LC-3, which would score the 72 critique rows and may reorder waves 4 to 9.
- Which root store TLS uses (LC-15), and how CLDR's locale scope is chosen (LC-9).
- Whether templates compose through traits (LC-13) well enough that `use`-composition after typed holes adds nothing.
- The `rce` binary, which carries the library, is 21.7 MB, with 37,123 lines in `ui/text.rce` and 39,164 in `net/idna.rce`; LC-9 keeps the fastest-growing tables out of it, and whether the remaining tables need a more compact encoding is open.

## Sources

- Adams, U. 2018. Ryū: fast float-to-string conversion. PLDI.
- Bierman, G., Russo, C., Mainland, G., Meijer, E. and Torgersen, M. 2012. Pause 'n' Play: formalizing asynchronous C#. ECOOP.
- Buschmann, F., Meunier, R., Rohnert, H., Sommerlad, P. and Stal, M. 1996. Pattern-Oriented Software Architecture, volume 1. Wiley.
- Claessen, K. and Hughes, J. 2000. QuickCheck: a lightweight tool for random testing of Haskell programs. ICFP.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L. and Stein, C. 2022. Introduction to Algorithms, fourth edition. MIT Press.
- Crochemore, M. and Perrin, D. 1991. Two-way string-matching. Journal of the ACM 38(3).
- Fowler, M. 2002. Patterns of Enterprise Application Architecture. Addison-Wesley.
- Gamma, E., Helm, R., Johnson, R. and Vlissides, J. 1994. Design Patterns. Addison-Wesley.
- Halbwachs, N., Caspi, P., Raymond, P. and Pilaud, D. 1991. The synchronous data flow programming language LUSTRE. Proceedings of the IEEE 79(9).
- Hohpe, G. and Woolf, B. 2003. Enterprise Integration Patterns. Addison-Wesley.
- Joshi, U. 2023. Patterns of Distributed Systems. Addison-Wesley.
- Lemire, D. 2021. Number parsing at a gigabyte per second. Software: Practice and Experience 51(8).
- Myers, E. W. 1986. An O(ND) difference algorithm and its variations. Algorithmica 1.
- Reynolds, J. C. 1972. Definitional interpreters for higher-order programming languages. Proceedings of the ACM Annual Conference.
- Sheard, T. and Peyton Jones, S. 2002. Template meta-programming for Haskell. Haskell Workshop.
- Sivaramakrishnan, K. C., Dolan, S., White, L., Kelly, T., Jaffer, S. and Madhavapeddy, A. 2021. Retrofitting effect handlers onto OCaml. PLDI.
- Wadler, P. 2003. A prettier printer. In The Fun of Programming. Palgrave.
- The domain sources the critique fetched (QuantLib, vacanza/holidays, python-stdnum, dinero.js, TigerBeetle, moov-io/iso8583, QuickFIX, NEP 1, GGPO, FastNoiseLite, glTF, heapless, python-control, filterpy, MAVLink, rosidl, PROJ, Turf, H3, Biopython, HL7 FHIR, pydicom, hnswlib, safetensors, ONNX, accname, axe-core, WebAuthn, klauspost/reedsolomon, CycloneDX, OpenFeature, the OCI image specification, the WebAssembly specification, ECMA-426, hunspell, mido, BIP-39 and the Go standard library) are named with their URLs in the `sources` field of each row.
