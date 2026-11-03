# RAP message fixtures: source and license

`spec_examples.json` holds small JSON examples copied verbatim from the
Reactive Agent Protocol (RAP) specification of the Infinity project:

- Source: https://github.com/hydro-project/infinity, `docs/docs/rap/spec/`
  (commit 1abc6fc), pages `basic/tool-invocation.md`, `basic/tool-result.md`,
  `basic/toolsets.md`, `basic/tool-cancellation.md` and
  `basic/thread-closure.md`. Each fixture names its page.
- License: Apache License 2.0
  (https://www.apache.org/licenses/LICENSE-2.0).
- Copyright notice of the project (its `NOTICE` file): "Infinity.
  Copyright 2026 Hydro Contributors. The Initial Developer of some parts of
  the framework is Amazon Web Services. Copyright 2026 Amazon.com, Inc. or
  its affiliates."

Only the examples are copied, unchanged; nothing else of the project is
used. `tests/host_rap.rs` (`rap_messages_match_the_spec_examples`) checks
the shapes `rce host serve rap` reads and writes against them.
