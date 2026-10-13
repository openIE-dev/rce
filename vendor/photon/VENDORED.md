# photon (vendored)

- Origin: the OpenIE web-browser repository, `crates/photon`, git commit
  `4c64e8e12a09363b04dd11ba3066bb95c8eee8f0` (2026-10-06 16:38:12 -0400).
- Taken with `git archive 4c64e8e1 crates/photon` (the committed tree).
  Archive sha256 `635ed6681a5750ea59e6dfda4a981480ebda83501ba4f1b8bd8937fae3eb416f`.
- A separate local copy (src only: lib, primitives, render, view) matches
  this commit byte for byte for those four files.
- `src/` is unchanged. `Cargo.toml` is rewritten: the upstream file inherited
  version/edition/rust-version/license (MPL-2.0) from the Nova workspace; they
  are now explicit, `serde`/`serde_json` come from this workspace,
  `ciborium = "0.2.2"`, and clippy lints are allowed for this package (upstream
  code has findings under the pinned 1.98.1 clippy; none were fixed here).
- License: MPL-2.0 (upstream), unlike the rest of this repo.
