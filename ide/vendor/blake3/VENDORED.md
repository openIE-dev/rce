blake3 1.8.7 from crates.io, library source (src/) unchanged. Removed: the C
sources (c/), benches, tools, media and the dev/build dependency tables.
build.rs is replaced by a pure-Rust script equal to upstream's `pure` path,
so the crate has no `cc` build dependency. Used through `[patch.crates-io]`
in the workspace Cargo.toml. To drop this copy, delete the patch entry once
upstream makes `cc` optional.
