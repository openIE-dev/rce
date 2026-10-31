# photon: MPL-2.0 source

The `rce-ide` binaries include photon (vendored in `vendor/photon`), which is
licensed under the Mozilla Public License 2.0 (text: LICENSE-MPL-2.0 in that
archive). The rest of this release is MIT OR Apache-2.0.

The complete source of the photon files used in this build is distributed
next to the binaries as `photon-source-mpl-2.0.tar.gz` (see SHA256SUMS), and
is also available at https://github.com/openIE-dev/rce/tree/main/ide/vendor/photon,
and as the asset `photon-source-mpl-2.0.tar.gz` of each release: releases
published from openIE-dev/rce (tags `ide-v*`) at
https://github.com/openIE-dev/rce/releases, and v0.1.0 to v0.1.3 at
https://github.com/openIE-dev/rce-ide/releases.

Origin: upstream commit 4c64e8e12a09363b04dd11ba3066bb95c8eee8f0; `src/` is
unchanged, `Cargo.toml` is rewritten for this workspace (see VENDORED.md).
