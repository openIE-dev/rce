//! `ide-sandbox`: where agent-produced RCE code runs.
//!
//! - [`wasi`]: RCE programs and tests run as a WASI command inside wasmi, a pure-Rust interpreter that
//!   also runs in the browser.
//!   Only the workspace directory is preopened (read-only, at `/ws`), the guest
//!   gets no sockets (WASI preview 1 has none, and no network capability is
//!   granted), no environment and no inherited stdio. Every run has a fuel
//!   limit, derived from the remaining joule budget, and reports the fuel it
//!   used as the simulator's `wasi_fuel` feature.
//! - [`net`]: declared network capabilities. The guest has no sockets; a
//!   manifest names endpoints, the host-side proxy grants exactly those once
//!   approved, injects supervisor credentials, and counts bytes.
//! - [`jail`]: a [`jail::Jail`] trait for native processes, with a Linux
//!   backend (Landlock filesystem and TCP rules, plus a seccomp filter) that
//!   fails closed: if the kernel cannot enforce every rule, nothing runs.
pub mod jail;
pub mod net;
pub mod proxy;
pub mod tls;
pub mod wasi;
