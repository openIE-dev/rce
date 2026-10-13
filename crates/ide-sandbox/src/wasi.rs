//! wasmi WASI (preview 1) runner. Pure Rust: wasmi interprets, wasi-common
//! with cap-std provides the capability-based filesystem.
use crate::net::{NetEvent, NetProxy, Proto};
use ide_energy::sim::Features;
use std::any::Any;
use std::path::Path;
use wasmi::{Caller, Config, Engine, Extern, Linker, Module, Store};
use wasmi_wasi::WasiCtx;
use wasmi_wasi::wasi_common::dir::{OpenResult, ReaddirCursor, ReaddirEntity, WasiDir};
use wasmi_wasi::wasi_common::file::{FdFlags, Filestat, OFlags};
use wasmi_wasi::wasi_common::pipe::WritePipe;
use wasmi_wasi::wasi_common::random::Deterministic;
use wasmi_wasi::wasi_common::sync::{Dir, ambient_authority, clocks_ctx, sched_ctx};
use wasmi_wasi::wasi_common::{Error as WasiError, ErrorExt, Table};

/// Directory name the workspace is mounted at inside the guest.
pub const GUEST_ROOT: &str = "/ws";
/// Cap on captured stdout and stderr, bytes each (output past it is dropped).
pub const OUTPUT_CAP: usize = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("wasm engine: {0}")]
    Engine(String),
    #[error("fuel limit: {0}")]
    Fuel(String),
    #[error("workspace: {0}")]
    Workspace(String),
}

/// A preopened directory the guest can read but never change: opening for
/// write, create, truncate or exclusive is refused, and every mutating
/// directory call keeps the trait's default "not supported".
struct ReadOnlyDir(Box<dyn WasiDir>);

#[wiggle::async_trait]
impl WasiDir for ReadOnlyDir {
    fn as_any(&self) -> &dyn Any {
        self
    }
    async fn open_file(
        &self,
        follow: bool,
        path: &str,
        oflags: OFlags,
        read: bool,
        write: bool,
        fdflags: FdFlags,
    ) -> Result<OpenResult, WasiError> {
        if write || oflags.intersects(OFlags::CREATE | OFlags::TRUNCATE | OFlags::EXCLUSIVE) {
            return Err(WasiError::perm());
        }
        match self
            .0
            .open_file(follow, path, oflags, read, false, fdflags)
            .await?
        {
            OpenResult::Dir(d) => Ok(OpenResult::Dir(Box::new(ReadOnlyDir(d)))),
            f => Ok(f),
        }
    }
    async fn readdir(
        &self,
        c: ReaddirCursor,
    ) -> Result<Box<dyn Iterator<Item = Result<ReaddirEntity, WasiError>> + Send>, WasiError> {
        self.0.readdir(c).await
    }
    async fn read_link(&self, p: &str) -> Result<std::path::PathBuf, WasiError> {
        self.0.read_link(p).await
    }
    async fn get_filestat(&self) -> Result<Filestat, WasiError> {
        self.0.get_filestat().await
    }
    async fn get_path_filestat(&self, p: &str, follow: bool) -> Result<Filestat, WasiError> {
        self.0.get_path_filestat(p, follow).await
    }
}

/// How much the guest may run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FuelLimit {
    /// A fixed number of fuel units.
    Fuel(u64),
    /// Joules left in the budget, and the calibrated cost of one fuel unit.
    /// Fuel = floor(joules / joules_per_fuel).
    Joules {
        remaining_j: f64,
        joules_per_fuel: f64,
    },
}

impl FuelLimit {
    pub fn fuel(self) -> Result<u64, SandboxError> {
        match self {
            FuelLimit::Fuel(0) => Err(SandboxError::Fuel("zero fuel".into())),
            FuelLimit::Fuel(n) => Ok(n),
            FuelLimit::Joules {
                remaining_j,
                joules_per_fuel,
            } => {
                if !(remaining_j.is_finite()
                    && joules_per_fuel.is_finite()
                    && joules_per_fuel > 0.0)
                {
                    return Err(SandboxError::Fuel(format!(
                        "need finite remaining joules and a positive joules_per_fuel (got {remaining_j}, {joules_per_fuel})"
                    )));
                }
                let f = (remaining_j.max(0.0) / joules_per_fuel).floor();
                if f < 1.0 {
                    return Err(SandboxError::Fuel("joule budget is spent".into()));
                }
                Ok(f.min(u64::MAX as f64) as u64)
            }
        }
    }
}

/// What a run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct RunReport {
    /// Guest exit code; `None` when it trapped (including running out of fuel).
    pub exit_code: Option<i32>,
    pub out_of_fuel: bool,
    pub trap: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub fuel_limit: u64,
    pub fuel_used: u64,
    /// Bytes moved through the network proxy (in + out).
    pub net_bytes: u64,
    /// Every proxied or refused request, for the receipt.
    pub net: Vec<NetEvent>,
}

impl RunReport {
    /// The simulator features of this run (gear `sandbox`).
    pub fn features(&self) -> Features {
        Features {
            wasi_fuel: self.fuel_used,
            net_bytes: self.net_bytes,
            gear: "sandbox".into(),
            ..Default::default()
        }
    }
}

impl RunReport {
    /// Receipt `detail` for this run: fuel, exit, and every network event
    /// (endpoint, decision, bytes; never credential values).
    pub fn receipt_detail(&self) -> serde_json::Value {
        serde_json::json!({
            "exit_code": self.exit_code,
            "out_of_fuel": self.out_of_fuel,
            "fuel_used": self.fuel_used,
            "net_bytes": self.net_bytes,
            "net": self.net,
        })
    }
}

/// A compiled guest, reusable across runs.
pub struct Sandbox {
    engine: Engine,
    module: Module,
}

type Out = WritePipe<std::io::Cursor<Vec<u8>>>;

fn take(p: Out) -> String {
    match p.try_into_inner() {
        Ok(c) => {
            let mut v = c.into_inner();
            v.truncate(OUTPUT_CAP);
            String::from_utf8_lossy(&v).into_owned()
        }
        Err(_) => String::new(),
    }
}

impl Sandbox {
    pub fn new(wasm: &[u8]) -> Result<Self, SandboxError> {
        let mut cfg = Config::default();
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg);
        let module = Module::new(&engine, wasm).map_err(|e| SandboxError::Engine(e.to_string()))?;
        Ok(Sandbox { engine, module })
    }

    /// Run the guest with `args` (argv[1..]), the workspace preopened
    /// read-only at [`GUEST_ROOT`], and the given fuel limit.
    pub fn run(
        &self,
        workspace: &Path,
        args: &[&str],
        limit: FuelLimit,
    ) -> Result<RunReport, SandboxError> {
        self.run_with_net(workspace, args, limit, &mut NetProxy::closed())
    }

    /// As [`Sandbox::run`], with the `rce_net.request` import routed through
    /// `net`. Events of this run are moved from the proxy into the report.
    pub fn run_with_net(
        &self,
        workspace: &Path,
        args: &[&str],
        limit: FuelLimit,
        net: &mut NetProxy,
    ) -> Result<RunReport, SandboxError> {
        let fuel = limit.fuel()?;
        let ws = workspace
            .canonicalize()
            .map_err(|e| SandboxError::Workspace(format!("{}: {e}", workspace.display())))?;
        let dir = Dir::open_ambient_dir(&ws, ambient_authority())
            .map_err(|e| SandboxError::Workspace(e.to_string()))?;
        // Fixed randomness: the guest's hash maps seed from WASI random, and a
        // fixed seed makes the same program and input cost the same fuel.
        // The checker needs no secret randomness.
        let mut ctx = WasiCtx::new(
            Box::new(Deterministic::new((0u8..=255).collect())),
            clocks_ctx(),
            sched_ctx(),
            Table::new(),
        );
        // No env, no stdin, no sockets: WASI preview 1 has no socket creation,
        // and no socket is preopened.
        let out: Out = WritePipe::new_in_memory();
        let err: Out = WritePipe::new_in_memory();
        ctx.set_stdout(Box::new(out.clone()));
        ctx.set_stderr(Box::new(err.clone()));
        ctx.push_arg("rce-guest")
            .map_err(|e| SandboxError::Engine(e.to_string()))?;
        for a in args {
            ctx.push_arg(a)
                .map_err(|e| SandboxError::Engine(e.to_string()))?;
        }
        let ro = ReadOnlyDir(Box::new(
            wasmi_wasi::wasi_common::sync::dir::Dir::from_cap_std(dir),
        ));
        ctx.push_preopened_dir(Box::new(ro), GUEST_ROOT)
            .map_err(|e| SandboxError::Workspace(e.to_string()))?;
        let proxy = std::mem::replace(net, NetProxy::closed());
        let mut store = Store::new(
            &self.engine,
            Host {
                wasi: ctx,
                net: proxy,
            },
        );
        store
            .set_fuel(fuel)
            .map_err(|e| SandboxError::Engine(e.to_string()))?;
        let mut linker: Linker<Host> = Linker::new(&self.engine);
        wasmi_wasi::add_to_linker(&mut linker, |h: &mut Host| &mut h.wasi)
            .map_err(|e| SandboxError::Engine(e.to_string()))?;
        linker
            .func_wrap("rce_net", "request", net_request)
            .map_err(|e| SandboxError::Engine(e.to_string()))?;
        let inst = linker
            .instantiate_and_start(&mut store, &self.module)
            .map_err(|e| SandboxError::Engine(e.to_string()))?;
        let start = inst
            .get_typed_func::<(), ()>(&store, "_start")
            .map_err(|e| SandboxError::Engine(e.to_string()))?;
        let r = start.call(&mut store, ());
        let left = store.get_fuel().unwrap_or(0);
        let (mut exit_code, mut out_of_fuel, mut trap) = (None, false, None);
        match r {
            Ok(()) => exit_code = Some(0),
            Err(e) => {
                if let Some(x) = e.i32_exit_status() {
                    exit_code = Some(x);
                } else if e.as_trap_code() == Some(wasmi::TrapCode::OutOfFuel) {
                    out_of_fuel = true;
                    trap = Some("out of fuel".into());
                } else {
                    trap = Some(e.to_string());
                }
            }
        }
        let Host {
            wasi,
            net: mut proxy,
        } = store.into_data();
        drop(wasi); // releases the stdout/stderr pipe clones
        let events = std::mem::take(&mut proxy.events);
        *net = proxy;
        let net_bytes = events.iter().map(|e| e.bytes_in + e.bytes_out).sum();
        Ok(RunReport {
            exit_code,
            out_of_fuel,
            trap,
            stdout: take(out),
            stderr: take(err),
            fuel_limit: fuel,
            fuel_used: fuel - left,
            net_bytes,
            net: events,
        })
    }

    /// `rce check FILE` in the guest. `rel` is relative to the workspace.
    pub fn check(&self, ws: &Path, rel: &str, limit: FuelLimit) -> Result<RunReport, SandboxError> {
        self.run(ws, &["check", &guest_path(rel)?], limit)
    }

    /// `rce test FILE` in the guest.
    pub fn test(&self, ws: &Path, rel: &str, limit: FuelLimit) -> Result<RunReport, SandboxError> {
        self.run(ws, &["test", &guest_path(rel)?], limit)
    }

    /// `rce run FILE [EVENTS]` in the guest.
    pub fn run_program(
        &self,
        ws: &Path,
        rel: &str,
        events: Option<&str>,
        limit: FuelLimit,
    ) -> Result<RunReport, SandboxError> {
        let p = guest_path(rel)?;
        let mut a = vec!["run", p.as_str()];
        if let Some(e) = events {
            a.push(e);
        }
        self.run(ws, &a, limit)
    }
}

/// Map a workspace-relative path into the guest; refuses absolute paths and `..`.
pub fn guest_path(rel: &str) -> Result<String, SandboxError> {
    let p = Path::new(rel);
    if rel.is_empty()
        || !p
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
    {
        return Err(SandboxError::Workspace(format!(
            "`{rel}`: only plain relative paths"
        )));
    }
    Ok(format!("{GUEST_ROOT}/{rel}"))
}

/// Store data: WASI plus the network proxy.
pub struct Host {
    wasi: WasiCtx,
    net: NetProxy,
}

/// `rce_net.request(proto, host_ptr, host_len, port, req_ptr, req_len, out_ptr, out_cap) -> i64`
///
/// Returns the full response length (at most `out_cap` bytes are written),
/// or a negative [`crate::net::NetError::code`]: -1 undeclared, -2 not
/// approved, -3 I/O, -4 bad request. -6 means bad guest memory arguments.
#[allow(clippy::too_many_arguments)]
fn net_request(
    mut c: Caller<'_, Host>,
    proto: i32,
    host_ptr: i32,
    host_len: i32,
    port: i32,
    req_ptr: i32,
    req_len: i32,
    out_ptr: i32,
    out_cap: i32,
) -> i64 {
    let Some(Extern::Memory(mem)) = c.get_export("memory") else {
        return -6;
    };
    let rd = |c: &Caller<'_, Host>, p: i32, n: i32| -> Option<Vec<u8>> {
        let (p, n) = (usize::try_from(p).ok()?, usize::try_from(n).ok()?);
        mem.data(c).get(p..p.checked_add(n)?).map(<[u8]>::to_vec)
    };
    let (Some(h), Some(req), Some(proto), Ok(port)) = (
        rd(&c, host_ptr, host_len),
        rd(&c, req_ptr, req_len),
        Proto::from_code(proto),
        u16::try_from(port),
    ) else {
        return -6;
    };
    let Ok(host) = String::from_utf8(h) else {
        return -6;
    };
    match c.data_mut().net.request(proto, &host, port, &req) {
        Ok(resp) => {
            let (Ok(p), Ok(cap)) = (usize::try_from(out_ptr), usize::try_from(out_cap)) else {
                return -6;
            };
            let n = resp.len().min(cap);
            match mem.data_mut(&mut c).get_mut(p..p.saturating_add(n)) {
                Some(dst) => {
                    dst.copy_from_slice(&resp[..n]);
                    resp.len() as i64
                }
                None => -6,
            }
        }
        Err(e) => e.code(),
    }
}
