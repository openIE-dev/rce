//! Drives the corpus through every backend × scheduler combination
//! and compares outputs.

use crate::compare::{compare, ToleranceMode};
use crate::corpus::{ConformanceCase, DtypeFamily};
use crate::report::{ReportLine, Verdict};
use rce_vm_backend_cpu::{
    interpret_with_config, InterpretConfig, OpDispatch, Value,
};
use rce_vm_backend_wgpu::WgpuExecutor;
use rce_vm_scheduler::Backends;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    CpuSync,
    CpuAsync,
    WgpuSync,
    WgpuAsync,
}

impl BackendId {
    pub fn label(&self) -> &'static str {
        match self {
            BackendId::CpuSync => "cpu·sync",
            BackendId::CpuAsync => "cpu·async",
            BackendId::WgpuSync => "wgpu·sync",
            BackendId::WgpuAsync => "wgpu·async",
        }
    }
}

pub struct RunOptions {
    /// If true, skip wgpu rows (CI without a GPU).
    pub skip_wgpu: bool,
    /// Per-family override map; missing entries default to 1e-4 / 1e-5.
    pub tolerance: ToleranceMode,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            skip_wgpu: false,
            tolerance: ToleranceMode::default(),
        }
    }
}

/// Run every case through every backend and produce a report. The CPU
/// sync output is the **reference**; everything else is compared
/// against it.
pub fn run_corpus(corpus: &[ConformanceCase], opts: &RunOptions) -> Vec<ReportLine> {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            return corpus
                .iter()
                .map(|c| ReportLine {
                    case: c.name.clone(),
                    backend: BackendId::CpuAsync,
                    verdict: Verdict::Error(format!("tokio runtime: {e}")),
                })
                .collect();
        }
    };
    let wgpu = if opts.skip_wgpu {
        None
    } else {
        match WgpuExecutor::new() {
            Ok(e) => Some(e),
            Err(_) => None,
        }
    };
    let wgpu_arc = wgpu.as_ref().map(|_| match WgpuExecutor::new() {
        Ok(e) => Some(Arc::new(e)),
        Err(_) => None,
    }).flatten();

    let mut report = Vec::with_capacity(corpus.len() * 4);

    for case in corpus {
        // 1. CPU sync — the reference.
        let cpu_sync_value = match run_cpu_sync(case, None) {
            Ok(v) => v,
            Err(e) => {
                report.push(ReportLine {
                    case: case.name.clone(),
                    backend: BackendId::CpuSync,
                    verdict: Verdict::Error(e),
                });
                continue;
            }
        };
        report.push(ReportLine {
            case: case.name.clone(),
            backend: BackendId::CpuSync,
            verdict: Verdict::Pass,
        });

        // 2. CPU async — must agree with reference.
        match run_cpu_async(&rt, case) {
            Ok(v) => report.push(line(case, BackendId::CpuAsync, &cpu_sync_value, &v, opts)),
            Err(e) => report.push(ReportLine {
                case: case.name.clone(),
                backend: BackendId::CpuAsync,
                verdict: Verdict::Error(e),
            }),
        }

        // 3. wgpu sync — agreement at f32 precision (tolerance mode).
        if let Some(exec) = wgpu.as_ref() {
            match run_cpu_sync(case, Some(exec)) {
                Ok(v) => {
                    let line_tol = effective_tolerance(case, opts);
                    let verdict = match compare(&cpu_sync_value, &v, line_tol) {
                        Ok(()) => Verdict::Pass,
                        Err(diff) => Verdict::Diverge(diff),
                    };
                    report.push(ReportLine {
                        case: case.name.clone(),
                        backend: BackendId::WgpuSync,
                        verdict,
                    });
                }
                Err(e) => report.push(ReportLine {
                    case: case.name.clone(),
                    backend: BackendId::WgpuSync,
                    verdict: Verdict::Error(e),
                }),
            }
        } else if !opts.skip_wgpu {
            report.push(ReportLine {
                case: case.name.clone(),
                backend: BackendId::WgpuSync,
                verdict: Verdict::Skipped("no wgpu adapter".into()),
            });
        }

        // 4. wgpu async.
        if let Some(arc) = wgpu_arc.as_ref() {
            match run_wgpu_async(&rt, case, arc.clone()) {
                Ok(v) => {
                    let line_tol = effective_tolerance(case, opts);
                    let verdict = match compare(&cpu_sync_value, &v, line_tol) {
                        Ok(()) => Verdict::Pass,
                        Err(diff) => Verdict::Diverge(diff),
                    };
                    report.push(ReportLine {
                        case: case.name.clone(),
                        backend: BackendId::WgpuAsync,
                        verdict,
                    });
                }
                Err(e) => report.push(ReportLine {
                    case: case.name.clone(),
                    backend: BackendId::WgpuAsync,
                    verdict: Verdict::Error(e),
                }),
            }
        } else if !opts.skip_wgpu {
            report.push(ReportLine {
                case: case.name.clone(),
                backend: BackendId::WgpuAsync,
                verdict: Verdict::Skipped("no wgpu adapter".into()),
            });
        }
    }
    report
}

fn line(
    case: &ConformanceCase,
    backend: BackendId,
    expected: &Value,
    got: &Value,
    opts: &RunOptions,
) -> ReportLine {
    let tol = effective_tolerance(case, opts);
    let verdict = match compare(expected, got, tol) {
        Ok(()) => Verdict::Pass,
        Err(diff) => Verdict::Diverge(diff),
    };
    ReportLine {
        case: case.name.clone(),
        backend,
        verdict,
    }
}

fn effective_tolerance(case: &ConformanceCase, opts: &RunOptions) -> ToleranceMode {
    // Int / Bool families demand bit-equality; Float / Mixed use the
    // configured (default 1e-4 relative) tolerance.
    match case.family {
        DtypeFamily::Int | DtypeFamily::Bool => ToleranceMode::Bitwise,
        DtypeFamily::Float | DtypeFamily::Mixed => opts.tolerance,
    }
}

fn run_cpu_sync(
    case: &ConformanceCase,
    wgpu: Option<&WgpuExecutor>,
) -> Result<Value, String> {
    let ops_storage: Vec<&dyn OpDispatch> = match wgpu {
        Some(e) => vec![e],
        None => Vec::new(),
    };
    let cfg = InterpretConfig {
        leaves: &[],
        ops: &ops_storage,
    };
    interpret_with_config(&case.graph, &case.inputs, cfg).map_err(|e| format!("{e}"))
}

fn run_cpu_async(
    rt: &tokio::runtime::Runtime,
    case: &ConformanceCase,
) -> Result<Value, String> {
    rt.block_on(rce_vm_scheduler::run_async(
        case.graph.clone(),
        case.inputs.clone(),
        Backends::default(),
    ))
    .map_err(|e| format!("{e}"))
}

fn run_wgpu_async(
    rt: &tokio::runtime::Runtime,
    case: &ConformanceCase,
    wgpu: Arc<WgpuExecutor>,
) -> Result<Value, String> {
    let mut backends = Backends::default();
    backends.ops.push(wgpu);
    rt.block_on(rce_vm_scheduler::run_async(
        case.graph.clone(),
        case.inputs.clone(),
        backends,
    ))
    .map_err(|e| format!("{e}"))
}
