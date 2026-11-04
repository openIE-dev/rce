//! GPU kernel launch attribution (CodeGreen / Nsight lesson), scaffold.
//! Given launch records (kernel, the source function that launched it,
//! start and end on the device clock) and a cumulative GPU energy counter
//! (NVML total energy), each kernel's share is its overlap with each
//! sample interval, via `apportion`. Concurrent kernels split an interval.
//!
//! Status: the attribution math is tested with synthetic samples. A real
//! launch source for work submitted through wgpu lives in
//! `rce_ide::gpu_launch` (GPU timestamp queries when the adapter has them,
//! else host-timed submit-to-done). Launches from other processes (CUDA,
//! ROCm, Metal outside wgpu) need vendor profilers and are not collected.
use crate::apportion::{apportion, Activity, Apportioned};
use crate::attrib::Sample;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub kernel: String,
    /// Form function (or host symbol) that issued the launch.
    pub caller: String,
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Something that yields launch records. Implemented for wgpu work in
/// `rce_ide::gpu_launch::WgpuLaunchSource`.
pub trait LaunchSource {
    fn drain(&mut self) -> Result<Vec<Launch>, String>;
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuAttribution {
    pub per_kernel: BTreeMap<String, f64>,
    pub per_caller: BTreeMap<String, f64>,
    pub idle: f64,
}

pub fn attribute(samples: &[Sample], launches: &[Launch]) -> GpuAttribution {
    let acts: Vec<Activity> = launches.iter().enumerate().map(|(i, l)| Activity {
        process: format!("{i}"), start_ns: l.start_ns, end_ns: l.end_ns, weight: 1.0,
    }).collect();
    let Apportioned { per_process, idle } = apportion(samples, &acts);
    let mut out = GpuAttribution { idle, ..Default::default() };
    for (k, j) in per_process {
        let l = &launches[k.parse::<usize>().unwrap_or(0)];
        *out.per_kernel.entry(l.kernel.clone()).or_default() += j;
        *out.per_caller.entry(l.caller.clone()).or_default() += j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernels_split_overlap_and_roll_up_to_callers() {
        let s = [Sample { t_ns: 0, joules: 0.0 }, Sample { t_ns: 10, joules: 10.0 }, Sample { t_ns: 20, joules: 30.0 }];
        let l = |k: &str, c: &str, a, b| Launch { kernel: k.into(), caller: c.into(), start_ns: a, end_ns: b };
        let r = attribute(&s, &[l("gemm", "train", 0, 10), l("gemm", "train", 10, 20), l("relu", "infer", 10, 20)]);
        assert!((r.per_kernel["gemm"] - 20.0).abs() < 1e-9);
        assert!((r.per_kernel["relu"] - 10.0).abs() < 1e-9);
        assert!((r.per_caller["train"] - 20.0).abs() < 1e-9);
        assert_eq!(r.idle, 0.0);
    }
}
