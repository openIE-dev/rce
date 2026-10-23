//! Native builders: the workspace services into graph3d data.
use crate::dash::SourceRow;
use crate::graph::{self, DefIn, Graph, Kind, Live};
use crate::overlay::Action;
use crate::time::FactIn;
use ide_energy::discover::{Probe, Status};
use ide_energy::meter::Meter;
use ide_receipt::Receipt;
use rce_memory::fact::Fact;
use rce_memory::form::FormIndex;
use std::time::{Duration, Instant};

/// The code graph of a Form version, from rce-memory's index (defs, refs,
/// calls, the checker's verdicts) plus data-flow edges by item kind.
pub fn code_graph(ix: &FormIndex) -> Graph {
    let defs: Vec<DefIn> = ix
        .defs
        .iter()
        .enumerate()
        .map(|(i, d)| DefIn {
            name: &d.name,
            kind: &d.kind,
            file: &d.file,
            line: d.line,
            rank: ix.rank.get(i).copied().unwrap_or(0.0),
            note: &d.note,
        })
        .collect();
    let refs: Vec<(usize, usize, bool)> = ix.refs.iter().map(|r| (r.from, r.to, r.call)).collect();
    graph::from_defs(&defs, &refs, &ix.verdicts, &ix.form_hash)
}

pub fn code_graph_of(program: &[(String, String)]) -> Graph {
    code_graph(&FormIndex::build(program))
}

fn strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| strings(x, out)),
        _ => {}
    }
}

/// One action per receipt. Names come from every string in the receipt's
/// detail (paths, edit text, item names); `device`/`rail` from the detail.
pub fn actions(rs: &[Receipt]) -> Vec<Action> {
    rs.iter()
        .map(|r| {
            let d = &r.body.draft;
            let mut names = vec![];
            strings(&d.detail, &mut names);
            let device = ["device", "rail"]
                .iter()
                .find_map(|k| d.detail.get(*k).and_then(|v| v.as_str()).map(String::from));
            Action {
                seq: r.body.seq,
                t_ms: (d.t_end_ns / 1_000_000) as i64,
                action: serde_json::to_value(&d.action)
                    .ok()
                    .map(|v| {
                        v.as_str()
                            .map(String::from)
                            .unwrap_or_else(|| v.to_string())
                    })
                    .unwrap_or_default(),
                e_sim_j: d.e_sim_j,
                e_meas_j: d.e_meas.map(|m| m.joules),
                source: d
                    .e_meas
                    .and_then(|m| serde_json::to_value(m.source).ok())
                    .and_then(|v| v.as_str().map(String::from)),
                delta_j: r.body.delta_j,
                device,
                names,
                synthetic: false,
            }
        })
        .collect()
}

pub fn facts(fs: &[Fact]) -> Vec<FactIn> {
    fs.iter()
        .map(|f| FactIn {
            subject: f.subject.clone(),
            predicate: f.predicate.clone(),
            object: f.object.clone(),
            valid_from: f.valid_from,
            valid_to: f.valid_to,
            recorded_at: f.recorded_at,
            expired_at: f.expired_at,
        })
        .collect()
}

pub fn status_str(s: &Status) -> String {
    match s {
        Status::Available => "available".into(),
        Status::NeedsSetup(m) => format!("needs setup: {m}"),
        Status::Unsupported(m) => format!("unsupported: {m}"),
        Status::Stale(m) => format!("stale: {m}"),
    }
}

pub fn row(p: &Probe) -> SourceRow {
    SourceRow {
        tag: p.tag.into(),
        domain: p.domain.clone(),
        detail: p.detail.clone(),
        status: status_str(&p.status),
        resolution: p.resolution.into(),
        accuracy: p.accuracy.into(),
        series: vec![],
        joules: None,
        synthetic: false,
    }
}

/// Live sampler over the sources this build can read in-process (RAPL,
/// hwmon, battery on Linux; SMC system power on macOS). Rows for every
/// probed source; only opened ones get samples.
type Open = (usize, Box<dyn Meter + Send>, Option<(Instant, f64)>);

pub struct Sampler {
    pub rows: Vec<SourceRow>,
    meters: Vec<Open>,
    t0: Instant,
    cpu: Option<(u64, u64)>,
    pub cpu_util: Option<f64>,
}

impl Sampler {
    pub fn open() -> Sampler {
        let probes = ide_energy::discover::probe_all();
        let mut rows: Vec<SourceRow> = probes.iter().map(row).collect();
        let mut meters: Vec<Open> = vec![];
        for (i, p) in probes.iter().enumerate() {
            if p.status != Status::Available {
                continue;
            }
            let m: Option<Box<dyn Meter + Send>> = match p.tag {
                "rapl" => ide_energy::meter::RaplMeter::open()
                    .ok()
                    .map(|m| Box::new(m) as _),
                "hwmon" | "jetson" => {
                    ide_energy::hwmon::scan(std::path::Path::new("/sys/class/hwmon"))
                        .into_iter()
                        .find(|d| p.detail.starts_with(&d.dir.display().to_string()))
                        .and_then(|d| ide_energy::hwmon::open(&d, Duration::from_millis(100)).ok())
                }
                "power-supply" => ide_energy::supply::open(
                    std::path::Path::new(&p.detail),
                    Duration::from_millis(100),
                )
                .ok()
                .map(|m| Box::new(m) as _),
                #[cfg(target_os = "macos")]
                "smc" => ide_energy::smc::SmcMeter::open(Duration::from_millis(100))
                    .ok()
                    .map(|m| Box::new(m) as _),
                _ => None,
            };
            match m {
                Some(m) => meters.push((i, m, None)),
                None => {
                    rows[i].status = "available (probe only; not sampled in the dashboard)".into();
                }
            }
        }
        Sampler {
            rows,
            meters,
            t0: Instant::now(),
            cpu: None,
            cpu_util: None,
        }
    }
    /// Take one sample from every open meter: W from the joule delta.
    pub fn tick(&mut self) {
        let now = Instant::now();
        let t_ms = now.duration_since(self.t0).as_millis() as i64;
        for (i, m, last) in self.meters.iter_mut() {
            match m.sample() {
                Ok(s) => {
                    if let Some((t, j)) = *last {
                        let dt = now.duration_since(t).as_secs_f64();
                        if dt > 0.0 {
                            let r = &mut self.rows[*i];
                            r.series.push((t_ms, (s.joules - j) / dt));
                            if r.series.len() > 60 {
                                r.series.remove(0);
                            }
                            r.joules = crate::dash::integrate(&r.series);
                        }
                    }
                    *last = Some((now, s.joules));
                }
                Err(e) => self.rows[*i].status = status_str(&ide_energy::discover::status_of(&e)),
            }
        }
        if let Some((busy, total)) = proc_stat() {
            if let Some((b0, t0)) = self.cpu
                && total > t0
            {
                self.cpu_util = Some((busy - b0) as f64 / (total - t0) as f64);
            }
            self.cpu = Some((busy, total));
        }
    }
}

/// Aggregate (busy, total) jiffies from /proc/stat (Linux).
pub fn proc_stat() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/stat").ok()?;
    let l = s.lines().next()?;
    let v: Vec<u64> = l
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let total: u64 = v.iter().sum();
    let idle = v.get(3)? + v.get(4).unwrap_or(&0);
    Some((total - idle, total))
}

/// Attach live telemetry to hardware nodes by domain: cpu -> packages,
/// gpu -> GPUs, system/battery -> the system node. Nodes with no source
/// get `no source` (never a zero).
pub fn attach_live(g: &mut Graph, rows: &[SourceRow], cpu_util: Option<f64>) {
    for n in g.nodes.iter_mut().filter(|n| n.kind.is_hw()) {
        let want = match n.kind {
            Kind::Package => "cpu",
            Kind::Gpu => "gpu",
            Kind::System => "system",
            Kind::Memory => "dram",
            Kind::Npu => "ane",
            _ => {
                continue;
            }
        };
        let r = rows
            .iter()
            .find(|r| r.domain == want && r.available())
            .or_else(|| rows.iter().find(|r| r.domain == want));
        let mut live = match r {
            Some(r) => Live {
                watts: r.watts(),
                joules: r.joules,
                util: None,
                source: Some(r.tag.clone()),
                accuracy: Some(r.accuracy.clone()),
                status: r.status.clone(),
                synthetic: r.synthetic,
            },
            None => Live {
                status: "no source".into(),
                ..Default::default()
            },
        };
        if n.kind == Kind::Package {
            live.util = cpu_util;
        }
        n.live = Some(live);
    }
}
