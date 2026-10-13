//! Hardware topology, discovered at runtime and never invented.
//!
//! Linux: `/sys/devices/system/cpu` (packages, clusters, cores, caches),
//! `/sys/devices/system/node` (NUMA memory), `/proc/meminfo`,
//! `/sys/bus/pci/devices` (GPUs, accelerators/NPUs, storage and network
//! controllers, with PCIe link speed and width), `/sys/block`,
//! `/sys/class/net`. macOS: `sysctl` (brand, performance levels, caches,
//! memory); Apple Silicon parts that sysctl does not enumerate (GPU, ANE,
//! unified memory fabric) are added as `declared` with no numbers.
//! Anything not found is absent.
use crate::graph::{EdgeKind, Graph, Kind, Node};
use std::collections::BTreeMap;
use std::path::Path;

fn hw(id: String, label: String, kind: Kind, detail: String) -> Node {
    Node {
        id,
        label,
        kind,
        file: None,
        line: None,
        refused: false,
        rank: 0.0,
        energy: Default::default(),
        live: None,
        detail,
    }
}

fn rd(p: &Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Parse a cpulist like `0-3,8,10-11`.
pub fn cpulist(s: &str) -> Vec<u32> {
    let mut v = vec![];
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<u32>(), b.parse::<u32>()) {
                    v.extend(a..=b)
                }
            }
            None => {
                if let Ok(a) = part.parse() {
                    v.push(a)
                }
            }
        }
    }
    v
}

/// Discover on this machine. `RCE_HWLOC_XML=topo.xml` (from
/// `lstopo topo.xml`) is used instead when set and readable; an unreadable
/// or invalid file falls back to native discovery with a warning.
pub fn discover() -> Graph {
    if let Ok(p) = std::env::var("RCE_HWLOC_XML") {
        match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|x| crate::hwloc::import(&x).map_err(|e| e.0)) {
            Ok(g) => return g,
            Err(e) => eprintln!("RCE_HWLOC_XML {p}: {e}; using native discovery"),
        }
    }
    if cfg!(target_os = "macos") {
        return macos(&|k| {
            std::process::Command::new("sysctl")
                .args(["-n", k])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        });
    }
    linux(Path::new("/"))
}

/// Linux discovery under `root` (a fake tree in tests).
pub fn linux(root: &Path) -> Graph {
    let mut g = Graph::default();
    let host = rd(&root.join("proc/sys/kernel/hostname")).unwrap_or_else(|| "this machine".into());
    let sys = g.add(hw(
        "hw:system".into(),
        host,
        Kind::System,
        "linux sysfs".into(),
    ));
    let model = rd(&root.join("proc/cpuinfo")).and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("model name") || l.starts_with("Model"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
    });
    let cpu_root = root.join("sys/devices/system/cpu");
    let mut cpus: Vec<u32> = rd(&cpu_root.join("online"))
        .map(|s| cpulist(&s))
        .unwrap_or_default();
    cpus.sort();
    let mut caches: BTreeMap<String, usize> = BTreeMap::new();
    let mut pkg_of: BTreeMap<String, usize> = BTreeMap::new();
    for c in &cpus {
        let d = cpu_root.join(format!("cpu{c}"));
        let t = d.join("topology");
        let pkg = rd(&t.join("physical_package_id")).unwrap_or_else(|| "?".into());
        let core = rd(&t.join("core_id")).unwrap_or_else(|| c.to_string());
        let cluster = rd(&t.join("cluster_id"));
        let pid = format!("hw:pkg{pkg}");
        let p = match pkg_of.get(&pid) {
            Some(&p) => p,
            None => {
                let p = g.add(hw(
                    pid.clone(),
                    format!("CPU package {pkg}"),
                    Kind::Package,
                    model.clone().unwrap_or_default(),
                ));
                g.link(sys, p, EdgeKind::Contains, 1.0, "");
                pkg_of.insert(pid.clone(), p);
                p
            }
        };
        // A cluster level only when it differs from the package (Arm big.LITTLE, hybrid).
        let parent = match cluster.filter(|c| c != "-1" && c != &pkg) {
            Some(cl) => {
                let cid = format!("{pid}/cluster{cl}");
                let i = g.add(hw(
                    cid,
                    format!("cluster {cl}"),
                    Kind::Cluster,
                    String::new(),
                ));
                g.link(p, i, EdgeKind::Contains, 1.0, "");
                i
            }
            None => p,
        };
        let core_id = format!("{pid}/core{core}");
        let k = g.add(hw(
            core_id,
            format!("core {core}"),
            Kind::Core,
            String::new(),
        ));
        g.link(parent, k, EdgeKind::Contains, 1.0, "");
        let thr = g.nodes[k].detail.clone();
        g.nodes[k].detail = if thr.is_empty() {
            format!("cpu {c}")
        } else {
            format!("{thr},{c}")
        };
        // Caches: one node per distinct (level, type, shared set).
        let cd = d.join("cache");
        let mut idx: Vec<_> = std::fs::read_dir(&cd)
            .map(|r| {
                r.flatten()
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("index"))
                    .collect()
            })
            .unwrap_or_default();
        idx.sort();
        for ix in idx {
            let e = cd.join(&ix);
            let (Some(level), Some(ty)) = (rd(&e.join("level")), rd(&e.join("type"))) else {
                continue;
            };
            let size = rd(&e.join("size")).unwrap_or_else(|| "size ?".into());
            let shared = rd(&e.join("shared_cpu_list")).unwrap_or_else(|| c.to_string());
            let key = format!("hw:pkg{pkg}/L{level}{ty}/{shared}");
            let n = match caches.get(&key) {
                Some(&n) => n,
                None => {
                    let short = match ty.as_str() {
                        "Data" => "d",
                        "Instruction" => "i",
                        _ => "",
                    };
                    let n = g.add(hw(
                        key.clone(),
                        format!("L{level}{short} {size}"),
                        Kind::Cache,
                        format!("{ty}, shared by cpus {shared}"),
                    ));
                    caches.insert(key, n);
                    n
                }
            };
            g.link(k, n, EdgeKind::Contains, 1.0, "");
        }
    }
    // Memory: NUMA nodes, else one DRAM node from /proc/meminfo.
    let node_root = root.join("sys/devices/system/node");
    let mut numa: Vec<String> = std::fs::read_dir(&node_root)
        .map(|r| {
            r.flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.starts_with("node") && n[4..].parse::<u32>().is_ok())
                .collect()
        })
        .unwrap_or_default();
    numa.sort();
    let meminfo = |s: &str| {
        s.lines()
            .find(|l| l.contains("MemTotal"))
            .and_then(|l| l.split_whitespace().rev().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
    };
    let pkgs: Vec<usize> = pkg_of.values().copied().collect();
    if numa.is_empty()
        && let Some(kb) = rd(&root.join("proc/meminfo")).as_deref().and_then(meminfo)
    {
        let m = g.add(hw(
            "hw:mem0".into(),
            format!("DRAM {}", gib(kb)),
            Kind::Memory,
            "/proc/meminfo".into(),
        ));
        for &p in &pkgs {
            g.link(p, m, EdgeKind::Link, 1.0, "memory bus");
        }
    }
    for n in &numa {
        let d = node_root.join(n);
        let kb = rd(&d.join("meminfo")).as_deref().and_then(meminfo);
        let m = g.add(hw(
            format!("hw:{n}"),
            format!(
                "NUMA {n} {}",
                kb.map(gib).unwrap_or_else(|| "size ?".into())
            ),
            Kind::Memory,
            rd(&d.join("cpulist"))
                .map(|c| format!("cpus {c}"))
                .unwrap_or_default(),
        ));
        let local: Vec<u32> = rd(&d.join("cpulist"))
            .map(|c| cpulist(&c))
            .unwrap_or_default();
        for c in local {
            let pkg = rd(&cpu_root.join(format!("cpu{c}/topology/physical_package_id")));
            if let Some(&p) = pkg.and_then(|p| pkg_of.get(&format!("hw:pkg{p}"))) {
                g.link(p, m, EdgeKind::Link, 1.0, "memory bus");
            }
        }
    }
    // PCI devices of interest.
    let pci_root = root.join("sys/bus/pci/devices");
    let mut devs: Vec<String> = std::fs::read_dir(&pci_root)
        .map(|r| {
            r.flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    devs.sort();
    let mut pci_bus = None;
    for d in devs {
        let p = pci_root.join(&d);
        let Some(class) = rd(&p.join("class"))
            .and_then(|c| u32::from_str_radix(c.trim_start_matches("0x"), 16).ok())
        else {
            continue;
        };
        let kind = match class >> 16 {
            0x03 => Kind::Gpu,
            0x12 => Kind::Accel,
            0x01 => Kind::Storage,
            0x02 => Kind::Net,
            _ => continue,
        };
        let vendor = rd(&p.join("vendor")).unwrap_or_default();
        let device = rd(&p.join("device")).unwrap_or_default();
        let speed = rd(&p.join("current_link_speed"));
        let width = rd(&p.join("current_link_width"));
        let link = match (&speed, &width) {
            (Some(s), Some(w)) if !s.contains("Unknown") => format!("PCIe {s} x{w}"),
            _ => "PCI".into(),
        };
        let vname = match vendor.as_str() {
            "0x10de" => "NVIDIA",
            "0x1002" => "AMD",
            "0x8086" => "Intel",
            "0x1af4" => "virtio",
            "0x1d0f" => "Amazon",
            "0x1234" => "QEMU",
            _ => "",
        };
        let label = format!("{} {} {}", vname, kind.as_str(), d)
            .trim()
            .to_string();
        let n = g.add(hw(
            format!("hw:pci/{d}"),
            label,
            kind,
            format!("{vendor}:{device} class {class:06x}"),
        ));
        let bus = *pci_bus.get_or_insert_with(|| {
            let b = g.add(hw(
                "hw:pci".into(),
                "PCI root".into(),
                Kind::Bus,
                String::new(),
            ));
            for &p in &pkgs {
                g.link(p, b, EdgeKind::Link, 1.0, "root complex");
            }
            if pkgs.is_empty() {
                g.link(sys, b, EdgeKind::Contains, 1.0, "");
            }
            b
        });
        g.link(bus, n, EdgeKind::Link, 1.0, &link);
    }
    // Block devices and network interfaces (not loop/ram/lo/virtual bridges).
    for (dir, kind, skip) in [
        (
            "sys/block",
            Kind::Storage,
            &["loop", "ram", "zram", "dm-"][..],
        ),
        (
            "sys/class/net",
            Kind::Net,
            &["lo", "docker", "veth", "br-", "virbr"][..],
        ),
    ] {
        let mut names: Vec<String> = std::fs::read_dir(root.join(dir))
            .map(|r| {
                r.flatten()
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        for nm in names
            .into_iter()
            .filter(|n| !skip.iter().any(|s| n.starts_with(s)))
        {
            let p = root.join(dir).join(&nm);
            let detail = if kind == Kind::Storage {
                let sectors = rd(&p.join("size")).and_then(|s| s.parse::<u64>().ok());
                let rot = rd(&p.join("queue/rotational"));
                format!(
                    "{}{}",
                    sectors.map(|s| gib(s / 2)).unwrap_or_default(),
                    match rot.as_deref() {
                        Some("0") => ", non-rotational",
                        Some("1") => ", rotational",
                        _ => "",
                    }
                )
            } else {
                match rd(&p.join("speed")).and_then(|s| s.parse::<i64>().ok()) {
                    Some(s) if s > 0 => format!("{s} Mb/s"),
                    _ => "speed not reported".into(),
                }
            };
            let n = g.add(hw(
                format!("hw:{dir}/{nm}"),
                nm.clone(),
                kind,
                detail.clone(),
            ));
            // Attach to its PCI function when the sysfs device link resolves to one.
            let parent = std::fs::canonicalize(p.join("device")).ok().and_then(|t| {
                t.ancestors()
                    .filter_map(|a| a.file_name().map(|f| f.to_string_lossy().to_string()))
                    .find_map(|f| g.index(&format!("hw:pci/{f}")))
            });
            let label = if kind == Kind::Net {
                detail
            } else {
                String::new()
            };
            match parent.or(pci_bus) {
                Some(par) => g.link(par, n, EdgeKind::Link, 1.0, &label),
                None => g.link(sys, n, EdgeKind::Link, 1.0, &label),
            }
        }
    }
    g
}

fn gib(kb: u64) -> String {
    let g = kb as f64 / 1024.0 / 1024.0;
    if g >= 1.0 {
        format!("{g:.1} GiB")
    } else {
        format!("{:.0} MiB", kb as f64 / 1024.0)
    }
}

/// macOS discovery from sysctl values (`get` returns `None` when absent).
pub fn macos(get: &dyn Fn(&str) -> Option<String>) -> Graph {
    let mut g = Graph::default();
    let model = get("hw.model").unwrap_or_else(|| "Mac".into());
    let sys = g.add(hw(
        "hw:system".into(),
        model,
        Kind::System,
        "macOS sysctl".into(),
    ));
    let brand = get("machdep.cpu.brand_string").unwrap_or_default();
    let p = g.add(hw(
        "hw:pkg0".into(),
        if brand.is_empty() {
            "CPU".into()
        } else {
            brand.clone()
        },
        Kind::Package,
        String::new(),
    ));
    g.link(sys, p, EdgeKind::Contains, 1.0, "");
    let levels = get("hw.nperflevels")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    for l in 0..levels {
        let name = get(&format!("hw.perflevel{l}.name")).unwrap_or_else(|| format!("level {l}"));
        let ncpu = get(&format!("hw.perflevel{l}.physicalcpu")).and_then(|s| s.parse::<u32>().ok());
        let c = g.add(hw(
            format!("hw:pkg0/cluster{l}"),
            format!(
                "{name} cluster{}",
                ncpu.map(|n| format!(" ({n} cores)")).unwrap_or_default()
            ),
            Kind::Cluster,
            String::new(),
        ));
        g.link(p, c, EdgeKind::Contains, 1.0, "");
        for (key, lab) in [("l2cachesize", "L2"), ("l1dcachesize", "L1d")] {
            if let Some(b) =
                get(&format!("hw.perflevel{l}.{key}")).and_then(|s| s.parse::<u64>().ok())
            {
                let n = g.add(hw(
                    format!("hw:pkg0/cluster{l}/{lab}"),
                    format!("{lab} {}", gib(b / 1024)),
                    Kind::Cache,
                    format!("perflevel{l}"),
                ));
                g.link(c, n, EdgeKind::Contains, 1.0, "");
            }
        }
        if let Some(n) = ncpu {
            for k in 0..n {
                let core = g.add(hw(
                    format!("hw:pkg0/cluster{l}/core{k}"),
                    format!("{name} core {k}"),
                    Kind::Core,
                    String::new(),
                ));
                g.link(c, core, EdgeKind::Contains, 1.0, "");
            }
        }
    }
    let apple = brand.starts_with("Apple");
    if let Some(b) = get("hw.memsize").and_then(|s| s.parse::<u64>().ok()) {
        let m = g.add(hw(
            "hw:mem0".into(),
            format!(
                "{} {}",
                if apple { "unified memory" } else { "DRAM" },
                gib(b / 1024)
            ),
            Kind::Memory,
            "hw.memsize".into(),
        ));
        g.link(
            p,
            m,
            EdgeKind::Link,
            1.0,
            if apple { "fabric" } else { "memory bus" },
        );
        if apple {
            // Present on every Apple Silicon SoC; sysctl does not enumerate them.
            for (id, label, kind) in [
                ("hw:gpu0", "Apple GPU", Kind::Gpu),
                ("hw:ane0", "Apple Neural Engine", Kind::Npu),
            ] {
                let n = g.add(hw(
                    id.into(),
                    label.into(),
                    kind,
                    "declared (Apple Silicon), not probed".into(),
                ));
                g.link(sys, n, EdgeKind::Contains, 1.0, "");
                g.link(n, m, EdgeKind::Link, 1.0, "fabric");
            }
        }
    }
    g
}
