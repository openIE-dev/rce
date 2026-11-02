//! macOS IOReport energy counters (Apple silicon), no root needed.
//!
//! IOReport is a private framework (`/usr/lib/libIOReport.dylib`) with no
//! SDK stub, so it is opened by name at run time (`dlopen`, `dlsym`); a
//! missing library or symbol makes every IOReport meter `Unsupported`, and
//! the binary still starts. CoreFoundation and IOKit are linked as usual.
//! Two groups carry energy:
//!
//! - `Energy Model`: per-rail counters (CPU, GPU, ANE, DRAM). On macOS 27
//!   (measured on an M5 Max, 27.0.1) its CPU, ANE and DRAM channels stay at
//!   0 under full load; only `GPU Energy` moves.
//! - `CLPC` / `Energy Counters`: the AppleCLPC aggregate counters for CPU,
//!   GPU and ANE (nJ). They are not listed by `IOReportCopyAllChannels`; the
//!   channel descriptors are built from the driver's registry id and report
//!   ids per CLPC driver and macOS major. The id table and the descriptor
//!   layout come from macmon (vladkens/macmon, MIT License, Copyright (c)
//!   2024 vladkens), `src/clpc.rs` and `src/sources.rs` at afdd572; this is
//!   a re-implementation, not a dependency.
//!
//! A rail is read from CLPC when it is available, else from the Energy
//! Model. There is no package channel: `IoReportMeter` reports one rail,
//! and `IoReportPackageMeter` reports the sum of the domains in [`PACKAGE`]
//! (CPU, GPU, ANE), naming each channel it sums ([`select_package`],
//! [`package_joules`]). Channels with an unknown unit are
//! skipped, never scaled by a guess. A CPU or DRAM counter that does not move
//! over a window is an error (`MeterError::Stale`), never 0 J: those rails
//! always draw power while the machine runs.
//!
//! The unit, rail and staleness logic is plain Rust and tested on every
//! platform; the FFI compiles only on macOS.

/// Joules from an IOReport integer and its unit label. Unknown units: `None`.
pub fn joules_from_unit(raw: i64, unit: &str) -> Option<f64> {
    if raw < 0 {
        return None;
    }
    let v = raw as f64;
    let j = match unit.trim() {
        "J" => v,
        "mJ" => v / 1e3,
        "uJ" | "µJ" | "μJ" => v / 1e6,
        "nJ" => v / 1e9,
        _ => return None,
    };
    j.is_finite().then_some(j)
}

/// One energy rail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rail {
    Cpu,
    Gpu,
    Ane,
    Dram,
}

impl Rail {
    pub fn name(self) -> &'static str {
        match self {
            Rail::Cpu => "cpu",
            Rail::Gpu => "gpu",
            Rail::Ane => "ane",
            Rail::Dram => "dram",
        }
    }
    /// Rails that draw power whenever the machine runs: a counter that does
    /// not move is broken, not idle.
    pub fn always_draws(self) -> bool {
        matches!(self, Rail::Cpu | Rail::Dram)
    }
}

/// One channel of a delta sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub group: String,
    pub name: String,
    pub raw: i64,
    pub unit: String,
}

impl Row {
    pub fn new(group: &str, name: &str, raw: i64, unit: &str) -> Row {
        Row {
            group: group.into(),
            name: name.into(),
            raw,
            unit: unit.into(),
        }
    }
}

/// Which rail an Energy Model channel name belongs to: "CPU Energy" totals
/// (and cluster or die channels ending in it) are CPU; "GPU*" without SRAM
/// is GPU; ANE*, DRAM*.
pub fn rail_of(name: &str) -> Option<Rail> {
    let n = name.trim();
    if n.ends_with("CPU Energy") {
        Some(Rail::Cpu)
    } else if n.starts_with("GPU") && !n.contains("SRAM") {
        Some(Rail::Gpu)
    } else if n.starts_with("ANE") {
        Some(Rail::Ane)
    } else if n.starts_with("DRAM") {
        Some(Rail::Dram)
    } else {
        None
    }
}

/// The CLPC channel name of a rail (CLPC has no DRAM counter).
pub fn clpc_name(rail: Rail) -> Option<&'static str> {
    match rail {
        Rail::Cpu => Some("CPU Energy"),
        Rail::Gpu => Some("GPU Energy"),
        Rail::Ane => Some("ANE"),
        Rail::Dram => None,
    }
}

/// Where a rail reading came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Clpc,
    EnergyModel,
}

/// The reading of one rail in one delta.
#[derive(Debug, Clone, PartialEq)]
pub enum RailReading {
    /// Joules, and the group they came from.
    Joules(f64, Origin),
    /// Channels exist with known units but none moved.
    NoChange(Origin),
    /// No channel of this rail with a known unit.
    Missing,
}

fn energy_model(rows: &[Row], rail: Rail) -> Option<f64> {
    let known: Vec<(&str, f64)> = rows
        .iter()
        .filter(|r| r.group == "Energy Model" && rail_of(&r.name) == Some(rail))
        .filter_map(|r| joules_from_unit(r.raw, &r.unit).map(|j| (r.name.trim(), j)))
        .collect();
    if known.is_empty() {
        return None;
    }
    // A total channel, when present, already covers the clusters and other
    // channels of that rail.
    let total = match rail {
        Rail::Gpu => Some("GPU Energy"),
        Rail::Cpu => Some("CPU Energy"),
        _ => None,
    };
    if let Some(t) = total
        && let Some((_, j)) = known.iter().find(|(n, _)| *n == t)
    {
        return Some(*j);
    }
    // Ultra: one "DIE_n_CPU Energy" per die, summed.
    if rail == Rail::Cpu && known.iter().any(|(n, _)| n.starts_with("DIE_")) {
        return Some(
            known
                .iter()
                .filter(|(n, _)| n.starts_with("DIE_"))
                .map(|(_, j)| j)
                .sum(),
        );
    }
    Some(known.iter().map(|(_, j)| j).sum())
}

fn clpc(rows: &[Row], rail: Rail) -> Option<f64> {
    let name = clpc_name(rail)?;
    let js: Vec<f64> = rows
        .iter()
        .filter(|r| r.group == "CLPC" && r.name == name)
        .filter_map(|r| joules_from_unit(r.raw, &r.unit))
        .collect();
    // Ultra: one CLPC driver per die, each with its own aggregate.
    (!js.is_empty()).then(|| js.iter().sum())
}

/// The reading of `rail` from one delta: CLPC first, then the Energy Model.
/// A source whose counters did not move falls through to the next one.
pub fn read_rail(rows: &[Row], rail: Rail) -> RailReading {
    let mut seen = None;
    for (o, j) in [
        (Origin::Clpc, clpc(rows, rail)),
        (Origin::EnergyModel, energy_model(rows, rail)),
    ] {
        match j {
            Some(j) if j > 0.0 => return RailReading::Joules(j, o),
            Some(_) => seen = seen.or(Some(o)),
            None => {}
        }
    }
    seen.map_or(RailReading::Missing, RailReading::NoChange)
}

/// Joules of `rail` in one delta, or `None` (missing, or a rail that always
/// draws power showing no change). A GPU or ANE that did not move is 0 J:
/// those rails can be idle.
pub fn rail_joules(rows: &[Row], rail: Rail) -> Option<f64> {
    match read_rail(rows, rail) {
        RailReading::Joules(j, _) => Some(j),
        RailReading::NoChange(_) if !rail.always_draws() => Some(0.0),
        _ => None,
    }
}

/// AppleCLPC report ids for macOS 27 (from macmon `src/clpc.rs`, MIT).
/// (bundle id, CPU key, GPU key, ANE key); the full id is index << 32 | key.
const CLPC_KEYS_27: &[(&str, u32, u32, u32)] = &[
    (
        "com.apple.driver.AppleT6000CLPCv3",
        0x4747_9059,
        0x4519_bf1e,
        0x9aaa_17f8,
    ),
    (
        "com.apple.driver.AppleT6002CLPC",
        0xb501_816b,
        0x9e2c_3e8b,
        0x14d5_a574,
    ),
    (
        "com.apple.driver.AppleT6020CLPC",
        0xe301_cd75,
        0x8f34_cc7a,
        0x7415_fb0f,
    ),
    (
        "com.apple.driver.AppleT6022CLPC",
        0x76f5_fd21,
        0x9062_4499,
        0x3c09_7307,
    ),
    (
        "com.apple.driver.AppleT6030CLPC",
        0xbe03_a01a,
        0x472d_6c9c,
        0xb899_0c38,
    ),
    (
        "com.apple.driver.AppleT6031CLPC",
        0xd81e_0084,
        0x1085_f98b,
        0x313a_3be1,
    ),
    (
        "com.apple.driver.AppleT6032CLPC",
        0xf4a2_1308,
        0x85e8_0513,
        0x63dc_8e25,
    ),
    (
        "com.apple.driver.AppleT6041CLPC",
        0x263c_f1f0,
        0x22e4_cf59,
        0x7849_fb5c,
    ),
    (
        "com.apple.driver.AppleT6050CLPC",
        0xd3ab_60bb,
        0x27cd_4ce5,
        0x8c0f_1c59,
    ),
    (
        "com.apple.driver.AppleT6050dCLPC",
        0x18b3_010e,
        0xedec_a11a,
        0x188b_d883,
    ),
    (
        "com.apple.driver.AppleT8103CLPCv3",
        0x2cdd_31f2,
        0x21be_cf42,
        0x0667_fb81,
    ),
    (
        "com.apple.driver.AppleT8112CLPC",
        0xb543_5137,
        0x638d_9d52,
        0xea08_9c36,
    ),
    (
        "com.apple.driver.AppleT8122CLPC",
        0x54b5_1c99,
        0xce2d_d9dc,
        0x960a_fe2f,
    ),
    (
        "com.apple.driver.AppleT8132CLPC",
        0x4a85_4d94,
        0x3b3e_3b79,
        0x1ff4_a9ec,
    ),
    (
        "com.apple.driver.AppleT8140CLPC",
        0x9bf5_4436,
        0xef70_589c,
        0x569d_941b,
    ),
    (
        "com.apple.driver.AppleT8142CLPC",
        0x4370_8744,
        0x6a01_0933,
        0xbc5e_fdf0,
    ),
    (
        "com.apple.driver.AppleT8152CLPC",
        0x485d_4bfc,
        0x1178_4654,
        0x9380_c6bb,
    ),
];

/// CLPC (report id, channel name) for a driver bundle on a macOS version;
/// `None` for unknown drivers and for macOS majors other than 27.
pub fn clpc_channels(bundle: &str, os_version: &str) -> Option<[(u64, &'static str); 3]> {
    if os_version.split('.').next() != Some("27") {
        return None;
    }
    let &(_, cpu, gpu, ane) = CLPC_KEYS_27.iter().find(|k| k.0 == bundle)?;
    let id = |i: u64, k: u32| (i << 32) | k as u64;
    Some([
        (id(16, cpu), "CPU Energy"),
        (id(25, gpu), "GPU Energy"),
        (id(24, ane), "ANE"),
    ])
}

/// The domains a package reading sums: CPU, GPU and ANE, the compute
/// domains of the SoC (the sum asitop and macmon print as package power).
/// DRAM is outside the boundary, as it is outside a RAPL `package` zone,
/// and on the M5 Max (macOS 27) it has no moving counter at all.
pub const PACKAGE: [Rail; 3] = [Rail::Cpu, Rail::Gpu, Rail::Ane];

/// One channel a package reading sums: every row of a delta with this
/// group and name (an Ultra has one CLPC driver per die, with the same
/// channel name on each), converted by the unit each row carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub rail: Rail,
    pub group: String,
    pub name: String,
    /// The unit label read from the channel when it was chosen.
    pub unit: String,
    /// How many rows of the probe delta carried this group and name.
    pub rows: usize,
}

impl Channel {
    /// `cpu: CLPC "CPU Energy" nJ`, with ` x2` when two rows are summed.
    pub fn describe(&self) -> String {
        let n = if self.rows > 1 {
            format!(" x{}", self.rows)
        } else {
            String::new()
        };
        format!(
            "{}: {} \"{}\" {}{n}",
            self.rail.name(),
            self.group,
            self.name,
            self.unit
        )
    }
}

/// The Energy Model channels of `rail` with a known unit, by the rule of
/// [`read_rail`]: the total channel (`CPU Energy`, `GPU Energy`) when there
/// is one, else the per-die `DIE_n_CPU Energy` channels, else every channel
/// of the rail.
fn em_names(rows: &[Row], rail: Rail) -> Vec<&Row> {
    let known: Vec<&Row> = rows
        .iter()
        .filter(|r| r.group == "Energy Model" && rail_of(&r.name) == Some(rail))
        .filter(|r| joules_from_unit(0, &r.unit).is_some())
        .collect();
    let total = match rail {
        Rail::Gpu => Some("GPU Energy"),
        Rail::Cpu => Some("CPU Energy"),
        _ => None,
    };
    if let Some(t) = total
        && known.iter().any(|r| r.name.trim() == t)
    {
        return known.into_iter().filter(|r| r.name.trim() == t).collect();
    }
    if rail == Rail::Cpu && known.iter().any(|r| r.name.trim().starts_with("DIE_")) {
        return known
            .into_iter()
            .filter(|r| r.name.trim().starts_with("DIE_"))
            .collect();
    }
    known
}

fn clpc_rows(rows: &[Row], rail: Rail) -> Vec<&Row> {
    let Some(name) = clpc_name(rail) else {
        return vec![];
    };
    rows.iter()
        .filter(|r| r.group == "CLPC" && r.name == name)
        .filter(|r| joules_from_unit(0, &r.unit).is_some())
        .collect()
}

fn channels(rail: Rail, rows: &[&Row]) -> Vec<Channel> {
    let mut out: Vec<Channel> = vec![];
    for r in rows {
        match out
            .iter_mut()
            .find(|c| c.group == r.group && c.name == r.name)
        {
            Some(c) => c.rows += 1,
            None => out.push(Channel {
                rail,
                group: r.group.clone(),
                name: r.name.clone(),
                unit: r.unit.trim().to_string(),
                rows: 1,
            }),
        }
    }
    out
}

/// The channels a package reading sums, chosen from one probe delta. For
/// each domain of [`PACKAGE`]: the CLPC channel when it moved, else the
/// Energy Model channels when they moved; when neither moved, CLPC if it
/// exists, else the Energy Model (an idle GPU or ANE is 0 J). A channel
/// whose unit label is not J, mJ, uJ or nJ is never chosen. Errors: a
/// domain with no channel of a known unit, or a CPU whose counters did not
/// move (the CPU always draws power, so a still counter is broken).
pub fn select_package(rows: &[Row]) -> Result<Vec<Channel>, String> {
    let moved = |rs: &[&Row]| rs.iter().any(|r| r.raw > 0);
    let mut out = vec![];
    for rail in PACKAGE {
        let (c, e) = (clpc_rows(rows, rail), em_names(rows, rail));
        let pick = if moved(&c) {
            c
        } else if moved(&e) {
            e
        } else if rail.always_draws() && !(c.is_empty() && e.is_empty()) {
            return Err(format!("{} counters did not move", rail.name()));
        } else if !c.is_empty() {
            c
        } else {
            e
        };
        if pick.is_empty() {
            return Err(format!("no {} channel with a known unit", rail.name()));
        }
        out.extend(channels(rail, &pick));
    }
    Ok(out)
}

/// Joules of one delta per domain of [`PACKAGE`], summing the chosen
/// channels, each row converted by its own unit label. Errors: a chosen
/// channel absent from the delta, a unit that is not J, mJ, uJ or nJ, or a
/// negative delta (a counter that went back: a reset, never counted as a
/// wraparound; the counters are 64-bit and do not reach their maximum).
pub fn package_joules(rows: &[Row], sel: &[Channel]) -> Result<[f64; 3], String> {
    let mut out = [0.0; 3];
    for c in sel {
        let mut seen = false;
        for r in rows
            .iter()
            .filter(|r| r.group == c.group && r.name == c.name)
        {
            seen = true;
            if r.raw < 0 {
                return Err(format!(
                    "{} \"{}\" went back by {} {}",
                    c.group,
                    c.name,
                    -(r.raw as i128),
                    r.unit
                ));
            }
            let j = joules_from_unit(r.raw, &r.unit).ok_or_else(|| {
                format!(
                    "{} \"{}\" has unit `{}`, not J, mJ, uJ or nJ",
                    c.group, c.name, r.unit
                )
            })?;
            let i = PACKAGE
                .iter()
                .position(|x| *x == c.rail)
                .expect("a package rail");
            out[i] += j;
        }
        if !seen {
            return Err(format!(
                "{} \"{}\" is missing from the sample",
                c.group, c.name
            ));
        }
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
pub use mac::{IoReportMeter, IoReportPackageMeter, os_version, sysctl};

#[cfg(target_os = "macos")]
mod mac {
    use super::{
        Channel, Origin, PACKAGE, Rail, RailReading, Row, clpc_channels, package_joules, read_rail,
        select_package,
    };
    use crate::attrib::Sample;
    use crate::meter::{Meter, MeterError, now_ns};
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::ptr;
    use std::sync::OnceLock;

    const UTF8: u32 = 0x0800_0100;
    const SINT64: isize = 4;
    type Cf = *const c_void;
    #[repr(C)]
    struct Callbacks([u8; 0]);

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        static kCFTypeArrayCallBacks: Callbacks;
        static kCFTypeDictionaryKeyCallBacks: Callbacks;
        static kCFTypeDictionaryValueCallBacks: Callbacks;
        fn CFStringCreateWithCString(alloc: Cf, s: *const c_char, enc: u32) -> Cf;
        fn CFStringGetCString(s: Cf, buf: *mut c_char, size: isize, enc: u32) -> u8;
        fn CFRelease(cf: Cf);
        fn CFDictionaryGetValue(d: Cf, k: Cf) -> Cf;
        fn CFDictionaryCreate(
            a: Cf,
            k: *const Cf,
            v: *const Cf,
            n: isize,
            kc: *const Callbacks,
            vc: *const Callbacks,
        ) -> Cf;
        fn CFDictionaryCreateMutableCopy(alloc: Cf, cap: isize, d: Cf) -> Cf;
        fn CFDictionarySetValue(d: Cf, k: Cf, v: Cf);
        fn CFArrayCreate(a: Cf, v: *const Cf, n: isize, cb: *const Callbacks) -> Cf;
        fn CFArrayCreateMutable(a: Cf, cap: isize, cb: *const Callbacks) -> Cf;
        fn CFArrayAppendValue(a: Cf, v: Cf);
        fn CFArrayGetCount(a: Cf) -> isize;
        fn CFArrayGetValueAtIndex(a: Cf, i: isize) -> Cf;
        fn CFNumberCreate(a: Cf, t: isize, v: *const c_void) -> Cf;
    }

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(n: *const c_char) -> Cf;
        fn IOServiceGetMatchingServices(port: u32, m: Cf, it: *mut u32) -> i32;
        fn IOIteratorNext(it: u32) -> u32;
        fn IOObjectRelease(o: u32) -> u32;
        fn IORegistryEntryGetRegistryEntryID(e: u32, id: *mut u64) -> i32;
        fn IORegistryEntryCreateCFProperty(e: u32, k: Cf, a: Cf, o: u32) -> Cf;
    }

    // libSystem: always linked.
    unsafe extern "C" {
        fn sysctlbyname(
            n: *const c_char,
            o: *mut c_void,
            ol: *mut usize,
            nn: *const c_void,
            nl: usize,
        ) -> i32;
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }
    const RTLD_LAZY: c_int = 0x1;
    const RTLD_LOCAL: c_int = 0x4;

    /// IOReport is a private framework with no SDK stub to link against, so
    /// it is opened by name at run time. Missing library or symbol: the
    /// meter is `Unsupported`, and the binary still starts.
    pub const LIBRARY: &CStr = c"/usr/lib/libIOReport.dylib";

    /// The IOReport functions this module calls, resolved by `dlsym`.
    struct Lib {
        copy_channels_in_group: unsafe extern "C" fn(Cf, Cf, u64, u64, u64) -> Cf,
        create_subscription: unsafe extern "C" fn(Cf, Cf, *mut Cf, u64, Cf) -> Cf,
        create_samples: unsafe extern "C" fn(Cf, Cf, Cf) -> Cf,
        create_samples_delta: unsafe extern "C" fn(Cf, Cf, Cf) -> Cf,
        channel_name: unsafe extern "C" fn(Cf) -> Cf,
        group: unsafe extern "C" fn(Cf) -> Cf,
        unit_label: unsafe extern "C" fn(Cf) -> Cf,
        integer_value: unsafe extern "C" fn(Cf, i32) -> i64,
    }

    fn dl_error() -> String {
        // SAFETY: dlerror returns null or a NUL-terminated thread-local string.
        let e = unsafe { dlerror() };
        if e.is_null() {
            "unknown error".into()
        } else {
            // SAFETY: e is non-null and NUL-terminated.
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    }

    /// The address of `name` in `h` as a function pointer of type `F`.
    ///
    /// # Safety
    /// `F` is a function pointer type matching the symbol's C signature.
    unsafe fn sym<F: Copy>(h: *mut c_void, name: &CStr) -> Result<F, String> {
        assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
        // SAFETY: h is a live dlopen handle; name is NUL-terminated.
        let p = unsafe { dlsym(h, name.as_ptr()) };
        if p.is_null() {
            return Err(format!(
                "{}: no symbol {} ({})",
                LIBRARY.to_string_lossy(),
                name.to_string_lossy(),
                dl_error()
            ));
        }
        // SAFETY: same size (asserted); the caller names the signature.
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
    }

    fn load() -> Result<Lib, String> {
        // SAFETY: LIBRARY is NUL-terminated; the handle is never closed.
        let h = unsafe { dlopen(LIBRARY.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
        if h.is_null() {
            return Err(format!(
                "dlopen {}: {}",
                LIBRARY.to_string_lossy(),
                dl_error()
            ));
        }
        // SAFETY: each type is the C signature of that IOReport function
        // (as declared by macmon and asitop's bindings).
        unsafe {
            Ok(Lib {
                copy_channels_in_group: sym(h, c"IOReportCopyChannelsInGroup")?,
                create_subscription: sym(h, c"IOReportCreateSubscription")?,
                create_samples: sym(h, c"IOReportCreateSamples")?,
                create_samples_delta: sym(h, c"IOReportCreateSamplesDelta")?,
                channel_name: sym(h, c"IOReportChannelGetChannelName")?,
                group: sym(h, c"IOReportChannelGetGroup")?,
                unit_label: sym(h, c"IOReportChannelGetUnitLabel")?,
                integer_value: sym(h, c"IOReportSimpleGetIntegerValue")?,
            })
        }
    }

    fn lib() -> Result<&'static Lib, MeterError> {
        static LIB: OnceLock<Result<Lib, String>> = OnceLock::new();
        LIB.get_or_init(load)
            .as_ref()
            .map_err(|e| MeterError::Unsupported("ioreport", e.clone()))
    }

    /// Owned CF object, released on drop.
    struct Own(Cf);
    impl Drop for Own {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: Own holds a +1 reference created by a CF Create/Copy call.
                unsafe { CFRelease(self.0) };
            }
        }
    }

    fn cfstr(s: &str) -> Own {
        let Ok(c) = CString::new(s) else {
            return Own(ptr::null());
        };
        // SAFETY: c is a valid NUL-terminated UTF-8 string.
        Own(unsafe { CFStringCreateWithCString(ptr::null(), c.as_ptr(), UTF8) })
    }

    fn num(v: i64) -> Own {
        // SAFETY: v outlives the call; kCFNumberSInt64Type matches i64.
        Own(unsafe { CFNumberCreate(ptr::null(), SINT64, &v as *const i64 as *const c_void) })
    }

    fn string(s: Cf) -> Option<String> {
        if s.is_null() {
            return None;
        }
        let mut buf = [0 as c_char; 256];
        // SAFETY: s is a CFString; buf is writable for its length.
        let ok = unsafe { CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as isize, UTF8) };
        // SAFETY: CFStringGetCString NUL-terminated buf on success.
        (ok != 0).then(|| {
            unsafe { CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
    }

    fn get(d: Cf, key: &str) -> Cf {
        let k = cfstr(key);
        if d.is_null() || k.0.is_null() {
            return ptr::null();
        }
        // SAFETY: d is a CFDictionary; the value is borrowed from it.
        unsafe { CFDictionaryGetValue(d, k.0) }
    }

    fn dict(kv: &[(&str, &Own)]) -> Own {
        let keys: Vec<Own> = kv.iter().map(|(k, _)| cfstr(k)).collect();
        let kp: Vec<Cf> = keys.iter().map(|k| k.0).collect();
        let vp: Vec<Cf> = kv.iter().map(|(_, v)| v.0).collect();
        // SAFETY: equal-length key and value arrays of live CF objects; the
        // CFType callbacks retain them.
        Own(unsafe {
            CFDictionaryCreate(
                ptr::null(),
                kp.as_ptr(),
                vp.as_ptr(),
                kp.len() as isize,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            )
        })
    }

    /// A CLPC energy channel descriptor (layout as in macmon
    /// `clpc_energy_channel`): simple integer, power category, one element, nJ.
    fn clpc_descriptor(driver: u64, id: u64, name: &str) -> Own {
        let ty = 1i64 | (2 << 16) | (1 << 32);
        let unit = (3i64 << 56) | (118 << 32);
        let (idn, tyn, nm) = (num(id as i64), num(ty), cfstr(name));
        let lg = [idn.0, tyn.0, nm.0];
        // SAFETY: three live CF objects, retained by the array.
        let legend =
            Own(unsafe { CFArrayCreate(ptr::null(), lg.as_ptr(), 3, &kCFTypeArrayCallBacks) });
        let unit_n = num(unit);
        let info = dict(&[("IOReportChannelUnit", &unit_n)]);
        let (drv, dn, gn, sgn) = (
            num(driver as i64),
            cfstr("AppleCLPC"),
            cfstr("CLPC"),
            cfstr("Energy Counters"),
        );
        let d = dict(&[
            ("DriverID", &drv),
            ("DriverName", &dn),
            ("IOReportGroupName", &gn),
            ("IOReportSubGroupName", &sgn),
            ("IOReportChannelInfo", &info),
            ("LegendChannel", &legend),
        ]);
        // IOReport's getters cache derived fields in the dictionary: mutable copy.
        // SAFETY: d is a live dictionary.
        Own(unsafe { CFDictionaryCreateMutableCopy(ptr::null(), 0, d.0) })
    }

    /// A string sysctl by name, such as `machdep.cpu.brand_string`,
    /// `hw.model`, `kern.osproductversion` (the version `sw_vers` prints)
    /// or `kern.osrelease`; `None` when it is absent or empty.
    pub fn sysctl(name: &str) -> Option<String> {
        let n = CString::new(name).ok()?;
        let mut len = 0usize;
        // SAFETY: a null buffer asks for the length only.
        let r = unsafe { sysctlbyname(n.as_ptr(), ptr::null_mut(), &mut len, ptr::null(), 0) };
        if r != 0 || len == 0 || len > 1 << 16 {
            return None;
        }
        let mut buf = vec![0u8; len];
        // SAFETY: buf is writable for len bytes.
        let r = unsafe {
            sysctlbyname(
                n.as_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
                ptr::null(),
                0,
            )
        };
        if r != 0 {
            return None;
        }
        buf.truncate(len);
        let s = String::from_utf8_lossy(&buf);
        let s = s.trim_end_matches('\0').trim();
        (!s.is_empty()).then(|| s.to_string())
    }

    /// `kern.osproductversion`, e.g. "27.0.1".
    pub fn os_version() -> Option<String> {
        sysctl("kern.osproductversion")
    }

    /// (registry id, bundle id) of each AppleCLPC driver.
    fn clpc_drivers() -> Vec<(u64, String)> {
        let mut out = vec![];
        let Ok(name) = CString::new("AppleCLPC") else {
            return out;
        };
        let mut it = 0u32;
        // SAFETY: IOServiceMatching's dictionary is consumed by the call.
        if unsafe { IOServiceGetMatchingServices(0, IOServiceMatching(name.as_ptr()), &mut it) }
            != 0
        {
            return out;
        }
        loop {
            // SAFETY: it is a live iterator.
            let e = unsafe { IOIteratorNext(it) };
            if e == 0 {
                break;
            }
            let mut id = 0u64;
            // SAFETY: e is a live registry entry, released below.
            let ok = unsafe { IORegistryEntryGetRegistryEntryID(e, &mut id) } == 0;
            let k = cfstr("CFBundleIdentifier");
            // SAFETY: the property is a +1 object, owned by Own.
            let b = Own(unsafe { IORegistryEntryCreateCFProperty(e, k.0, ptr::null(), 0) });
            if ok && let Some(b) = string(b.0) {
                out.push((id, b));
            }
            // SAFETY: e and (after the loop) it are released once.
            unsafe { IOObjectRelease(e) };
        }
        unsafe { IOObjectRelease(it) };
        out
    }

    fn rows(l: &Lib, delta: Cf) -> Vec<Row> {
        let ch = get(delta, "IOReportChannels");
        if ch.is_null() {
            return vec![];
        }
        let mut out = vec![];
        // SAFETY: ch is the CFArray of channel dictionaries inside delta;
        // the IOReport getters borrow from each item.
        for i in 0..unsafe { CFArrayGetCount(ch) } {
            let it = unsafe { CFArrayGetValueAtIndex(ch, i) };
            if it.is_null() {
                continue;
            }
            let (Some(g), Some(n), Some(u)) = (
                string(unsafe { (l.group)(it) }),
                string(unsafe { (l.channel_name)(it) }),
                string(unsafe { (l.unit_label)(it) }),
            ) else {
                continue;
            };
            if g != "Energy Model" && g != "CLPC" {
                continue;
            }
            let raw = unsafe { (l.integer_value)(it, 0) };
            out.push(Row {
                group: g,
                name: n,
                raw,
                unit: u.trim().to_string(),
            });
        }
        out
    }

    /// A subscription to the `Energy Model` group plus the CLPC counters of
    /// every known driver, and its last sample.
    struct Sub {
        lib: &'static Lib,
        chan: Own,
        sub: Own,
        last: Own,
        last_t: u64,
        clpc: usize,
        os: String,
    }

    impl Sub {
        fn open() -> Result<Sub, MeterError> {
            let u = |m: String| MeterError::Unsupported("ioreport", m);
            let l = lib()?;
            let g = cfstr("Energy Model");
            // SAFETY: every returned object is checked for null and owned once.
            let copied = Own(unsafe { (l.copy_channels_in_group)(g.0, ptr::null(), 0, 0, 0) });
            if copied.0.is_null() {
                return Err(u("no Energy Model group".into()));
            }
            let chan = Own(unsafe { CFDictionaryCreateMutableCopy(ptr::null(), 0, copied.0) });
            let sel = Own(unsafe { CFArrayCreateMutable(ptr::null(), 0, &kCFTypeArrayCallBacks) });
            if chan.0.is_null() || sel.0.is_null() {
                return Err(u("channel copy failed".into()));
            }
            let old = get(chan.0, "IOReportChannels");
            if !old.is_null() {
                for i in 0..unsafe { CFArrayGetCount(old) } {
                    unsafe { CFArrayAppendValue(sel.0 as _, CFArrayGetValueAtIndex(old, i)) };
                }
            }
            let os = os_version().unwrap_or_default();
            let mut clpc = 0;
            for (driver, bundle) in clpc_drivers() {
                if let Some(chs) = clpc_channels(&bundle, &os) {
                    for (id, name) in chs {
                        let d = clpc_descriptor(driver, id, name);
                        unsafe { CFArrayAppendValue(sel.0 as _, d.0) };
                    }
                    clpc += 1;
                }
            }
            if unsafe { CFArrayGetCount(sel.0) } == 0 {
                return Err(u("the Energy Model group has no channels".into()));
            }
            let k = cfstr("IOReportChannels");
            unsafe { CFDictionarySetValue(chan.0 as _, k.0, sel.0) };
            let mut subbed: Cf = ptr::null();
            let sub = Own(unsafe {
                (l.create_subscription)(ptr::null(), chan.0, &mut subbed, 0, ptr::null())
            });
            let subbed = Own(subbed);
            if sub.0.is_null() || subbed.0.is_null() {
                return Err(u("subscription failed".into()));
            }
            let last = Own(unsafe { (l.create_samples)(sub.0, subbed.0, ptr::null()) });
            if last.0.is_null() {
                return Err(u("first sample failed".into()));
            }
            Ok(Sub {
                lib: l,
                chan: subbed,
                sub,
                last,
                last_t: now_ns(),
                clpc,
                os,
            })
        }

        /// The rows of the delta since the last sample.
        fn delta(&mut self) -> Result<Vec<Row>, MeterError> {
            let l = self.lib;
            // SAFETY: sub and chan are live; samples are owned.
            let now = Own(unsafe { (l.create_samples)(self.sub.0, self.chan.0, ptr::null()) });
            if now.0.is_null() {
                return Err(MeterError::Read(
                    "ioreport",
                    "IOReportCreateSamples failed".into(),
                ));
            }
            let d = Own(unsafe { (l.create_samples_delta)(self.last.0, now.0, ptr::null()) });
            self.last = now;
            self.last_t = now_ns();
            if d.0.is_null() {
                return Err(MeterError::Read(
                    "ioreport",
                    "IOReportCreateSamplesDelta failed".into(),
                ));
            }
            Ok(rows(l, d.0))
        }
    }

    /// Cumulative joules of one rail since open.
    pub struct IoReportMeter {
        rail: Rail,
        s: Sub,
        total_j: f64,
        origin: Option<Origin>,
    }

    // The CF objects are only touched through &mut self.
    unsafe impl Send for IoReportMeter {}

    /// Windows shorter than this are too short to call a counter stale.
    const STALE_MIN_NS: u64 = 20_000_000;

    impl IoReportMeter {
        /// Subscribe to the Energy Model group plus the CLPC counters of
        /// every known driver. For a rail that always draws power, one
        /// 100 ms check runs first: if its counters do not move, the rail is
        /// `Unsupported` on this machine.
        pub fn open(rail: Rail) -> Result<Self, MeterError> {
            let s = Sub::open()?;
            let mut m = IoReportMeter {
                rail,
                s,
                total_j: 0.0,
                origin: None,
            };
            if rail.always_draws() {
                std::thread::sleep(std::time::Duration::from_millis(100));
                match m.delta() {
                    Ok(RailReading::Joules(_, o)) => m.origin = Some(o),
                    Ok(other) => {
                        return Err(MeterError::Unsupported(
                            "ioreport",
                            format!(
                                "{} counters did not move over 100 ms ({other:?}; macOS {}, \
                                 {} known CLPC driver(s))",
                                rail.name(),
                                m.s.os,
                                m.s.clpc
                            ),
                        ));
                    }
                    Err(e) => return Err(e),
                }
                m.total_j = 0.0;
            }
            Ok(m)
        }

        /// Which group the rail was last read from.
        pub fn origin(&self) -> Option<Origin> {
            self.origin
        }

        /// Number of AppleCLPC drivers with known report ids.
        pub fn clpc_drivers(&self) -> usize {
            self.s.clpc
        }

        fn delta(&mut self) -> Result<RailReading, MeterError> {
            Ok(read_rail(&self.s.delta()?, self.rail))
        }
    }

    impl Meter for IoReportMeter {
        fn source(&self) -> &'static str {
            "ioreport"
        }
        fn sample(&mut self) -> Result<Sample, MeterError> {
            let t0 = self.s.last_t;
            let r = self.delta()?;
            let t = self.s.last_t;
            match r {
                RailReading::Joules(j, o) => {
                    self.origin = Some(o);
                    self.total_j += j;
                }
                RailReading::NoChange(o) if self.rail.always_draws() => {
                    if t.saturating_sub(t0) >= STALE_MIN_NS {
                        return Err(MeterError::Stale(
                            "ioreport",
                            format!(
                                "{} counters ({o:?}) did not move over {} ms",
                                self.rail.name(),
                                (t - t0) / 1_000_000
                            ),
                        ));
                    }
                }
                RailReading::NoChange(_) => {}
                RailReading::Missing => {
                    return Err(MeterError::Read(
                        "ioreport",
                        format!("no {} channel with a known unit", self.rail.name()),
                    ));
                }
            }
            Ok(Sample {
                t_ns: t,
                joules: self.total_j,
            })
        }
    }

    /// Cumulative joules of the package boundary ([`PACKAGE`]: CPU + GPU +
    /// ANE) since open, from the channels [`select_package`] chose over a
    /// 100 ms probe. The IOReport values are 64-bit integer deltas
    /// (`IOReportCreateSamplesDelta`), which do not reach their maximum
    /// (2^63 nJ is about 9.2 GJ, 2.9 years at 100 W); a negative delta is
    /// a read error, never a wraparound, so [`IoReportPackageMeter::wraps`]
    /// is 0.
    pub struct IoReportPackageMeter {
        s: Sub,
        sel: Vec<Channel>,
        total: [f64; 3],
    }

    // The CF objects are only touched through &mut self.
    unsafe impl Send for IoReportPackageMeter {}

    impl IoReportPackageMeter {
        /// Subscribe, wait 100 ms, and choose the channels from that delta.
        /// No library, no group, no channel of a known unit for a domain, or
        /// a CPU counter that did not move: `Unsupported`, with the reason.
        pub fn open() -> Result<Self, MeterError> {
            let mut s = Sub::open()?;
            std::thread::sleep(std::time::Duration::from_millis(100));
            let rows = s.delta()?;
            let sel = select_package(&rows).map_err(|e| {
                MeterError::Unsupported(
                    "ioreport",
                    format!(
                        "{e} (over 100 ms; {} Energy Model and CLPC channels; macOS {}, {} known CLPC driver(s))",
                        rows.len(),
                        s.os,
                        s.clpc
                    ),
                )
            })?;
            Ok(IoReportPackageMeter {
                s,
                sel,
                total: [0.0; 3],
            })
        }

        /// The channels read, by domain.
        pub fn channels(&self) -> &[Channel] {
            &self.sel
        }

        /// Joules per domain of [`PACKAGE`] since open.
        pub fn totals(&self) -> [(Rail, f64); 3] {
            [0, 1, 2].map(|i| (PACKAGE[i], self.total[i]))
        }

        /// Wraparounds counted: 0, since a 64-bit delta does not wrap and
        /// a negative one is refused (see the type's documentation).
        pub fn wraps(&self) -> u64 {
            0
        }

        /// Number of AppleCLPC drivers with known report ids.
        pub fn clpc_drivers(&self) -> usize {
            self.s.clpc
        }
    }

    impl Meter for IoReportPackageMeter {
        fn source(&self) -> &'static str {
            "ioreport"
        }
        fn sample(&mut self) -> Result<Sample, MeterError> {
            let t0 = self.s.last_t;
            let rows = self.s.delta()?;
            let t = self.s.last_t;
            let j =
                package_joules(&rows, &self.sel).map_err(|e| MeterError::Read("ioreport", e))?;
            // The CPU always draws power: a still CPU counter over a window
            // long enough to see it move is broken, not idle.
            if j[0] <= 0.0 && t.saturating_sub(t0) >= STALE_MIN_NS {
                return Err(MeterError::Stale(
                    "ioreport",
                    format!("cpu counters did not move over {} ms", (t - t0) / 1_000_000),
                ));
            }
            for (a, b) in self.total.iter_mut().zip(j) {
                *a += b;
            }
            Ok(Sample {
                t_ns: t,
                joules: self.total.iter().sum(),
            })
        }
    }
}
