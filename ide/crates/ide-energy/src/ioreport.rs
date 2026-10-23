//! macOS IOReport energy counters (Apple silicon), no root needed.
//!
//! IOReport is a private framework (`/usr/lib/libIOReport.dylib`). Two
//! groups carry energy:
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
//! Model. There is no package channel, so this meter reports one rail and
//! never a sum presented as the package. Channels with an unknown unit are
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

#[cfg(target_os = "macos")]
pub use mac::{IoReportMeter, os_version};

#[cfg(target_os = "macos")]
mod mac {
    use super::{Origin, Rail, RailReading, Row, clpc_channels, read_rail};
    use crate::attrib::Sample;
    use crate::meter::{Meter, MeterError, now_ns};
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::ptr;

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

    #[link(name = "IOReport")]
    unsafe extern "C" {
        fn IOReportCopyChannelsInGroup(g: Cf, sg: Cf, a: u64, b: u64, c: u64) -> Cf;
        fn IOReportCreateSubscription(a: Cf, desired: Cf, subbed: *mut Cf, id: u64, b: Cf) -> Cf;
        fn IOReportCreateSamples(sub: Cf, chan: Cf, a: Cf) -> Cf;
        fn IOReportCreateSamplesDelta(prev: Cf, next: Cf, a: Cf) -> Cf;
        fn IOReportChannelGetChannelName(item: Cf) -> Cf;
        fn IOReportChannelGetGroup(item: Cf) -> Cf;
        fn IOReportChannelGetUnitLabel(item: Cf) -> Cf;
        fn IOReportSimpleGetIntegerValue(item: Cf, col: i32) -> i64;
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

    unsafe extern "C" {
        fn sysctlbyname(
            n: *const c_char,
            o: *mut c_void,
            ol: *mut usize,
            nn: *const c_void,
            nl: usize,
        ) -> i32;
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

    /// `kern.osproductversion`, e.g. "27.0.1".
    pub fn os_version() -> Option<String> {
        let n = CString::new("kern.osproductversion").ok()?;
        let mut buf = [0u8; 64];
        let mut len = buf.len();
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
        (r == 0 && len > 0).then(|| String::from_utf8_lossy(&buf[..len - 1]).into_owned())
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

    fn rows(delta: Cf) -> Vec<Row> {
        let ch = get(delta, "IOReportChannels");
        if ch.is_null() {
            return vec![];
        }
        let mut out = vec![];
        // SAFETY: ch is the CFArray of channel dictionaries inside delta.
        for i in 0..unsafe { CFArrayGetCount(ch) } {
            let it = unsafe { CFArrayGetValueAtIndex(ch, i) };
            if it.is_null() {
                continue;
            }
            let (Some(g), Some(n), Some(u)) = (
                string(unsafe { IOReportChannelGetGroup(it) }),
                string(unsafe { IOReportChannelGetChannelName(it) }),
                string(unsafe { IOReportChannelGetUnitLabel(it) }),
            ) else {
                continue;
            };
            if g != "Energy Model" && g != "CLPC" {
                continue;
            }
            let raw = unsafe { IOReportSimpleGetIntegerValue(it, 0) };
            out.push(Row {
                group: g,
                name: n,
                raw,
                unit: u.trim().to_string(),
            });
        }
        out
    }

    /// Cumulative joules of one rail since open.
    pub struct IoReportMeter {
        rail: Rail,
        chan: Own,
        sub: Own,
        last: Own,
        last_t: u64,
        total_j: f64,
        origin: Option<Origin>,
        clpc: usize,
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
            let u = |m: String| MeterError::Unsupported("ioreport", m);
            let g = cfstr("Energy Model");
            // SAFETY: every returned object is checked for null and owned once.
            let copied = Own(unsafe { IOReportCopyChannelsInGroup(g.0, ptr::null(), 0, 0, 0) });
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
            let k = cfstr("IOReportChannels");
            unsafe { CFDictionarySetValue(chan.0 as _, k.0, sel.0) };
            let mut subbed: Cf = ptr::null();
            let sub = Own(unsafe {
                IOReportCreateSubscription(ptr::null(), chan.0, &mut subbed, 0, ptr::null())
            });
            let subbed = Own(subbed);
            if sub.0.is_null() || subbed.0.is_null() {
                return Err(u("subscription failed".into()));
            }
            let last = Own(unsafe { IOReportCreateSamples(sub.0, subbed.0, ptr::null()) });
            if last.0.is_null() {
                return Err(u("first sample failed".into()));
            }
            let mut m = IoReportMeter {
                rail,
                chan: subbed,
                sub,
                last,
                last_t: now_ns(),
                total_j: 0.0,
                origin: None,
                clpc,
            };
            if rail.always_draws() {
                std::thread::sleep(std::time::Duration::from_millis(100));
                match m.delta() {
                    Ok(RailReading::Joules(_, o)) => m.origin = Some(o),
                    Ok(other) => {
                        return Err(u(format!(
                            "{} counters did not move over 100 ms ({other:?}; macOS {os}, \
                             {clpc} known CLPC driver(s))",
                            rail.name()
                        )));
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
            self.clpc
        }

        fn delta(&mut self) -> Result<RailReading, MeterError> {
            // SAFETY: sub and chan are live; samples are owned.
            let now = Own(unsafe { IOReportCreateSamples(self.sub.0, self.chan.0, ptr::null()) });
            if now.0.is_null() {
                return Err(MeterError::Read(
                    "ioreport",
                    "IOReportCreateSamples failed".into(),
                ));
            }
            let d = Own(unsafe { IOReportCreateSamplesDelta(self.last.0, now.0, ptr::null()) });
            self.last = now;
            self.last_t = now_ns();
            if d.0.is_null() {
                return Err(MeterError::Read(
                    "ioreport",
                    "IOReportCreateSamplesDelta failed".into(),
                ));
            }
            Ok(read_rail(&rows(d.0), self.rail))
        }
    }

    impl Meter for IoReportMeter {
        fn source(&self) -> &'static str {
            "ioreport"
        }
        fn sample(&mut self) -> Result<Sample, MeterError> {
            let t0 = self.last_t;
            let r = self.delta()?;
            let t = self.last_t;
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
}
