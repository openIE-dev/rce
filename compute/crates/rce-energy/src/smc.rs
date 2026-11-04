//! Apple SMC power keys (no root needed), polled and integrated.
//!
//! `PSTR` is total system power in watts, as read by macmon
//! (vladkens/macmon, MIT, `get_sys_power`). Other `P*` keys exist (81 on an
//! M5 Max, macOS 27.0.1) but Apple does not document which rail each one is,
//! so only `PSTR` is used and it is tagged as the whole system, never as CPU.
//! The SMC protocol (struct layout, selector 2, commands 5/8/9) follows
//! macmon `src/sources.rs`; this is a re-implementation.

/// Decode an SMC value from its type FourCC and bytes. Only types seen on
/// power keys are decoded; others are `None`, never guessed.
pub fn decode(ty: &str, bytes: &[u8]) -> Option<f64> {
    let v = match (ty, bytes.len()) {
        ("flt ", 4) => f32::from_le_bytes(bytes.try_into().ok()?) as f64,
        ("ui8 ", 1) => bytes[0] as f64,
        ("ui16", 2) => u16::from_be_bytes(bytes.try_into().ok()?) as f64,
        ("ui32", 4) => u32::from_be_bytes(bytes.try_into().ok()?) as f64,
        _ => return None,
    };
    v.is_finite().then_some(v)
}

/// A four-character SMC key as the protocol's big-endian u32.
pub fn key(k: &str) -> Option<u32> {
    let b = k.as_bytes();
    (b.len() == 4).then(|| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(target_os = "macos")]
pub use mac::{Smc, SmcMeter};

#[cfg(target_os = "macos")]
mod mac {
    use super::{decode, key};
    use crate::meter::MeterError;
    use crate::power::PolledMeter;
    use std::ffi::{CString, c_char, c_void};
    use std::time::Duration;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(n: *const c_char) -> *const c_void;
        fn IOServiceGetMatchingServices(p: u32, m: *const c_void, it: *mut u32) -> i32;
        fn IOIteratorNext(it: u32) -> u32;
        fn IOObjectRelease(o: u32) -> u32;
        fn IORegistryEntryGetName(e: u32, name: *mut c_char) -> i32;
        fn IOServiceOpen(s: u32, task: u32, ty: u32, c: *mut u32) -> i32;
        fn IOServiceClose(c: u32) -> i32;
        fn IOConnectCallStructMethod(
            c: u32,
            sel: u32,
            i: *const c_void,
            il: usize,
            o: *mut c_void,
            ol: *mut usize,
        ) -> i32;
        fn mach_task_self() -> u32;
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Ver {
        major: u8,
        minor: u8,
        build: u8,
        reserved: u8,
        release: u16,
    }
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct PLimit {
        version: u16,
        length: u16,
        cpu: u32,
        gpu: u32,
        mem: u32,
    }
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct KeyInfo {
        size: u32,
        ty: u32,
        attr: u8,
    }
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct KeyData {
        key: u32,
        vers: Ver,
        plimit: PLimit,
        info: KeyInfo,
        result: u8,
        status: u8,
        data8: u8,
        data32: u32,
        bytes: [u8; 32],
    }

    /// An open AppleSMC connection.
    pub struct Smc {
        conn: u32,
    }

    unsafe impl Send for Smc {}

    impl Smc {
        pub fn open() -> Result<Smc, MeterError> {
            let u = |m: String| MeterError::Unsupported("smc", m);
            let n = CString::new("AppleSMC").map_err(|e| u(e.to_string()))?;
            let mut it = 0u32;
            // SAFETY: the matching dictionary is consumed by the call.
            if unsafe { IOServiceGetMatchingServices(0, IOServiceMatching(n.as_ptr()), &mut it) }
                != 0
            {
                return Err(u("AppleSMC not found".into()));
            }
            let mut conn = 0u32;
            loop {
                // SAFETY: it is a live iterator; each entry is released.
                let e = unsafe { IOIteratorNext(it) };
                if e == 0 {
                    break;
                }
                let mut name = [0 as c_char; 128];
                unsafe { IORegistryEntryGetName(e, name.as_mut_ptr()) };
                // SAFETY: IORegistryEntryGetName NUL-terminates into 128 bytes.
                let nm = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
                if conn == 0 && nm == "AppleSMCKeysEndpoint" {
                    unsafe { IOServiceOpen(e, mach_task_self(), 0, &mut conn) };
                }
                unsafe { IOObjectRelease(e) };
            }
            unsafe { IOObjectRelease(it) };
            if conn == 0 {
                return Err(u("AppleSMCKeysEndpoint not openable".into()));
            }
            Ok(Smc { conn })
        }

        fn call(&self, i: &KeyData) -> Result<KeyData, String> {
            let mut o = KeyData::default();
            let mut ol = std::mem::size_of::<KeyData>();
            // SAFETY: i and o are KeyData-sized buffers.
            let r = unsafe {
                IOConnectCallStructMethod(
                    self.conn,
                    2,
                    i as *const KeyData as *const c_void,
                    std::mem::size_of::<KeyData>(),
                    &mut o as *mut KeyData as *mut c_void,
                    &mut ol,
                )
            };
            if r != 0 {
                return Err(format!("IOConnectCallStructMethod: {r}"));
            }
            match o.result {
                0 => Ok(o),
                132 => Err("key not found".into()),
                x => Err(format!("SMC error {x}")),
            }
        }

        /// Read one key as a number (see [`decode`]).
        pub fn read(&self, k: &str) -> Result<f64, String> {
            let id = key(k).ok_or("SMC key must be 4 bytes")?;
            let info = self
                .call(&KeyData {
                    key: id,
                    data8: 9,
                    ..Default::default()
                })?
                .info;
            let o = self.call(&KeyData {
                key: id,
                info,
                data8: 5,
                ..Default::default()
            })?;
            let ty = String::from_utf8_lossy(&info.ty.to_be_bytes()).into_owned();
            let n = (info.size as usize).min(32);
            decode(&ty, &o.bytes[..n])
                .ok_or_else(|| format!("{k}: type {ty:?} size {n} not decoded"))
        }
    }

    impl Drop for Smc {
        fn drop(&mut self) {
            unsafe { IOServiceClose(self.conn) };
        }
    }

    /// System power from SMC `PSTR`, polled every `period` and integrated.
    pub struct SmcMeter;

    impl SmcMeter {
        pub fn open(period: Duration) -> Result<PolledMeter, MeterError> {
            let smc = Smc::open()?;
            smc.read("PSTR")
                .map_err(|e| MeterError::Unsupported("smc", format!("PSTR: {e}")))?;
            PolledMeter::start("smc", period, move || {
                smc.read("PSTR").map_err(|e| MeterError::Read("smc", e))
            })
        }
    }
}
