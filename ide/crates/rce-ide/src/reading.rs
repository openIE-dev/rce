//! The D.11 reading of a package counter. It lives in RCE's one meter
//! library (`rce-energy`, here under the name `ide_energy`), which the host
//! also uses for `host.Energy` (HC-38).
pub use ide_energy::reading::*;

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) use ide_energy::reading::fixtures::{prov, zone};
}
