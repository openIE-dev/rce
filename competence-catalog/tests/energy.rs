//! measured_j: the hardware meter measures real joules, or honestly reports that
//! no source can measure here — never a fabricated reading (D.11 provenance).

use competence_catalog::energy::EnergyMeter;

#[test]
fn meter_measures_or_honestly_reports_none() {
    match EnergyMeter::open() {
        Ok(mut m) => {
            // A working meter reads the package draw over a window of real work.
            let r = m.measure_mean(20, || {
                std::hint::black_box((0..50_000).sum::<u64>());
            });
            assert!(r.is_ok(), "a working meter must measure: {r:?}");
            println!("meter {} measured {} µJ/act over 20 acts", m.source(), r.unwrap());
        }
        Err(e) => {
            // No meter on this host: the failure is honest, not a fake 0.
            println!("no hardware meter here (cost stays Modelled): {e}");
        }
    }
}

#[test]
fn zero_acts_is_refused() {
    if let Ok(mut m) = EnergyMeter::open() {
        assert!(m.measure_mean(0, || {}).is_err(), "n=0 must not yield a reading");
    }
}
