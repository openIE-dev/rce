//! Print the measured external-corpus (Rosetta Code) lift coverage and holdout hit-rate.

use competence_catalog::rosetta;

fn main() {
    println!("real-corpus lift coverage (Rosetta Code, verbatim):");
    for (task, variant, src) in rosetta::CORPUS {
        println!("  {:10} {:18} lift_ok={}", task, variant, rosetta::lift_ok(src));
    }
    let (l, t) = rosetta::lift_coverage();
    println!("lift_coverage = {l}/{t}");
    let (h, n) = rosetta::external_holdout();
    println!("external_holdout (admitted gcd+factorial; requests all 6 real variants) = {h}/{n}");
}