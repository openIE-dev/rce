//! Appendix D.12: ladders and labels (LL-1 to LL-4).
//!
//! The research papers describe cost-ordered ladders of their own (Lookup,
//! Formula, Solver, Model and others) and evidence classes of their own
//! (`HwShunt`, "datasheet", "demonstrated" and others). This module is the
//! one table that places each rung on RCE's gears G0 to G5 (compile
//! CHARTER) and each class on RCE's energy labels of D.11 (`unpriced` <
//! `est` < `reported` < `measured`). `tests/ladder_map.rs` checks the rules
//! and that the tables printed in D.12 are these tables.

/// An RCE energy label (D.11), weakest first, so `min` is "the weakest wins".
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Label {
    /// No joule figure: `j` is null, never 0.
    Unpriced,
    /// A model of the work, including a provider's own figure.
    Est,
    /// An OS or on-chip counter (RAPL, NVML, IOReport).
    Reported,
    /// A wall or shunt meter on a named rail.
    Measured,
}

impl Label {
    pub fn word(self) -> &'static str {
        match self {
            Label::Unpriced => "unpriced",
            Label::Est => "est",
            Label::Reported => "reported",
            Label::Measured => "measured",
        }
    }
    /// The label of a figure built from two terms: the weaker one (D.11).
    pub fn combine(self, other: Label) -> Label {
        self.min(other)
    }
}

/// An RCE gear (compile CHARTER): G0 stored result by hash, G1 proven
/// rewrite rules, G2 cheapest extraction, G3 to G5 proposers whose every
/// output is checked before use.
pub type Gear = u8;

/// Where a foreign rung sits in RCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// The gears `lo..=hi`.
    Gears(Gear, Gear),
    /// The check every output of G1 to G5 passes before use.
    Gate,
    /// Not a gear: the Form's state and input that the gears read.
    Input,
}

impl Place {
    pub fn text(self) -> String {
        match self {
            Place::Gears(lo, hi) if lo == hi => format!("G{lo}"),
            Place::Gears(lo, hi) => format!("G{lo} to G{hi}"),
            Place::Gate => "gate".into(),
            Place::Input => "input".into(),
        }
    }
}

/// One rung of a foreign ladder, in the ladder's own order.
#[derive(Clone, Copy, Debug)]
pub struct Rung {
    pub ladder: &'static str,
    pub rung: &'static str,
    pub place: Place,
}

/// Where a foreign evidence class sits on the D.11 labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map {
    /// Always this label.
    To(Label),
    /// The label of the cited source's own figure; `est` when the source
    /// names no counter or meter.
    Source,
    /// A proved bound: carried as a bound, never as a joule figure.
    Bound,
}

impl Map {
    pub fn text(self) -> &'static str {
        match self {
            Map::To(l) => l.word(),
            Map::Source => "source",
            Map::Bound => "bound",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Class {
    pub scheme: &'static str,
    pub class: &'static str,
    pub map: Map,
}

const fn r(ladder: &'static str, rung: &'static str, place: Place) -> Rung {
    Rung { ladder, rung, place }
}
const fn c(scheme: &'static str, class: &'static str, map: Map) -> Class {
    Class { scheme, class, map }
}

/// LL-1: every rung of every ladder the papers name.
pub const LADDERS: &[Rung] = &[
    r("gates", "Lookup", Place::Gears(0, 0)),
    r("gates", "Formula", Place::Gears(1, 1)),
    r("gates", "Solver", Place::Gears(2, 2)),
    r("gates", "Model", Place::Gears(3, 5)),
    r("mei", "Lookup", Place::Gears(0, 0)),
    r("mei", "Formula", Place::Gears(1, 1)),
    r("mei", "Solver", Place::Gears(2, 2)),
    r("mei", "Model", Place::Gears(3, 4)),
    r("mei", "fifth rung", Place::Gears(5, 5)),
    r("spellcheck", "Lookup", Place::Gears(0, 0)),
    r("spellcheck", "Formula", Place::Gears(1, 1)),
    r("spellcheck", "Model", Place::Gears(3, 5)),
    r("jouleos-eoc", "state-construct", Place::Input),
    r("jouleos-eoc", "retrieve", Place::Gears(0, 0)),
    r("jouleos-eoc", "refine", Place::Gears(1, 5)),
    r("jouleos-eoc", "check", Place::Gate),
];

/// LL-2 and LL-3: every evidence class the papers name.
pub const CLASSES: &[Class] = &[
    c("joule-contract", "HwShunt", Map::To(Label::Measured)),
    c("joule-contract", "ModelBased", Map::To(Label::Est)),
    c("joule-contract", "Estimator", Map::To(Label::Est)),
    c("mei", "real protocol", Map::To(Label::Unpriced)),
    c("mei", "emulator-measured", Map::To(Label::Est)),
    c("mei", "datasheet", Map::To(Label::Est)),
    c("mei", "measured-kernel model", Map::To(Label::Est)),
    c("mei", "looked-up", Map::To(Label::Est)),
    c("mei", "calculation", Map::To(Label::Est)),
    c("mei", "plant model", Map::To(Label::Est)),
    c("latency", "established by proof", Map::Bound),
    c("latency", "demonstrated", Map::Source),
    c("latency", "opinion", Map::To(Label::Unpriced)),
    c("latency", "model output", Map::To(Label::Est)),
];

/// The rungs of one ladder, in its order.
pub fn ladder(name: &str) -> Vec<Rung> {
    LADDERS.iter().copied().filter(|r| r.ladder == name).collect()
}

/// The ladder names, in table order.
pub fn ladder_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for r in LADDERS {
        if !v.contains(&r.ladder) {
            v.push(r.ladder);
        }
    }
    v
}

/// LL-1: a ladder is well placed when its input rungs come first, its gate
/// rungs last, its gear rungs in non-decreasing gear order, no two rungs
/// share a gear, and the gears it names start at G0.
pub fn ladder_ok(rungs: &[Rung]) -> Result<(), String> {
    let mut stage = 0; // 0 input, 1 gears, 2 gate
    let mut last_hi: Option<Gear> = None;
    let mut first_lo: Option<Gear> = None;
    for r in rungs {
        match r.place {
            Place::Input if stage == 0 => {}
            Place::Gears(lo, hi) if stage <= 1 => {
                if lo > hi || hi > 5 {
                    return Err(format!("{}: {} names no gear range", r.ladder, r.rung));
                }
                if let Some(p) = last_hi
                    && lo <= p
                {
                        return Err(format!("{}: {} is not above the rung before it", r.ladder, r.rung));
                }
                first_lo.get_or_insert(lo);
                last_hi = Some(hi);
                stage = 1;
            }
            Place::Gate => stage = 2,
            _ => return Err(format!("{}: {} is out of order", r.ladder, r.rung)),
        }
    }
    match first_lo {
        Some(0) => Ok(()),
        _ => Err(format!("{}: the ladder does not start at G0", rungs.first().map_or("", |r| r.ladder))),
    }
}

/// LL-2: the label a class gives a figure. `source` is the label of the
/// cited source's own figure, when it names a counter or meter.
/// `None` means the class gives no joule figure (a bound).
pub fn label_of(class: &Class, source: Option<Label>) -> Option<Label> {
    match class.map {
        Map::To(l) => Some(l),
        Map::Source => Some(source.unwrap_or(Label::Est)),
        Map::Bound => None,
    }
}

/// LL-3: the `j` a figure carries under its label: null (None) when the
/// label is `unpriced` or the class is a bound, never 0.
pub fn joules(class: &Class, source: Option<Label>, j: f64) -> Option<f64> {
    match label_of(class, source) {
        None | Some(Label::Unpriced) => None,
        Some(_) => Some(j),
    }
}

/// The class named `class` in `scheme`.
pub fn class(scheme: &str, class: &str) -> Option<&'static Class> {
    CLASSES.iter().find(|c| c.scheme == scheme && c.class == class)
}
