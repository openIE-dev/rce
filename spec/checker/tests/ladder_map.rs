//! D.12 ladders and labels, LL-1 to LL-4: the foreign ladders placed on the
//! gears and the foreign evidence classes placed on the D.11 labels.
use rce_energy::reading::Basis;
use rce_spec_check::ladder::{
    CLASSES, LADDERS, Label, Map, Place, Rung, class, joules, label_of, ladder, ladder_names,
    ladder_ok,
};

const CH15: &str = include_str!("../../book/15-handler-spec-profile.md");

#[test]
fn every_ladder_is_in_gear_order() {
    let names = ladder_names();
    assert_eq!(names, ["gates", "mei", "spellcheck", "jouleos-eoc"]);
    for n in names {
        let rungs = ladder(n);
        ladder_ok(&rungs).unwrap_or_else(|e| panic!("{e}"));
        // every ladder with a model rung reaches a proposer gear, gated
        let top = rungs.iter().filter_map(|r| match r.place {
            Place::Gears(_, hi) => Some(hi),
            _ => None,
        });
        assert_eq!(top.max(), Some(5), "{n} reaches G5");
    }
}

#[test]
fn a_rung_out_of_order_is_refused() {
    let bad = [
        Rung { ladder: "x", rung: "Model", place: Place::Gears(3, 5) },
        Rung { ladder: "x", rung: "Lookup", place: Place::Gears(0, 0) },
    ];
    assert!(ladder_ok(&bad).is_err());
    let gate_first = [
        Rung { ladder: "x", rung: "check", place: Place::Gate },
        Rung { ladder: "x", rung: "Lookup", place: Place::Gears(0, 0) },
    ];
    assert!(ladder_ok(&gate_first).is_err());
    let no_g0 = [Rung { ladder: "x", rung: "Formula", place: Place::Gears(1, 1) }];
    assert!(ladder_ok(&no_g0).is_err());
    let overlap = [
        Rung { ladder: "x", rung: "a", place: Place::Gears(0, 2) },
        Rung { ladder: "x", rung: "b", place: Place::Gears(2, 3) },
    ];
    assert!(ladder_ok(&overlap).is_err());
}

#[test]
fn only_a_shunt_meter_gives_measured() {
    for c in CLASSES {
        let fixed = match c.map {
            Map::To(l) => l,
            _ => continue,
        };
        assert_ne!(fixed, Label::Reported, "{}/{}", c.scheme, c.class);
        assert_eq!(fixed == Label::Measured, c.class == "HwShunt", "{}/{}", c.scheme, c.class);
    }
    // no MEI figure is from an external meter
    assert!(CLASSES.iter().filter(|c| c.scheme == "mei").all(|c| label_of(c, None) <= Some(Label::Est)));
}

#[test]
fn source_is_est_without_a_counter_or_meter() {
    let d = class("latency", "demonstrated").unwrap();
    assert_eq!(label_of(d, None), Some(Label::Est));
    assert_eq!(label_of(d, Some(Label::Reported)), Some(Label::Reported));
    assert_eq!(label_of(d, Some(Label::Measured)), Some(Label::Measured));
    assert_eq!(label_of(class("latency", "model output").unwrap(), None), Some(Label::Est));
}

#[test]
fn a_sum_takes_the_weakest_label() {
    use Label::*;
    assert!(Unpriced < Est && Est < Reported && Reported < Measured);
    assert_eq!(Measured.combine(Est), Est);
    assert_eq!(Reported.combine(Measured), Reported);
    assert_eq!(Est.combine(Unpriced), Unpriced);
}

#[test]
fn label_words_match_the_meter_library() {
    assert_eq!(Label::Reported.word(), Basis::Reported.word());
    assert_eq!(Label::Measured.word(), Basis::Measured.word());
}

#[test]
fn unpriced_and_bound_carry_no_joules() {
    assert_eq!(joules(class("latency", "opinion").unwrap(), None, 0.0), None);
    assert_eq!(joules(class("mei", "real protocol").unwrap(), None, 3.0), None);
    assert_eq!(joules(class("latency", "established by proof").unwrap(), None, 1.0), None);
    assert_eq!(label_of(class("latency", "established by proof").unwrap(), None), None);
    assert_eq!(joules(class("mei", "datasheet").unwrap(), None, 2.5), Some(2.5));
}

/// The rows of the first table in D.12 whose header starts with `head`.
fn table(head: &str) -> Vec<[String; 3]> {
    let d12 = CH15.split("### D.12").nth(1).expect("D.12 is in chapter 15");
    let d12 = d12.split("\n### ").next().unwrap();
    let mut lines = d12.lines().skip_while(|l| !l.starts_with(head)).skip(2);
    let mut rows = Vec::new();
    while let Some(l) = lines.next().filter(|l| l.starts_with('|')) {
        let cells: Vec<String> = l.trim_matches('|').split('|').map(|s| s.trim().to_string()).collect();
        rows.push([cells[0].clone(), cells[1].clone(), cells[2].clone()]);
    }
    rows
}

#[test]
fn the_d12_tables_are_the_code_tables() {
    let ladders: Vec<[String; 3]> = LADDERS
        .iter()
        .map(|r| [r.ladder.to_string(), r.rung.to_string(), r.place.text()])
        .collect();
    assert_eq!(table("| Ladder | Rung | RCE place |"), ladders);
    let classes: Vec<[String; 3]> = CLASSES
        .iter()
        .map(|c| [c.scheme.to_string(), c.class.to_string(), c.map.text().to_string()])
        .collect();
    assert_eq!(table("| Scheme | Class | RCE label |"), classes);
}
