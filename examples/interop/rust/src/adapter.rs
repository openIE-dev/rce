// The boundary between a host and one program printed by `rce emit rust`.
// It is included once per emitted module (`m`), because every emitted file
// carries its own runtime and so its own value type `m::V`.
use serde_json::Value as J;

pub struct Program {
    state: m::State,
}

pub enum Outcome {
    Applied(Vec<m::V>),
    Kept(String),
}

const INT_TYPES: [&str; 10] = ["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "usize"];

impl Program {
    pub fn new() -> Result<Program, String> {
        m::fuel_reset();
        let state = m::initial_state().map_err(|_| "the initial state traps".to_string())?;
        Ok(Program { state })
    }

    /// An event of the program from a JSON object {"type": NAME, field: value, ..}.
    pub fn event(&self, j: &J) -> Result<m::V, String> {
        let o = j.as_object().ok_or("an event is a JSON object")?;
        let name = o.get("type").and_then(J::as_str).ok_or("an event names its type")?;
        let fields = m::event_fields(name).ok_or_else(|| format!("no event {name}"))?;
        let mut fs = vec![];
        for (f, t) in fields {
            let x = o.get(f).ok_or_else(|| format!("missing field {f}"))?;
            let bad = || format!("field {f} is not a {t}");
            let v = match t {
                "text" => m::V::Text(x.as_str().ok_or_else(bad)?.to_string()),
                "bool" => m::V::Bool(x.as_bool().ok_or_else(bad)?),
                "f64" => m::V::F(x.as_f64().ok_or_else(bad)?),
                t if INT_TYPES.contains(&t) => {
                    let n = x.as_i64().map(i128::from).or_else(|| x.as_u64().map(i128::from)).ok_or_else(bad)?;
                    m::chk(n, t).map_err(|_| format!("field {f}: {n} is outside {t}"))?
                }
                _ => return Err(format!("field {f} has type {t}, which this adapter does not read")),
            };
            fs.push((f.to_string(), v));
        }
        Ok(m::V::Rec(name.to_string(), fs))
    }

    /// Apply one event; a kept step leaves the state as it was (HC-21).
    pub fn step(&mut self, ev: &m::V) -> Outcome {
        m::fuel_reset(); // HC-7a: full fuel for each input
        match m::delta(&self.state, ev) {
            Ok((s, outs)) => {
                self.state = s;
                Outcome::Applied(outs)
            }
            Err(r) => Outcome::Kept(r),
        }
    }

    pub fn show_state(&self) -> String {
        m::show_state(&self.state)
    }

    /// The lines `rce run` prints for event number k.
    pub fn lines(&self, k: u64, ev: &m::V, o: &Outcome) -> Vec<String> {
        match o {
            Outcome::Kept(r) => vec![format!("{k} {} kept: {r}", m::show(ev))],
            Outcome::Applied(outs) => {
                let mut v = vec![format!("{k} {} {}", m::show(ev), self.show_state())];
                v.extend(outs.iter().map(|o| format!("  emit {}", m::show(o))));
                v
            }
        }
    }

    /// The state as JSON: {region: {field: value}}.
    pub fn state_json(&self) -> J {
        J::Object(self.state.iter().map(|(r, fs)| (r.to_string(), J::Object(fs.iter().map(|(n, v)| (n.to_string(), value_json(v))).collect()))).collect())
    }
}

fn value_json(v: &m::V) -> J {
    use m::V;
    match v {
        V::Int(x, _) => i64::try_from(*x).map(J::from).or_else(|_| u64::try_from(*x).map(J::from)).unwrap_or_else(|_| J::String(x.to_string())),
        V::F(x) | V::F32(x) => J::from(*x),
        V::Bool(b) => J::Bool(*b),
        V::Text(s) => J::String(s.clone()),
        V::Unit | V::Opt(None) => J::Null,
        V::Opt(Some(x)) => value_json(x),
        V::Tuple(xs) | V::List(xs) | V::Set(xs) => J::Array(xs.iter().map(value_json).collect()),
        V::Map(ps) => J::Array(ps.iter().map(|(k, v)| J::Array(vec![value_json(k), value_json(v)])).collect()),
        V::Rec(n, fs) => {
            let mut o: serde_json::Map<String, J> = fs.iter().map(|(k, x)| (k.clone(), value_json(x))).collect();
            o.insert("$type".into(), J::String(n.clone()));
            J::Object(o)
        }
        other => J::String(m::show(other)),
    }
}

impl crate::Host for Program {
    fn initial_line(&self) -> String {
        format!("0 (initial) {}", self.show_state())
    }
    fn apply(&mut self, seq: u64, line: &str) -> J {
        let ev = match serde_json::from_str::<J>(line).map_err(|e| e.to_string()).and_then(|j| self.event(&j)) {
            Ok(ev) => ev,
            Err(why) => return serde_json::json!({"outcome": "not-admitted", "why": why}),
        };
        let o = self.step(&ev);
        let outcome = match &o {
            Outcome::Applied(_) => "applied".to_string(),
            Outcome::Kept(r) => format!("kept: {r}"),
        };
        serde_json::json!({"seq": seq, "outcome": outcome, "lines": self.lines(seq, &ev, &o), "state": self.state_json()})
    }
}
