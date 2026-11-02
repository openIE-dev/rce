//! One meter shared by every action in a process, with receipt energy split
//! by [`crate::apportion`] so overlapping actions are never double counted.
//!
//! Before, each action read the cumulative counter at its start and end and
//! booked the whole difference. Two actions that overlap in time then both
//! book the overlap, so the receipts add up to more than the meter saw.
//!
//! Here every `begin` and `end` takes a sample, so the set of open actions is
//! constant between consecutive samples. Each interval's joules are split
//! equally among the actions open in it (weight 1.0 each), or booked as idle
//! when none is open. Because interval boundaries are exactly the action
//! boundaries, an action's share is final when it ends: later actions cannot
//! change it. Invariant, tested: sum of shares + idle == counter total.
//!
//! Scope: one process. Actions in other processes that read the same
//! hardware counter through their own `SharedMeter` are not seen here.
use crate::apportion::{Activity, apportion};
use crate::attrib::Sample;
use crate::meter::{Meter, MeterError};
use std::sync::{Arc, Mutex};

struct Open {
    id: u64,
    start_ns: u64,
}

struct Inner {
    meter: Box<dyn Meter>,
    /// Samples since the last moment no action was open (the first one is
    /// that moment).
    samples: Vec<Sample>,
    open: Vec<Open>,
    closed: Vec<Activity>,
    next_id: u64,
    booked: f64,
    idle: f64,
    /// Counter value at the first sample ever taken.
    origin: Option<f64>,
}

/// Cloneable handle; clones share the meter and the books.
#[derive(Clone)]
pub struct SharedMeter {
    inner: Arc<Mutex<Inner>>,
    source: &'static str,
}

/// An open action. Pass it back to [`SharedMeter::end`].
#[derive(Debug)]
pub struct Span {
    id: u64,
}

/// Totals since the meter was wrapped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Books {
    pub booked: f64,
    pub idle: f64,
    /// Counter delta from the first sample to the latest.
    pub total: f64,
}

impl SharedMeter {
    // Arc, not Rc: clones are shared across views; meters are not Send, so
    // a SharedMeter stays on one thread, which the type already enforces.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(meter: Box<dyn Meter>) -> Self {
        let source = meter.source();
        SharedMeter {
            inner: Arc::new(Mutex::new(Inner {
                meter,
                samples: vec![],
                open: vec![],
                closed: vec![],
                next_id: 0,
                booked: 0.0,
                idle: 0.0,
                origin: None,
            })),
            source,
        }
    }

    pub fn source(&self) -> &'static str {
        self.source
    }

    fn take(i: &mut Inner) -> Result<Sample, MeterError> {
        let s = i.meter.sample()?;
        if i.origin.is_none() {
            i.origin = Some(s.joules);
        }
        // One clock throughout; never let time go backwards in the log.
        let s = match i.samples.last() {
            Some(l) if s.t_ns <= l.t_ns => Sample { t_ns: l.t_ns + 1, joules: s.joules.max(l.joules) },
            _ => s,
        };
        i.samples.push(s);
        Ok(s)
    }

    /// Open an action. `None` when the meter cannot be read now.
    pub fn begin(&self) -> Option<Span> {
        let mut i = self.inner.lock().ok()?;
        let s = Self::take(&mut i).ok()?;
        let id = i.next_id;
        i.next_id += 1;
        i.open.push(Open { id, start_ns: s.t_ns });
        Some(Span { id })
    }

    /// Close an action and return its joules. `None` when the meter cannot
    /// be read now (the action stays open in no one's books: its intervals
    /// fall to the other open actions or to idle, and it is reported as
    /// unmeasured, never as 0 J).
    pub fn end(&self, span: Span) -> Option<f64> {
        let mut i = self.inner.lock().ok()?;
        let pos = i.open.iter().position(|o| o.id == span.id)?;
        let o = i.open.remove(pos);
        let s = match Self::take(&mut i) {
            Ok(s) => s,
            Err(_) => return None,
        };
        let me = format!("#{}", o.id);
        i.closed.push(Activity { process: me.clone(), start_ns: o.start_ns, end_ns: s.t_ns, weight: 1.0 });
        // Joules of this action: its share of every interval in its window,
        // given everyone open in it (open ones run at least to now).
        let mut acts = i.closed.clone();
        for x in &i.open {
            acts.push(Activity { process: format!("#{}", x.id), start_ns: x.start_ns, end_ns: s.t_ns, weight: 1.0 });
        }
        let win: Vec<Sample> = i.samples.iter().copied().filter(|x| x.t_ns >= o.start_ns && x.t_ns <= s.t_ns).collect();
        let mine = apportion(&win, &acts).per_process.get(&me).copied().unwrap_or(0.0);
        i.booked += mine;
        if i.open.is_empty() {
            // Settle idle for the retained history and drop it.
            let all = apportion(&i.samples, &i.closed);
            i.idle += all.idle;
            let last = *i.samples.last().expect("sample just taken");
            i.samples = vec![last];
            i.closed.clear();
        }
        Some(mine)
    }

    /// Books so far. Idle in intervals still holding open actions is not
    /// settled yet, so call this when no action is open for an exact balance.
    pub fn books(&self) -> Books {
        let i = self.inner.lock().expect("shared meter lock");
        let total = match (i.origin, i.samples.last()) {
            (Some(o), Some(l)) => l.joules - o,
            _ => 0.0,
        };
        Books { booked: i.booked, idle: i.idle, total }
    }

    /// Record an idle-only sample (moves the books forward with no action).
    pub fn tick(&self) -> Result<(), MeterError> {
        let mut i = self.inner.lock().map_err(|e| MeterError::Read("shared", e.to_string()))?;
        Self::take(&mut i)?;
        if i.open.is_empty() {
            let all = apportion(&i.samples, &i.closed);
            i.idle += all.idle;
            let last = *i.samples.last().expect("sample just taken");
            i.samples = vec![last];
            i.closed.clear();
        }
        Ok(())
    }
}
