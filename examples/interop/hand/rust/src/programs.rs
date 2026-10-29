//! The eight programs written by hand as idiomatic Rust, for comparison
//! with what `rce emit rust` prints: a state struct, an event enum read by
//! serde, and `step(&mut self, event) -> Result<outputs, kept reason>` that
//! changes nothing when it returns Err. Checked on the streams of
//! streams.json only (overflow is not checked, there is no fuel).
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::{self, Display};

pub type Step<O> = Result<Vec<O>, &'static str>;

pub trait Program: Default + Display {
    type Event: for<'de> Deserialize<'de>;
    type Out: Display;
    fn step(&mut self, e: Self::Event) -> Step<Self::Out>;
}

/// No outputs.
pub enum Never {}
impl Display for Never {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        match *self {}
    }
}

fn show_map<K, V>(m: &BTreeMap<K, V>, k: impl Fn(&K) -> String, v: impl Fn(&V) -> String) -> String {
    let ps: Vec<String> = m.iter().map(|(a, b)| format!("{}: {}", k(a), v(b))).collect();
    format!("{{{}}}", ps.join(", "))
}

// ---- counter (01_counter.rce)
#[derive(Default)]
pub struct Counter { n: u64 }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum CounterEvent { Inc, Dec }
impl Program for Counter {
    type Event = CounterEvent;
    type Out = Never;
    fn step(&mut self, e: CounterEvent) -> Step<Never> {
        match e {
            CounterEvent::Inc => self.n += 1,
            CounterEvent::Dec => self.n = self.n.checked_sub(1).ok_or("Underflow")?,
        }
        Ok(vec![])
    }
}
impl Display for Counter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Counter {{ n = {} }}", self.n) }
}

// ---- bank (04_bank_account.rce)
#[derive(Default)]
pub struct Account { balance: i64 }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum BankEvent { Deposit { amount: i64 }, Withdraw { amount: i64 } }
pub struct Withdrawn { amount: i64 }
impl Display for Withdrawn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Withdrawn(amount: {})", self.amount) }
}
impl Program for Account {
    type Event = BankEvent;
    type Out = Withdrawn;
    fn step(&mut self, e: BankEvent) -> Step<Withdrawn> {
        match e {
            BankEvent::Deposit { amount } if amount > 0 => self.balance += amount,
            BankEvent::Deposit { .. } => {}
            BankEvent::Withdraw { amount } => {
                if amount > self.balance {
                    return Err("Insufficient");
                }
                self.balance -= amount;
                return Ok(vec![Withdrawn { amount }]);
            }
        }
        Ok(vec![])
    }
}
impl Display for Account {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Account {{ balance = {} }}", self.balance) }
}

// ---- multi_region (16_multi_region.rce)
#[derive(Default)]
pub struct Shop { items: Vec<String>, adds: u64, removes: u64 }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ShopEvent { AddItem { name: String }, RemoveItem { name: String }, Reset }
impl Program for Shop {
    type Event = ShopEvent;
    type Out = Never;
    fn step(&mut self, e: ShopEvent) -> Step<Never> {
        match e {
            ShopEvent::AddItem { name } => { self.items.push(name); self.adds += 1; }
            ShopEvent::RemoveItem { name } => { self.items.retain(|i| *i != name); self.removes += 1; }
            ShopEvent::Reset => self.items.clear(),
        }
        Ok(vec![])
    }
}
impl Display for Shop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let items: Vec<String> = self.items.iter().map(|i| format!("{i:?}")).collect();
        write!(f, "Cart {{ items = [{}] }}; Stats {{ adds = {}, removes = {} }}", items.join(", "), self.adds, self.removes)
    }
}

// ---- traffic_light (17_traffic_light.rce)
#[derive(Default, Debug, Clone, Copy)]
pub enum Light { #[default] Red, Green, Yellow }
#[derive(Default)]
pub struct Signal { light: Light }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum SignalEvent { Tick }
impl Program for Signal {
    type Event = SignalEvent;
    type Out = Never;
    fn step(&mut self, SignalEvent::Tick: SignalEvent) -> Step<Never> {
        self.light = match self.light { Light::Red => Light::Green, Light::Green => Light::Yellow, Light::Yellow => Light::Red };
        Ok(vec![])
    }
}
impl Display for Signal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Signal {{ light = {:?} }}", self.light) }
}

// ---- door (30_state_machine_guarded.rce)
#[derive(Default)]
pub struct Door { open: bool, locked: bool }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum DoorEvent { Open, Close, Lock, Unlock }
impl Program for Door {
    type Event = DoorEvent;
    type Out = Never;
    fn step(&mut self, e: DoorEvent) -> Step<Never> {
        match e {
            DoorEvent::Open if self.locked => return Err("Locked"),
            DoorEvent::Open => self.open = true,
            DoorEvent::Close => self.open = false,
            DoorEvent::Lock if self.open => return Err("DoorOpen"),
            DoorEvent::Lock => self.locked = true,
            DoorEvent::Unlock => self.locked = false,
        }
        Ok(vec![])
    }
}
impl Display for Door {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Door {{ open = {}, locked = {} }}", self.open, self.locked) }
}

// ---- inventory (31_inventory.rce)
#[derive(Default)]
pub struct Inventory { stock: BTreeMap<String, u32> }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum InventoryEvent { Receive { sku: String, qty: u32 }, Ship { sku: String, qty: u32 } }
pub struct Shipped { sku: String, qty: u32 }
impl Display for Shipped {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Shipped(sku: {:?}, qty: {})", self.sku, self.qty) }
}
impl Program for Inventory {
    type Event = InventoryEvent;
    type Out = Shipped;
    fn step(&mut self, e: InventoryEvent) -> Step<Shipped> {
        match e {
            InventoryEvent::Receive { sku, qty } => { *self.stock.entry(sku).or_default() += qty; Ok(vec![]) }
            InventoryEvent::Ship { sku, qty } => {
                let have = self.stock.get(&sku).copied().unwrap_or(0);
                if have < qty {
                    return Err("OutOfStock");
                }
                self.stock.insert(sku.clone(), have - qty);
                Ok(vec![Shipped { sku, qty }])
            }
        }
    }
}
impl Display for Inventory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Inventory {{ stock = {} }}", show_map(&self.stock, |k| format!("{k:?}"), u32::to_string)) }
}

// ---- rcedb (55_rcedb.rce)
#[derive(Default)]
pub struct Db { rows: BTreeMap<String, String>, writes: u64, durable: u64 }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum DbEvent { Put { key: String, val: String }, Delete { key: String }, Synced { upto: u64 }, Scan { start: String } }
pub struct Row { key: String, val: String }
impl Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Row(key: {:?}, val: {:?})", self.key, self.val) }
}
impl Program for Db {
    type Event = DbEvent;
    type Out = Row;
    fn step(&mut self, e: DbEvent) -> Step<Row> {
        match e {
            DbEvent::Put { key, val } => { self.rows.insert(key, val); self.writes += 1; }
            DbEvent::Delete { key } => { self.rows.remove(&key).ok_or("Missing")?; self.writes += 1; }
            DbEvent::Synced { upto } => self.durable = self.durable.max(upto),
            DbEvent::Scan { start } => {
                return Ok(self.rows.range(start..).map(|(k, v)| Row { key: k.clone(), val: v.clone() }).collect());
            }
        }
        Ok(vec![])
    }
}
impl Display for Db {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Db {{ rows = {}, writes = {}, durable = {} }}", show_map(&self.rows, |k| format!("{k:?}"), |v| format!("{v:?}")), self.writes, self.durable)
    }
}

// ---- float_keys (58_float_keys.rce)
/// An f64 ordered by IEEE 754 totalOrder, so -0.0 and 0.0 are two keys.
#[derive(Clone, Copy)]
pub struct Key(f64);
impl PartialEq for Key { fn eq(&self, o: &Key) -> bool { self.0.total_cmp(&o.0).is_eq() } }
impl Eq for Key {}
impl PartialOrd for Key { fn partial_cmp(&self, o: &Key) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for Key { fn cmp(&self, o: &Key) -> std::cmp::Ordering { self.0.total_cmp(&o.0) } }
#[derive(Default)]
pub struct Tally { m: BTreeMap<Key, i64> }
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum TallyEvent { Add { x: f64 }, Flip { x: f64 } }
impl Program for Tally {
    type Event = TallyEvent;
    type Out = Never;
    fn step(&mut self, e: TallyEvent) -> Step<Never> {
        let k = match e { TallyEvent::Add { x } => x, TallyEvent::Flip { x } => x * -1.0 };
        *self.m.entry(Key(k)).or_default() += 1;
        Ok(vec![])
    }
}
impl Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Tally {{ m = {} }}", show_map(&self.m, |k| format!("{}", k.0), i64::to_string)) }
}
