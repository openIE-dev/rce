//! The kernel library of `rce emit wasm` (spec 10.18, EW-2).
//!
//! Built for `wasm32-unknown-unknown` by `scripts/build-wasm-kernels.sh`,
//! this crate is [`rce_vm_backend_cpu::reference_node`], the step the
//! reference evaluation takes for one node, with an interface a program
//! module calls through its imports. A program module emitted for a graph
//! term holds one descriptor per node and calls [`rce_node`] on each in the
//! reference order; values stay in this module's slots, addressed by
//! number. Because each node runs the reference's own kernel, an exact node
//! is bit for bit the reference's value, and every other node is too.
//!
//! The interface (every number an `i32`):
//!
//! - `rce_alloc(len) -> ptr`: room for `len` bytes in this memory.
//! - `rce_put(slot, ptr, len) -> status`: a tensor in the tensor encoding
//!   (below) into `slot`.
//! - `rce_put_bytes(slot, ptr, len) -> status`: raw bytes (a blob) into `slot`.
//! - `rce_node(ptr, len) -> status`: run the node the JSON descriptor at
//!   `ptr` names.
//! - `rce_copy(from, to) -> status`.
//! - `rce_get(slot) -> ptr`, `rce_get_len() -> len`: the tensor encoding
//!   of the tensor in `slot`.
//! - `rce_msg() -> ptr`, `rce_msg_len() -> len`: the text of the last
//!   failure.
//!
//! Status: 0 done; 1 trap `non-finite float`; 2 a trap of RCE (integer
//! overflow, division by zero, index out of range), its text in `rce_msg`;
//! 3 the node cannot run (its text in `rce_msg`); 4 a blob's bytes do not
//! hash to its digest.
//!
//! The tensor encoding: u32 element code (0 f32, 1 f64, 2 i32), u32 rank,
//! rank u32 dimensions, then the elements row-major, little-endian.

use rce_graph::graph::GraphNode;
use rce_vm_backend_cpu::{reference_node, DType, HostTensor, InterpretError, Value};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// A node descriptor, the data segment the program module holds for it.
#[derive(serde::Deserialize)]
pub struct Desc {
    /// the node's index in the term's graph
    pub node: usize,
    /// the slot its value goes to
    pub out: u32,
    /// the slots of its operands, in port order
    #[serde(default)]
    pub args: Vec<u32>,
    pub run: Run,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Run {
    /// the reference step of this node
    Node(GraphNode),
    /// a parameter or tensor literal: the tensor already put in `args[0]`
    Value,
    /// a blob: the bytes in `args[0]` must hash to `digest` (BLAKE3), and
    /// are the elements of a tensor of `elem` and `shape` (an `i8` blob
    /// is widened to `i32` lanes)
    Blob { digest: String, elem: String, shape: Vec<usize> },
}

enum Slot {
    V(Value),
    Bytes(Vec<u8>),
}

#[derive(Default)]
struct State {
    slots: BTreeMap<u32, Slot>,
    msg: Vec<u8>,
    got: Vec<u8>,
}

thread_local! {
    static ST: RefCell<State> = RefCell::new(State::default());
}

pub const DONE: i32 = 0;
pub const NON_FINITE: i32 = 1;
pub const TRAP: i32 = 2;
pub const CANNOT_RUN: i32 = 3;
pub const BAD_DIGEST: i32 = 4;

fn fail(code: i32, m: String) -> i32 {
    ST.with(|s| s.borrow_mut().msg = m.into_bytes());
    code
}

/// The tensor encoding of `t`.
pub fn encode(t: &HostTensor) -> Option<Vec<u8>> {
    let code: u32 = match t.dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::I32 => 2,
        _ => return None,
    };
    let mut out = vec![];
    out.extend(code.to_le_bytes());
    out.extend((t.shape.len() as u32).to_le_bytes());
    for d in &t.shape {
        out.extend((*d as u32).to_le_bytes());
    }
    out.extend(&t.data);
    Some(out)
}

/// The tensor a tensor encoding holds.
pub fn decode(b: &[u8]) -> Result<HostTensor, String> {
    let word = |i: usize| -> Result<u32, String> {
        b.get(4 * i..4 * i + 4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).ok_or_else(|| "a short tensor encoding".to_string())
    };
    let dtype = match word(0)? {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::I32,
        c => return Err(format!("element code {c}")),
    };
    let rank = word(1)? as usize;
    let shape = (0..rank).map(|i| word(2 + i).map(|d| d as usize)).collect::<Result<Vec<_>, _>>()?;
    HostTensor::new(shape, dtype, b[4 * (2 + rank)..].to_vec()).map_err(|e| e.to_string())
}

/// Run one descriptor against the slots.
fn run(d: Desc) -> i32 {
    let args: Result<Vec<Slot>, String> = ST.with(|s| {
        let s = s.borrow();
        d.args
            .iter()
            .map(|a| match s.slots.get(a) {
                Some(Slot::V(v)) => Ok(Slot::V(v.clone())),
                Some(Slot::Bytes(b)) => Ok(Slot::Bytes(b.clone())),
                None => Err(format!("node {}: slot {a} is empty", d.node)),
            })
            .collect()
    });
    let args = match args {
        Ok(a) => a,
        Err(m) => return fail(CANNOT_RUN, m),
    };
    let r = match d.run {
        Run::Node(n) => {
            let vs: Vec<Value> = args.into_iter().filter_map(|a| if let Slot::V(v) = a { Some(v) } else { None }).collect();
            reference_node(d.node, &n, vs)
        }
        Run::Value => match args.into_iter().next() {
            // the value's own finite check, as the reference makes it
            Some(Slot::V(v)) => finite(d.node, v),
            _ => return fail(CANNOT_RUN, format!("node {}: no value to take", d.node)),
        },
        Run::Blob { digest, elem, shape } => {
            let Some(Slot::Bytes(b)) = args.into_iter().next() else {
                return fail(CANNOT_RUN, format!("node {}: no blob bytes", d.node));
            };
            let got = blake3::hash(&b).to_hex().to_string();
            if got != digest {
                return fail(BAD_DIGEST, format!("node {}: the blob bytes hash to {got}, not {digest}", d.node));
            }
            let (data, dtype) = match elem.as_str() {
                "i8" => (b.iter().flat_map(|x| (*x as i8 as i32).to_le_bytes()).collect(), DType::I32),
                e => match DType::from_name(e) {
                    Some(t) => (b, t),
                    None => return fail(CANNOT_RUN, format!("node {}: a blob of {e}", d.node)),
                },
            };
            match HostTensor::new(shape, dtype, data) {
                Ok(t) => finite(d.node, Value::Tensor(t)),
                Err(e) => Err(e),
            }
        }
    };
    match r {
        Ok(v) => {
            ST.with(|s| s.borrow_mut().slots.insert(d.out, Slot::V(v)));
            DONE
        }
        Err(InterpretError::NonFinite(_) | InterpretError::DivByZero(_)) => fail(NON_FINITE, "non-finite float".into()),
        Err(InterpretError::Trap(_, t)) => fail(TRAP, t),
        Err(e) => fail(CANNOT_RUN, format!("node {}: {e}", d.node)),
    }
}

fn finite(node: usize, v: Value) -> Result<Value, InterpretError> {
    let ok = match &v {
        Value::Tensor(t) if t.dtype == DType::F32 => t.as_f32()?.iter().all(|x| x.is_finite()),
        Value::Tensor(t) if t.dtype == DType::F64 => t.as_f64()?.iter().all(|x| x.is_finite()),
        _ => true,
    };
    if ok { Ok(v) } else { Err(InterpretError::NonFinite(node)) }
}

/// # Safety
/// `ptr..ptr+len` lies in this module's memory.
unsafe fn bytes<'a>(ptr: i32, len: i32) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(ptr as usize as *const u8, len as usize) }
}

#[unsafe(no_mangle)]
pub extern "C" fn rce_alloc(len: i32) -> i32 {
    let v = vec![0u8; len.max(1) as usize].into_boxed_slice();
    Box::leak(v).as_mut_ptr() as usize as i32
}

/// # Safety
/// `ptr..ptr+len` lies in this module's memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rce_put(slot: i32, ptr: i32, len: i32) -> i32 {
    match decode(unsafe { bytes(ptr, len) }) {
        Ok(t) => {
            ST.with(|s| s.borrow_mut().slots.insert(slot as u32, Slot::V(Value::Tensor(t))));
            DONE
        }
        Err(m) => fail(CANNOT_RUN, m),
    }
}

/// # Safety
/// `ptr..ptr+len` lies in this module's memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rce_put_bytes(slot: i32, ptr: i32, len: i32) -> i32 {
    let b = unsafe { bytes(ptr, len) }.to_vec();
    ST.with(|s| s.borrow_mut().slots.insert(slot as u32, Slot::Bytes(b)));
    DONE
}

/// # Safety
/// `ptr..ptr+len` lies in this module's memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rce_node(ptr: i32, len: i32) -> i32 {
    match serde_json::from_slice::<Desc>(unsafe { bytes(ptr, len) }) {
        Ok(d) => run(d),
        Err(e) => fail(CANNOT_RUN, format!("a node descriptor that does not decode: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rce_copy(from: i32, to: i32) -> i32 {
    ST.with(|s| {
        let mut s = s.borrow_mut();
        match s.slots.get(&(from as u32)) {
            Some(Slot::V(v)) => {
                let v = v.clone();
                s.slots.insert(to as u32, Slot::V(v));
                DONE
            }
            _ => {
                s.msg = format!("slot {from} holds no value").into_bytes();
                CANNOT_RUN
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rce_get(slot: i32) -> i32 {
    ST.with(|s| {
        let mut s = s.borrow_mut();
        let enc = match s.slots.get(&(slot as u32)) {
            Some(Slot::V(Value::Tensor(t))) => encode(t).unwrap_or_default(),
            _ => vec![],
        };
        s.got = enc;
        s.got.as_ptr() as usize as i32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn rce_get_len() -> i32 {
    ST.with(|s| s.borrow().got.len() as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn rce_msg() -> i32 {
    ST.with(|s| s.borrow().msg.as_ptr() as usize as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn rce_msg_len() -> i32 {
    ST.with(|s| s.borrow().msg.len() as i32)
}
