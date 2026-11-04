//! The typing of a graph term (spec 6.13, GT-2 and GT-3).
//!
//! A graph term of RCE's core is a [`FunctionGraph`](crate::graph::FunctionGraph)
//! together with its [`TermTypes`]: the element type and shape of each
//! parameter and of the result, and the equivalence level of every
//! operation node. A `.fg` file holds the graph in its `Graph` section and
//! these types in its `TypeTable` section, so the file's content hash, the
//! term's key, covers the body, the shapes and the levels.

use serde::{Deserialize, Serialize};

/// The element type and shape of a tensor value. An empty shape is a
/// scalar of the element type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    /// `f32` or `f64`.
    pub dtype: String,
    pub shape: Vec<u64>,
}

/// The equivalence level an operation carries (spec 6.13, GT-3). It states
/// what any realisation of the operation other than the reference (another
/// backend, an emitted kernel, a fusion) must reproduce of the reference's
/// result. The levels are those of the RCE Compile study's E3
/// (`compile/CHARTER.md` 4.3), with an accumulation class named by the
/// operation itself, so a realisation that keeps the operation's order is
/// held to `Exact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Level {
    /// Bit for bit the reference's result.
    Exact,
    /// Each element within `ulps` units in the last place of the element
    /// type of the reference's element.
    Within { ulps: u32 },
    /// No equality is claimed: the operation lowers precision, and a
    /// realisation of it is a change of program, recorded as such.
    Lossy,
}

impl Level {
    /// Whether a realisation held to `self` is also held to `other`:
    /// exact is stronger than every bound, a smaller bound than a larger
    /// one, and every level than lossy.
    pub fn at_least(self, other: Level) -> bool {
        match (self, other) {
            (_, Level::Lossy) => true,
            (Level::Lossy, _) => false,
            (Level::Exact, _) => true,
            (Level::Within { .. }, Level::Exact) => false,
            (Level::Within { ulps: a }, Level::Within { ulps: b }) => a <= b,
        }
    }

    /// The level as RCE text names it: `exact`, `within N`, `lossy`.
    pub fn text(self) -> String {
        match self {
            Level::Exact => "exact".into(),
            Level::Within { ulps } => format!("within {ulps}"),
            Level::Lossy => "lossy".into(),
        }
    }
}

/// The level of one operation node of the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeLevel {
    pub node: u32,
    pub level: Level,
}

/// The typing of a graph term: parameters in order (the graph's `Param`
/// nodes are named `p0`, `p1`, ... in this order, so the term is name-free),
/// the result, and the level of every `Apply` node in ascending node order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TermTypes {
    pub params: Vec<TensorType>,
    pub result: TensorType,
    pub levels: Vec<NodeLevel>,
}
