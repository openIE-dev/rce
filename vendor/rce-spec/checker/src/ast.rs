//! Abstract syntax of RCE (RCE-SPEC.md section 3).
#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Path(Vec<String>, Vec<Ty>),
    Unit,
    Tuple(Vec<Ty>),
    Slice(Box<Ty>),
    Array(Box<Ty>, Box<Expr>),
    Chan(Box<Ty>),
    Iter(Box<Ty>),
    Fn(Vec<Ty>, Box<Ty>),
    Ref(bool, Box<Ty>),
    Opt(Box<Ty>),
    Union(Box<Ty>, Box<Ty>),
    Region(Box<Ty>, String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arg {
    pub label: Option<String>,
    pub e: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Name(String, usize),
    SelfRef(usize),
    Int(i128, usize),
    Float(f64, usize),
    Str(String, usize),
    Bool(bool, usize),
    None(usize),
    Unit(usize),
    Tuple(Vec<Expr>, usize),
    List(Vec<Expr>, usize),
    ListRepeat(Box<Expr>, Box<Expr>, usize),
    ListComp(Box<Expr>, Box<Pat>, Box<Expr>, Option<Box<Expr>>, usize),
    Map(Vec<(Expr, Expr)>, usize),
    ChanNew(Ty, Option<Box<Expr>>, usize),
    Call(Box<Expr>, Vec<Arg>, usize),
    Index(Box<Expr>, Vec<Expr>, usize),
    Slice(Box<Expr>, Option<Box<Expr>>, Option<Box<Expr>>, usize),
    Member(Box<Expr>, String, usize),
    TupleMember(Box<Expr>, usize, usize),
    Neg(Box<Expr>, usize),
    BitNot(Box<Expr>, usize),
    Borrow(bool, Box<Expr>, usize),
    Recv(Box<Expr>, usize),
    Comptime(Box<Expr>, usize),
    Cast(Box<Expr>, Ty, usize),
    Bin(String, Box<Expr>, Box<Expr>, usize),
    Not(Box<Expr>, usize),
    Catch(Box<Expr>, Option<String>, Box<Expr>, usize),
    Try(Box<Expr>, Option<Box<Expr>>, usize),
    If(Box<Expr>, Box<Expr>, Box<Expr>, usize),
    Lambda(Vec<Param>, Box<Expr>, usize),
}

impl Expr {
    pub fn line(&self) -> usize {
        use Expr::*;
        match self {
            Name(_, l)
            | SelfRef(l)
            | Int(_, l)
            | Float(_, l)
            | Str(_, l)
            | Bool(_, l)
            | None(l)
            | Unit(l)
            | Tuple(_, l)
            | List(_, l)
            | ListRepeat(_, _, l)
            | ListComp(_, _, _, _, l)
            | Map(_, l)
            | ChanNew(_, _, l)
            | Call(_, _, l)
            | Index(_, _, l)
            | Slice(_, _, _, l)
            | Member(_, _, l)
            | TupleMember(_, _, l)
            | Neg(_, l)
            | BitNot(_, l)
            | Borrow(_, _, l)
            | Recv(_, l)
            | Comptime(_, l)
            | Cast(_, _, l)
            | Bin(_, _, _, l)
            | Not(_, l)
            | Catch(_, _, _, l)
            | Try(_, _, l)
            | If(_, _, _, l)
            | Lambda(_, _, l) => *l,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pat {
    Wild,
    Path(Vec<String>),
    Ctor(Vec<String>, Vec<(Option<String>, Pat)>),
    Lit(Expr),
    Range(Expr, bool, Expr),
    Tuple(Vec<Pat>),
    Alt(Vec<Pat>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    Arrow(Expr),
    Block(Vec<Stmt>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: Vec<Stmt>,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelArm {
    Recv(Pat, Expr, Vec<Stmt>),
    Send(Expr, Expr, Vec<Stmt>),
    After(Expr, Vec<Stmt>),
    Default(Vec<Stmt>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expr(Expr),
    Assign(String, Expr, Expr),
    Send(Expr, Expr),
    Let(Pat, Option<Ty>, Expr, usize),
    Var(String, Option<Ty>, Expr, usize),
    If(Vec<(Expr, Vec<Stmt>)>, Option<Vec<Stmt>>),
    While(Expr, Option<Expr>, Vec<Stmt>),
    Loop(Option<Expr>, Vec<Stmt>),
    For(Pat, Expr, Vec<Stmt>, usize),
    Match(Expr, Vec<Arm>, usize),
    Return(Option<Expr>, usize),
    Break(usize),
    Continue(usize),
    Pass,
    Emit(Expr),
    Fail(Expr),
    Spawn(Expr),
    Select(Vec<SelArm>, usize),
    Arena(String, Option<Expr>, Vec<Stmt>, usize),
    Device(String, Option<Expr>, Vec<Stmt>, usize),
    Yield(Expr, usize),
    Comptime(Vec<Stmt>),
    Check(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    Named {
        comptime: bool,
        label: Option<String>,
        name: String,
        ty: Ty,
    },
    SelfVal,
    SelfRef(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GParam {
    pub comptime: bool,
    pub name: String,
    pub bounds: Vec<Ty>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnSig {
    pub comptime: bool,
    pub name: String,
    pub generics: Vec<GParam>,
    pub params: Vec<Param>,
    pub ret: Option<Ty>,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub sig: FnSig,
    pub body: Body,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: Ty,
    pub init: Option<Expr>,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub events: Vec<(String, Vec<Arg>)>,
    pub refused: Option<String>,
    pub value: Option<Expr>,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Handler {
    pub pat: Pat,
    pub guard: Option<Expr>,
    pub body: Body,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraitMember {
    Type(String),
    Fn(FnSig),
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Item {
    Use(Vec<String>, Option<String>),
    Intent(String, Vec<Example>, Vec<Expr>),
    Example(Example),
    Check(Expr),
    Const {
        name: String,
        ty: Option<Ty>,
        e: Expr,
        line: usize,
    },
    TypeAlias {
        name: String,
        generics: Vec<GParam>,
        ty: Ty,
        line: usize,
    },
    Struct {
        name: String,
        generics: Vec<GParam>,
        fields: Vec<Field>,
        line: usize,
        linear: bool,
    },
    State {
        name: String,
        fields: Vec<Field>,
        line: usize,
        /// `replicated state` (C.11.4)
        replicated: bool,
    },
    Event {
        name: String,
        payload: Vec<(String, Ty)>,
        line: usize,
        /// `interrupt event ... priority p` (C.11.7)
        interrupt: Option<Expr>,
    },
    Error {
        name: String,
        payload: Vec<(String, Ty)>,
        line: usize,
    },
    Kind {
        name: String,
        generics: Vec<GParam>,
        variants: Vec<(String, Vec<(String, Ty)>)>,
        line: usize,
    },
    Trait {
        name: String,
        generics: Vec<GParam>,
        members: Vec<TraitMember>,
        line: usize,
    },
    Impl {
        generics: Vec<GParam>,
        ty: Ty,
        trait_: Option<Ty>,
        members: Vec<FnDecl>,
        line: usize,
    },
    Fn(FnDecl),
    Extern {
        abi: String,
        sig: FnSig,
        line: usize,
    },
    Step {
        region: String,
        pat: Pat,
        guard: Option<Expr>,
        body: Body,
        line: usize,
        /// `deadline d` in logical ticks (C.11.6)
        deadline: Option<Expr>,
    },
    Actor {
        name: String,
        fields: Vec<Field>,
        handlers: Vec<Handler>,
        fns: Vec<FnDecl>,
        line: usize,
    },
    // draft 0.6b (the draft 0.6 concurrency chapter, not published here C.11)
    Protocol {
        name: String,
        steps: Vec<ProtoStep>,
        line: usize,
    },
    Node {
        name: String,
        actors: Vec<String>,
        line: usize,
    },
    Supervisor {
        name: String,
        strategy: String,
        children: Vec<SupChild>,
        line: usize,
    },
}

/// A step of a `protocol` (C.11.3), from one endpoint's point of view.
#[derive(Debug, Clone, PartialEq)]
pub enum ProtoStep {
    Send(Ty),
    Recv(Ty),
    Offer(Vec<(String, Vec<ProtoStep>)>),
    Choose(Vec<(String, Vec<ProtoStep>)>),
    Loop(Vec<ProtoStep>),
    Break,
}

/// `child name: Type [restart bound n]` of a supervisor (C.11.11).
#[derive(Debug, Clone, PartialEq)]
pub struct SupChild {
    pub name: String,
    pub ty: String,
    pub restart: u32,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub path: Option<Vec<String>>,
    pub items: Vec<Item>,
    /// names of the items declared `pub` (4.1: visible to importing modules)
    pub pubs: std::collections::BTreeSet<String>,
}
