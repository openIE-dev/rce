import RCE.Form
import RCE.Conc
/-! T7 extended (spec 6.6): name-free hashing of the log and of the ids of
    channels, actors and tasks, and the choices of `select`.
    The hash function `H` is arbitrary (any function; SHA-256 in the spec):
    the theorems hold for every `H` because what is hashed is a *name-free
    key* in which every identifier is replaced by its declaration position. -/
namespace RCE.NameFree

/-- Declaration position of a name (De Bruijn-style); `length` if absent. -/
def idx : List String → String → Nat
  | [], _ => 0
  | a :: as, x => if a = x then 0 else idx as x + 1

theorem idx_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (x : String) :
    ∀ ds : List String, idx (ds.map ρ) (ρ x) = idx ds x
  | [] => rfl
  | a :: as => by
    simp only [List.map, idx]
    have : (ρ a = ρ x) ↔ (a = x) := ⟨hρ _ _, fun h => h ▸ rfl⟩
    by_cases h : a = x
    · simp [h]
    · rw [if_neg (fun h' => h (this.1 h')), if_neg h, idx_rename ρ hρ x as]

/-! ### Log hashes -/

/-- Canonical, name-free key of a log entry. -/
inductive CK where
  | ev (i : Nat) (args : List Val)
  | emit (v : Val)
  | user (i : Nat)
  | trap (m : Trap)
  deriving DecidableEq, Repr

def outKey (ds : List String) : Out → CK
  | .emit v => .emit v
  | .refused (.user x) => .user (idx ds x)
  | .refused (.trap m) => .trap m

def entryKey (ds : List String) (en : Event × List Out) : List CK :=
  .ev (idx ds en.1.1) en.1.2 :: en.2.map (outKey ds)

/-- The hash chain of a log, for any hash function `H`. -/
def logHashes (H : List CK → Nat) (ds : List String) (log : List (Event × List Out)) : List Nat :=
  log.map fun en => H (entryKey ds en)

def renEv (ρ : String → String) (e : Event) : Event := (ρ e.1, e.2)

def renEntry (ρ : String → String) (en : Event × List Out) : Event × List Out :=
  (renEv ρ en.1, en.2.map (Out.rename ρ))

def renM (ρ : String → String) (m : Machine) : Machine := ⟨m.state, m.log.map (renEntry ρ)⟩

theorem admit_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (P : Program) (m : Machine) (e : Event) :
    admit (P.rename ρ) (renM ρ m) (renEv ρ e) = renM ρ (admit P m e) := by
  simp only [admit, renM, renEv]
  rw [name_independence ρ hρ P m.state e]
  simp [renEntry, renEv]

theorem run_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (P : Program) :
    ∀ (es : List Event) (m : Machine),
    (es.map (renEv ρ)).foldl (admit (P.rename ρ)) (renM ρ m) = renM ρ (es.foldl (admit P) m) := by
  intro es
  induction es with
  | nil => intro m; rfl
  | cons e es ih => intro m; simp only [List.map, List.foldl]; rw [admit_rename ρ hρ, ih]

theorem outKey_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (ds : List String) (o : Out) :
    outKey (ds.map ρ) (o.rename ρ) = outKey ds o := by
  cases o with
  | emit v => rfl
  | refused r => cases r with
    | user x => simp [Out.rename, Refusal.rename, outKey, idx_rename ρ hρ]
    | trap m => rfl

/-- **T7 for log hashes**: renaming every event and error name (any
    injective ρ, applied to the declarations, the program and the input
    events) leaves the state and every log hash unchanged, for every hash
    function `H`. -/
theorem log_hashes_name_independent (H : List CK → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (ds : List String) (P : Program) (s0 : State) (es : List Event) :
    (run (P.rename ρ) s0 (es.map (renEv ρ))).state = (run P s0 es).state ∧
    logHashes H (ds.map ρ) (run (P.rename ρ) s0 (es.map (renEv ρ))).log = logHashes H ds (run P s0 es).log := by
  have h := run_rename ρ hρ P es ⟨s0, []⟩
  simp only [renM, List.map_nil] at h
  unfold run
  rw [h]
  refine ⟨rfl, ?_⟩
  simp only [logHashes, List.map_map]
  congr 1
  funext en
  simp only [Function.comp, renEntry, entryKey, renEv, List.map_map, idx_rename ρ hρ]
  congr 2
  apply List.map_congr_left
  intro o _
  exact outKey_rename ρ hρ ds o

/-! ### Ids of channels, actors and tasks, and `select` -/

/-- Surface concurrency code: channels and actors are referred to by name. -/
inductive NInstr where
  | send (c : String) (r : Nat)
  | recv (c : String) (r : Nat)
  | close (c : String)
  | emit (r : Nat)
  | alloc (r : Nat)
  | spawn (body : List NInstr) (args : List Nat)
  | tell (a : String) (r : Nat)
  | select (arms : List (String × Nat)) (dflt : Bool)

/-- Declarations, in source order: channel names and actor names. -/
structure Decls where
  chans : List String
  actors : List String

/-- The id of a channel: H of its name-free key (creating task, position of
    its declaration). Actors are positional (their index in `actors`). -/
def chanId (H : Nat × Nat → Nat) (d : Decls) (c : String) : Nat := H (0, idx d.chans c)

mutual
def compileI (H : Nat × Nat → Nat) (d : Decls) : NInstr → Conc.Instr
  | .send c r => .send (chanId H d c) r
  | .recv c r => .recv (chanId H d c) r
  | .close c => .close (chanId H d c)
  | .emit r => .emit r
  | .alloc r => .alloc r
  | .spawn b as => .spawn (compileL H d b) as
  | .tell a r => .tell (idx d.actors a) r
  | .select arms dflt => .select (compileA H d arms) dflt
def compileL (H : Nat × Nat → Nat) (d : Decls) : List NInstr → List Conc.Instr
  | [] => []
  | i :: is => compileI H d i :: compileL H d is
def compileA (H : Nat × Nat → Nat) (d : Decls) : List (String × Nat) → List (Nat × Nat)
  | [] => []
  | (c, r) :: as => (chanId H d c, r) :: compileA H d as
end

mutual
def renI (ρ : String → String) : NInstr → NInstr
  | .send c r => .send (ρ c) r
  | .recv c r => .recv (ρ c) r
  | .close c => .close (ρ c)
  | .emit r => .emit r
  | .alloc r => .alloc r
  | .spawn b as => .spawn (renL ρ b) as
  | .tell a r => .tell (ρ a) r
  | .select arms dflt => .select (renA ρ arms) dflt
def renL (ρ : String → String) : List NInstr → List NInstr
  | [] => []
  | i :: is => renI ρ i :: renL ρ is
def renA (ρ : String → String) : List (String × Nat) → List (String × Nat)
  | [] => []
  | (c, r) :: as => (ρ c, r) :: renA ρ as
end

def Decls.ren (ρ : String → String) (d : Decls) : Decls := ⟨d.chans.map ρ, d.actors.map ρ⟩

theorem chanId_ren (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (d : Decls) (c : String) :
    chanId H (d.ren ρ) (ρ c) = chanId H d c := by
  simp [chanId, Decls.ren, idx_rename ρ hρ]

mutual
theorem compileI_ren (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (d : Decls) :
    ∀ i : NInstr, compileI H (d.ren ρ) (renI ρ i) = compileI H d i
  | .send c r => by simp only [renI, compileI, chanId_ren H ρ hρ]
  | .recv c r => by simp only [renI, compileI, chanId_ren H ρ hρ]
  | .close c => by simp only [renI, compileI, chanId_ren H ρ hρ]
  | .emit r => rfl
  | .alloc r => rfl
  | .spawn b as => by simp only [renI, compileI, compileL_ren H ρ hρ d b]
  | .tell a r => by simp only [renI, compileI, Decls.ren, idx_rename ρ hρ]
  | .select arms dflt => by simp only [renI, compileI, compileA_ren H ρ hρ d arms]
theorem compileL_ren (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (d : Decls) :
    ∀ l : List NInstr, compileL H (d.ren ρ) (renL ρ l) = compileL H d l
  | [] => rfl
  | i :: is => by simp only [renL, compileL, compileI_ren H ρ hρ d i, compileL_ren H ρ hρ d is]
theorem compileA_ren (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (d : Decls) :
    ∀ l : List (String × Nat), compileA H (d.ren ρ) (renA ρ l) = compileA H d l
  | [] => rfl
  | (c, r) :: as => by simp only [renA, compileA, chanId_ren H ρ hρ, compileA_ren H ρ hρ d as]
end

/-- A concurrent program: declarations, channel capacities by declaration
    position, actors, and `main`. -/
structure CProg where
  decls : Decls
  caps : List Nat
  actors : List Conc.Actor
  main : List NInstr

def CProg.ren (ρ : String → String) (p : CProg) : CProg := ⟨p.decls.ren ρ, p.caps, p.actors, renL ρ p.main⟩

/-- Channels are created empty with their declared capacity at their id. -/
def CProg.capOf (H : Nat × Nat → Nat) (p : CProg) (n : Nat) : Nat :=
  match (List.range p.decls.chans.length).find? (fun i => H (0, i) == n) with
  | some i => p.caps.getD i 0
  | none => 0

def CProg.chans (H : Nat × Nat → Nat) (p : CProg) : Nat → Conc.Chan :=
  fun n => ⟨p.capOf H n, [], false⟩

def CProg.start (H : Nat × Nat → Nat) (p : CProg) : Conc.Cfg :=
  Conc.initial (compileL H p.decls p.main) (p.chans H) p.actors

/-- **T7 for ids and select**: renaming channels and actors (any injective
    ρ) gives the same compiled code and the same initial configuration, for
    every hash function: every channel id, actor id, task id (task ids are
    creation positions), every tick, every `select` choice (the least ready
    channel id) and every log entry of every run is unchanged. -/
theorem conc_name_independent (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (p : CProg) : (p.ren ρ).start H = p.start H := by
  simp only [CProg.start, CProg.ren]
  rw [compileL_ren H ρ hρ]
  have : CProg.chans H ⟨p.decls.ren ρ, p.caps, p.actors, renL ρ p.main⟩ = CProg.chans H p := by
    funext n; simp [CProg.chans, CProg.capOf, Decls.ren]
  rw [this]

theorem runs_name_independent (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (p : CProg) (c : Conc.Cfg) : Conc.Reach ((p.ren ρ).start H) c ↔ Conc.Reach (p.start H) c := by
  rw [conc_name_independent H ρ hρ]

theorem select_name_independent (H : Nat × Nat → Nat) (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (d : Decls) (arms : List (String × Nat)) (rdy : Nat → Bool) :
    Conc.minReady rdy (compileA H (d.ren ρ) (renA ρ arms)) = Conc.minReady rdy (compileA H d arms) := by
  rw [compileA_ren H ρ hρ]

end RCE.NameFree
