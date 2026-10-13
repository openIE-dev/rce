/-! Concurrency (spec 6.8): tasks, bounded channels with staged delivery,
    `select` choosing the least ready channel, actors with mailboxes, and
    ticks.  Theorems: determinism of a tick, progress (every configuration
    ticks; if no task moves, every task is blocked or done, i.e. the run
    ends with a defined outcome), and no sharing of mutable data between
    tasks (an ownership invariant preserved by every tick). -/
namespace RCE.Conc

/-- A location is (owning task id, per-task allocation index) (6.6: ids are
    derived from the creating task, never from a global counter). -/
abbrev Loc := Nat × Nat

inductive CV where
  | data (n : Nat)
  | ref (l : Loc)
  | none
  deriving DecidableEq, Repr

def CV.isRef : CV → Bool
  | .ref _ => true
  | _ => false

inductive Instr where
  | send (c r : Nat)
  | recv (c r : Nat)
  | close (c : Nat)
  | emit (r : Nat)
  | alloc (r : Nat)
  | spawn (body : List Instr) (args : List Nat)
  | tell (a r : Nat)
  /-- `select` over receive arms (channel, register), with or without `default`. -/
  | select (arms : List (Nat × Nat)) (dflt : Bool)

structure Task where
  id : Nat
  code : List Instr
  regs : Nat → CV
  allocs : Nat

structure Chan where
  cap : Nat
  buf : List CV
  closed : Bool

inductive LogE where
  | emit (task : Nat) (v : CV)
  | backpressure (task c : Nat)
  | trap (task : Nat)
  deriving DecidableEq, Repr

structure Actor where
  st : Nat
  mbox : List CV
  h : Nat → CV → Nat

/-- The world threaded through the tasks of one tick. Sends, actor messages
    and spawns are staged and delivered when the tick ends (Form law 2). -/
structure W where
  chans : Nat → Chan
  staged : List (Nat × CV)
  mail : List (Nat × CV)
  born : List Task
  log : List LogE
  nextId : Nat
  moved : Nat

def setR (f : Nat → CV) (i : Nat) (v : CV) : Nat → CV := fun j => if j = i then v else f j
def setC (f : Nat → Chan) (i : Nat) (v : Chan) : Nat → Chan := fun j => if j = i then v else f j

/-- What a receive on a channel yields now: a value, `none` when closed and
    empty, or nothing (blocked). -/
def recvFrom (ch : Chan) : Option (CV × Chan) :=
  match ch.buf with
  | v :: rest => some (v, { ch with buf := rest })
  | [] => if ch.closed then some (.none, ch) else Option.none

def pending (w : W) (c : Nat) : Nat := (w.staged.filter (fun p => p.1 == c)).length

def full (w : W) (c : Nat) : Prop := (w.chans c).buf.length + pending w c ≥ (w.chans c).cap

instance (w : W) (c : Nat) : Decidable (full w c) := by unfold full; exact inferInstance

def ready (w : W) (c : Nat) : Bool := (recvFrom (w.chans c)).isSome

/-- The ready arm with the least channel id (canonical order; first arm on ties). -/
def minReady (rdy : Nat → Bool) : List (Nat × Nat) → Option (Nat × Nat)
  | [] => .none
  | a :: as =>
    match minReady rdy as with
    | .none => if rdy a.1 then some a else .none
    | some b => if rdy a.1 ∧ a.1 ≤ b.1 then some a else some b

def argRegs (regs : Nat → CV) (args : List Nat) : Nat → CV :=
  fun i => match args.get? i with
    | some r => regs r
    | .none => .none

/-- One communication step of one task (6.8). Blocked tasks stay as they are. -/
inductive TStep : W → Task → W → Task → Prop where
  | done {w t} : t.code = [] → TStep w t w t
  | sendOk {w t c r k} : t.code = .send c r :: k → (w.chans c).closed = false → ¬ full w c →
      TStep w t { w with staged := w.staged ++ [(c, t.regs r)], moved := w.moved + 1 } { t with code := k }
  | sendFull {w t c r k} : t.code = .send c r :: k → (w.chans c).closed = false → full w c →
      TStep w t { w with log := w.log ++ [.backpressure t.id c] } t
  | sendClosed {w t c r k} : t.code = .send c r :: k → (w.chans c).closed = true →
      TStep w t { w with log := w.log ++ [.trap t.id], moved := w.moved + 1 } { t with code := [] }
  | recvOk {w t c r k v ch} : t.code = .recv c r :: k → recvFrom (w.chans c) = some (v, ch) →
      TStep w t { w with chans := setC w.chans c ch, moved := w.moved + 1 } { t with code := k, regs := setR t.regs r v }
  | recvBlocked {w t c r k} : t.code = .recv c r :: k → recvFrom (w.chans c) = .none → TStep w t w t
  | close {w t c k} : t.code = .close c :: k →
      TStep w t { w with chans := setC w.chans c { w.chans c with closed := true }, moved := w.moved + 1 } { t with code := k }
  | emit {w t r k} : t.code = .emit r :: k →
      TStep w t { w with log := w.log ++ [.emit t.id (t.regs r)], moved := w.moved + 1 } { t with code := k }
  | alloc {w t r k} : t.code = .alloc r :: k →
      TStep w t { w with moved := w.moved + 1 } { t with code := k, regs := setR t.regs r (.ref (t.id, t.allocs)), allocs := t.allocs + 1 }
  | spawn {w t b as k} : t.code = .spawn b as :: k →
      TStep w t { w with born := w.born ++ [⟨w.nextId, b, argRegs t.regs as, 0⟩], nextId := w.nextId + 1, moved := w.moved + 1 } { t with code := k }
  | tell {w t a r k} : t.code = .tell a r :: k →
      TStep w t { w with mail := w.mail ++ [(a, t.regs r)], moved := w.moved + 1 } { t with code := k }
  | selRecv {w t arms d k c r v ch} : t.code = .select arms d :: k → minReady (ready w) arms = some (c, r) →
      recvFrom (w.chans c) = some (v, ch) →
      TStep w t { w with chans := setC w.chans c ch, moved := w.moved + 1 } { t with code := k, regs := setR t.regs r v }
  | selDefault {w t arms k} : t.code = .select arms true :: k → minReady (ready w) arms = .none →
      TStep w t { w with moved := w.moved + 1 } { t with code := k }
  | selBlocked {w t arms k} : t.code = .select arms false :: k → minReady (ready w) arms = .none → TStep w t w t

/-- All tasks of a tick, in id order (list order). -/
inductive Run : W → List Task → W → List Task → Prop where
  | nil {w} : Run w [] w []
  | cons {w t ts w1 t' w2 ts'} : TStep w t w1 t' → Run w1 ts w2 ts' → Run w (t :: ts) w2 (t' :: ts')

structure Cfg where
  tasks : List Task
  chans : Nat → Chan
  actors : List Actor
  log : List LogE
  nextId : Nat

def deliver (chans : Nat → Chan) : List (Nat × CV) → Nat → Chan
  | [] => chans
  | (c, v) :: rest => deliver (setC chans c { chans c with buf := (chans c).buf ++ [v] }) rest

def postMail (as : List Actor) (m : List (Nat × CV)) : List Actor :=
  as.enum.map fun (i, a) => { a with mbox := a.mbox ++ (m.filter (fun p => p.1 == i)).map Prod.snd }

/-- Each actor handles at most one message per tick, in mailbox order. -/
def runActor (a : Actor) : Actor :=
  match a.mbox with
  | [] => a
  | v :: rest => { a with st := a.h a.st v, mbox := rest }

def start (c : Cfg) : W := ⟨c.chans, [], [], [], c.log, c.nextId, 0⟩

def finish (c : Cfg) (w : W) (ts : List Task) : Cfg :=
  ⟨ts ++ w.born, deliver w.chans w.staged, postMail (c.actors.map runActor) w.mail, w.log, w.nextId⟩

/-- A tick: every task takes one communication step on the staged world,
    then sends, messages and spawns are delivered. -/
inductive Tick : Cfg → Cfg → Nat → Prop where
  | mk {c w ts} : Run (start c) c.tasks w ts → Tick c (finish c w ts) w.moved

/-! ## T1 for ticks: determinism -/

theorem tstep_det {w t w1 t1 w2 t2} (h1 : TStep w t w1 t1) (h2 : TStep w t w2 t2) : w1 = w2 ∧ t1 = t2 := by
  cases h1 <;> cases h2 <;> simp_all

theorem run_det {w ts w1 ts1 w2 ts2} (h1 : Run w ts w1 ts1) (h2 : Run w ts w2 ts2) : w1 = w2 ∧ ts1 = ts2 := by
  induction h1 generalizing w2 ts2 with
  | nil => cases h2; exact ⟨rfl, rfl⟩
  | cons hs _ ih =>
    cases h2 with
    | cons hs2 hr2 =>
      obtain ⟨rfl, rfl⟩ := tstep_det hs hs2
      obtain ⟨rfl, rfl⟩ := ih hr2
      exact ⟨rfl, rfl⟩

theorem tick_det {c c1 c2 m1 m2} (h1 : Tick c c1 m1) (h2 : Tick c c2 m2) : c1 = c2 ∧ m1 = m2 := by
  cases h1 with
  | mk r1 => cases h2 with
    | mk r2 => obtain ⟨rfl, rfl⟩ := run_det r1 r2; exact ⟨rfl, rfl⟩

/-! ## T4 for ticks: progress -/

theorem minReady_spec {rdy : Nat → Bool} : ∀ {as : List (Nat × Nat)} {a}, minReady rdy as = some a → rdy a.1 = true
  | [], _, h => by simp [minReady] at h
  | b :: bs, a, h => by
    simp only [minReady] at h
    cases hm : minReady rdy bs with
    | none =>
      rw [hm] at h; simp only at h
      by_cases hb : rdy b.1 = true
      · simp [hb] at h; subst h; exact hb
      · simp [hb] at h
    | some x =>
      rw [hm] at h; simp only at h
      by_cases hb : rdy b.1 = true ∧ b.1 ≤ x.1
      · simp only [hb, and_self, ite_true, Option.some.injEq] at h; subst h; exact hb.1
      · rw [if_neg hb] at h; cases h; exact minReady_spec hm

theorem tstep_total (w : W) (t : Task) : ∃ w' t', TStep w t w' t' := by
  cases hc : t.code with
  | nil => exact ⟨_, _, .done hc⟩
  | cons i k =>
    cases i with
    | send c r =>
      cases hcl : (w.chans c).closed with
      | true => exact ⟨_, _, .sendClosed hc hcl⟩
      | false =>
        by_cases hf : full w c
        · exact ⟨_, _, .sendFull hc hcl hf⟩
        · exact ⟨_, _, .sendOk hc hcl hf⟩
    | recv c r =>
      cases hr : recvFrom (w.chans c) with
      | none => exact ⟨_, _, .recvBlocked hc hr⟩
      | some p => obtain ⟨v, ch⟩ := p; exact ⟨_, _, .recvOk hc hr⟩
    | close c => exact ⟨_, _, .close hc⟩
    | emit r => exact ⟨_, _, .emit hc⟩
    | alloc r => exact ⟨_, _, .alloc hc⟩
    | spawn b as => exact ⟨_, _, .spawn hc⟩
    | tell a r => exact ⟨_, _, .tell hc⟩
    | select arms d =>
      cases hm : minReady (ready w) arms with
      | none =>
        cases d with
        | true => exact ⟨_, _, .selDefault hc hm⟩
        | false => exact ⟨_, _, .selBlocked hc hm⟩
      | some a =>
        obtain ⟨c, r⟩ := a
        have hr := minReady_spec hm
        simp only [ready] at hr
        cases hrf : recvFrom (w.chans c) with
        | none => rw [hrf] at hr; simp at hr
        | some p => obtain ⟨v, ch⟩ := p; exact ⟨_, _, .selRecv hc hm hrf⟩

theorem run_total (w : W) : ∀ ts, ∃ w' ts', Run w ts w' ts'
  | [] => ⟨_, _, .nil⟩
  | t :: ts => by
    obtain ⟨w1, t', h⟩ := tstep_total w t
    obtain ⟨w2, ts', hr⟩ := run_total w1 ts
    exact ⟨_, _, .cons h hr⟩

/-- A task can take a communication step now. -/
def enabled (w : W) (t : Task) : Prop :=
  match t.code with
  | [] => False
  | .send c _ :: _ => (w.chans c).closed = true ∨ ¬ full w c
  | .recv c _ :: _ => ready w c = true
  | .select arms d :: _ => minReady (ready w) arms ≠ .none ∨ d = true
  | _ :: _ => True

theorem tstep_moves {w t w' t'} (h : TStep w t w' t') (he : enabled w t) : w'.moved = w.moved + 1 := by
  cases h <;> simp_all [enabled, ready]

theorem tstep_idle {w t w' t'} (h : TStep w t w' t') (he : ¬ enabled w t) :
    t' = t ∧ w'.chans = w.chans ∧ w'.staged = w.staged ∧ w'.moved = w.moved ∧ w'.born = w.born ∧
    w'.mail = w.mail ∧ w'.nextId = w.nextId := by
  cases h <;> simp_all [enabled, ready]

theorem tstep_mono {w t w' t'} (h : TStep w t w' t') : w.moved ≤ w'.moved := by
  cases h <;> simp

theorem run_mono {w ts w' ts'} (h : Run w ts w' ts') : w.moved ≤ w'.moved := by
  induction h with
  | nil => exact Nat.le_refl _
  | cons hs _ ih => exact Nat.le_trans (tstep_mono hs) ih

theorem enabled_congr {w1 w2 : W} (hc : w1.chans = w2.chans) (hs : w1.staged = w2.staged) (t : Task) :
    enabled w1 t ↔ enabled w2 t := by
  have hr : ready w1 = ready w2 := by funext c; simp [ready, hc]
  have hf : ∀ c, full w1 c ↔ full w2 c := by intro c; simp [full, pending, hc, hs]
  unfold enabled
  split <;> simp_all

theorem run_progress {w ts w' ts'} (h : Run w ts w' ts') (he : ∃ t ∈ ts, enabled w t) : w.moved < w'.moved := by
  induction h with
  | nil => obtain ⟨_, ht, _⟩ := he; simp at ht
  | @cons w t ts w1 t' w2 ts' hs hr ih =>
    by_cases h0 : enabled w t
    · have := tstep_moves hs h0; have := run_mono hr; omega
    · obtain ⟨-, hc, hst, hm, -⟩ := tstep_idle hs h0
      obtain ⟨u, hu, heu⟩ := he
      simp only [List.mem_cons] at hu
      rcases hu with rfl | hu
      · exact absurd heu h0
      · have := ih ⟨u, hu, (enabled_congr hc hst u).2 heu⟩; omega

theorem run_idle {w ts w' ts'} (h : Run w ts w' ts') (hn : ∀ t ∈ ts, ¬ enabled w t) :
    ts' = ts ∧ w'.chans = w.chans ∧ w'.staged = w.staged ∧ w'.moved = w.moved ∧ w'.born = w.born := by
  induction h with
  | nil => simp
  | @cons w t ts w1 t' w2 ts' hs hr ih =>
    obtain ⟨rfl, hc, hst, hm, hb, -⟩ := tstep_idle hs (hn t (List.mem_cons_self _ _))
    have := ih (fun u hu => by rw [enabled_congr hc hst u]; exact hn u (List.mem_cons_of_mem _ hu))
    obtain ⟨rfl, h1, h2, h3, h4⟩ := this
    exact ⟨rfl, h1.trans hc, h2.trans hst, h3.trans hm, h4.trans hb⟩

/-- T4 (6.8): every configuration ticks; if some task is enabled, some task
    moves; if none is, no task, channel or spawn changes, so the run ends with
    a defined outcome (all tasks done, or `Deadlock`). -/
theorem progress (c : Cfg) : ∃ c' m, Tick c c' m ∧
    ((∃ t ∈ c.tasks, enabled (start c) t) → 0 < m) ∧
    ((∀ t ∈ c.tasks, ¬ enabled (start c) t) → m = 0 ∧ c'.tasks = c.tasks ∧ c'.chans = c.chans) := by
  obtain ⟨w, ts, hr⟩ := run_total (start c) c.tasks
  refine ⟨_, _, .mk hr, fun he => ?_, fun hn => ?_⟩
  · have := run_progress hr he; simpa [start] using this
  · obtain ⟨rfl, hc, hs, hm, hb⟩ := run_idle hr hn
    simp [finish, start] at hc hs hm hb ⊢
    simp [hm, hb, hs, hc, deliver]

/-! ## T6 for tasks: no sharing of mutable data

Registers are typed by a register kind map `RK` (true = holds a reference to
mutable data, false = plain data).  The typing rules mirror W-OWN-3 and
W-SPAWN-1: what is sent, emitted, told to an actor, received, or passed to a
spawned task is plain data; references only come from `alloc` in the task
that owns them. -/

inductive Wt (RK : Nat → Bool) : List Instr → Prop where
  | nil : Wt RK []
  | send {c r k} : RK r = false → Wt RK k → Wt RK (.send c r :: k)
  | recv {c r k} : RK r = false → Wt RK k → Wt RK (.recv c r :: k)
  | close {c k} : Wt RK k → Wt RK (.close c :: k)
  | emit {r k} : RK r = false → Wt RK k → Wt RK (.emit r :: k)
  | alloc {r k} : RK r = true → Wt RK k → Wt RK (.alloc r :: k)
  | spawn {b as k} : Wt RK b → (∀ a ∈ as, RK a = false) → Wt RK k → Wt RK (.spawn b as :: k)
  | tell {a r k} : RK r = false → Wt RK k → Wt RK (.tell a r :: k)
  | select {arms d k} : (∀ p ∈ arms, RK p.2 = false) → Wt RK k → Wt RK (.select arms d :: k)

/-- A task is well formed: its code is typed, data registers hold no
    references, and every reference it holds is to a location it allocated. -/
structure TaskOK (RK : Nat → Bool) (t : Task) : Prop where
  wt : Wt RK t.code
  data : ∀ r, RK r = false → (t.regs r).isRef = false
  own : ∀ r l, t.regs r = .ref l → l.1 = t.id ∧ l.2 < t.allocs

def DataL (l : List CV) : Prop := ∀ v ∈ l, v.isRef = false

structure WOK (RK : Nat → Bool) (base : Nat) (w : W) : Prop where
  chans : ∀ c, DataL (w.chans c).buf
  staged : ∀ p ∈ w.staged, p.2.isRef = false
  mail : ∀ p ∈ w.mail, p.2.isRef = false
  born : ∀ t ∈ w.born, TaskOK RK t ∧ base ≤ t.id ∧ t.id < w.nextId
  bornNodup : (w.born.map Task.id).Nodup
  le : base ≤ w.nextId

theorem nodup_append' {l1 l2 : List Nat} : (l1 ++ l2).Nodup ↔ l1.Nodup ∧ l2.Nodup ∧ ∀ a ∈ l1, ∀ b ∈ l2, a ≠ b :=
  List.pairwise_append

theorem minReady_mem {rdy : Nat → Bool} : ∀ {as : List (Nat × Nat)} {a}, minReady rdy as = some a → a ∈ as
  | [], _, h => by simp [minReady] at h
  | b :: bs, a, h => by
    simp only [minReady] at h
    cases hm : minReady rdy bs with
    | none =>
      rw [hm] at h; simp only at h
      by_cases hb : rdy b.1 = true
      · simp [hb] at h; subst h; exact List.mem_cons_self _ _
      · simp [hb] at h
    | some x =>
      rw [hm] at h; simp only at h
      by_cases hb : rdy b.1 = true ∧ b.1 ≤ x.1
      · simp only [hb, and_self, ite_true, Option.some.injEq] at h; subst h; exact List.mem_cons_self _ _
      · rw [if_neg hb] at h; cases h; exact List.mem_cons_of_mem _ (minReady_mem hm)

theorem recvFrom_data {ch v ch'} (hd : DataL ch.buf) (h : recvFrom ch = some (v, ch')) :
    v.isRef = false ∧ DataL ch'.buf := by
  unfold recvFrom at h
  split at h
  · rename_i x rest hb
    simp at h; obtain ⟨rfl, rfl⟩ := h
    rw [hb] at hd
    exact ⟨hd _ (List.mem_cons_self _ _), fun u hu => hd u (List.mem_cons_of_mem _ hu)⟩
  · split at h
    · simp at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨rfl, hd⟩
    · simp at h

theorem setC_data {f : Nat → Chan} {c ch} (hf : ∀ c, DataL (f c).buf) (hc : DataL ch.buf) :
    ∀ c', DataL (setC f c ch c').buf := by
  intro c'; unfold setC; split
  · exact hc
  · exact hf c'

theorem taskOK_setR_data {RK t r v} (h : TaskOK RK t) (hv : v.isRef = false) (k : List Instr)
    (hk : Wt RK k) : TaskOK RK { t with code := k, regs := setR t.regs r v } := by
  refine ⟨hk, fun r' hr' => ?_, fun r' l hl => ?_⟩
  · simp only [setR]; split
    · exact hv
    · exact h.data r' hr'
  · simp only [setR] at hl; split at hl
    · subst hl; simp [CV.isRef] at hv
    · exact h.own r' l hl

theorem tstep_ok {RK base w t w' t'} (hs : TStep w t w' t') (ht : TaskOK RK t) (hw : WOK RK base w) :
    TaskOK RK t' ∧ WOK RK base w' := by
  cases hs with
  | done => exact ⟨ht, hw⟩
  | sendOk hc _ _ =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | send hr hk =>
      refine ⟨⟨hk, ht.data, ht.own⟩, ⟨hw.chans, ?_, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
      intro p hp; simp only [List.mem_append, List.mem_singleton] at hp
      rcases hp with hp | rfl
      · exact hw.staged p hp
      · exact ht.data _ hr
  | sendFull => exact ⟨ht, ⟨hw.chans, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | sendClosed => exact ⟨⟨.nil, ht.data, ht.own⟩, ⟨hw.chans, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | recvOk hc hrf =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | recv hr hk =>
      obtain ⟨hv, hch⟩ := recvFrom_data (hw.chans _) hrf
      exact ⟨taskOK_setR_data ht hv _ hk, ⟨setC_data hw.chans hch, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | recvBlocked => exact ⟨ht, hw⟩
  | close hc =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | close hk => exact ⟨⟨hk, ht.data, ht.own⟩, ⟨setC_data hw.chans (hw.chans _), hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | emit hc =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | emit _ hk => exact ⟨⟨hk, ht.data, ht.own⟩, ⟨hw.chans, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | alloc hc =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | alloc hr hk =>
      refine ⟨⟨hk, fun r' hr' => ?_, fun r' l hl => ?_⟩, ?_⟩
      · simp only [setR]; split
        · rename_i h; subst h; rw [hr] at hr'; cases hr'
        · exact ht.data r' hr'
      · simp only [setR] at hl; split at hl
        · cases hl; exact ⟨rfl, Nat.lt_succ_self _⟩
        · obtain ⟨h1, h2⟩ := ht.own r' l hl; exact ⟨h1, Nat.lt_succ_of_lt h2⟩
      · exact ⟨hw.chans, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩
  | spawn hc =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | spawn hb has hk =>
      refine ⟨⟨hk, ht.data, ht.own⟩, ⟨hw.chans, hw.staged, hw.mail, ?_, ?_, Nat.le_succ_of_le hw.le⟩⟩
      · intro u hu; simp only [List.mem_append, List.mem_singleton] at hu
        rcases hu with hu | rfl
        · obtain ⟨h1, h2, h3⟩ := hw.born u hu; exact ⟨h1, h2, Nat.lt_succ_of_lt h3⟩
        · refine ⟨⟨hb, fun r _ => ?_, fun r l hl => ?_⟩, hw.le, Nat.lt_succ_self _⟩
          · simp only [argRegs]; split
            · rename_i a ha; exact ht.data a (has a (List.get?_mem ha))
            · rfl
          · simp only [argRegs] at hl; split at hl
            · rename_i a ha; have := ht.data a (has a (List.get?_mem ha)); rw [hl] at this; simp [CV.isRef] at this
            · cases hl
      · simp only [List.map_append, List.map_cons, List.map_nil]
        rw [nodup_append']
        refine ⟨hw.bornNodup, List.pairwise_singleton _ _, ?_⟩
        intro a ha b hb' hab; simp at hb'
        simp only [List.mem_map] at ha; obtain ⟨u, hu, rfl⟩ := ha
        have := (hw.born u hu).2.2; omega
  | tell hc =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | tell hr hk =>
      refine ⟨⟨hk, ht.data, ht.own⟩, ⟨hw.chans, hw.staged, ?_, hw.born, hw.bornNodup, hw.le⟩⟩
      intro p hp; simp only [List.mem_append, List.mem_singleton] at hp
      rcases hp with hp | rfl
      · exact hw.mail p hp
      · exact ht.data _ hr
  | selRecv hc hm hrf =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | select harms hk =>
      obtain ⟨hv, hch⟩ := recvFrom_data (hw.chans _) hrf
      have hr := harms _ (minReady_mem hm)
      exact ⟨taskOK_setR_data ht hv _ hk, ⟨setC_data hw.chans hch, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | selDefault hc =>
    have hwt := ht.wt; rw [hc] at hwt; cases hwt with
    | select _ hk => exact ⟨⟨hk, ht.data, ht.own⟩, ⟨hw.chans, hw.staged, hw.mail, hw.born, hw.bornNodup, hw.le⟩⟩
  | selBlocked => exact ⟨ht, hw⟩

theorem tstep_id {w t w' t'} (hs : TStep w t w' t') : t'.id = t.id := by
  cases hs <;> rfl

theorem run_ok {RK base w ts w' ts'} (hr : Run w ts w' ts') (ht : ∀ t ∈ ts, TaskOK RK t) (hw : WOK RK base w) :
    (∀ t ∈ ts', TaskOK RK t) ∧ WOK RK base w' ∧ ts'.map Task.id = ts.map Task.id ∧ w.nextId ≤ w'.nextId := by
  induction hr with
  | nil => exact ⟨fun _ h => by simp at h, hw, rfl, Nat.le_refl _⟩
  | @cons w t ts w1 t' w2 ts' hs _ ih =>
    obtain ⟨h1, h2⟩ := tstep_ok hs (ht _ (List.mem_cons_self _ _)) hw
    obtain ⟨h3, h4, h5, h6⟩ := ih (fun u hu => ht u (List.mem_cons_of_mem _ hu)) h2
    refine ⟨fun u hu => ?_, h4, ?_, ?_⟩
    · simp only [List.mem_cons] at hu; rcases hu with rfl | hu
      · exact h1
      · exact h3 u hu
    · simp [h5, tstep_id hs]
    · have : w.nextId ≤ w1.nextId := by cases hs <;> simp
      exact Nat.le_trans this h6

/-- The invariant of a configuration. -/
structure Inv (RK : Nat → Bool) (c : Cfg) : Prop where
  tasks : ∀ t ∈ c.tasks, TaskOK RK t
  chans : ∀ ch, DataL (c.chans ch).buf
  actors : ∀ a ∈ c.actors, DataL a.mbox
  ids : ∀ t ∈ c.tasks, t.id < c.nextId
  nodup : (c.tasks.map Task.id).Nodup

theorem deliver_data {f : Nat → Chan} (hf : ∀ c, DataL (f c).buf) :
    ∀ (st : List (Nat × CV)), (∀ p ∈ st, p.2.isRef = false) → ∀ c, DataL (deliver f st c).buf := by
  intro st
  induction st generalizing f with
  | nil => intro _; simpa [deliver] using hf
  | cons p rest ih =>
    intro hst
    obtain ⟨c0, v⟩ := p
    simp only [deliver]
    apply ih
    · intro c'; unfold setC; split
      · intro u hu; simp only [List.mem_append, List.mem_singleton] at hu
        rcases hu with hu | rfl
        · exact hf c0 u hu
        · exact hst _ (List.mem_cons_self _ _)
      · exact hf c'
    · intro q hq; exact hst q (List.mem_cons_of_mem _ hq)

theorem runActor_data {a : Actor} (h : DataL a.mbox) : DataL (runActor a).mbox := by
  unfold runActor; split
  · exact h
  · rename_i v rest hm; rw [hm] at h; exact fun u hu => h u (List.mem_cons_of_mem _ hu)

theorem postMail_data {as : List Actor} (ha : ∀ a ∈ as, DataL a.mbox) {m : List (Nat × CV)}
    (hm : ∀ p ∈ m, p.2.isRef = false) : ∀ a ∈ postMail as m, DataL a.mbox := by
  intro a hamem
  simp only [postMail, List.mem_map] at hamem
  obtain ⟨⟨i, b⟩, hib, rfl⟩ := hamem
  have hb : b ∈ as := List.snd_mem_of_mem_enum hib
  intro u hu
  simp only [List.mem_append, List.mem_map, List.mem_filter] at hu
  rcases hu with hu | ⟨p, ⟨hp, _⟩, rfl⟩
  · exact ha b hb u hu
  · exact hm p hp

/-- T6 for ticks: the ownership invariant is preserved by every tick. -/
theorem tick_inv {RK c c' m} (hi : Inv RK c) (ht : Tick c c' m) : Inv RK c' := by
  cases ht with
  | mk hr =>
    have hw0 : WOK RK c.nextId (start c) :=
      ⟨hi.chans, by simp [start], by simp [start], by simp [start], by simp [start], by simp [start]⟩
    obtain ⟨h1, h2, h3, h4⟩ := run_ok hr hi.tasks hw0
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · intro t htm; simp only [finish, List.mem_append] at htm
      rcases htm with htm | htm
      · exact h1 t htm
      · exact (h2.born t htm).1
    · exact deliver_data h2.chans _ h2.staged
    · apply postMail_data
      · intro a ha; simp only [List.mem_map] at ha; obtain ⟨b, hb, rfl⟩ := ha; exact runActor_data (hi.actors b hb)
      · exact h2.mail
    · intro t htm; simp only [finish, List.mem_append] at htm
      rcases htm with htm | htm
      · have : t.id ∈ c.tasks.map Task.id := by rw [← h3]; exact List.mem_map_of_mem _ htm
        simp only [List.mem_map] at this; obtain ⟨u, hu, he⟩ := this
        have := hi.ids u hu; simp [start] at h4; simp [finish]; omega
      · exact (h2.born t htm).2.2
    · simp only [finish, List.map_append]
      rw [nodup_append', h3]
      refine ⟨hi.nodup, h2.bornNodup, ?_⟩
      intro a ha b hb hab; subst hab
      simp only [List.mem_map] at ha hb
      obtain ⟨u, hu, rfl⟩ := ha; obtain ⟨v, hv, he⟩ := hb
      have := hi.ids u hu; have := (h2.born v hv).2.1; simp [start] at this; omega

/-- T6 (5.8): no two tasks ever hold a reference to the same mutable
    location: references are only to locations a task allocated itself, and
    task ids are distinct. -/
theorem no_sharing {RK c} (hi : Inv RK c) {t1 t2 : Task} (h1 : t1 ∈ c.tasks) (h2 : t2 ∈ c.tasks)
    (hne : t1.id ≠ t2.id) {r1 r2 l} (hr1 : t1.regs r1 = .ref l) (hr2 : t2.regs r2 = .ref l) : False :=
  hne (((hi.tasks t1 h1).own r1 l hr1).1.symm.trans ((hi.tasks t2 h2).own r2 l hr2).1)

/-- Reachable configurations: zero or more ticks. -/
inductive Reach : Cfg → Cfg → Prop where
  | refl {c} : Reach c c
  | tick {c c1 c2 m} : Tick c c1 m → Reach c1 c2 → Reach c c2

theorem reach_inv {RK c c'} (hi : Inv RK c) (h : Reach c c') : Inv RK c' := by
  induction h with
  | refl => exact hi
  | tick ht _ ih => exact ih (tick_inv hi ht)

/-- A program starts as one task (`main`, id 0) with empty channels and actors. -/
def initial (code : List Instr) (chans : Nat → Chan) (actors : List Actor) : Cfg :=
  ⟨[⟨0, code, fun _ => .none, 0⟩], chans, actors, [], 1⟩

theorem initial_inv {RK code chans actors} (hc : Wt RK code) (hch : ∀ c, (chans c).buf = [])
    (ha : ∀ a ∈ actors, a.mbox = []) : Inv RK (initial code chans actors) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro t ht; simp [initial] at ht; subst ht
    exact ⟨hc, fun _ _ => rfl, fun _ _ h => by cases h⟩
  · intro c v hv; simp only [initial] at hv; rw [hch] at hv; cases hv
  · intro a hamem v hv; rw [ha a hamem] at hv; cases hv
  · intro t ht; simp [initial] at ht; subst ht; simp [initial]
  · simp [initial]

/-- T6 for whole runs: in every reachable configuration of a well-typed
    program, distinct tasks never share a mutable location. -/
theorem tasks_never_share {RK code chans actors c'} (hc : Wt RK code) (hch : ∀ c, (chans c).buf = [])
    (ha : ∀ a ∈ actors, a.mbox = []) (hr : Reach (initial code chans actors) c') :
    (c'.tasks.map Task.id).Nodup ∧
    ∀ t1 ∈ c'.tasks, ∀ t2 ∈ c'.tasks, t1.id ≠ t2.id → ∀ r1 r2 l, t1.regs r1 = .ref l → t2.regs r2 ≠ .ref l := by
  have hi := reach_inv (initial_inv hc hch ha) hr
  exact ⟨hi.nodup, fun t1 h1 t2 h2 hne r1 r2 l hr1 hr2 => no_sharing hi h1 h2 hne hr1 hr2⟩

end RCE.Conc
