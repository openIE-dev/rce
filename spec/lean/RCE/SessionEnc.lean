import RCE.SessionRec
/-! The encoding of surface protocols (C.11.3: `send`, `recv`, `choose`,
    `offer`, `loop` with `break`) into the recursive session types of
    `SessionRec`, mechanized.

    Surface protocols `SP` are step sequences written as trees: each step
    holds its continuation, `done` ends the current sequence (the end of a
    loop body repeats the loop; at top level it ends the protocol), and
    `loop body rest` runs `body` until a `break`, then `rest`. Choice arms
    are binary and carry their own continuation (steps after a `choose` or
    `offer` block are copied into both arms). A `break` outside any loop is
    refused by W-SES; here it ends the protocol.

    `enc s e k` encodes `s` given the type `e` that follows the end of the
    sequence and `k` that follows a `break`:
    `enc (loop b r) e k = mu (enc b (var 0) (shift 0 (enc r e k)))`.

    Proved: the encoding of a protocol is closed (`enc_top_closed`);
    it commutes with duality (`enc_dual`), so the two endpoints
    `enc s` and `enc (sdual s)` are compatible and enjoy progress
    (`enc_comp`, `enc_progress`); and it is a weak bisimulation between
    the surface protocol, run with a stack of loop frames, and its encoding
    under one-endpoint transitions with `mu` unfolding as the silent step
    (`enc_forward`, `enc_backward`): each surface step is matched by the
    same action of the encoding (a loop entry or repeat by an unfolding, a
    `break` by no step), and every transition of the encoding comes from a
    surface step after some `break`s. -/
namespace RCE.SessionRec

inductive SP where
  | done
  | send : SP → SP
  | recv : SP → SP
  | choose : SP → SP → SP
  | offer : SP → SP → SP
  | loop : SP → SP → SP
  | brk

def sdual : SP → SP
  | .done => .done
  | .send k => .recv (sdual k)
  | .recv k => .send (sdual k)
  | .choose a b => .offer (sdual a) (sdual b)
  | .offer a b => .choose (sdual a) (sdual b)
  | .loop b r => .loop (sdual b) (sdual r)
  | .brk => .brk

/-- add one to the variables at or above `c` -/
def shift (c : Nat) : P → P
  | .done => .done
  | .send a => .send (shift c a)
  | .recv a => .recv (shift c a)
  | .choose a b => .choose (shift c a) (shift c b)
  | .offer a b => .offer (shift c a) (shift c b)
  | .mu a => .mu (shift (c + 1) a)
  | .var n => if c ≤ n then .var (n + 1) else .var n

def enc : SP → P → P → P
  | .done, e, _ => e
  | .brk, _, k => k
  | .send s, e, k => .send (enc s e k)
  | .recv s, e, k => .recv (enc s e k)
  | .choose a b, e, k => .choose (enc a e k) (enc b e k)
  | .offer a b, e, k => .offer (enc a e k) (enc b e k)
  | .loop b r, e, k => .mu (enc b (.var 0) (shift 0 (enc r e k)))

/-! ### duality -/

theorem dual_shift : ∀ (c : Nat) (p : P), dual (shift c p) = shift c (dual p)
  | _, .done => rfl
  | c, .send a | c, .recv a => by simp [shift, dual, dual_shift c a]
  | c, .choose a b | c, .offer a b => by simp [shift, dual, dual_shift c a, dual_shift c b]
  | c, .mu a => by simp [shift, dual, dual_shift (c + 1) a]
  | c, .var n => by simp only [shift]; split <;> rfl

theorem enc_dual : ∀ (s : SP) (e k : P), dual (enc s e k) = enc (sdual s) (dual e) (dual k)
  | .done, _, _ | .brk, _, _ => rfl
  | .send s, e, k | .recv s, e, k => by simp [enc, sdual, dual, enc_dual s e k]
  | .choose a b, e, k | .offer a b, e, k => by
    simp [enc, sdual, dual, enc_dual a e k, enc_dual b e k]
  | .loop b r, e, k => by
    simp only [enc, sdual, dual]
    rw [enc_dual b, dual_shift, enc_dual r]; rfl

/-! ### closedness -/

theorem closedAt_shift : ∀ {n : Nat} (c : Nat) (p : P), closedAt n p → closedAt (n + 1) (shift c p)
  | _, _, .done, _ => trivial
  | _, c, .send a, h | _, c, .recv a, h => closedAt_shift c a h
  | _, c, .choose a b, h | _, c, .offer a b, h => ⟨closedAt_shift c a h.1, closedAt_shift c b h.2⟩
  | _, c, .mu a, h => closedAt_shift (c + 1) a h
  | n, c, .var m, h => by
    simp only [shift]; split <;> simp only [closedAt] at h ⊢ <;> omega

theorem enc_closed : ∀ {n : Nat} (s : SP) {e k : P}, closedAt n e → closedAt n k →
    closedAt n (enc s e k)
  | _, .done, _, _, he, _ => he
  | _, .brk, _, _, _, hk => hk
  | _, .send s, _, _, he, hk | _, .recv s, _, _, he, hk => enc_closed s he hk
  | _, .choose a b, _, _, he, hk | _, .offer a b, _, _, he, hk =>
    ⟨enc_closed a he hk, enc_closed b he hk⟩
  | n, .loop b r, _, _, he, hk => by
    show closedAt (n + 1) _
    exact enc_closed b (show (0 : Nat) < n + 1 by omega) (closedAt_shift 0 _ (enc_closed r he hk))

theorem enc_top_closed (s : SP) : closedAt 0 (enc s .done .done) :=
  enc_closed s trivial trivial

/-- the endpoints of a protocol and of its dual are compatible -/
theorem enc_comp (s : SP) : Comp (enc s .done .done) (enc (sdual s) .done .done) := by
  have := (t28r (enc s .done .done)).1
  rwa [enc_dual] at this

/-- and never deadlock: both done, or a joint step exists, along every run -/
theorem enc_progress (s : SP) :
    (enc s .done .done = .done ∧ enc (sdual s) .done .done = .done) ∨
      ∃ a' b', Red (enc s .done .done) (enc (sdual s) .done .done) a' b' :=
  t28r_progress (enc_comp s) (enc_top_closed s)

/-! ### substitution and shifting -/

theorem shift_closed : ∀ {m : Nat} (j : Nat) (p : P), closedAt m p → m ≤ j → shift j p = p
  | _, _, .done, _, _ => rfl
  | _, j, .send a, h, hj | _, j, .recv a, h, hj => by simp [shift, shift_closed j a h hj]
  | _, j, .choose a b, h, hj | _, j, .offer a b, h, hj => by
    simp [shift, shift_closed j a h.1 hj, shift_closed j b h.2 hj]
  | _, j, .mu a, h, hj => by simp [shift, shift_closed (j + 1) a h (by omega)]
  | m, j, .var n, h, hj => by
    simp only [closedAt] at h; simp only [shift]; rw [if_neg (by omega)]

theorem subst_closed : ∀ {m : Nat} (k : Nat) (c p : P), closedAt m p → m ≤ k → subst k c p = p
  | _, _, _, .done, _, _ => rfl
  | _, k, c, .send a, h, hk | _, k, c, .recv a, h, hk => by simp [subst, subst_closed k c a h hk]
  | _, k, c, .choose a b, h, hk | _, k, c, .offer a b, h, hk => by
    simp [subst, subst_closed k c a h.1 hk, subst_closed k c b h.2 hk]
  | _, k, c, .mu a, h, hk => by simp [subst, subst_closed (k + 1) c a h (by omega)]
  | _, k, c, .var n, h, hk => by
    simp only [closedAt] at h; simp only [subst]; rw [if_neg (by omega)]

theorem subst_shift_comm : ∀ (j k : Nat) {c : P} (p : P), closedAt 0 c → j ≤ k →
    subst (k + 1) c (shift j p) = shift j (subst k c p)
  | _, _, _, .done, _, _ => rfl
  | j, k, _, .send a, hc, hj | j, k, _, .recv a, hc, hj => by
    simp [shift, subst, subst_shift_comm j k a hc hj]
  | j, k, _, .choose a b, hc, hj | j, k, _, .offer a b, hc, hj => by
    simp [shift, subst, subst_shift_comm j k a hc hj, subst_shift_comm j k b hc hj]
  | j, k, _, .mu a, hc, hj => by
    simp only [shift, subst]; rw [subst_shift_comm (j + 1) (k + 1) a hc (by omega)]
  | j, k, c, .var n, hc, hj => by
    by_cases h1 : j ≤ n
    · simp only [shift, if_pos h1, subst]
      by_cases h2 : n = k
      · subst h2; simp [shift_closed j c hc (Nat.zero_le _)]
      · rw [if_neg (by omega), if_neg h2]; simp [shift, h1]
    · simp only [shift, if_neg h1, subst]
      rw [if_neg (by omega), if_neg (by omega)]; simp [shift, h1]

theorem subst_enc : ∀ (k : Nat) {c : P} (s : SP) (e kk : P), closedAt 0 c →
    subst k c (enc s e kk) = enc s (subst k c e) (subst k c kk)
  | _, _, .done, _, _, _ | _, _, .brk, _, _, _ => rfl
  | k, _, .send s, e, kk, hc | k, _, .recv s, e, kk, hc => by
    simp [enc, subst, subst_enc k s e kk hc]
  | k, _, .choose a b, e, kk, hc | k, _, .offer a b, e, kk, hc => by
    simp [enc, subst, subst_enc k a e kk hc, subst_enc k b e kk hc]
  | k, c, .loop b r, e, kk, hc => by
    simp only [enc, subst]
    rw [subst_enc (k + 1) b _ _ hc, subst_shift_comm 0 k _ hc (Nat.zero_le _), subst_enc k r e kk hc]
    simp [subst]

/-- unfolding an encoded loop gives its body with the loop as the
    continuation at the end and the rest after a break -/
theorem unfold_loop (b : SP) (R : P) (hR : closedAt 0 R) :
    subst 0 (.mu (enc b (.var 0) (shift 0 R))) (enc b (.var 0) (shift 0 R)) =
      enc b (.mu (enc b (.var 0) (shift 0 R))) R := by
  have hL : closedAt 0 (P.mu (enc b (.var 0) (shift 0 R))) :=
    enc_closed b (show (0 : Nat) < 0 + 1 by omega) (closedAt_shift 0 R hR)
  rw [subst_enc 0 b _ _ hL, shift_closed 0 R hR (Nat.le_refl _), subst_closed 0 _ R hR (Nat.le_refl _)]
  simp [subst]

/-! ### the bisimulation -/

inductive Act where
  | snd | rcv | chL | chR | ofL | ofR

/-- one endpoint's transitions; `none` is the silent unfolding of `mu` -/
inductive PStep : P → Option Act → P → Prop
  | snd {k} : PStep (.send k) (some .snd) k
  | rcv {k} : PStep (.recv k) (some .rcv) k
  | chL {a b} : PStep (.choose a b) (some .chL) a
  | chR {a b} : PStep (.choose a b) (some .chR) b
  | ofL {a b} : PStep (.offer a b) (some .ofL) a
  | ofR {a b} : PStep (.offer a b) (some .ofR) b
  | unf {a} : PStep (.mu a) none (subst 0 (.mu a) a)

/-- a surface configuration: the current steps and the enclosing loops
    (body, rest), innermost first -/
abbrev Cfg := SP × List (SP × SP)

inductive SStep : Cfg → Option Act → Cfg → Prop
  | snd {k F} : SStep (.send k, F) (some .snd) (k, F)
  | rcv {k F} : SStep (.recv k, F) (some .rcv) (k, F)
  | chL {a b F} : SStep (.choose a b, F) (some .chL) (a, F)
  | chR {a b F} : SStep (.choose a b, F) (some .chR) (b, F)
  | ofL {a b F} : SStep (.offer a b, F) (some .ofL) (a, F)
  | ofR {a b F} : SStep (.offer a b, F) (some .ofR) (b, F)
  /-- the end of a loop body repeats the loop -/
  | cont {b r F} : SStep (.done, (b, r) :: F) none (b, (b, r) :: F)
  | enter {b r F} : SStep (.loop b r, F) none (b, (b, r) :: F)
  | brk {b r F} : SStep (.brk, (b, r) :: F) none (r, F)

/-- the continuations the frames give: (end of body, after break) -/
def encF : List (SP × SP) → P × P
  | [] => (.done, .done)
  | (b, r) :: F => (enc (.loop b r) (encF F).1 (encF F).2, enc r (encF F).1 (encF F).2)

def encC (c : Cfg) : P := enc c.1 (encF c.2).1 (encF c.2).2

theorem encF_closed : ∀ F : List (SP × SP), closedAt 0 (encF F).1 ∧ closedAt 0 (encF F).2
  | [] => ⟨trivial, trivial⟩
  | (b, r) :: F => by
    have ⟨h1, h2⟩ := encF_closed F
    exact ⟨enc_closed (.loop b r) h1 h2, enc_closed r h1 h2⟩

/-- **forward**: every surface step is matched by the encoding: the same
    visible action, an unfolding for a loop entry or repeat, and no step at
    all for a `break` (the encoding already continues with the rest) -/
theorem enc_forward {c c' : Cfg} {α : Option Act} (h : SStep c α c') :
    PStep (encC c) α (encC c') ∨ (α = none ∧ encC c = encC c') := by
  cases h with
  | snd => exact Or.inl .snd
  | rcv => exact Or.inl .rcv
  | chL => exact Or.inl .chL
  | chR => exact Or.inl .chR
  | ofL => exact Or.inl .ofL
  | ofR => exact Or.inl .ofR
  | cont =>
    rename_i b r F
    left
    have ⟨h1, h2⟩ := encF_closed F
    have := @PStep.unf (enc b (.var 0) (shift 0 (enc r (encF F).1 (encF F).2)))
    simp only [encC, enc, encF] at this ⊢
    rw [unfold_loop b _ (enc_closed r h1 h2)] at this
    exact this
  | enter =>
    rename_i b r F
    left
    have ⟨h1, h2⟩ := encF_closed F
    have := @PStep.unf (enc b (.var 0) (shift 0 (enc r (encF F).1 (encF F).2)))
    simp only [encC, enc, encF] at this ⊢
    rw [unfold_loop b _ (enc_closed r h1 h2)] at this
    exact this
  | brk => exact Or.inr ⟨rfl, rfl⟩

/-- surface `break` steps, which leave the encoding unchanged -/
inductive Brks : Cfg → Cfg → Prop
  | refl {c} : Brks c c
  | step {b r F c} : Brks (r, F) c → Brks (.brk, (b, r) :: F) c

theorem brks_enc {c c' : Cfg} (h : Brks c c') : encC c = encC c' := by
  induction h with
  | refl => rfl
  | step _ ih => exact ih

/-- **backward**: every transition of the encoding comes from a surface
    step, possibly after some `break`s, and lands on the encoding of the
    surface successor -/
theorem enc_backward : ∀ (F : List (SP × SP)) (s : SP) {α : Option Act} {q : P},
    PStep (encC (s, F)) α q → ∃ c1 c', Brks (s, F) c1 ∧ SStep c1 α c' ∧ q = encC c'
  | F, .send k, _, _, h => by
    cases h; exact ⟨_, _, .refl, .snd, rfl⟩
  | F, .recv k, _, _, h => by
    cases h; exact ⟨_, _, .refl, .rcv, rfl⟩
  | F, .choose a b, _, _, h => by
    cases h
    · exact ⟨_, _, .refl, .chL, rfl⟩
    · exact ⟨_, _, .refl, .chR, rfl⟩
  | F, .offer a b, _, _, h => by
    cases h
    · exact ⟨_, _, .refl, .ofL, rfl⟩
    · exact ⟨_, _, .refl, .ofR, rfl⟩
  | F, .loop b r, _, _, h => by
    have ⟨h1, h2⟩ := encF_closed F
    simp only [encC, enc] at h
    cases h
    refine ⟨_, _, .refl, .enter, ?_⟩
    have := unfold_loop b _ (enc_closed r h1 h2)
    simp only [encC, encF, enc]; exact this
  | [], .done, _, _, h => by simp only [encC, enc, encF] at h; cases h
  | (b, r) :: F, .done, _, _, h => by
    have ⟨h1, h2⟩ := encF_closed F
    simp only [encC, enc, encF] at h
    cases h
    refine ⟨_, _, .refl, .cont, ?_⟩
    have := unfold_loop b _ (enc_closed r h1 h2)
    simp only [encC, encF, enc]; exact this
  | [], .brk, _, _, h => by simp only [encC, enc, encF] at h; cases h
  | (b, r) :: F, .brk, α, q, h => by
    have h' : PStep (encC (r, F)) α q := by simpa only [encC, enc, encF] using h
    obtain ⟨c1, c', hb, hs, hq⟩ := enc_backward F r h'
    exact ⟨c1, c', .step hb, hs, hq⟩

end RCE.SessionRec
