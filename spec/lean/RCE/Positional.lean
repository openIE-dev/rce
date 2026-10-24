import RCE.NameFree
/-! The positional embedding of 6.1 (T7, resolution A of 2026-10-23): a model
    of what `spec/checker` hashes. The checker holds *named* values (a record
    carries its type name and its field names) and a module's declarations
    (the fields of every struct, event and error in declaration order, the
    variants of every kind in order, the events in order). Its `state_hash`
    and entry `hash` are over the *positional form*: a record of a declared
    type is the list of its field values in declaration order, a variant is
    `[k, fields]` with `k` the variant's index in its kind, an event's type
    is its index among the events, and a record of an undeclared type (the
    host's `host.*` records) keeps its names.

    Two theorems link this to the De Bruijn model of `RCE/NameFree.lean`:

    * `pos_rename`: the positional form is unchanged by any renaming that is
      injective and fixes every name the module does not declare, which is
      what `rce fmt --nameless` does (10.6). So for *any* hash function the
      state hash and the entry chain of a renamed program equal the
      original's (`state_hash_name_independent`, `chain_name_independent`).
    * `eventIndex_eq_idx`: the `type` the checker hashes for an event is the
      declaration position `idx` that `NameFree.entryKey` hashes, so the two
      encodings agree on the one component they share, and both are
      name-free by the same lemma (`idx_rename`). -/
namespace RCE.Positional
open RCE.NameFree (idx idx_rename)

/-! ### Named values and declarations (the checker's side) -/

mutual
/-- A value as the checker holds it: scalars are already name-free
    (`Nat` stands for any scalar), lists are positional, a record carries
    its type (or variant) name and its named fields. -/
inductive NV where
  | scalar (n : Nat)
  | list (xs : NVs)
  | record (ty : String) (fs : Fs)
inductive NVs where
  | nil
  | cons (v : NV) (vs : NVs)
inductive Fs where
  | nil
  | cons (f : String) (v : NV) (fs : Fs)
end

/-- The declarations of a module that the embedding reads. -/
structure Decls where
  /-- struct, event and error types: name and field names in declaration order -/
  types : List (String × List String)
  /-- kinds: each a list of (variant name, field names) in declaration order -/
  kinds : List (List (String × List String))
  /-- events in declaration order -/
  events : List String

/-- The positional form (the `FV` of 6.1 restricted to what the embedding
    produces): scalars, lists, and named records only for undeclared types. -/
inductive PV where
  | scalar (n : Nat)
  | list (xs : List PV)
  | record (fs : List (String × PV))

/-! ### Lookups -/

def declFields : List (String × List String) → String → Option (List String)
  | [], _ => none
  | (n, fs) :: rest, ty => if n = ty then some fs else declFields rest ty

/-- The index of a variant within one kind and its fields. -/
def variantIn : List (String × List String) → String → Option (Nat × List String)
  | [], _ => none
  | (v, fs) :: rest, n => if v = n then some (0, fs) else (variantIn rest n).map fun p => (p.1 + 1, p.2)

def variantIndex : List (List (String × List String)) → String → Option (Nat × List String)
  | [], _ => none
  | k :: ks, n => match variantIn k n with
    | some p => some p
    | none => variantIndex ks n

def lookupF : List (String × PV) → String → PV
  | [], _ => .scalar 0
  | (g, v) :: rest, f => if g = f then v else lookupF rest f

/-- Fields in declaration order; a field the value lacks reads as null. -/
def reorder (decl : List String) (pf : List (String × PV)) : List PV :=
  decl.map (lookupF pf)

/-! ### The embedding (interp.rs `form_value_pos`, `state_pos_json`, `event_pos_json`) -/

mutual
def pos (d : Decls) : NV → PV
  | .scalar n => .scalar n
  | .list xs => .list (posL d xs)
  | .record ty fs =>
    match declFields d.types ty with
    | some decl => .list (reorder decl (posF d fs))
    | none => match variantIndex d.kinds ty with
      | some (k, decl) => .list [.scalar k, .list (reorder decl (posF d fs))]
      | none => .record (posF d fs)
def posL (d : Decls) : NVs → List PV
  | .nil => []
  | .cons v vs => pos d v :: posL d vs
def posF (d : Decls) : Fs → List (String × PV)
  | .nil => []
  | .cons f v fs => (f, pos d v) :: posF d fs
end

/-- A region's fields in declaration order, as the checker keeps them. -/
abbrev Region := String × List (String × NV)

/-- The hashed form of a state: regions in order, each its field values in order. -/
def posState (d : Decls) (s : List Region) : PV :=
  .list (s.map fun r => .list (r.2.map fun f => pos d f.2))

/-- The event index the checker writes as `type`: its position among the
    declared events (`length` when undeclared, as `idx`). -/
def eventIndex (d : Decls) (name : String) : Nat := idx d.events name

/-- The hashed form of an event `{"payload": P, "type": k}`. -/
def posEvent (d : Decls) (e : String × NV) : Nat × PV := (eventIndex d e.1, pos d e.2)

/-! ### Renaming, as `rce fmt --nameless` does it (10.6) -/

/-- A type name the module declares (struct, event, error or variant). -/
def Declared (d : Decls) (ty : String) : Prop :=
  (declFields d.types ty).isSome ∨ (variantIndex d.kinds ty).isSome

def Decls.rename (ρ : String → String) (d : Decls) : Decls :=
  ⟨d.types.map fun p => (ρ p.1, p.2.map ρ),
   d.kinds.map fun k => k.map fun p => (ρ p.1, p.2.map ρ),
   d.events.map ρ⟩

mutual
/-- Renaming a value: a record of a declared type has its type and field
    names renamed; a record of an undeclared type (a host record, whose keys
    the host fixes) keeps them. Values inside are renamed either way. -/
def NV.rename (d : Decls) (ρ : String → String) : NV → NV
  | .scalar n => .scalar n
  | .list xs => .list (NVs.rename d ρ xs)
  | .record ty fs =>
    if (declFields d.types ty).isSome ∨ (variantIndex d.kinds ty).isSome
    then .record (ρ ty) (Fs.renameAll d ρ fs)
    else .record ty (Fs.renameVals d ρ fs)
def NVs.rename (d : Decls) (ρ : String → String) : NVs → NVs
  | .nil => .nil
  | .cons v vs => .cons (NV.rename d ρ v) (NVs.rename d ρ vs)
def Fs.renameAll (d : Decls) (ρ : String → String) : Fs → Fs
  | .nil => .nil
  | .cons f v fs => .cons (ρ f) (NV.rename d ρ v) (Fs.renameAll d ρ fs)
def Fs.renameVals (d : Decls) (ρ : String → String) : Fs → Fs
  | .nil => .nil
  | .cons f v fs => .cons f (NV.rename d ρ v) (Fs.renameVals d ρ fs)
end

def renameRegion (d : Decls) (ρ : String → String) (r : Region) : Region :=
  (ρ r.1, r.2.map fun f => (ρ f.1, NV.rename d ρ f.2))

/-- The renamings `rce fmt --nameless` produces: injective, and fixing every
    type name the module does not declare. -/
structure Nameless (d : Decls) (ρ : String → String) : Prop where
  inj : ∀ a b, ρ a = ρ b → a = b
  fix : ∀ x, ¬ Declared d x → ρ x = x

/-! ### Lemmas on the lookups -/

theorem declFields_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (ty : String) :
    ∀ ts : List (String × List String),
    declFields (ts.map fun p => (ρ p.1, p.2.map ρ)) (ρ ty) = (declFields ts ty).map (·.map ρ)
  | [] => rfl
  | (n, fs) :: rest => by
    simp only [List.map, declFields]
    by_cases h : n = ty
    · subst h; simp
    · rw [if_neg (fun h' => h (hρ _ _ h')), if_neg h, declFields_rename ρ hρ ty rest]

theorem variantIn_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (n : String) :
    ∀ k : List (String × List String),
    variantIn (k.map fun p => (ρ p.1, p.2.map ρ)) (ρ n) = (variantIn k n).map fun p => (p.1, p.2.map ρ)
  | [] => rfl
  | (v, fs) :: rest => by
    simp only [List.map, variantIn]
    by_cases h : v = n
    · subst h; simp
    · rw [if_neg (fun h' => h (hρ _ _ h')), if_neg h, variantIn_rename ρ hρ n rest]
      cases variantIn rest n <;> rfl

theorem variantIndex_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (n : String) :
    ∀ ks : List (List (String × List String)),
    variantIndex (ks.map fun k => k.map fun p => (ρ p.1, p.2.map ρ)) (ρ n) =
      (variantIndex ks n).map fun p => (p.1, p.2.map ρ)
  | [] => rfl
  | k :: ks => by
    simp only [List.map, variantIndex, variantIn_rename ρ hρ n k]
    cases variantIn k n with
    | none => simp [variantIndex_rename ρ hρ n ks]
    | some p => rfl

theorem lookupF_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b) (f : String) :
    ∀ pf : List (String × PV), lookupF (pf.map fun p => (ρ p.1, p.2)) (ρ f) = lookupF pf f
  | [] => rfl
  | (g, v) :: rest => by
    simp only [List.map, lookupF]
    by_cases h : g = f
    · subst h; simp
    · rw [if_neg (fun h' => h (hρ _ _ h')), if_neg h, lookupF_rename ρ hρ f rest]

theorem reorder_rename (ρ : String → String) (hρ : ∀ a b, ρ a = ρ b → a = b)
    (decl : List String) (pf : List (String × PV)) :
    reorder (decl.map ρ) (pf.map fun p => (ρ p.1, p.2)) = reorder decl pf := by
  simp only [reorder, List.map_map]
  apply List.map_congr_left
  intro f _
  exact lookupF_rename ρ hρ f pf

/-- An undeclared type stays undeclared after a nameless renaming. -/
theorem declFields_undeclared (d : Decls) (ρ : String → String) (h : Nameless d ρ) (ty : String)
    (hty : ¬ Declared d ty) : declFields (d.rename ρ).types ty = none := by
  have hfix := h.fix ty hty
  rw [← hfix, Decls.rename, declFields_rename ρ h.inj]
  have : declFields d.types ty = none := by
    cases hd : declFields d.types ty with
    | none => rfl
    | some _ => exact absurd (Or.inl (by simp [hd])) hty
  rw [this]; rfl

theorem variantIndex_undeclared (d : Decls) (ρ : String → String) (h : Nameless d ρ) (ty : String)
    (hty : ¬ Declared d ty) : variantIndex (d.rename ρ).kinds ty = none := by
  have hfix := h.fix ty hty
  rw [← hfix, Decls.rename, variantIndex_rename ρ h.inj]
  have : variantIndex d.kinds ty = none := by
    cases hd : variantIndex d.kinds ty with
    | none => rfl
    | some _ => exact absurd (Or.inr (by simp [hd])) hty
  rw [this]; rfl

/-! ### The theorem: the positional form is name-free -/

mutual
theorem pos_rename (d : Decls) (ρ : String → String) (h : Nameless d ρ) :
    ∀ v : NV, pos (d.rename ρ) (NV.rename d ρ v) = pos d v
  | .scalar n => rfl
  | .list xs => by simp only [NV.rename, pos, posL_rename d ρ h xs]
  | .record ty fs => by
    by_cases hdecl : (declFields d.types ty).isSome ∨ (variantIndex d.kinds ty).isSome
    · simp only [NV.rename, if_pos hdecl, pos]
      rw [posF_renameAll d ρ h fs]
      have ht := declFields_rename ρ h.inj ty d.types
      have hv := variantIndex_rename ρ h.inj ty d.kinds
      simp only [Decls.rename] at ht hv ⊢
      rw [ht, hv]
      cases hd1 : declFields d.types ty with
      | some decl => simp [reorder_rename ρ h.inj]
      | none =>
        simp only [Option.map_none']
        cases hd2 : variantIndex d.kinds ty with
        | some p => simp [reorder_rename ρ h.inj]
        | none => rw [hd1, hd2] at hdecl; simp at hdecl
    · simp only [NV.rename, if_neg hdecl, pos]
      rw [posF_renameVals d ρ h fs, declFields_undeclared d ρ h ty hdecl, variantIndex_undeclared d ρ h ty hdecl]
      have h1 : declFields d.types ty = none := by
        cases hd : declFields d.types ty with
        | none => rfl
        | some _ => exact absurd (Or.inl (by simp [hd])) hdecl
      have h2 : variantIndex d.kinds ty = none := by
        cases hd : variantIndex d.kinds ty with
        | none => rfl
        | some _ => exact absurd (Or.inr (by simp [hd])) hdecl
      rw [h1, h2]
theorem posL_rename (d : Decls) (ρ : String → String) (h : Nameless d ρ) :
    ∀ xs : NVs, posL (d.rename ρ) (NVs.rename d ρ xs) = posL d xs
  | .nil => rfl
  | .cons v vs => by simp only [NVs.rename, posL, pos_rename d ρ h v, posL_rename d ρ h vs]
/-- Renamed fields embed to the renamed field list (names still attached). -/
theorem posF_renameAll (d : Decls) (ρ : String → String) (h : Nameless d ρ) :
    ∀ fs : Fs, posF (d.rename ρ) (Fs.renameAll d ρ fs) = (posF d fs).map fun p => (ρ p.1, p.2)
  | .nil => rfl
  | .cons f v fs => by simp only [Fs.renameAll, posF, List.map, pos_rename d ρ h v, posF_renameAll d ρ h fs]
theorem posF_renameVals (d : Decls) (ρ : String → String) (h : Nameless d ρ) :
    ∀ fs : Fs, posF (d.rename ρ) (Fs.renameVals d ρ fs) = posF d fs
  | .nil => rfl
  | .cons f v fs => by simp only [Fs.renameVals, posF, pos_rename d ρ h v, posF_renameVals d ρ h fs]
end

theorem posState_rename (d : Decls) (ρ : String → String) (h : Nameless d ρ) (s : List Region) :
    posState (d.rename ρ) (s.map (renameRegion d ρ)) = posState d s := by
  simp only [posState, List.map_map]
  congr 1
  apply List.map_congr_left
  intro r _
  simp only [Function.comp, renameRegion, List.map_map]
  congr 1
  apply List.map_congr_left
  intro f _
  exact pos_rename d ρ h f.2

/-- The event index is the De Bruijn position of `RCE/NameFree.lean`
    (`entryKey` hashes `idx ds name`): the two encodings share this
    component by definition. -/
theorem eventIndex_eq_idx (d : Decls) (name : String) : eventIndex d name = idx d.events name := rfl

theorem posEvent_rename (d : Decls) (ρ : String → String) (h : Nameless d ρ) (e : String × NV) :
    posEvent (d.rename ρ) (ρ e.1, NV.rename d ρ e.2) = posEvent d e := by
  unfold posEvent
  rw [pos_rename d ρ h]
  simp only [eventIndex, Decls.rename, idx_rename ρ h.inj]

/-! ### Any hash of the positional form, and the chain -/

/-- `state_hash` for any hash function `H` of the positional form. -/
def stateHash (H : PV → Nat) (d : Decls) (s : List Region) : Nat := H (posState d s)

theorem state_hash_name_independent (H : PV → Nat) (d : Decls) (ρ : String → String) (h : Nameless d ρ)
    (s : List Region) : stateHash H (d.rename ρ) (s.map (renameRegion d ρ)) = stateHash H d s := by
  simp only [stateHash, posState_rename d ρ h]

/-- One entry's hash: `H2 (event, prev_hash, seq, state_hash)` over the
    positional event, as `host.rs` and `interp.rs` compute it. -/
def entryHash (H : PV → Nat) (H2 : (Nat × PV) × Nat × Nat × Nat → Nat) (d : Decls)
    (e : String × NV) (prev seq : Nat) (s : List Region) : Nat :=
  H2 (posEvent d e, prev, seq, stateHash H d s)

/-- The chain of a log: each entry's hash from the previous one (genesis 0). -/
def chain (H : PV → Nat) (H2 : (Nat × PV) × Nat × Nat × Nat → Nat) (d : Decls) :
    List ((String × NV) × List Region) → Nat → Nat → List Nat
  | [], _, _ => []
  | (e, s) :: rest, prev, seq =>
    let h := entryHash H H2 d e prev seq s
    h :: chain H H2 d rest h (seq + 1)

def renameLog (d : Decls) (ρ : String → String) (log : List ((String × NV) × List Region)) :
    List ((String × NV) × List Region) :=
  log.map fun p => ((ρ p.1.1, NV.rename d ρ p.1.2), p.2.map (renameRegion d ρ))

/-- **T7 for the implementation's hashes.** For every pair of hash functions
    the whole chain of a nameless-renamed log equals the original's: the
    state hashes, the entry hashes and the previous-hash links. -/
theorem chain_name_independent (H : PV → Nat) (H2 : (Nat × PV) × Nat × Nat × Nat → Nat) (d : Decls)
    (ρ : String → String) (h : Nameless d ρ) :
    ∀ (log : List ((String × NV) × List Region)) (prev seq : Nat),
    chain H H2 (d.rename ρ) (renameLog d ρ log) prev seq = chain H H2 d log prev seq
  | [], _, _ => rfl
  | (e, s) :: rest, prev, seq => by
    simp only [renameLog, List.map, chain, entryHash]
    rw [posEvent_rename d ρ h e, state_hash_name_independent H d ρ h s]
    have ih := chain_name_independent H H2 d ρ h rest (H2 (posEvent d e, prev, seq, stateHash H d s)) (seq + 1)
    simp only [renameLog] at ih
    rw [ih]

end RCE.Positional
