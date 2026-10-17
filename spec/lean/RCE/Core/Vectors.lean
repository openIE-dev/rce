import RCE.Core.Lang
/-! S4 seed: executable vectors for the core interpreter, checked by
    `decide`. These are the shape of the differential cases run against
    the reference interpreter with the conformance suite. -/
namespace RCE.Core

def u8 : Ty := .int 0 255

/-- 200 + 100 at u8 is a checked overflow trap (spec 6.3 N2). -/
example : eval [] 10 [] (.bin .add u8 (.lit (.int 200)) (.lit (.int 100))) = .err (.overflow 300 0 255) := by decide
/-- -7 / 2 truncates toward zero, -7 % 2 has the dividend's sign. -/
example : eval [] 10 [] (.bin .div (.int (-128) 127) (.lit (.int (-7))) (.lit (.int 2))) = .ok (.int (-3)) := by decide
example : eval [] 10 [] (.bin .rem (.int (-128) 127) (.lit (.int (-7))) (.lit (.int 2))) = .ok (.int (-1)) := by decide
/-- A declared error is caught; a trap passes through `catch`. -/
example : eval [] 10 [] (.catch_ (.raise 3) (.lit (.int 1))) = .ok (.int 1) := by decide
example : eval [] 10 [] (.catch_ (.bin .div u8 (.lit (.int 1)) (.lit (.int 0))) (.lit (.int 1))) = .err .divZero := by decide
/-- Enum match: `case (inr 5) of inl x => x | inr y => y + 1`. -/
example : eval [] 10 [] (.case (.inr (.lit (.int 5))) (.var 0) (.bin .add u8 (.var 0) (.lit (.int 1)))) = .ok (.int 6) := by decide
/-- Text concatenation. -/
example : eval [] 10 [] (.bin .cat .text (.lit (.text "ab")) (.lit (.text "c"))) = .ok (.text "abc") := by decide
/-- An unbounded recursive call ends in exhaustion, never a hang. -/
example : eval [(u8, u8, .call 0 (.var 0))] 50 [] (.call 0 (.lit (.int 1))) = .oof := by decide

/-- `try none else 7` fails with the declared error 7, which `catch` handles. -/
example : eval [] 10 [] (.catch_ (.try_ (.inl (.lit .unit)) 7) (.lit (.int 0))) = .ok (.int 0) := by decide
example : eval [] 10 [] (.try_ (.inr (.lit (.int 4))) 7) = .ok (.int 4) := by decide
/-- `300 as u8` traps; `200 as u8` converts. -/
example : eval [] 10 [] (.cast 0 255 (.lit (.int 300))) = .err (.overflow 300 0 255) := by decide
example : eval [] 10 [] (.cast 0 255 (.lit (.int 200))) = .ok (.int 200) := by decide
/-- `eval`'s fuel bounds nesting depth: `f(1)` with `f(x) = x` nests the
    call node, the entry tick and the body, so it needs depth 3. Total tick
    counts, comparable with the reference, are `evalC` (Count.lean). -/
example : eval [(u8, u8, .var 0)] 3 [] (.call 0 (.lit (.int 1))) = .ok (.int 1) := by decide
example : eval [(u8, u8, .var 0)] 2 [] (.call 0 (.lit (.int 1))) = .oof := by decide

end RCE.Core

/-! S1 bit operators and N2 remainder (book 6.3). -/
namespace RCE.Core
private abbrev i8 : Ty := .int (-128) 127
private abbrev u8' : Ty := .int 0 255
example : eval [] 10 [] (.bin .rem i8 (.lit (.int (-128))) (.lit (.int (-1)))) = .err (.overflow 128 (-128) 127) := by decide
example : eval [] 10 [] (.bin .band i8 (.lit (.int (-6))) (.lit (.int 13))) = .ok (.int 8) := by decide
example : eval [] 10 [] (.bin .bor i8 (.lit (.int (-6))) (.lit (.int 13))) = .ok (.int (-1)) := by decide
example : eval [] 10 [] (.bin .bxor i8 (.lit (.int (-6))) (.lit (.int 13))) = .ok (.int (-9)) := by decide
example : eval [] 10 [] (.bin .shl i8 (.lit (.int 64)) (.lit (.int 1))) = .ok (.int (-128)) := by decide
example : eval [] 10 [] (.bin .shl u8' (.lit (.int 200)) (.lit (.int 1))) = .ok (.int 144) := by decide
example : eval [] 10 [] (.bin .shr i8 (.lit (.int (-5))) (.lit (.int 1))) = .ok (.int (-3)) := by decide
example : eval [] 10 [] (.bin .shl u8' (.lit (.int 1)) (.lit (.int 8))) = .err .shift := by decide
end RCE.Core
