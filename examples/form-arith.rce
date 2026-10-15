spec "arith-edges" v0.1.0
reason "Edge semantics of Form arithmetic across targets (add sub mul div rem fdiv min max fadd fsub fmul)" by "david"
currency max_age=24h

objective O1 "Compute"
  metric computed target 1
  reason "edge test" by "david"

acceptance A1 validates O1
  property "for all a, b: every target computes a op b with the Form's edge semantics"
  reason "one defined behavior" by "david"

deliverable D1 fulfills O1
  schema "AddResult"
  reason "add results" by "david"

deliverable D2 fulfills O1
  schema "SubResult"
  reason "sub results" by "david"

deliverable D3 fulfills O1
  schema "MulResult"
  reason "mul results" by "david"

deliverable D4 fulfills O1
  schema "DivResult"
  reason "div results" by "david"

deliverable D5 fulfills O1
  schema "RemResult"
  reason "rem results" by "david"

deliverable D6 fulfills O1
  schema "FdivResult"
  reason "fdiv results" by "david"

deliverable D7 fulfills O1
  schema "MinResult"
  reason "min results" by "david"

deliverable D8 fulfills O1
  schema "MaxResult"
  reason "max results" by "david"

deliverable D9 fulfills O1
  schema "FaddResult"
  reason "fadd results" by "david"

deliverable D10 fulfills O1
  schema "FsubResult"
  reason "fsub results" by "david"

deliverable D11 fulfills O1
  schema "FmulResult"
  reason "fmul results" by "david"

role R1 "add" execution_node
  reason "computes add" by "david"

role R2 "sub" execution_node
  reason "computes sub" by "david"

role R3 "mul" execution_node
  reason "computes mul" by "david"

role R4 "div" execution_node
  reason "computes div" by "david"

role R5 "rem" execution_node
  reason "computes rem" by "david"

role R6 "fdiv" execution_node
  reason "computes fdiv" by "david"

role R7 "min" execution_node
  reason "computes min" by "david"

role R8 "max" execution_node
  reason "computes max" by "david"

role R9 "fadd" execution_node
  reason "computes fadd" by "david"

role R10 "fsub" execution_node
  reason "computes fsub" by "david"

role R11 "fmul" execution_node
  reason "computes fmul" by "david"

dependency DEP1 crate "numbers"
  reason "operand pairs" by "david"

flow F1 from DEP1 to R1
  reason "operands to add" by "david"

flow G1 from R1 to D1
  reason "add results recorded" by "david"

flow F2 from DEP1 to R2
  reason "operands to sub" by "david"

flow G2 from R2 to D2
  reason "sub results recorded" by "david"

flow F3 from DEP1 to R3
  reason "operands to mul" by "david"

flow G3 from R3 to D3
  reason "mul results recorded" by "david"

flow F4 from DEP1 to R4
  reason "operands to div" by "david"

flow G4 from R4 to D4
  reason "div results recorded" by "david"

flow F5 from DEP1 to R5
  reason "operands to rem" by "david"

flow G5 from R5 to D5
  reason "rem results recorded" by "david"

flow F6 from DEP1 to R6
  reason "operands to fdiv" by "david"

flow G6 from R6 to D6
  reason "fdiv results recorded" by "david"

flow F7 from DEP1 to R7
  reason "operands to min" by "david"

flow G7 from R7 to D7
  reason "min results recorded" by "david"

flow F8 from DEP1 to R8
  reason "operands to max" by "david"

flow G8 from R8 to D8
  reason "max results recorded" by "david"

flow F9 from DEP1 to R9
  reason "operands to fadd" by "david"

flow G9 from R9 to D9
  reason "fadd results recorded" by "david"

flow F10 from DEP1 to R10
  reason "operands to fsub" by "david"

flow G10 from R10 to D10
  reason "fsub results recorded" by "david"

flow F11 from DEP1 to R11
  reason "operands to fmul" by "david"

flow G11 from R11 to D11
  reason "fmul results recorded" by "david"

escalation E1 from "div" to R4 timeout 300ms
  reason "self-loop for timeout testing" by "david"

change_control approvers ["david"] review_period 24h
