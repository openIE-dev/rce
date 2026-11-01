def sumSq (n : Nat) : Nat := (List.range (n + 1)).foldl (fun a i => a + i * i) 0

def main : IO Unit := IO.println s!"sum of squares to 10: {sumSq 10}"
