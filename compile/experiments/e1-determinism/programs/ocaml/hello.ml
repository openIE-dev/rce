let () =
  let squares = List.init 10 (fun i -> (i + 1) * (i + 1)) in
  Printf.printf "hello %d\n" (List.fold_left ( + ) 0 squares)
