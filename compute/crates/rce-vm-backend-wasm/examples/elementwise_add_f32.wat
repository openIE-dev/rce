;; rce Wasm leaf — element-wise f32 add.
;; Conforms to v0.2 of the leaf tensor ABI (docs/spec/fg-wasm-leaf-abi.md).
;;
;; Two equal-shape f32 input tensors -> sum tensor of the same shape.
;; Bump-allocates outputs; assumes the caller (host) frees inputs.

(module
  (memory (export "memory") 1)

  ;; Bump heap starts at 1KB so the first 1024 bytes are reserved for
  ;; per-instance state. Real modules may want a real allocator.
  (global $heap (mut i32) (i32.const 1024))

  (func $alloc (export "rce_alloc") (param $sz i32) (result i32)
    (local $p i32)
    ;; Round up size to 4-byte alignment so descriptor reads stay aligned.
    (local.set $sz
      (i32.and
        (i32.add (local.get $sz) (i32.const 3))
        (i32.const 0xfffffffc)))
    (local.set $p (global.get $heap))
    (global.set $heap (i32.add (global.get $heap) (local.get $sz)))
    (local.get $p))

  (func (export "rce_free") (param $p i32) (param $sz i32) nop)

  ;; Optional ABI version hint.
  (global (export "rce_abi_version") i32 (i32.const 2))

  ;; rce_dispatch_tensor(count, ins_ptr) -> out_desc_ptr
  (func (export "rce_dispatch_tensor")
        (param $count i32) (param $ins i32)
        (result i32)
    (local $a_desc i32)
    (local $b_desc i32)
    (local $rank i32)
    (local $shape_bytes i32)
    (local $numel i32)
    (local $i i32)
    (local $a_data i32)
    (local $b_data i32)
    (local $out_data i32)
    (local $out_desc i32)

    ;; Decline if we don't get exactly 2 inputs.
    (if (i32.ne (local.get $count) (i32.const 2))
      (then (return (i32.const 0))))

    ;; Read input descriptor pointers from the inputs array.
    (local.set $a_desc (i32.load (local.get $ins)))
    (local.set $b_desc (i32.load (i32.add (local.get $ins) (i32.const 4))))

    ;; Decline if dtypes differ from f32 (code 0).
    (if (i32.or
          (i32.ne (i32.load (i32.add (local.get $a_desc) (i32.const 4))) (i32.const 0))
          (i32.ne (i32.load (i32.add (local.get $b_desc) (i32.const 4))) (i32.const 0)))
      (then (return (i32.const 0))))

    ;; Read rank (assumed equal across both inputs).
    (local.set $rank (i32.load (local.get $a_desc)))
    (local.set $shape_bytes (i32.mul (local.get $rank) (i32.const 4)))

    ;; Decline on rank mismatch.
    (if (i32.ne (local.get $rank) (i32.load (local.get $b_desc)))
      (then (return (i32.const 0))))

    ;; Compute numel by walking the shape array of $a_desc.
    (local.set $numel (i32.const 1))
    (local.set $i (i32.const 0))
    (block $shape_done
      (loop $shape_next
        (br_if $shape_done (i32.ge_u (local.get $i) (local.get $rank)))
        (local.set $numel
          (i32.mul
            (local.get $numel)
            (i32.load
              (i32.add
                (i32.add (local.get $a_desc) (i32.const 16))
                (i32.mul (local.get $i) (i32.const 4))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $shape_next)))

    ;; Read input data pointers.
    (local.set $a_data (i32.load (i32.add (local.get $a_desc) (i32.const 8))))
    (local.set $b_data (i32.load (i32.add (local.get $b_desc) (i32.const 8))))

    ;; Allocate output data buffer (numel * 4 bytes for f32).
    (local.set $out_data
      (call $alloc (i32.mul (local.get $numel) (i32.const 4))))

    ;; Allocate output descriptor (16 + shape_bytes).
    (local.set $out_desc
      (call $alloc (i32.add (i32.const 16) (local.get $shape_bytes))))

    ;; Write descriptor: rank, dtype=0 (f32), data_ptr, reserved=0, then shape.
    (i32.store (local.get $out_desc) (local.get $rank))
    (i32.store (i32.add (local.get $out_desc) (i32.const 4)) (i32.const 0))
    (i32.store (i32.add (local.get $out_desc) (i32.const 8)) (local.get $out_data))
    (i32.store (i32.add (local.get $out_desc) (i32.const 12)) (i32.const 0))
    (memory.copy
      (i32.add (local.get $out_desc) (i32.const 16))
      (i32.add (local.get $a_desc) (i32.const 16))
      (local.get $shape_bytes))

    ;; out[i] = a[i] + b[i] for i in 0..numel.
    (local.set $i (i32.const 0))
    (block $add_done
      (loop $add_next
        (br_if $add_done (i32.ge_u (local.get $i) (local.get $numel)))
        (f32.store
          (i32.add (local.get $out_data) (i32.mul (local.get $i) (i32.const 4)))
          (f32.add
            (f32.load
              (i32.add (local.get $a_data) (i32.mul (local.get $i) (i32.const 4))))
            (f32.load
              (i32.add (local.get $b_data) (i32.mul (local.get $i) (i32.const 4))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $add_next)))

    (local.get $out_desc)))
