import csv
L=["L0 Form core/hash","L1 trace log storage","L2 compiled runtime+fuel","L3 concurrent runtime","L4 event admission+clock","L5 output sinks (display/audio/file/net)","L6 input vocabulary sources","L7 env failure as event + caps","L8 WASM component FFI","L9 text beyond scalars","L10 recursion (lang)","L11 data-structure perf","L12 energy receipts","L13 bounded E (lang)"]
# codes: P provided, Pa partial, A absent ; hazard text
envs={
"Linux syscalls (glibc)":"P P P P P P P Pa Pa A P P Pa P|signals, scheduler order, clock, /proc ambient, io_uring kernel surface",
"Linux io_uring":"P P P P P Pa Pa Pa A A P P A P|completion order nondeterministic; 60% kCTF exploits (Google 2023)",
"Linux epoll":"P P P P P Pa Pa Pa A A P P A P|readiness order, spurious wakeups",
"macOS Darwin/kqueue/Mach":"P P P P P P P Pa Pa Pa P P Pa P|GCD scheduling order, Mach ports ambient, IOReport energy only",
"Cocoa/Metal":"P P P P P P P A A P P P Pa P|GPU float not bit-exact, main-thread rule",
"Windows Win32/NT/IOCP":"P P P P P P P Pa Pa P P P Pa P|completion order, undocumented NT syscall numbers vary per build",
"DirectX 12":"P P P P Pa P A A A A P P A P|GPU float, driver variance",
"Android (NDK/ART)":"P P P P P P P Pa Pa P P P Pa P|lifecycle kills, io_uring blocked by seccomp",
"iOS":"P P P P P P P Pa Pa P P P Pa P|no JIT for 3rd party (W^X), background suspension",
"FreeBSD (kqueue/Capsicum)":"P P P P P Pa P P Pa A P P Pa P|Capsicum caps fit L7; small GUI/audio",
"UEFI":"P Pa P A Pa Pa Pa A A A P P A P|boot services only, single core, timer events",
"no_std Rust":"P Pa P A A A A A A A P Pa A P|everything host-supplied; no alloc unless provided",
"Zephyr RTOS":"P Pa P P P Pa P Pa Pa A P Pa Pa P|preemptive priority scheduling, ISR timing",
"seL4":"P Pa P P Pa Pa Pa P Pa A P Pa A P|capabilities native; drivers user-level; small ecosystem",
"FreeRTOS/other RTOS":"P Pa P P P Pa Pa A A A P Pa A P|C memory safety, tick ISR",
"Browser Web APIs":"P Pa P Pa P P P Pa P P P P A P|event loop order, timers clamped, perf.now coarsened",
"WebGPU":"P A Pa Pa A P A A A A P P A P|WGSL float not bit-exact (reassoc, FTZ allowed)",
"WASI 0.2":"P P P Pa P Pa Pa P P A P P A P|pollable order, wall-clock import",
"WASI 0.3 (CM async)":"P P P Pa P Pa Pa P P A P P A P|future/stream completion order host-chosen",
"Wasmtime runtime":"P P P Pa P Pa Pa P P A P P A P|CVE-2026-34971 aarch64 Cranelift escape; fuel+epoch available",
"libc glibc":"P P P P P Pa Pa A A Pa P P A P|locale, malloc order, errno, C memory safety",
"musl":"P P P P P Pa Pa A A A P P A P|smaller, slower malloc, DNS differences",
"Rust std":"P P P P P Pa Pa A Pa A P P A P|HashMap random seed, thread scheduling",
"Go runtime":"P P P P P Pa Pa A A A P P A P|map iteration randomized, select random, GC",
"JVM":"P P P P P P P A A P P P A P|JIT, GC pauses, HashMap order by identity",
".NET CLR":"P P P P P P P A A P P P A P|JIT, GC, string hash randomized",
"CPython":"P P Pa Pa P Pa Pa A A Pa P Pa A P|PYTHONHASHSEED, GIL/free-threading, slow",
"Node.js":"P P Pa Pa P Pa Pa Pa A P P P A P|libuv completion order, npm supply chain",
"Deno":"P P Pa Pa P Pa Pa P A P P P A P|permission flags (caps), V8",
"Bun":"P P Pa Pa P Pa Pa Pa A P P P A P|JSC, young runtime",
"Erlang BEAM/OTP":"P P Pa P P Pa Pa P A Pa P Pa A P|preemptive reductions, mailbox order across nodes",
"Haskell RTS":"P P P P P Pa Pa A A A P Pa A P|lazy eval, GC, unsafePerformIO",
"OCaml 5 effects":"P P P P P Pa Pa A A A P P A P|domains parallel, effects untyped",
"Zig std":"P P P Pa P Pa Pa A A A P P A P|manual memory, Io interface in flux",
"Swift":"P P P P P P P A A P P P A P|actor scheduling, ARC",
"Mojo":"P P P Pa P Pa Pa A A A P P A P|young, partly closed",
"rr / Pernosco":"P P A A P A A A A A P A A P|needs PMU, single core, x86/aarch64 Linux only",
"FoundationDB simulation":"P P P P P A A P A A P P A P|requires code written against simulated interfaces",
"TigerBeetle VOPR":"P P P Pa P A A P A A P P A P|same; fixed IO interface",
"Antithesis":"P P P P P A A P A A P P A P|deterministic hypervisor, commercial",
"Elm architecture":"P Pa Pa A P P P Pa A P P Pa A P|Cmd/Sub = effects as messages; browser only",
"Hermit (Meta)":"P Pa P Pa P A A A A A P P A P|maintenance mode, not security boundary, unsupported syscalls pass through",
"Deterministic WASM (fuel, no threads, NaN canon)":"P P P Pa P A A P P A P P A P|NaN bits, relaxed-simd nondeterministic",
"Fuchsia/Zircon":"P P P P P P P P Pa P P P A P|handles=caps, component framework",
"CloudABI (abandoned)":"P P P Pa P Pa Pa P A A P P A P|project discontinued, superseded by WASI",
}
M={"P":"provided","Pa":"partial","A":"absent"}
with open("MATRIX.csv","w",newline="") as f:
  w=csv.writer(f); w.writerow(["environment","host_need","status","dominant_hazard"])
  for e,v in envs.items():
    s,h=v.split("|")
    for l,c in zip(L,s.split()): w.writerow([e,l,M[c],h])
import json; json.dump({e:v.split("|")[0].split() for e,v in envs.items()},open("figures/m.json","w"))
