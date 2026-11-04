# rce-vm-web

Browser bindings for the rce Graph VM. Compiles to `wasm32-unknown-unknown` and ships as the `@rce/runtime` npm package.

Same Rust source as the one `rce` toolchain (`spec/checker`): the CPU interpreter from `rce-vm-backend-cpu` plus the wgpu backend from `rce-vm-backend-wgpu`, which targets WebGPU in the browser through wgpu's web target.

## Build

```bash
# 1. Compile to wasm32 in release mode.
cargo build --target wasm32-unknown-unknown -p rce-vm-web --release

# 2. Generate the JS shim + npm-style package.
mkdir -p crates/rce-vm-web/pkg
wasm-bindgen \
    --target web \
    --out-dir crates/rce-vm-web/pkg \
    --no-typescript \
    target/wasm32-unknown-unknown/release/rce_vm_web.wasm

# (Optional) Optimize the wasm size further.
wasm-opt -O3 -o crates/rce-vm-web/pkg/rce_vm_web_bg.wasm \
    crates/rce-vm-web/pkg/rce_vm_web_bg.wasm
```

## Demo (local)

```bash
# From the repo root, write the graph terms of an RCE program as .fg files
# (spec 10.16); each is named by its content hash:
rce graph terms compute/examples/dense.rce crates/rce-vm-web/demo/

# Serve.
cd crates/rce-vm-web && python3 -m http.server 8000

# Open http://localhost:8000/demo/ in Chrome (M113+).
```

## API

```js
import init, { Vm, Module } from "@rce/runtime";

await init();

const vm = new Vm();
await vm.initWgpu();      // optional; rejects if WebGPU is unavailable.

const fgBytes = new Uint8Array(await (await fetch("/model.fg")).arrayBuffer());
const stBytes = new Uint8Array(await (await fetch("/model.safetensors")).arrayBuffer());
const module = Module.load(fgBytes, stBytes);

const out = vm.run(module, JSON.stringify({ x: [[1, 2, 3, 4]] }));
console.log(JSON.parse(out));
```

## Status

- CPU interpreter: full coverage of the v0.3 op set (matmul, broadcasting, reduce, softmax, activations, transpose, reshape, concat, fused matmul+scale+softmax)
- WebGPU backend: f32 only by WGSL spec; matmul (2D + batched 3D), elementwise add/sub/mul/div with full broadcasting, softmax(-1), 2D transpose, relu/sigmoid/tanh/gelu, fused matmul+scale+softmax
- Wasm leaves (`OpKind::CodeBlock { language: "wasm" }`): not yet supported in browser; planned for v0.2 via the WebAssembly host API
- Tensor inputs accepted as nested JSON arrays; outputs returned as JSON-encoded arrays
- Bundle size: ~940 KB raw, ~260 KB gzipped

## Browser compatibility

WebGPU is required for GPU acceleration:

- Chrome stable (M113+) ✓
- Safari Technology Preview ✓
- Firefox Nightly with `dom.webgpu.enabled` ✓

CPU mode runs everywhere wasm runs.

## License

MIT OR Apache-2.0, as for the rest of the RCE repository.
