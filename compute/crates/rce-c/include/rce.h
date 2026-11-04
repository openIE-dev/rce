/*
 * rce.h — stable C ABI for the rce Graph VM.
 *
 * Generated from crates/rce-c — do not edit by hand.
 *
 * Build:   cargo build --release -p rce-c
 * Library: target/release/librce.{a,so,dylib}
 *
 * License: MIT OR Apache-2.0 (matches the RCE repository).
 */


#ifndef RCE_H
#define RCE_H

/* Generated with cbindgen:0.29.4 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

/**
 * Status codes returned from every fallible function.
 */
typedef enum {
  RceStatus_Ok = 0,
  /**
   * Null pointer where a non-null pointer was required.
   */
  RceStatus_NullArgument = 1,
  /**
   * `.fg` parse / format error.
   */
  RceStatus_Parse = 2,
  /**
   * safetensors sidecar load error.
   */
  RceStatus_Weights = 3,
  /**
   * Interpret-time error (op unsupported, type mismatch, OOM, etc.).
   */
  RceStatus_Interpret = 4,
  /**
   * JSON input could not be parsed or did not match the graph's parameters.
   */
  RceStatus_Inputs = 5,
  /**
   * wgpu adapter could not be obtained.
   */
  RceStatus_NoGpu = 6,
  /**
   * Generic internal error; check `rce_last_error()`.
   */
  RceStatus_Internal = 7,
} RceStatus;

/**
 * An opaque module — a parsed `.fg` graph with optional bound weights.
 */
typedef struct RceModule RceModule;

/**
 * An opaque VM — a CPU interpreter, optionally with a wgpu backend
 * pre-initialised. Handle is reusable across many `rce_run` calls.
 */
typedef struct RceVm RceVm;

/**
 * Return a borrowed null-terminated string with the last error set on
 * this thread, or NULL if no error has occurred. The pointer is valid
 * until the next failing call on this thread.
 *
 * # Safety
 * The returned pointer is valid for the lifetime of the calling thread
 * up to the next failing rce_* call. Do not free it.
 */
 const char *rce_last_error(void) ;

/**
 * Free a string returned by `rce_run`.
 *
 * # Safety
 * `s` must be a pointer returned by `rce_run` (or NULL); passing any
 * other pointer is undefined behaviour.
 */
 void rce_string_free(char *s) ;

/**
 * Parse a `.fg` byte buffer (and optional safetensors sidecar) into an
 * opaque module handle. On success, `*out_module` is set to a heap-
 * allocated `RceModule` that the caller must free with
 * `rce_module_free`.
 *
 * # Safety
 * `fg_bytes` must point to `fg_len` readable bytes. If `st_bytes` is
 * non-NULL, it must point to `st_len` readable bytes. `out_module`
 * must be a valid `*mut *mut RceModule`.
 */

RceStatus rce_module_load(const uint8_t *fg_bytes,
                          uintptr_t fg_len,
                          const uint8_t *st_bytes,
                          uintptr_t st_len,
                          RceModule **out_module)
;

/**
 * Free a module handle.
 *
 * # Safety
 * `m` must be a pointer returned by `rce_module_load` (or NULL); any
 * other pointer is undefined behaviour. After this call `m` is invalid.
 */
 void rce_module_free(RceModule *m) ;

/**
 * Construct a VM. Pass `try_wgpu = 1` to attempt wgpu initialisation;
 * the call still succeeds if wgpu is unavailable (CPU-only mode).
 *
 * # Safety
 * `out_vm` must be a valid `*mut *mut RceVm`.
 */
 RceStatus rce_vm_new(int32_t try_wgpu, RceVm **out_vm) ;

/**
 * Toggle automatic f64 → f32 input casting. The CPU runtime is dtype-
 * strict; if your graph expects f32 weights but you're sending f64
 * JSON inputs, set this to 1 and inputs are cast on entry. Equivalent
 * to the `--cast-f32` CLI flag.
 *
 * # Safety
 * `vm` must be a valid handle from `rce_vm_new`.
 */
 void rce_vm_set_cast_f32(RceVm *vm, int32_t on) ;

/**
 * Free a VM handle.
 *
 * # Safety
 * `vm` must be a pointer returned by `rce_vm_new` (or NULL); any
 * other pointer is undefined behaviour.
 */
 void rce_vm_free(RceVm *vm) ;

/**
 * Run a module against bound inputs. Inputs and outputs are JSON-
 * encoded; tensors are nested arrays (e.g. `{"x": [[1,2],[3,4]]}`).
 * On success, `*out_json` is set to a heap-allocated null-terminated
 * UTF-8 string the caller must free with `rce_string_free`.
 *
 * # Safety
 * `vm` and `m` must be valid handles. `input_json` must be a non-null
 * null-terminated UTF-8 string. `out_json` must be a valid
 * `*mut *mut c_char`.
 */
 RceStatus rce_run(RceVm *vm, const RceModule *m, const char *input_json, char **out_json) ;

#endif  /* RCE_H */
