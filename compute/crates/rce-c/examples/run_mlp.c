/*
 * run_mlp.c — example C consumer of the rce C ABI.
 *
 * Loads /tmp/mlp.fg + /tmp/mlp.safetensors (produced by `rce make-sample-mlp`),
 * runs the graph, prints the output, and frees everything.
 *
 * Build (from repo root):
 *   cargo build --release -p rce-c
 *   clang -o /tmp/run_mlp \
 *       -I crates/rce-c/include \
 *       crates/rce-c/examples/run_mlp.c \
 *       -L target/release -lrce \
 *       -Wl,-rpath,$(pwd)/target/release
 *
 * Run:
 *   /tmp/run_mlp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rce.h"

static unsigned char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, n, f);
    fclose(f);
    if ((long)got != n) { free(buf); return NULL; }
    *out_len = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    const char *fg_path = argc > 1 ? argv[1] : "/tmp/mlp.fg";
    const char *st_path = argc > 2 ? argv[2] : "/tmp/mlp.safetensors";

    size_t fg_len = 0, st_len = 0;
    unsigned char *fg = read_file(fg_path, &fg_len);
    unsigned char *st = read_file(st_path, &st_len);
    if (!fg || !st) return 1;

    RceModule *module = NULL;
    RceStatus s = rce_module_load(fg, fg_len, st, st_len, &module);
    if (s != RceStatus_Ok) {
        fprintf(stderr, "rce_module_load failed (%d): %s\n", (int)s, rce_last_error());
        return 1;
    }

    RceVm *vm = NULL;
    s = rce_vm_new(0, &vm);  // CPU only
    if (s != RceStatus_Ok) {
        fprintf(stderr, "rce_vm_new failed (%d): %s\n", (int)s, rce_last_error());
        return 1;
    }
    /* Sample MLP has f32 weights; opt into auto-casting f64 JSON inputs. */
    rce_vm_set_cast_f32(vm, 1);

    char *out = NULL;
    const char *input = "{\"x\": [[1.0, 2.0, 3.0, 4.0]]}";
    s = rce_run(vm, module, input, &out);
    if (s != RceStatus_Ok) {
        fprintf(stderr, "rce_run failed (%d): %s\n", (int)s, rce_last_error());
        return 1;
    }

    printf("input : %s\n", input);
    printf("output: %s\n", out);

    rce_string_free(out);
    rce_vm_free(vm);
    rce_module_free(module);
    free(fg);
    free(st);
    return 0;
}
