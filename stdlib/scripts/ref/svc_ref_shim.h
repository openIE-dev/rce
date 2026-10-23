/* Types and helpers the copied systemd functions use (scripts/ref/svc_ref_build.py). */
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#define WHITESPACE " \t\n\r"
#define ELEMENTSOF(x) (sizeof(x) / sizeof((x)[0]))
#define SAFE_ATO_REFUSE_PLUS_MINUS (1U << 30)
#define SAFE_ATO_REFUSE_LEADING_ZERO (1U << 29)
#define SAFE_ATO_REFUSE_LEADING_WHITESPACE (1U << 28)
#define SAFE_ATO_ALL_FLAGS (SAFE_ATO_REFUSE_PLUS_MINUS|SAFE_ATO_REFUSE_LEADING_ZERO|SAFE_ATO_REFUSE_LEADING_WHITESPACE)
#define SAFE_ATO_MASK_FLAGS(base) ((base) & ~SAFE_ATO_ALL_FLAGS)
#define FLAGS_SET(v, f) ((~(v) & (f)) == 0)
#define _IN_SET(x, ...) ({ int _v = (x); int _a[] = {__VA_ARGS__}; bool _f = false; for (size_t _i = 0; _i < ELEMENTSOF(_a); _i++) if (_a[_i] == _v) _f = true; _f; })
#define IN_SET(x, ...) _IN_SET(x, __VA_ARGS__)
#define assert_not_reached() abort()
static inline const char *startswith(const char *s, const char *p) { size_t n = strlen(p); return strncmp(s, p, n) == 0 ? s + n : NULL; }
#define STARTSWITH_SET(s, a, b) (startswith(s, a) ?: startswith(s, b))
static inline bool streq_ptr(const char *a, const char *b) { return a && b && strcmp(a, b) == 0; }
#define streq(a, b) (strcmp((a), (b)) == 0)
static inline bool SIGNAL_VALID(int signo) { return signo > 0 && signo < _NSIG; }
typedef struct { uint64_t w[4]; } Bitmap;
static bool bitmap_isset(const Bitmap *b, unsigned n) { return n < 256 && (b->w[n / 64] >> (n % 64)) & 1; }
static void bitmap_set(Bitmap *b, unsigned n) { b->w[n / 64] |= 1ULL << (n % 64); }
typedef struct ExitStatusSet { Bitmap status; Bitmap signal; } ExitStatusSet;
typedef struct ExitStatusMapping { const char *name; int class; } ExitStatusMapping;
typedef enum ExitClean { EXIT_CLEAN_DAEMON, EXIT_CLEAN_COMMAND } ExitClean;
