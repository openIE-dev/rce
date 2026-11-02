/* E4 kernel: the n-th Fibonacci number modulo 2^32 (n at most 30), by naive recursion. */
#include <stdint.h>

static uint32_t fib(uint32_t n) { return n < 2u ? n : fib(n - 1u) + fib(n - 2u); }

uint32_t run(uint32_t n) {
    if (n > 30u) n = 30u;
    return fib(n);
}
