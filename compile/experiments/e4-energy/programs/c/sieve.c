/* E4 kernel: the number of primes below n (n at most 8192), by the sieve of Eratosthenes.
   A composite is marked with this call's stamp, so the array needs no clearing between calls. */
#include <stdint.h>

#define CAP 8192u
static uint32_t mark[CAP];
static uint32_t stamp;

uint32_t run(uint32_t n) {
    if (n > CAP) n = CAP;
    uint32_t s = ++stamp, count = 0;
    for (uint32_t i = 2; i < n; i++) {
        if (mark[i] != s) {
            count++;
            for (uint32_t j = i * i; j < n; j += i) mark[j] = s;
        }
    }
    return count;
}
