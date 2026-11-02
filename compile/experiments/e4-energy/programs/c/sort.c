/* E4 kernel: insertion sort of n (at most 1024) values from a linear congruential generator,
   returning the sum of value times (position + 1), modulo 2^32. */
#include <stdint.h>

#define CAP 1024u
static uint32_t a[CAP];

uint32_t run(uint32_t n) {
    if (n > CAP) n = CAP;
    uint32_t x = 12345u;
    for (uint32_t i = 0; i < n; i++) {
        x = x * 1103515245u + 12345u;
        a[i] = (x >> 16) & 0x7fffu;
    }
    for (uint32_t i = 1; i < n; i++) {
        uint32_t v = a[i], j = i;
        while (j > 0 && a[j - 1] > v) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = v;
    }
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++) sum += a[i] * (i + 1u);
    return sum;
}
