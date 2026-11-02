/* E4 kernel: total Collatz steps of the starting values 1..n, each capped at 1000 steps. */
#include <stdint.h>

uint32_t run(uint32_t n) {
    uint32_t total = 0;
    for (uint32_t v = 1; v <= n; v++) {
        uint32_t x = v, steps = 0;
        while (x != 1u && steps < 1000u) {
            x = (x & 1u) ? 3u * x + 1u : x >> 1;
            steps++;
        }
        total += steps;
    }
    return total;
}
