/* E4 kernel: the total number of set bits in n values from a linear congruential generator,
   counted by clearing the lowest set bit. */
#include <stdint.h>

uint32_t run(uint32_t n) {
    uint32_t x = 7u, total = 0;
    for (uint32_t i = 0; i < n; i++) {
        x = x * 22695477u + 1u;
        uint32_t y = x;
        while (y) {
            y &= y - 1u;
            total++;
        }
    }
    return total;
}
