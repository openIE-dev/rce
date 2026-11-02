/* E4 kernel: a running sum of squares mixed by a multiply, modulo 2^32. */
#include <stdint.h>

uint32_t run(uint32_t n) {
    uint32_t s = 0;
    for (uint32_t i = 0; i < n; i++) s = s * 31u + i * i;
    return s;
}
