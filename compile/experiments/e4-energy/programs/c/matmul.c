/* E4 kernel: the product of two n-by-n integer matrices (n at most 24), returning the sum of
   each entry times (row + column + 1), modulo 2^32. */
#include <stdint.h>

#define CAP 24u
static uint32_t A[CAP][CAP], B[CAP][CAP], C[CAP][CAP];

uint32_t run(uint32_t n) {
    if (n > CAP) n = CAP;
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++) {
            A[i][j] = i * j + 1u;
            B[i][j] = i + 2u * j + 3u;
        }
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++) {
            uint32_t s = 0;
            for (uint32_t k = 0; k < n; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++) sum += C[i][j] * (i + j + 1u);
    return sum;
}
