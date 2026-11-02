/* E4 kernel: CRC-32 (reflected polynomial 0xEDB88320), bit by bit, of n bytes from a
   linear congruential generator. */
#include <stdint.h>

uint32_t run(uint32_t n) {
    uint32_t crc = 0xffffffffu, x = 1u;
    for (uint32_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        crc ^= x >> 24;
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
