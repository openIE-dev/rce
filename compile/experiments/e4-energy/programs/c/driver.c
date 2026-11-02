/* E4 native driver: calls the kernel's run(n) R times and prints run(n) and a checksum of all R calls.
   The kernel is in another translation unit and is compiled without link-time optimisation,
   so the calls cannot be removed or moved out of the loop. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t run(uint32_t n);

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: PROGRAM N R\n");
        return 2;
    }
    uint32_t n = (uint32_t)strtoul(argv[1], 0, 10);
    unsigned long r = strtoul(argv[2], 0, 10);
    uint32_t first = run(n), x = 0;
    for (unsigned long i = 0; i < r; i++) x ^= run(n) + (uint32_t)i;
    printf("%u %u\n", first, x);
    return 0;
}
