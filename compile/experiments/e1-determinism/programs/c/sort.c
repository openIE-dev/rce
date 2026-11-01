#include <stdio.h>
#include <stdlib.h>

static int cmp(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}

int main(void) {
    long v[64];
    unsigned long s = 12345;
    for (int i = 0; i < 64; i++) {
        s = s * 6364136223846793005UL + 1442695040888963407UL;
        v[i] = (long)(s >> 33) % 1000;
    }
    qsort(v, 64, sizeof v[0], cmp);
    long sum = 0;
    for (int i = 0; i < 64; i++) sum += v[i] * (i + 1);
    printf("%ld %ld %ld\n", v[0], v[63], sum);
    return 0;
}
