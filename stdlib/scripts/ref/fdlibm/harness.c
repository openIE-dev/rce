/* Oracle for lib/num/special.rce: FDLIBM 5.3 (netlib) compiled with gcc on
   x86-64 (SSE2, -O0 -ffp-contract=off -fno-builtin). Reads "fn hexbits"
   lines, prints "fn hexbits hexresult" (sign for lgamma appended). */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
double __ieee754_exp(double), __ieee754_log(double), log1p(double), expm1(double),
       erf(double), erfc(double), __ieee754_lgamma_r(double, int *);
static double d(uint64_t u) { double x; memcpy(&x, &u, 8); return x; }
static uint64_t u(double x) { uint64_t v; memcpy(&v, &x, 8); return v; }
int main(void) {
  char fn[32]; unsigned long long b;
  while (scanf("%31s %llx", fn, &b) == 2) {
    double x = d(b), y; int sg = 0;
    if (!strcmp(fn, "exp")) y = __ieee754_exp(x);
    else if (!strcmp(fn, "log")) y = __ieee754_log(x);
    else if (!strcmp(fn, "log1p")) y = log1p(x);
    else if (!strcmp(fn, "expm1")) y = expm1(x);
    else if (!strcmp(fn, "erf")) y = erf(x);
    else if (!strcmp(fn, "erfc")) y = erfc(x);
    else if (!strcmp(fn, "lgamma")) y = __ieee754_lgamma_r(x, &sg);
    else return 2;
    printf("%s %016llx %016llx %d\n", fn, b, (unsigned long long)u(y), sg);
  }
  return 0;
}
