#!/bin/sh
# Build the FDLIBM 5.3 oracle (sources from https://www.netlib.org/fdlibm/,
# sha256 in SHA256SUMS) used by scripts/gen/special_gen.py.
set -e
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS >/dev/null
gcc -O0 -ffp-contract=off -fno-builtin -D__LITTLE_ENDIAN -w -o fdlibm_ref harness.c e_exp.c e_log.c s_log1p.c s_expm1.c e_lgamma_r.c s_erf.c k_sin.c k_cos.c s_floor.c s_fabs.c
