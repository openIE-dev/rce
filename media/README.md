# Media: signals and buffers

Programs that process fixed-size buffers of samples, the shape of most
audio and image code.

| Program | What it shows |
|---|---|
| `gain.rce` | lane-wise arithmetic on arrays (`buf * gs`, `a + b`) and clipping to the `i16` range |
| `moving_average.rce` | a three-sample moving average over a list |

Lane-wise arithmetic applies an operator to each element of two arrays of
the same length. The result is defined element by element, so it is the
same on every target.

Limits in draft 0.5: there are no stream or sample-rate types and no FFT
primitive.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
