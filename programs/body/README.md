# Body: control and sensing

Programs that read sensors and decide what a device should do. Each
reading is an event, and an impossible reading keeps the state instead of
being acted on.

| Program | What it shows |
|---|---|
| `thermostat.rce` | a heater with hysteresis, using integer tenths of a degree |
| `arm.rce` | a proportional controller for one joint, using `f64` |

Limits in draft 0.5: there are no duration, date or time-of-day types, and
no construct for continuous dynamics. Time is the program's logical clock.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
