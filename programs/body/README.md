# Body: control and sensing

Programs that read sensors and decide what a device should do. Each
reading is an event, and an impossible reading keeps the state instead of
being acted on.

| Program | What it shows |
|---|---|
| `thermostat.rce` | a heater with hysteresis, using integer tenths of a degree |
| `arm.rce` | a proportional controller for one joint, using `f64` |

Limits in the current draft: there are no duration, date or time-of-day
types in the language, and no construct for continuous dynamics. Time is the
program's logical clock. The standard library module `time` (`use time`)
gives durations, instants and proleptic Gregorian dates as `i64`
nanoseconds and day counts, with RFC 3339 text, for time a program receives
in an event.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
