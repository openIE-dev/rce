# Body: control and sensing

Programs that read sensors and decide what a device should do. Each
reading is an event, and an impossible reading is refused instead of being
acted on.

| Program | What it shows |
|---|---|
| `thermostat.rce` | a heater with hysteresis, using integer tenths of a degree |
| `arm.rce` | a proportional controller for one joint, using `f64` |

Time enters a program as logged events: the program's logical clock, or
readings that `os.clock` admits. The `time` module gives durations,
instants, calendar dates and RFC 3339 text over those readings. Continuous
dynamics are written as discrete steps, as `arm.rce` does; integrators and
controllers as library modules are rows of the library coverage ledger
(`spec/data/coverage-map.json`: `num.calc`, `ctl.pid`, `ctl.filter`).

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
