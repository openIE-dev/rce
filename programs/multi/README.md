# Multi: two regions, one clock

Programs whose state is split over two regions that the same events drive.
Each region holds what one part of the device knows (a door and a car, a
queue and a toner cartridge, a lobby and a match), each refuses what it must
refuse on its own, and every program says something a reader outside it can
see: an emitted value or a refusal. They were written as the population of
the fifth dimension run of the research programme (specification 6.10, `rce
dim`), which needs observable programs with more than one region, and they
are ordinary teaching programs first.

| Program | Regions | What it shows |
|---|---|---|
| `elevator.rce` | car, door | a car that never moves with its door open |
| `vending.rce` | credit, stock | a sale needs a coin and a can |
| `crossing.rce` | signal, pedestrian | a walk phase of one tick after a press |
| `parking.rce` | barrier, lot | a barrier that never admits a car to a full lot |
| `printer.rce` | queue, toner | printing takes a job and a unit of toner |
| `lobby.rce` | lobby, match | a match that holds its players until it ends |
| `card.rce` | PIN, account | a card that locks after three wrong PINs |
| `washer.rce` | cycle, door | a door locked from start to finish |
| `session.rce` | auth, rate limit | login throttled after failures, relaxed by time |
| `player.rce` | transport, volume | a player whose output is the track it plays |
| `climate.rce` | mode, setpoint | heat when cold, cool when hot |
| `reservoir.rce` | tank, pump | a level that neither overflows nor runs dry unnoticed |

A step belongs to one region and reads only that region's fields, so a
condition that spans two regions is written as two steps on the same
event: the elevator's door refuses `Up` while it is open and the car moves
on `Up` otherwise, and a refusal in either region refuses the event for
both (6.2, T8). Every state field has a small domain (a count with a cap, a
flag, a phase), so `rce dim` explores each program completely.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
