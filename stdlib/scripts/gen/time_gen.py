# Usage: python3 scripts/gen/time_gen.py, then replace the lines after
# "# CPython datetime cross-checks" in tests/time/time_checks.rce.
# Generates the CPython cross-check section of tests/time/time_checks.rce:
# random proleptic Gregorian dates in years 1..9999 (seed 20261015) with
# days since 1970-01-01, weekday, ISO week date, day of year, and RFC 3339
# formatting of random instants. CPython's datetime is the oracle.
import random, datetime as dt
r = random.Random(20261015)
E = dt.date(1970, 1, 1).toordinal()
for i in range(60):
    d = dt.date.fromordinal(r.randint(1, dt.date(9999, 12, 31).toordinal()))
    iy, iw, iwd = d.isocalendar()
    print('check date_ok(%d, %d, %d, %d, %d, [%d, %d, %d], %d)' % (d.year, d.month, d.day, d.toordinal() - E, (d.weekday() + 1) % 7, iy, iw, iwd, d.timetuple().tm_yday))
for i in range(20):
    s = r.randint(-8000000000, 8000000000); ns = r.randint(0, 999999999); off = r.choice([0, 0, 60, -300, 330, 345, -570, 840])
    tz = dt.timezone(dt.timedelta(minutes=off))
    t = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=s)
    lt = t.astimezone(tz)
    base = lt.strftime('%Y-%m-%dT%H:%M:%S')
    o = 'Z' if off == 0 else ('%s%02d:%02d' % ('-' if off < 0 else '+', abs(off) // 60, abs(off) % 60))
    print('check fmt_ok(%d, %d, "%s")' % (s * 10**9 + ns, off, base + '.' + '%09d' % ns + o))
