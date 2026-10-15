# Usage: python3 scripts/gen/text_gen.py vectors/text/UTF-8-test.txt, then
# replace the lines after "# Markus Kuhn, UTF-8-test.txt" in
# tests/text/utf8_checks.rce.
# Generates the Kuhn section of tests/text/utf8_checks.rce from
# vectors/text/UTF-8-test.txt: one check per line holding a non-ASCII byte
# (expected values from CPython 3 bytes.decode, which follows the Unicode
# maximal-subpart practice for U+FFFD), then one check for the whole file.
import sys
d = open(sys.argv[1], 'rb').read()
def exp(b):
    try:
        b.decode('utf-8'); err = -1
    except UnicodeDecodeError as e:
        err = e.start
    cps = [ord(c) for c in b.decode('utf-8', 'replace')]
    return err, cps
for no, l in enumerate(d.split(b'\n'), 1):
    if not any(x > 127 for x in l): continue
    err, cps = exp(l)
    print('# line %d' % no); print('check kuhn("%s", %d, [%s])' % (l.hex(), err, ', '.join(map(str, cps))))
err, cps = exp(d)
print('# whole file: first error, code points, U+FFFD count'); print('check kuhn_file("%s", %d, %d, %d)' % (d.hex(), err, len(cps), cps.count(0xFFFD)))
