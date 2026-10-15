# Applies the IRP's own confEcma.xsl to every testNNN*.txml of
# vectors/ui/scxml-irp-full, writing the .scxml next to it (the
# ECMAScript-datamodel documents the suite defines).
import glob, sys
from lxml import etree
d = sys.argv[1] if len(sys.argv) > 1 else 'vectors/ui/scxml-irp-full'
x = etree.XSLT(etree.parse(d + '/confEcma.xsl'))
for f in sorted(glob.glob(d + '/*/*.txml')):
    open(f[:-5] + '.scxml', 'wb').write(etree.tostring(x(etree.parse(f))))
