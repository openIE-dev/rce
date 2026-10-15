# Applies the IRP's own confEcma.xsl to every testNNN*.txml of
# vectors/ui/scxml-irp-full, writing the .scxml next to it (the
# ECMAScript-datamodel documents the suite defines).
# confEcma.xsl is XSLT 2.0 (xsl:analyze-string), so it needs an XSLT 2+
# processor: Saxon-HE via `pip install saxonche` (lxml/libxslt is 1.0 only
# and silently yields empty cond attributes).
import glob, os, sys
from saxonche import PySaxonProcessor
d = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'vectors/ui/scxml-irp-full')
with PySaxonProcessor(license=False) as p:
    x = p.new_xslt30_processor().compile_stylesheet(stylesheet_file=d + '/confEcma.xsl')
    for f in sorted(glob.glob(d + '/*/*.txml')):
        x.transform_to_file(source_file=f, output_file=f[:-5] + '.scxml')
