# txml -> ecmascript SCXML, following confEcma.xsl for the conf:
# attributes and elements the section-3 tests use
import re, sys, os
import xml.etree.ElementTree as ET
SC = 'http://www.w3.org/2005/07/scxml'; CF = 'http://www.w3.org/2005/scxml-conformance'
ET.register_namespace('', SC)
def q(t): return '{%s}%s' % (SC, t)
def idval(v):
    m = re.match(r'([0-9]+)([=<>]=?)(.*)', v); op = '==' if m.group(2) == '=' else m.group(2)
    return 'Var%s %s %s' % (m.group(1), op, m.group(3))
ATTR = {
 'targetpass': lambda v: ('target', 'pass'), 'targetfail': lambda v: ('target', 'fail'),
 'datamodel': lambda v: ('datamodel', 'ecmascript'), 'id': lambda v: ('id', 'Var' + v),
 'location': lambda v: ('location', 'Var' + v), 'invalidLocation': lambda v: ('location', 'foo.bar.baz '),
 'expr': lambda v: ('expr', v), 'illegalTarget': lambda v: ('target', 'baz'),
 'delay': lambda v: ('delayexpr', "'%ss'" % v), 'idVal': lambda v: ('cond', idval(v)),
 'eventNameVal': lambda v: ('cond', "_event.name == '%s'" % v), 'inState': lambda v: ('cond', "In('%s')" % v),
 'true': lambda v: ('cond', 'true'), 'false': lambda v: ('cond', 'false'),
}
def fix(e):
    for k in list(e.attrib):
        if k.startswith('{%s}' % CF):
            n = k.split('}')[1]; a, v = ATTR[n](e.attrib.pop(k)); e.set(a, v)
    out = []
    for c in list(e):
        if c.tag == '{%s}pass' % CF or c.tag == '{%s}fail' % CF:
            r = 'pass' if c.tag.endswith('pass') else 'fail'
            f = ET.Element(q('final'), id=r); oe = ET.SubElement(f, q('onentry'))
            ET.SubElement(oe, q('log'), label='Outcome', expr="'%s'" % r); f.tail = c.tail; out.append(f)
        elif c.tag == '{%s}incrementID' % CF:
            i = c.get('id'); f = ET.Element(q('assign'), location='Var' + i, expr='Var%s + 1' % i); f.tail = c.tail; out.append(f)
        elif c.tag.startswith('{%s}' % CF):
            raise SystemExit('unknown conf element ' + c.tag)
        else:
            fix(c); out.append(c)
    for c in list(e): e.remove(c)
    for c in out: e.append(c)
for f in sys.argv[2:]:
    t = ET.parse(f); fix(t.getroot())
    t.write(os.path.join(sys.argv[1], os.path.basename(f).replace('.txml', '.scxml')), xml_declaration=True, encoding='utf-8')
