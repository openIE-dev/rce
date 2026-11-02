# The checks a generator does not write (W-MUT-1, spec 6.12): a check that
# no single edit of its module makes fail carries no information about the
# module, and `rce check` refuses it. scripts/gen/vacuous.tsv lists them,
# one `GENERATOR<TAB>CHECK` line each.
import builtins, os
_rows = {}
for _l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vacuous.tsv'), encoding='utf-8'):
    if _l.startswith('#') or '\t' not in _l:
        continue
    _g, _c = _l.rstrip('\n').split('\t', 1)
    _rows.setdefault(_g, set()).add(_c)


def drop(gen, lines):
    """The lines of a list, without the checks `gen` does not write."""
    return [l for l in lines if l not in _rows.get(gen, ())]


def drop_text(gen, text):
    """A text, without the lines that are checks `gen` does not write."""
    return '\n'.join(drop(gen, text.split('\n')))


def printer(gen):
    """`print`, writing nothing for a check `gen` does not write."""
    def p(*a, **k):
        if k.get('file') is None and len(a) == 1 and a[0] in _rows.get(gen, ()):
            return
        builtins.print(*a, **k)
    return p
