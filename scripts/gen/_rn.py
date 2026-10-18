"""Generators write .rce text through rn(): reserved words used as names get the
replacement of scripts/rename-reserved.map (the same token rule as scripts/rename-reserved.py)."""
import importlib.util, os
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rename_reserved', os.path.join(_here, '..', 'rename-reserved.py'))
_src = open(os.path.join(_here, '..', 'rename-reserved.py')).read().replace('sys.exit(main(sys.argv[1:]))', '')
_ns = {}
exec(compile(_src, 'rename-reserved.py', 'exec'), _ns)
_map = _ns['load'](os.path.join(_here, '..', 'rename-reserved.map'))

import re
# W-LEX-4: invisible, bidi and control characters are never written literally into a source text
_LEX = re.compile('[\u202a-\u202e\u2066-\u2069\u061c\u200e\u200f\u200b-\u200d\u2060-\u2064\ufeff\u00ad\u034f\u180e\u115f\u1160\u17b4\u17b5\u3164\uffa0\ufff9-\ufffb\U000e0000-\U000e007f\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufdd0-\ufdef\U00040000-\U000dffff\ufffe\uffff]')
def esc(text):
    return _LEX.sub(lambda m: '\\u{%X}' % ord(m.group(0)), text)

def rn(text):
    return esc(_ns['rename'](text, _map))
