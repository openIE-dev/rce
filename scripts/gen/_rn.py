"""Generators write .rce text through rn(): reserved words used as names get the
replacement of scripts/rename-reserved.map (the same token rule as scripts/rename-reserved.py)."""
import importlib.util, os
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('rename_reserved', os.path.join(_here, '..', 'rename-reserved.py'))
_src = open(os.path.join(_here, '..', 'rename-reserved.py')).read().replace('sys.exit(main(sys.argv[1:]))', '')
_ns = {}
exec(compile(_src, 'rename-reserved.py', 'exec'), _ns)
_map = _ns['load'](os.path.join(_here, '..', 'rename-reserved.map'))

def rn(text):
    return _ns['rename'](text, _map)
