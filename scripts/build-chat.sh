#!/bin/sh
# apps/chat.rce = apps/chat-app.rce without its `use crypto` line and its
# `crypto.` qualifiers, then stdlib/lib/crypto.rce without its module, use
# and check lines (its checks run in `rce test stdlib/lib/crypto.rce`). A
# single file, so `rce host serve apps/chat.rce` reads no module.
# tests/chat_e2ee.rs (generated_chat_is_in_sync) builds the same bytes.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
python3 - "$here" <<'PY'
import re, sys
root = sys.argv[1]
def unqualify(line, mods):
    out, i, code = [], 0, ''
    pat = re.compile(r'(?<![\w.])(?:%s)\.(?=[A-Za-z_])' % '|'.join(mods))
    while i < len(line):
        c = line[i]
        if c == '#':
            break
        if c == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            out.append(pat.sub('', code)); code = ''
            out.append(line[i:j + 1]); i = j + 1
            continue
        code += c; i += 1
    out.append(pat.sub('', code))
    return ''.join(out) + line[i:]
app = open(root + '/apps/chat-app.rce').read().split('\n')
if app and app[-1] == '':
    app = app[:-1]
out = [unqualify(l, ['crypto']) for l in app if l != 'use crypto']
out.append('')
out.append('# ---------------------------------------------------------------- stdlib/lib/crypto.rce (generated, do not edit here)')
lib = open(root + '/stdlib/lib/crypto.rce').read().split('\n')
if lib and lib[-1] == '':
    lib = lib[:-1]
for l in lib:
    if l.startswith('module ') or l.startswith('use ') or l.startswith('check '):
        continue
    out.append(unqualify(l, ['crypto']))
open(root + '/apps/chat.rce', 'w').write('\n'.join(out) + '\n')
PY
