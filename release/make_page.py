# Render RELEASE.md into index.html (draft page; links point to dist/).
import html, re
md = open('RELEASE.md').read().split('\n')
def inline(t): return re.sub(r'`([^`]+)`', r'<code>\1</code>', html.escape(t))
h = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>RCE IDE 0.1.0 (draft)</title>',
     '<style>body{font:16px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;color:#1b1d22}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.4rem;text-align:left;vertical-align:top}code{font-size:.85em;word-break:break-all}.draft{background:#fff4d6;padding:.5rem 1rem;border-radius:6px}</style></head><body>',
     '<p class="draft">Release notes for v0.1.0. Download links point to files next to this page; the same files are on GitHub Releases.</p>']
ul = tb = False
def close():
    global ul, tb
    if ul: h.append('</ul>'); ul = False
    if tb: h.append('</table>'); tb = False
for l in md:
    if l.startswith('# '): close(); h.append(f'<h1>{inline(l[2:])}</h1>')
    elif l.startswith('## '): close(); h.append(f'<h2>{inline(l[3:])}</h2>')
    elif l.startswith('| Platform'): close(); h.append('<table><tr><th>Platform</th><th>File</th><th>Size</th><th>SHA-256</th></tr>'); tb = True
    elif l.startswith('|---'): pass
    elif l.startswith('| '):
        c = [x.strip() for x in l.strip('|').split('|')]; f = c[1].strip('`')
        h.append(f'<tr><td>{inline(c[0])}</td><td><a href="dist/{f}">{f}</a></td><td>{c[2]}</td><td><code>{c[3].strip("`")}</code></td></tr>')
    elif l.startswith('- '):
        if not ul: h.append('<ul>'); ul = True
        h.append(f'<li>{inline(l[2:])}</li>')
    elif l.strip(): close(); h.append(f'<p>{inline(l)}</p>')
close()
h.append('<p><a href="dist/SHA256SUMS">SHA256SUMS</a></p></body></html>')
open('index.html', 'w').write('\n'.join(h) + '\n')
