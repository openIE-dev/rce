#!/usr/bin/env python3
"""The WCAG 2.2 table of ui.rules, checked three ways (run by scripts/check.sh):

1. criteria() in lib/ui/rules.rce lists exactly the level A and AA success
   criteria of vectors/ui/wcag22-sc.tsv (taken from w3c/wcag; 4.1.1 Parsing
   is obsolete and absent), in document order, with the same level;
2. every row has exactly one class of E, L, P and H; every E, L and P row
   names a rule that rules.rce defines (fn r_RULE), lists in rule_names()
   and dispatches by name, and no two rows share a rule; H rows name none;
3. the classification table of spec/proposals/human-interface.md has the
   same rows (id, level, class, rule).
Prints one summary line; exits 1 on the first disagreement."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(STD)


def fail(msg):
    print('FAIL rules_table_probes: ' + msg)
    sys.exit(1)


src = open(os.path.join(STD, 'lib/ui/rules.rce')).read()
wcag = [l.split('\t') for l in open(os.path.join(STD, 'vectors/ui/wcag22-sc.tsv')).read().split('\n') if l]
rows = re.findall(r'c\.push\(cr\("([0-9.]+)", "(A|AA)", "([A-Z])", "([a-z_-]+)", "([^"]*)"\)\)', src)
if [(r[0], r[1]) for r in rows] != [(w[0], w[2]) for w in wcag]:
    fail('criteria() does not list the A and AA criteria of vectors/ui/wcag22-sc.tsv in order')
names = re.search(r'pub fn rule_names\(\) -> List\[text\]\n    return \[([^\]]*)\]', src).group(1)
names = re.findall(r'"([a-z_]+)"', names)
seen = set()
counts = {'E': 0, 'L': 0, 'P': 0, 'H': 0}
for sc, lv, cl, rule, human in rows:
    if cl not in counts:
        fail('%s has class %s' % (sc, cl))
    counts[cl] += 1
    if cl == 'H':
        if rule != '-' or not human:
            fail('%s is H and must name no rule and state the human part' % sc)
        continue
    if cl == 'P' and not human:
        fail('%s is P and states no human part' % sc)
    if rule in seen:
        fail('rule %s is named twice' % rule)
    seen.add(rule)
    if rule not in names:
        fail('rule %s of %s is not in rule_names()' % (rule, sc))
    if not re.search(r'\nfn r_%s\(' % rule, src):
        fail('rule %s of %s has no fn r_%s' % (rule, sc, rule))
    if not re.search(r'name == "%s"\n        r_%s\(' % (rule, rule), src):
        fail('rule %s of %s is not dispatched' % (rule, sc))
if seen != set(names):
    fail('rule_names() and the criteria disagree: %s' % sorted(set(names) ^ seen))
prop = open(os.path.join(REPO, 'spec/proposals/human-interface.md')).read()
table = re.findall(r'^\| ([0-9]+\.[0-9]+\.[0-9]+) \| [^|]+ \| (A|AA) \| ([ELPH]) \| `?([a-z_-]+)`? \|', prop, re.M)
if [(t[0], t[1], t[2], t[3]) for t in table] != [(r[0], r[1], r[2], r[3]) for r in rows]:
    fail('the table of spec/proposals/human-interface.md differs from criteria()')
print('rules_table_probes: %d criteria (E %d, L %d, P %d, H %d), %d rules, proposal table agrees' %
      (len(rows), counts['E'], counts['L'], counts['P'], counts['H'], len(seen)))
