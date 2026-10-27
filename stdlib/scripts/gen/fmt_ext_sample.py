# Usage: python3 -I scripts/gen/fmt_ext_sample.py PNF_DIR > vectors/fmt/parse-number-fxx-sample.txt
# Draws the float-parsing sample from a checkout of
# github.com/nigeltao/parse-number-fxx-test-data (commit
# 55d79b184b7d8fac2e143e89dc19b766ec4e54b8): a fixed number of lines of
# each data/*.txt file, chosen by random.Random(20261027), every line of
# more-test-cases.txt, and every line longer than 100 characters of the
# lemire and wuffs files (the long-digit cases). Lines are kept verbatim
# ("f16 f32 f64 string"), each prefixed with its file name.
import os, random, sys
d = os.path.join(sys.argv[1], 'data')
N = {'exhaustive-float16.txt': 60, 'freetype-2-7.txt': 150, 'google-double-conversion.txt': 250,
     'google-wuffs.txt': 200, 'ibm-fpgen.txt': 250, 'lemire-fast-double-parser.txt': 200,
     'lemire-fast-float.txt': 200, 'more-test-cases.txt': 0, 'remyoudompheng-fptest-0.txt': 100,
     'remyoudompheng-fptest-1.txt': 100, 'remyoudompheng-fptest-2.txt': 100,
     'remyoudompheng-fptest-3.txt': 100, 'tencent-rapidjson.txt': 200, 'ulfjack-ryu.txt': 250}
r = random.Random(20261027)
for name in sorted(N):
    lines = open(os.path.join(d, name)).read().split('\n')
    lines = [l for l in lines if l]
    if name == 'more-test-cases.txt':
        pick = list(range(len(lines)))
    else:
        pick = sorted(r.sample(range(len(lines)), N[name]))
        if name.startswith(('lemire', 'google-wuffs')):
            pick = sorted(set(pick) | {i for i, l in enumerate(lines) if len(l) > 131})
    for i in pick:
        print(name, lines[i])
