# SPDX-License-Identifier: MIT OR Apache-2.0
# Word frequencies of a fixed text, sorted by count and then by word, and
# Counter.most_common, which keeps first-seen order among equal counts.
import re
from collections import Counter

TEXT = """It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it was the
epoch of incredulity, it was the season of Light, it was the season of Darkness"""
words = re.findall(r"[a-z]+", TEXT.lower())
counts = Counter(words)
for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(c, w)
print(len(words), len(counts))
print(counts.most_common(6))
