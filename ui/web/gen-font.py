#!/usr/bin/env python3
"""Write ui/web/rce-cell.ttf, the font of the ui.layout model (rce.ui.layout/1).

Every printable ASCII character (U+0020 to U+007E) has an advance of
exactly one em (1024 units per em, so that a font size of F CSS px scales
every advance to exactly F px with no binary rounding); every glyph but
the space is a square from the descender to the ascender. Ascender 768,
descender 256 and line gap 0 in hhea, OS/2 typo and OS/2 win metrics, so a line of text in this
font at font-size F CSS px is F px high with line-height 1 and every
character advances F px, the metric ui.layout states. The output is the
same bytes on every run (fixed timestamps; fontTools pinned by the caller).

Usage: python3 gen-font.py OUT.ttf  (with no argument, writes rce-cell.ttf
next to this file)."""
import os
import sys
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPM, ASC, DESC = 1024, 768, 256
FIXED_TIME = 3786912000  # 2024-01-01T00:00:00Z in seconds since 1904-01-01


def square():
    p = TTGlyphPen(None)
    p.moveTo((0, -DESC))
    p.lineTo((0, ASC))
    p.lineTo((UPM, ASC))
    p.lineTo((UPM, -DESC))
    p.closePath()
    return p.glyph()


def empty():
    return TTGlyphPen(None).glyph()


def build(out):
    names = [".notdef", "space"] + ["c%02x" % c for c in range(0x21, 0x7F)]
    cmap = {0x20: "space"}
    for c in range(0x21, 0x7F):
        cmap[c] = "c%02x" % c
    f = FontBuilder(UPM, isTTF=True)
    f.setupGlyphOrder(names)
    f.setupCharacterMap(cmap)
    glyphs = {n: (empty() if n == "space" else square()) for n in names}
    f.setupGlyf(glyphs)
    f.setupHorizontalMetrics({n: (UPM, 0) for n in names})
    f.setupHorizontalHeader(ascent=ASC, descent=-DESC, lineGap=0)
    f.setupNameTable({"familyName": "RCE Cell", "styleName": "Regular"})
    f.setupOS2(version=4, sTypoAscender=ASC, sTypoDescender=-DESC, sTypoLineGap=0,
               usWinAscent=ASC, usWinDescent=DESC, fsSelection=0x40 | 0x80,
               xAvgCharWidth=UPM)
    f.setupPost()
    f.font["head"].created = FIXED_TIME
    f.font["head"].modified = FIXED_TIME
    f.font.recalcTimestamp = False
    f.save(out)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "rce-cell.ttf"))
