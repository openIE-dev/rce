#!/usr/bin/env python3
"""Reference for lib/ui/input.rce: crossterm 0.29.0 (crate sha256
d8b9f2e4c67f833b660cdb0a3523065869fb35570177239812ed4c905aeff87b).
Writes src/parse.rs from vectors/ui/crossterm_parse.rs (the crate's
src/event/sys/unix/parse.rs) with only these changes: crate paths point at
the crossterm crate, raw mode is a flag set by the harness, and each
assert_eq!/assert!(is_err) in its tests also prints the asserted value in
canonical form (src/canon.rs). Then:
  cargo test -q -- --nocapture --test-threads=1 | grep -o 'REC.*' | cut -f2- > vectors/ui/crossterm_official.tsv
  cargo run -q --release > vectors/ui/crossterm_live.tsv"""
import os
H = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(H, '../../../vectors/ui/crossterm_parse.rs')).read()
s = s.replace('crate::event::', 'crossterm::event::').replace('super::super::super::InternalEvent', 'crate::InternalEvent')
s = s.replace('crate::terminal::sys::is_raw_mode_enabled()', 'crate::raw()')
a, b = s.split('#[cfg(test)]\nmod tests {', 1)
b = b.replace('assert_eq!(', 'rec!(')
b = b.replace('        assert!(parse_utf8_char(', '        rec_err!(parse_utf8_char(').replace('.is_err());', ');')
b = '''
    macro_rules! rec_err {
        ($l:expr) => {{ let l = $l; println!("REC\\t{}\\t{}", stringify!($l), crate::canon::res(&l)); assert!(l.is_err()); }};
    }
    macro_rules! rec {
        ($l:expr, $r:expr $(, $m:expr)? $(,)?) => {{
            let l = $l; let r = $r;
            println!("REC\\t{}\\t{}", stringify!($l).replace('\\n', " "), crate::canon::Canon::canon(&l));
            assert_eq!(l, r);
        }};
    }
''' + b
open(os.path.join(H, 'src/parse.rs'), 'w').write(a + '#[cfg(test)]\nmod tests {' + b)
