#!/bin/sh
# Regenerate the fixtures of the renderer's tests with the reference checker:
# - the frame sink of `rce host run frames` of test/shop.rce on the inputs of
#   shop.events, on a new log D, and `rce host replay frames` of that D,
#   which adds each frame's ui.rules findings (Appendix E, HC-29);
# - the frame sink of test/layout.rce on the inputs of layout.events
#   (layout.frames.jsonl) and the Display lines of that run
#   (layout.boxes.txt): the targets ui.rules finds under 2.5.8 and the boxes
#   ui.layout gives each view in each mode.
# spec/checker/tests/host_frames.rs requires all four files to be current.
# Set RCE to the rce binary (default: rce on PATH).
set -eu
RCE=${RCE:-rce}
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$RCE" host run frames "$tmp/frames.jsonl" shop.rce "$tmp/shop.log" "$(cat shop.events)" > /dev/null
"$RCE" host replay frames shop.rce "$tmp/shop.log" > shop.replay.txt
cp "$tmp/frames.jsonl" shop.frames.jsonl
"$RCE" host run frames "$tmp/layout.jsonl" layout.rce "$tmp/layout.log" "$(cat layout.events)" > "$tmp/layout.out"
sed -n 's/^ *emit Display(data: "\(view .*\)")$/\1/p' "$tmp/layout.out" > layout.boxes.txt
cp "$tmp/layout.jsonl" layout.frames.jsonl
