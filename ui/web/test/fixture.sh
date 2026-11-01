#!/bin/sh
# Regenerate the fixture of the renderer's test with the reference checker:
# the frame sink of `rce host run frames` of test/shop.rce on the inputs of
# shop.events, on a new log D, and `rce host replay frames` of that D, which
# adds each frame's ui.rules findings (Appendix E, HC-29).
# spec/checker/tests/host_frames.rs requires both files to be current.
# Set RCE to the rce binary (default: rce on PATH).
set -eu
RCE=${RCE:-rce}
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$RCE" host run frames "$tmp/frames.jsonl" shop.rce "$tmp/shop.log" "$(cat shop.events)" > /dev/null
"$RCE" host replay frames shop.rce "$tmp/shop.log" > shop.replay.txt
cp "$tmp/frames.jsonl" shop.frames.jsonl
