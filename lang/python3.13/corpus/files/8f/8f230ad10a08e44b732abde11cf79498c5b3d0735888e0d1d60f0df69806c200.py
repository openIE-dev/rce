# SPDX-License-Identifier: MIT OR Apache-2.0
# A program writes to both standard output and standard error and exits
# with status 0.
import sys

print("to stdout 1")
print("to stderr", file=sys.stderr)
sys.stderr.write("written\n")
print("to stdout 2")
