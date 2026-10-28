# SPDX-License-Identifier: MIT OR Apache-2.0
# sys.exit raises SystemExit; an integer argument becomes the exit status,
# and finally blocks still run.
import sys

try:
    print("about to exit")
    sys.exit(3)
finally:
    print("finally runs")
print("never printed")
