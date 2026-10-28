# SPDX-License-Identifier: MIT OR Apache-2.0
# sys.exit with a string prints it to standard error and exits with status
# 1; with None it exits with status 0.
import sys

print("before")
sys.exit("fatal: configuration missing")
