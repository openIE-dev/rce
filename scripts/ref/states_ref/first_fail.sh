#!/bin/sh
# index of the first failing lv check
cd /workspace/m53/rce
L=$(cat states.rce | wc -l); H=$(awk '/^check l[vx]/{print NR; exit}' $1)
cat states.rce $1 > /tmp/ff.rce
n=$(/tmp/rce-ide-0.1.3-x86_64-unknown-linux-gnu/rce test /tmp/ff.rce 2>&1 | tr ';' '\n' | grep false | head -1 | sed 's/.*line \([0-9]*\):.*/\1/')
echo $(( n - L - H ))
