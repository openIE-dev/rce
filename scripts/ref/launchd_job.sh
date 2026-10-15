#!/bin/sh
# $1: how to end: ok (exit 0), fail (exit 3), xN (exit N), kill/segv/abrt/term/int/hup (signal itself)
echo run >> "$2"
case "$1" in ok) exit 0;; fail) exit 3;; x[0-9]*) exit ${1#x};; kill) kill -KILL $$;; segv) kill -SEGV $$;; abrt) kill -ABRT $$;; term) kill -TERM $$;; int) kill -INT $$;; hup) kill -HUP $$;; *) kill -$1 $$;; esac
