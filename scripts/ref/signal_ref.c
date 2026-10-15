/* Reference program for scripts/gen/signal_gen.py (Linux, glibc).
 * Each input line is a script run in a fresh child with every signal
 * blocked and caught by a no-op handler. Ops (space separated tokens,
 * ";" between ops):
 *   send N V   sigqueue(getpid(), N, V)
 *   ign N      sigaction SIG_IGN      dfl N   sigaction SIG_DFL
 *   catch N    sigaction no-op handler
 *   wait A,B,. sigtimedwait on {A,B,..} with zero timeout
 * Output per op: "ok", "N:V" for a dequeued signal, or "none". */
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
static void noop(int s) { (void)s; }
static void set(int n, void (*h)(int)) {
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = h; sigemptyset(&sa.sa_mask);
    sigaction(n, &sa, NULL);
}
static void run(char *line) {
    sigset_t all; sigfillset(&all); sigprocmask(SIG_SETMASK, &all, NULL);
    for (int n = 1; n < 65; n++) if (n != 9 && n != 19 && (n < 32 || n > 33)) set(n, noop);
    int first = 1;
    for (char *op = strtok(line, ";"); op; op = strtok(NULL, ";")) {
        char k[16]; char a[512]; long v = 0;
        a[0] = 0;
        sscanf(op, "%15s %511s %ld", k, a, &v);
        if (!first) putchar(';');
        first = 0;
        if (!strcmp(k, "send")) { union sigval sv; sv.sival_int = (int)v; printf(sigqueue(getpid(), atoi(a), sv) == 0 ? "ok" : "err"); }
        else if (!strcmp(k, "ign")) { set(atoi(a), SIG_IGN); printf("ok"); }
        else if (!strcmp(k, "dfl")) { set(atoi(a), SIG_DFL); printf("ok"); }
        else if (!strcmp(k, "catch")) { set(atoi(a), noop); printf("ok"); }
        else if (!strcmp(k, "wait")) {
            sigset_t w; sigemptyset(&w);
            for (char *p = a; *p; ) { sigaddset(&w, atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
            siginfo_t si; struct timespec z = {0, 0};
            int s = sigtimedwait(&w, &si, &z);
            if (s < 0) printf("none"); else printf("%d:%d", s, si.si_value.sival_int);
        }
    }
    putchar('\n'); fflush(stdout);
}
int main(void) {
    char line[1 << 16];
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = 0;
        pid_t p = fork();
        if (p == 0) { run(line); _exit(0); }
        int st; waitpid(p, &st, 0);
    }
    return 0;
}
