/* Driver for the copied systemd functions (scripts/ref/svc_ref_build.py).
 * Reads one tab-separated command per line, prints one answer per line:
 *   es WORD              exit_status_from_string
 *   name N               exit_status_to_string(N, FULL) and exit_status_class(N)
 *   sig WORD             signal_from_string
 *   set RVALUE           config_parse_set_status's sets ("s:..;g:..")
 *   clean CODE STATUS MODE SET   is_clean_exit (SET "-" = NULL)
 *   result TYPE CODE STATUS SET  service_sigchld_event's result for the
 *                        main process (clean mode daemon unless oneshot)
 *   restart RESTART TYPE RESULT CODE STATUS PREVENT FORCE FORBID
 *                        service_shall_restart: "1 reason" / "0 reason"
 * CODE is exited, killed or dumped. A set's words are split at WHITESPACE
 * as extract_first_word() does for words without quotes or backslashes. */
static const char *const restarts[] = { "no", "on-success", "on-failure", "on-abnormal", "on-watchdog", "on-abort", "always" };
static const char *const types[] = { "simple", "exec", "forking", "oneshot", "dbus", "notify", "notify-reload", "idle" };
static const char *const results[] = { "success", "resources", "protocol", "timeout", "exit-code", "signal", "core-dump", "watchdog", "start-limit-hit", "oom-kill", "condition" };
static int find(const char *const *t, size_t n, const char *s) { for (size_t i = 0; i < n; i++) if (streq(t[i], s)) return (int) i; abort(); }
static int code_of(const char *s) { return streq(s, "exited") ? CLD_EXITED : streq(s, "killed") ? CLD_KILLED : streq(s, "dumped") ? CLD_DUMPED : (abort(), 0); }
static void parse_set(const char *rvalue, ExitStatusSet *x) {
        memset(x, 0, sizeof *x);
        char buf[4096]; snprintf(buf, sizeof buf, "%s", rvalue);
        for (char *w = strtok(buf, WHITESPACE); w; w = strtok(NULL, WHITESPACE)) {
                int r = exit_status_from_string(w);
                if (r >= 0) { bitmap_set(&x->status, r); continue; }
                r = signal_from_string(w);
                if (r < 0) continue;
                bitmap_set(&x->signal, r);
        }
}
static void show_set(const ExitStatusSet *x) {
        printf("s:"); const char *sep = "";
        for (int i = 0; i < 256; i++) if (bitmap_isset(&x->status, i)) { printf("%s%d", sep, i); sep = ","; }
        printf(";g:"); sep = "";
        for (int i = 0; i < 256; i++) if (bitmap_isset(&x->signal, i)) { printf("%s%d", sep, i); sep = ","; }
        printf("\n");
}
int main(void) {
        char line[8192];
        while (fgets(line, sizeof line, stdin)) {
                line[strcspn(line, "\n")] = 0;
                char *f[10] = {0}; int n = 0;
                for (char *p = line; n < 10; ) { f[n++] = p; char *t = strchr(p, '\t'); if (!t) break; *t = 0; p = t + 1; }
                if (streq(f[0], "es")) printf("%d\n", exit_status_from_string(f[1]));
                else if (streq(f[0], "name")) { const char *a = exit_status_to_string(atoi(f[1]), EXIT_STATUS_FULL), *b = exit_status_class(atoi(f[1])); printf("%s %s\n", a ? a : "-", b ? b : "-"); }
                else if (streq(f[0], "sig")) printf("%d\n", signal_from_string(f[1]));
                else if (streq(f[0], "set")) { ExitStatusSet x; parse_set(f[1], &x); show_set(&x); }
                else if (streq(f[0], "clean")) {
                        ExitStatusSet x; parse_set(f[4], &x);
                        printf("%d\n", is_clean_exit(code_of(f[1]), atoi(f[2]), streq(f[3], "daemon") ? EXIT_CLEAN_DAEMON : EXIT_CLEAN_COMMAND, streq(f[4], "-") ? NULL : &x));
                } else if (streq(f[0], "result")) {
                        ExitStatusSet x; parse_set(f[4], &x);
                        int code = code_of(f[2]), status = atoi(f[3]);
                        ExitClean clean_mode = streq(f[1], "oneshot") ? EXIT_CLEAN_COMMAND : EXIT_CLEAN_DAEMON;
                        ServiceResult r;
                        if (is_clean_exit(code, status, clean_mode, &x)) r = SERVICE_SUCCESS;
                        else if (code == CLD_EXITED) r = SERVICE_FAILURE_EXIT_CODE;
                        else if (code == CLD_KILLED) r = SERVICE_FAILURE_SIGNAL;
                        else r = SERVICE_FAILURE_CORE_DUMP;
                        printf("%s\n", results[r]);
                } else if (streq(f[0], "restart")) {
                        Service s = {0}; const char *reason = NULL;
                        s.restart = find(restarts, ELEMENTSOF(restarts), f[1]);
                        s.type = find(types, ELEMENTSOF(types), f[2]);
                        s.result = find(results, ELEMENTSOF(results), f[3]);
                        s.main_exec_status.code = code_of(f[4]); s.main_exec_status.status = atoi(f[5]);
                        parse_set(f[6], &s.restart_prevent_status); parse_set(f[7], &s.restart_force_status);
                        s.forbid_restart = streq(f[8], "1");
                        bool r = service_shall_restart(&s, &reason);
                        printf("%d %s\n", r, reason);
                } else abort();
                fflush(stdout);
        }
        return 0;
}
