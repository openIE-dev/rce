# Records launchd's relaunch decision: one user agent per (KeepAlive, exit)
# case in gui/$UID, RunAtLoad, ThrottleInterval 1; after 6 s, relaunched
# iff the job ran more than once. Prints TSV: keepalive  exit  runs>1.
import os, subprocess, time, plistlib, json
uid = os.getuid()
KA = {'none': None, 'false': False, 'true': True, 'empty': {}, 'se_t': {'SuccessfulExit': True}, 'se_f': {'SuccessfulExit': False},
      'cr_t': {'Crashed': True}, 'cr_f': {'Crashed': False}, 'se_t_cr_t': {'SuccessfulExit': True, 'Crashed': True},
      'se_f_cr_f': {'SuccessfulExit': False, 'Crashed': False}, 'se_f_cr_t': {'SuccessfulExit': False, 'Crashed': True}, 'se_t_cr_f': {'SuccessfulExit': True, 'Crashed': False}}
import sys
EX = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ok', 'fail', 'kill', 'segv', 'abrt', 'term', 'int', 'hup']
if len(sys.argv) > 1: KA = {k: KA[k] for k in sys.argv[1].split(',')}
labels = []
for k, ka in KA.items():
    for e in EX:
        L = 'com.dcharlot.rcetest.%s.%s' % (k.replace('_', '-'), e)
        cnt = '/tmp/ld/%s.count' % L
        if os.path.exists(cnt): os.remove(cnt)
        d = {'Label': L, 'ProgramArguments': ['/tmp/ld/job.sh', e, cnt], 'RunAtLoad': True, 'ThrottleInterval': 1}
        if ka is not None: d['KeepAlive'] = ka
        p = '/tmp/ld/%s.plist' % L
        plistlib.dump(d, open(p, 'wb'))
        subprocess.run(['launchctl', 'bootstrap', 'gui/%d' % uid, p], check=True)
        labels.append((k, e, L, cnt))
time.sleep(6)
for k, e, L, cnt in labels:
    info = subprocess.run(['launchctl', 'print', 'gui/%d/%s' % (uid, L)], capture_output=True, text=True).stdout
    subprocess.run(['launchctl', 'bootout', 'gui/%d/%s' % (uid, L)], capture_output=True)
    runs = len(open(cnt).read().split()) if os.path.exists(cnt) else 0
    last = [l.strip() for l in info.split('\n') if 'last exit code' in l or 'last terminating signal' in l]
    print('%s\t%s\t%d\t%s' % (k, e, 1 if runs > 1 else 0, '; '.join(last)))
