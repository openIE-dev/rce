// Throwaway test service for recording SCM failure-action behaviour.
// binPath: rcesvc.exe NAME MODE COUNTFILE
//   MODE crash: the process ends abruptly 500 ms after start (no SERVICE_STOPPED)
//        stop0: reports SERVICE_STOPPED with exit code 0 after 500 ms
//        stop1: reports SERVICE_STOPPED with exit code 1 after 500 ms
using System; using System.IO; using System.ServiceProcess; using System.Threading;
class Svc : ServiceBase {
    string mode, count;
    Svc(string name, string mode, string count) { ServiceName = name; this.mode = mode; this.count = count; }
    protected override void OnStart(string[] a) {
        File.AppendAllText(count, "run\n");
        new Thread(() => {
            Thread.Sleep(500);
            if (mode == "crash") System.Diagnostics.Process.GetCurrentProcess().Kill();
            ExitCode = mode == "stop1" ? 1 : 0;
            Stop();
        }).Start();
    }
    static void Main(string[] args) { ServiceBase.Run(new Svc(args[0], args[1], args[2])); }
}
