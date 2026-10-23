# Records the Service Control Manager's failure actions: one throwaway
# service per case (sc create ... start= demand), sc failure / sc failureflag
# set, sc start, wait, count the runs, sc delete. Prints TSV:
#   case  actions  flag  mode  runs
param([string]$Dir = "C:\rcesvc")
$exe = "$Dir\rcesvc.exe"
$cases = @(
  @("none",  "",                                   0),
  @("r1",    "restart/500",                        0),
  @("r1f",   "restart/500",                        1),
  @("r2",    "restart/500/restart/500",            0),
  @("r2f",   "restart/500/restart/500",            1),
  @("r3",    "restart/500/restart/500/restart/500",0),
  @("rn",    "restart/500/none/0",                 0),
  @("nr",    "none/0/restart/500",                 0),
  @("nrf",   "none/0/restart/500",                 1),
  @("rnr",   "restart/500/none/0/restart/500",     0)
)
$modes = @("crash", "stop0", "stop1")
$all = @()
foreach ($c in $cases) { foreach ($m in $modes) {
  $name = "RceSvcTest_$($c[0])_$m"; $cnt = "$Dir\$name.count"
  Remove-Item $cnt -ErrorAction SilentlyContinue
  sc.exe create $name binPath= "`"$exe`" $name $m $cnt" start= demand | Out-Null
  if ($c[1] -ne "") { sc.exe failure $name reset= 86400 actions= $($c[1]) | Out-Null }
  sc.exe failureflag $name $($c[2]) | Out-Null
  $all += ,@($c, $m, $name, $cnt)
}}
foreach ($x in $all) { sc.exe start $x[2] | Out-Null }
Start-Sleep -Seconds 12
foreach ($x in $all) {
  $runs = 0; if (Test-Path $x[3]) { $runs = (Get-Content $x[3]).Count }
  sc.exe stop $x[2] 2>&1 | Out-Null; Start-Sleep -Milliseconds 200
  sc.exe delete $x[2] | Out-Null
  "{0}`t{1}`t{2}`t{3}`t{4}" -f $x[0][0], $x[0][1], $x[0][2], $x[1], $runs
}
