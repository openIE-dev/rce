import Lake
open Lake DSL
package rce
@[default_target] lean_lib RCE

lean_exe «rce-lean-core» where
  root := `CoreMain
