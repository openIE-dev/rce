# Energy telemetry in ide-energy

Every source implements one trait, `ide_energy::meter::Meter`:
`source()` is the tag written into receipts (`ide_receipt::MeterSource`),
`sample()` returns cumulative joules since open with a timestamp. A source
that cannot measure returns an error (`Unsupported`, `Read`, or `Stale` for
a counter that did not move while the hardware certainly drew power). No
source returns 0 J in place of a measurement.

`rce-ide telemetry probe` (or `ide_energy::discover::probe_all`) opens every
source this build knows, reads it once, and prints what works, what needs a
permission, what is unsupported and why, plus the tracked sources below.
`rce-ide telemetry setup` grants the permissions some sources need. It shows
each step in full (file path, exact content, every command, the security
cost) and runs it only after the user types `yes` for that step; root
commands go through `sudo`, which asks for the password. With stdin not a
terminal it asks nothing and changes nothing.

## Implemented

| Tag | Platform | Domain | How | Resolution | Accuracy |
|---|---|---|---|---|---|
| `rapl` | Linux (Intel, AMD Zen) | CPU package (and DRAM on servers) | powercap `intel-rapl:N/energy_uj`, wrap handled, `psys` skipped | energy counter, ~1 ms update | measured on Haswell+ and Zen; model-based before |
| `ioreport` | macOS, Apple silicon | CPU, GPU, ANE (DRAM where it moves) | IOReport `CLPC/Energy Counters` (macOS 27) or `Energy Model` | nJ/mJ counter, on demand | Apple's on-chip energy model, not a shunt |
| `smc` | macOS | whole system | SMC key `PSTR` (W), polled every 100 ms, trapezoid integrated | 100 ms | SMC sensor reading; faster changes smoothed |
| `powermetrics` | macOS | CPU, GPU, ANE, combined | `sudo -n powermetrics`, only after setup | 200 ms average power | Apple's tool, same counters as IOReport |
| `nvml` | Linux, Windows (NVIDIA) | GPU board | `nvmlDeviceGetTotalEnergyConsumption` (feature `nvml`) | mJ counter (Volta+) | about +/-5 W on datacenter boards |
| `hwmon` | Linux | GPU (amdgpu, i915, xe, nouveau), CPU (amd_energy, zenpower), other | `energy*_input` (uJ counter) else `power*_average`/`power*_input` (uW, polled) | driver-defined | driver-defined |
| `jetson` | Linux (NVIDIA Jetson, INA3221 boards) | board rails | hwmon `in*_input` x `curr*_input`, sum/VDD_IN channels skipped | 100 ms | shunt, about 1-2 % |
| `power-supply` | Linux, Android, Raspberry Pi with battery | whole device | `power_now` or `current_now` x `voltage_now`, only while discharging | 100 ms | fuel gauge, often averaged over seconds |

### macOS details (measured on the 128GB M5 Max, macOS 27.0.1)

- `Energy Model` channels `CPU Energy`, `ANE0`, `DRAM0` and the per-core
  `PCPUDTL*` channels read 0 over every window, under load average above 100.
  Only `GPU Energy` moves. The old meter therefore reported 0.000 J for CPU.
- The AppleCLPC aggregate counters (`CLPC` group, `Energy Counters`
  subgroup) move for CPU and GPU. They are not listed by
  `IOReportCopyAllChannels`; the channel descriptors are built from the
  driver's registry id and per-driver report ids (driver on this Mac:
  `com.apple.driver.AppleT6050CLPC`). CLPC GPU agreed with Energy Model GPU
  within 0.1 % over the same 4 s window (39.20 J vs 39.22 J).
- DRAM has no moving counter on this machine: `ioreport` DRAM is
  `Unsupported` there.
- The id table and descriptor layout come from macmon
  (https://github.com/vladkens/macmon at afdd572, MIT License, Copyright (c)
  2024 vladkens; `src/clpc.rs`, `src/sources.rs`, `src/shared.rs`,
  `src/metrics.rs`). macmon is pure Rust (core-foundation, libc). ide-energy
  does not depend on it; the reads are re-implemented over raw FFI.
- SMC: 81 `P*` keys exist on this Mac. Apple does not document which rail
  each is, so only `PSTR` (system total, as macmon uses it) is read.

### Permissions

- RAPL: `energy_uj` is root-only since Linux 5.10 (CVE-2020-8694,
  "PLATYPUS"). `setup` can add group `power`, add the user to it, and install
  `/etc/udev/rules.d/60-rce-powercap.rules`:

      ACTION=="add", SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chgrp power /sys%p/energy_uj", RUN+="/bin/chmod g+r /sys%p/energy_uj"

  Anyone in `power` can then read energy counters, which is the side
  channel the CVE closed.
- powermetrics: `setup` can install `/etc/sudoers.d/rce-powermetrics`
  (`USER ALL=(root) NOPASSWD: /usr/bin/powermetrics`, mode 0440, checked
  with `visudo -cf`). The meter is used only when that file exists and
  `sudo -n -l` confirms it; it never relies on a cached sudo timestamp.
- IOReport, SMC, hwmon (most drivers), power_supply: no setup.

### Measurement on the M5 Max (2026-10-12, 14:52 to 14:54 EDT)

The machine was not idle: other users' rustc jobs held the load average at
114 to 188 throughout, so "idle" means "no work of ours", and the compile's
own share cannot be separated from the background.

| Source | 5 s with no work of ours | `cargo build --release` of rce-spec-check (62.85 s, exit 0) |
|---|---|---|
| `ioreport` CPU (CLPC) | 131.649 J / 5.029 s = 26.18 W | 1346.149 J / 62.847 s = 21.42 W |
| `ioreport` GPU (CLPC) | 12.050 J = 2.40 W | 215.375 J = 3.43 W |
| `ioreport` ANE | 0.000 J (idle; allowed for ANE) | 0.000 J |
| `ioreport` DRAM | Unsupported (counter does not move) | Unsupported |
| `smc` PSTR (system) | 352.286 J / 5.003 s = 70.42 W | 4437.447 J / 62.897 s = 70.55 W |
| `powermetrics` CPU | no sample yet at window start | 1179.030 J / 62.875 s = 18.75 W |

powermetrics ran in that one measurement because `sudo -n` succeeded
without a prompt on this Mac at the time (no rce sudoers entry exists). The
probe was then changed to require the setup's sudoers file, and now reports
powermetrics as "needs setup" there.

## Tracked, not implemented

| Tag | Domain | Why not yet | Reference |
|---|---|---|---|
| `windows-emi` | CPU, GPU, system | Energy Meter Interface (`IOCTL_EMI_GET_MEASUREMENT`); Win32 FFI, no Windows machine to test | https://learn.microsoft.com/en-us/windows-hardware/drivers/powermeter/energy-meter-interface |
| `amd-smi` | GPU, CPU | C library; amdgpu hwmon covers GPU power | https://github.com/ROCm/amdsmi |
| `level-zero` | GPU, NPU | Sysman C API; i915/xe hwmon covers GPU energy | https://spec.oneapi.io/level-zero/latest/sysman/api.html |
| `intel-npu` | NPU | no public energy counter found | https://docs.kernel.org/accel/intel_vpu/ |
| `arm-scmi` | CPU | appears as hwmon/powercap on some SoCs (then covered); no generic PMU energy event | https://docs.kernel.org/hwmon/scmi-hwmon.html |
| `qualcomm` | NPU, SoC | vendor tools or Android power HAL only | https://source.android.com/docs/core/power/values |
| `tpu` | accelerator | Cloud Monitoring only, not on the host | https://cloud.google.com/tpu/docs/tpu-monitoring-library |
| `tegrastats` | board | same INA3221 data as `jetson` | https://docs.nvidia.com/jetson/ |
| `rpi-pmic` | board | Pi 5 PMIC via `vcgencmd pmic_read_adc`, not sysfs | https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd |
| `probe-rs` | MCU | needs a probe with a power monitor and probe-rs support | https://probe.rs |
| `ipmi-dcmi` | server | DCMI Get Power Reading, root and a BMC | DCMI 1.5 specification |
| `redfish` | server | BMC credentials | https://www.dmtf.org/standards/redfish |
| `ios` | phone | no public energy API | https://developer.apple.com/documentation/xcode/analyzing-your-app-s-battery-use |
| `freebsd` | CPU | RAPL MSRs via cpuctl(4), root | https://man.freebsd.org/cgi/man.cgi?cpuctl(4) |
| `usb-meter` | external | per-device protocols | https://sigrok.org/wiki/Supported_hardware |
| `smart-pdu` | external | SNMP or vendor HTTP, per model | vendor documentation |

Android phones are covered by `power-supply` (the kernel attributes
BatteryManager reads) when the app can read sysfs; an in-app
BatteryManager backend is tracked with `qualcomm`.

## Tests

`crates/ide-energy/tests/energy.rs`: unit handling and rail selection with
the channel values recorded on the M5 Max (CLPC preferred, a stuck CPU or
DRAM counter is never 0 J, an idle ANE is), CLPC ids, trapezoid
integration and polled-meter error propagation, SMC decoding, hwmon and
power_supply on fake sysfs trees (Jetson VDD_IN skipped, counter reset is an
error, no reading from a charging battery), powermetrics line parsing and
its error without setup, discovery reasons, and setup: nothing runs without consent,
the full file content and security note are shown first.

## Calibration (`rce-ide telemetry calibrate`)

Runs a fixed set of workloads under every meter `open_all` finds (RAPL,
IOReport rails, SMC, NVML, hwmon, battery; not powermetrics) and fits the
E_sim coefficients per meter by least squares:

- `empty` (the per-action floor), `check {small,medium,large} xK` (K real
  `rce_check` calls, K = 10..160, over 1, 2 or 3 bundled RCE programs, so
  passes and form nodes vary independently), `test large x20` (`rce_test`).
- Features are the ones rce-mcp records for the same call. Runs are
  interleaved over `--reps` passes (default 5).
- Measured joules are whole-domain (idle draw included), as in receipts.
- Only the `checker` gear is calibrated. Terms no workload varies (wasi
  fuel, tokens, net bytes) are held at 0 and listed as not calibrated;
  agent gears keep predicting nothing.
- A meter that is stale or errors for a window contributes nothing for it;
  one with no usable windows is listed as unusable. With no meter at all
  the command fails; it never writes zeros.

Output: `$RCE_IDE_CALIBRATION`, else `$XDG_CONFIG_HOME/rce-ide/calibration.json`,
else `~/.config/rce-ide/calibration.json` (`--out` overrides). The file's top level
is the first meter's simulator (`rce-mcp --calibration FILE` takes it as is),
with the full report under `report`; `calibration.<meter>.json` beside it per
meter. The IDE loads the default file at startup. `--fake-meter WATTS` (a
constant-power stand-in, tagged `fake`) is for testing without hardware.
