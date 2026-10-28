// SPDX-License-Identifier: MIT OR Apache-2.0
// process.exitCode sets the status the process exits with once the program
// ends.
console.log("about to set the exit status");
process.exitCode = 3;
console.log("still running");
