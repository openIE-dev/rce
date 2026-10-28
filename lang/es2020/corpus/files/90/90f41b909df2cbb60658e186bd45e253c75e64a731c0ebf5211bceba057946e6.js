// SPDX-License-Identifier: MIT OR Apache-2.0
// Reading a property of null throws a TypeError; here it is not caught.
const config = { server: null };
console.log("reading the port");
console.log(config.server.port);
