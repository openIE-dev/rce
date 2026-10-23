//! `acp-client`: run outside coding agents through the Agent Client
//! Protocol (v1, <https://agentclientprotocol.com>), stdio transport.
//!
//! The agent is a subprocess speaking newline-delimited JSON-RPC 2.0. What
//! it gets:
//!
//! - **Only the rce-mcp tools.** `session/new` passes exactly one MCP server,
//!   `rce-mcp` over stdio, rooted at the workspace with its own receipt
//!   ledger. The client advertises no `fs`, no `terminal`, no auth terminal,
//!   no elicitation; any such call is answered `-32601` and recorded.
//! - **Enforcement.** Permission requests are granted only for tool calls
//!   that name an rce-mcp tool and are not `execute`/`delete`/`move`/`fetch`;
//!   everything else is rejected. Agents can still use their own built-in
//!   tools without asking: after every turn the workspace is diffed, and a
//!   change not backed by a passing `propose_edit` receipt in the rce-mcp
//!   ledger is reverted and reported.
//! - **A jail.** The agent process (and the rce-mcp it starts, which
//!   inherits the restriction) runs under Landlock + seccomp: read/write
//!   the workspace, read/execute its own binary, rce-mcp and the shared
//!   libraries, nothing else, and no network unless ports are granted
//!   ([`AgentJail`]). So its built-in read tools cannot see outside the
//!   workspace either. Fails closed where the jail is unavailable.
//! - **One signed receipt per turn** in the IDE ledger (`acp.turn`), linking
//!   the rce-mcp sub-ledger head, the tool calls, permission decisions,
//!   refused methods, reverted files, stop reason and transport bytes.
//!
//! [`g4::AcpG4`] plugs this into mei-agent as an optional G4.
//!
//! ACP v2 is a draft (it drops `fs`/`terminal` and has clients provide
//! tools as MCP servers, which is what this client already does). Version
//! negotiation here asks for v1 and refuses anything else.
pub mod client;
pub mod g4;
pub mod rpc;
pub mod snapshot;

pub use client::{AcpClient, AcpConfig, AgentJail, McpServer, Policy, TurnReport};

/// The protocol version this client speaks.
pub const PROTOCOL_VERSION: u64 = 1;
