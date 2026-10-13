//! An ACP agent as mei-agent's optional G4.
use crate::client::AcpClient;
use mei_agent::{ExternalG4, ExternalTurn};

pub struct AcpG4(pub AcpClient);

impl ExternalG4 for AcpG4 {
    fn name(&self) -> String {
        format!(
            "acp:{}",
            self.0.agent_info["name"].as_str().unwrap_or("agent")
        )
    }
    fn turn(&mut self, prompt: &str) -> Result<ExternalTurn, String> {
        let r = self.0.prompt(prompt).map_err(|e| e.to_string())?;
        if !r.unmediated.is_empty() {
            // Reverted already; tell mei so the receipt says why.
            return Err(format!(
                "agent wrote outside rce-mcp: {:?} (reverted {:?})",
                r.unmediated, r.reverted
            ));
        }
        let receipt = r.receipt.clone();
        let mediated = r.mediated.clone();
        Ok(ExternalTurn {
            receipt,
            mediated,
            detail: serde_json::to_value(&r).unwrap_or_default(),
            e_meas: None,
        })
    }
}
