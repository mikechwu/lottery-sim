use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DrawStatus {
    DrawComplete = 0,
    DrawTimedOut = 1,
    DrawAborted = 2,
    DrawCancelled = 3,
}

impl DrawStatus {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::DrawComplete),
            1 => Some(Self::DrawTimedOut),
            2 => Some(Self::DrawAborted),
            3 => Some(Self::DrawCancelled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallExit {
    pub entity_id: u32,
    pub exit_frame: u32,
}

/// DrawOutcome per v1 §5.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DrawOutcome {
    pub status: DrawStatus,
    pub exits: Vec<BallExit>,
}

impl DrawOutcome {
    /// Serialize to binary per v4 §H.2.6.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(self.status as u8);
        buf.push(self.exits.len() as u8);
        for exit in &self.exits {
            buf.extend_from_slice(&exit.entity_id.to_le_bytes());
            buf.extend_from_slice(&exit.exit_frame.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 2 {
            return Err("DrawOutcome too short".into());
        }
        let status = DrawStatus::from_u8(data[0])
            .ok_or_else(|| format!("Unknown draw status: {}", data[0]))?;
        let exit_count = data[1] as usize;
        if data.len() < 2 + exit_count * 8 {
            return Err("DrawOutcome truncated".into());
        }
        let mut exits = Vec::with_capacity(exit_count);
        for i in 0..exit_count {
            let off = 2 + i * 8;
            let entity_id = u32::from_le_bytes(data[off..off+4].try_into().unwrap());
            let exit_frame = u32::from_le_bytes(data[off+4..off+8].try_into().unwrap());
            exits.push(BallExit { entity_id, exit_frame });
        }
        Ok(Self { status, exits })
    }
}
