use sha2::{Sha256, Digest};

/// Per-body canonical state: 56 bytes per v2 §1.6.
/// Fields in order: entity_id, pos(xyz), rot(xyzw), linvel(xyz), angvel(xyz), is_sleeping, is_active.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyState {
    pub entity_id: u32,
    pub position: [f32; 3],
    pub orientation: [f32; 4], // quaternion xyzw
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub is_sleeping: bool,
    pub is_active: bool,
}

impl BodyState {
    /// Serialize to canonical 56-byte little-endian representation per v2 §1.6.
    pub fn to_bytes(&self) -> [u8; 56] {
        let mut buf = [0u8; 56];
        buf[0..4].copy_from_slice(&self.entity_id.to_le_bytes());
        // position: 3 x f32 = 12 bytes at offset 4
        for i in 0..3 {
            buf[4 + i * 4..8 + i * 4].copy_from_slice(&self.position[i].to_bits().to_le_bytes());
        }
        // orientation: 4 x f32 = 16 bytes at offset 16
        for i in 0..4 {
            buf[16 + i * 4..20 + i * 4].copy_from_slice(&self.orientation[i].to_bits().to_le_bytes());
        }
        // linear_velocity: 3 x f32 = 12 bytes at offset 32
        for i in 0..3 {
            buf[32 + i * 4..36 + i * 4].copy_from_slice(&self.linear_velocity[i].to_bits().to_le_bytes());
        }
        // angular_velocity: 3 x f32 = 12 bytes at offset 44
        for i in 0..3 {
            buf[44 + i * 4..48 + i * 4].copy_from_slice(&self.angular_velocity[i].to_bits().to_le_bytes());
        }
        buf[54] = self.is_sleeping as u8;
        buf[55] = self.is_active as u8;
        buf
    }
}

/// Canonical Physics State Fingerprint per v2 §1.
#[derive(Debug, Clone, PartialEq)]
pub struct Cpsf {
    pub frame_number: u32,
    pub bodies: Vec<BodyState>,
}

impl Cpsf {
    /// Serialize to canonical byte buffer per v2 §1.6:
    /// [frame_number: 4 LE][body_count: 4 LE][bodies...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.bodies.len() * 56);
        buf.extend_from_slice(&self.frame_number.to_le_bytes());
        buf.extend_from_slice(&(self.bodies.len() as u32).to_le_bytes());
        for body in &self.bodies {
            buf.extend_from_slice(&body.to_bytes());
        }
        buf
    }

    /// Compute SHA-256 hash per v2 §1.7.
    pub fn hash(&self) -> [u8; 32] {
        let bytes = self.to_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }
}

/// A checkpoint entry: (frame, hash).
#[derive(Debug, Clone, PartialEq)]
pub struct CpsfCheckpoint {
    pub frame: u32,
    pub hash: [u8; 32],
}

impl CpsfCheckpoint {
    pub fn to_bytes(&self) -> [u8; 36] {
        let mut buf = [0u8; 36];
        buf[0..4].copy_from_slice(&self.frame.to_le_bytes());
        buf[4..36].copy_from_slice(&self.hash);
        buf
    }

    pub fn from_bytes(data: &[u8; 36]) -> Self {
        let frame = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&data[4..36]);
        Self { frame, hash }
    }
}
