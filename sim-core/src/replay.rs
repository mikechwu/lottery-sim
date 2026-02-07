use sha2::{Sha256, Digest};
use crate::cpsf::CpsfCheckpoint;
use crate::event::EventStream;
use crate::outcome::DrawOutcome;
use crate::params::SimulationParameters;

/// Magic number: ASCII "LOTTORPL" per v4 §H.2.1.
pub const REPLAY_MAGIC: [u8; 8] = *b"LOTTORPL";
/// Format version per v4 §H.2.2.
pub const REPLAY_FORMAT_VERSION: u32 = 1;

/// Replay header per v4 §H.2.3.
#[derive(Debug, Clone)]
pub struct ReplayHeader {
    pub wasm_binary_hash: [u8; 32],
    pub seed: u64,
    pub sim_version_major: u32,
    pub sim_version_minor: u32,
    pub sim_version_patch: u32,
    pub params_hash_short: u32, // first 4 bytes of SHA-256
    pub params_blob: Vec<u8>,
    // Camera/mapping fields (v4 §E.2)
    pub mapping_version: u16,
    pub camera_pos: [f32; 3],
    pub camera_target: [f32; 3],
    pub camera_fov_rad: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub drum_plane_z: f32,
    pub tan_half_fov: f32,
}

/// Full replay container per v4 §H.
#[derive(Debug, Clone)]
pub struct ReplayContainer {
    pub header: ReplayHeader,
    pub event_stream: EventStream,
    pub checkpoints: Vec<CpsfCheckpoint>,
    pub outcome: DrawOutcome,
}

impl ReplayContainer {
    /// Serialize to the canonical binary format per v4 §H.
    /// Returns the full container bytes WITH checksum trailer appended.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic (8 bytes)
        buf.extend_from_slice(&REPLAY_MAGIC);
        // Format version (4 bytes LE)
        buf.extend_from_slice(&REPLAY_FORMAT_VERSION.to_le_bytes());

        // Header
        let header_bytes = self.serialize_header();
        buf.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&header_bytes);

        // Event stream
        let event_bytes = self.event_stream.to_bytes();
        buf.extend_from_slice(&event_bytes);

        // CPSF checkpoints
        buf.extend_from_slice(&(self.checkpoints.len() as u32).to_le_bytes());
        for cp in &self.checkpoints {
            buf.extend_from_slice(&cp.to_bytes());
        }

        // Draw outcome
        let outcome_bytes = self.outcome.to_bytes();
        buf.extend_from_slice(&outcome_bytes);

        // Checksum trailer: SHA-256 of everything above
        let checksum = Self::compute_checksum(&buf);
        buf.extend_from_slice(&checksum);

        buf
    }

    fn serialize_header(&self) -> Vec<u8> {
        let h = &self.header;
        let mut buf = Vec::new();
        buf.extend_from_slice(&h.wasm_binary_hash);
        buf.extend_from_slice(&h.seed.to_le_bytes());
        buf.extend_from_slice(&h.sim_version_major.to_le_bytes());
        buf.extend_from_slice(&h.sim_version_minor.to_le_bytes());
        buf.extend_from_slice(&h.sim_version_patch.to_le_bytes());
        buf.extend_from_slice(&h.params_hash_short.to_le_bytes());
        buf.extend_from_slice(&(h.params_blob.len() as u32).to_le_bytes());
        buf.extend_from_slice(&h.params_blob);
        buf.extend_from_slice(&h.mapping_version.to_le_bytes());
        for v in &h.camera_pos { buf.extend_from_slice(&v.to_bits().to_le_bytes()); }
        for v in &h.camera_target { buf.extend_from_slice(&v.to_bits().to_le_bytes()); }
        buf.extend_from_slice(&h.camera_fov_rad.to_bits().to_le_bytes());
        buf.extend_from_slice(&h.viewport_width.to_bits().to_le_bytes());
        buf.extend_from_slice(&h.viewport_height.to_bits().to_le_bytes());
        buf.extend_from_slice(&h.drum_plane_z.to_bits().to_le_bytes());
        buf.extend_from_slice(&h.tan_half_fov.to_bits().to_le_bytes());
        buf
    }

    fn compute_checksum(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Parse a replay container from bytes.
    /// Verifies magic, version, and checksum.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ReplayError> {
        if data.len() < 8 + 4 + 4 + 32 {
            return Err(ReplayError::FormatError("File too short".into()));
        }

        // Magic
        if data[0..8] != REPLAY_MAGIC {
            return Err(ReplayError::FormatError("Invalid magic number".into()));
        }

        // Format version
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != REPLAY_FORMAT_VERSION {
            return Err(ReplayError::UnsupportedVersion(version));
        }

        // Checksum verification: last 32 bytes
        let checksum_start = data.len() - 32;
        let stored_checksum = &data[checksum_start..];
        let computed_checksum = Self::compute_checksum(&data[..checksum_start]);
        if stored_checksum != computed_checksum {
            return Err(ReplayError::IntegrityViolation);
        }

        // Parse header
        let header_len = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        if 16 + header_len > checksum_start {
            return Err(ReplayError::FormatError("Header overflows file".into()));
        }
        let header = Self::parse_header(&data[16..16 + header_len])?;

        let mut offset = 16 + header_len;

        // Parse event stream
        let event_stream = EventStream::from_bytes(&data[offset..checksum_start])
            .map_err(|e| ReplayError::FormatError(e))?;
        // Advance past event stream bytes
        let event_count = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        for _ in 0..event_count {
            offset += 9; // fixed fields
            let payload_len = u16::from_le_bytes(data[offset..offset+2].try_into().unwrap()) as usize;
            offset += 2 + payload_len;
        }

        // Parse CPSF checkpoints
        if offset + 4 > checksum_start {
            return Err(ReplayError::FormatError("Missing checkpoint count".into()));
        }
        let cp_count = u32::from_le_bytes(data[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        let mut checkpoints = Vec::with_capacity(cp_count);
        for _ in 0..cp_count {
            if offset + 36 > checksum_start {
                return Err(ReplayError::FormatError("Truncated checkpoint".into()));
            }
            let cp_bytes: [u8; 36] = data[offset..offset+36].try_into().unwrap();
            checkpoints.push(CpsfCheckpoint::from_bytes(&cp_bytes));
            offset += 36;
        }

        // Parse draw outcome
        let outcome = DrawOutcome::from_bytes(&data[offset..checksum_start])
            .map_err(|e| ReplayError::FormatError(e))?;

        Ok(Self { header, event_stream, checkpoints, outcome })
    }

    fn parse_header(data: &[u8]) -> Result<ReplayHeader, ReplayError> {
        if data.len() < 56 {
            return Err(ReplayError::FormatError("Header too short".into()));
        }
        let mut wasm_binary_hash = [0u8; 32];
        wasm_binary_hash.copy_from_slice(&data[0..32]);
        let seed = u64::from_le_bytes(data[32..40].try_into().unwrap());
        let sim_version_major = u32::from_le_bytes(data[40..44].try_into().unwrap());
        let sim_version_minor = u32::from_le_bytes(data[44..48].try_into().unwrap());
        let sim_version_patch = u32::from_le_bytes(data[48..52].try_into().unwrap());
        let params_hash_short = u32::from_le_bytes(data[52..56].try_into().unwrap());
        let params_len = u32::from_le_bytes(data[56..60].try_into().unwrap()) as usize;
        if 60 + params_len + 46 > data.len() {
            return Err(ReplayError::FormatError("Header params overflow".into()));
        }
        let params_blob = data[60..60 + params_len].to_vec();
        let p = 60 + params_len;
        let mapping_version = u16::from_le_bytes(data[p..p+2].try_into().unwrap());
        let read_f32 = |off: usize| -> f32 {
            f32::from_bits(u32::from_le_bytes(data[off..off+4].try_into().unwrap()))
        };
        let camera_pos = [read_f32(p+2), read_f32(p+6), read_f32(p+10)];
        let camera_target = [read_f32(p+14), read_f32(p+18), read_f32(p+22)];
        let camera_fov_rad = read_f32(p+26);
        let viewport_width = read_f32(p+30);
        let viewport_height = read_f32(p+34);
        let drum_plane_z = read_f32(p+38);
        let tan_half_fov = read_f32(p+42);

        Ok(ReplayHeader {
            wasm_binary_hash,
            seed,
            sim_version_major,
            sim_version_minor,
            sim_version_patch,
            params_hash_short,
            params_blob,
            mapping_version,
            camera_pos,
            camera_target,
            camera_fov_rad,
            viewport_width,
            viewport_height,
            drum_plane_z,
            tan_half_fov,
        })
    }

    /// Create a header from simulation params and seed.
    pub fn make_header(params: &SimulationParameters, seed: u64, wasm_hash: [u8; 32]) -> ReplayHeader {
        let params_blob = params.serialize_bincode();
        let full_hash = params.hash();
        let params_hash_short = u32::from_le_bytes(full_hash[0..4].try_into().unwrap());
        ReplayHeader {
            wasm_binary_hash: wasm_hash,
            seed,
            sim_version_major: 1,
            sim_version_minor: 0,
            sim_version_patch: 0,
            params_hash_short,
            params_blob,
            mapping_version: 1,
            camera_pos: [0.0, 0.0, 2.0],
            camera_target: [0.0, 0.0, 0.0],
            camera_fov_rad: 1.0472, // ~60 degrees
            viewport_width: 1920.0,
            viewport_height: 1080.0,
            drum_plane_z: 0.0,
            tan_half_fov: 0.57735, // tan(30deg)
        }
    }
}

#[derive(Debug)]
pub enum ReplayError {
    IntegrityViolation,
    UnsupportedVersion(u32),
    FormatError(String),
    BinaryMismatch { expected: [u8; 32], actual: [u8; 32] },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IntegrityViolation => write!(f, "INTEGRITY_VIOLATION: checksum mismatch"),
            Self::UnsupportedVersion(v) => write!(f, "UNSUPPORTED_VERSION: format version {}", v),
            Self::FormatError(msg) => write!(f, "FORMAT_ERROR: {}", msg),
            Self::BinaryMismatch { .. } => write!(f, "BINARY_MISMATCH: wasm binary hash mismatch"),
        }
    }
}

impl std::error::Error for ReplayError {}
