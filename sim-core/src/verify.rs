use serde::{Serialize, Deserialize};
use crate::replay::{ReplayContainer, ReplayError};
use crate::sim::Simulation;
use crate::params::SimulationParameters;

/// Verification verdict per v5 §C / v4 §H.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationVerdict {
    Verified,
    IntegrityViolation,
    BinaryMismatch,
    CheckpointMismatch,
    FormatError,
    UnsupportedVersion,
}

impl std::fmt::Display for VerificationVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verified => write!(f, "VERIFIED"),
            Self::IntegrityViolation => write!(f, "INTEGRITY_VIOLATION"),
            Self::BinaryMismatch => write!(f, "BINARY_MISMATCH"),
            Self::CheckpointMismatch => write!(f, "CHECKPOINT_MISMATCH"),
            Self::FormatError => write!(f, "FORMAT_ERROR"),
            Self::UnsupportedVersion => write!(f, "UNSUPPORTED_VERSION"),
        }
    }
}

/// Verification Evidence Bundle (VEB) per v5 §C.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationEvidenceBundle {
    pub verdict: String,
    pub replay_checksum_ok: bool,
    pub binary_hash_match: bool,
    pub checkpoints_verified: u32,
    pub checkpoints_total: u32,
    pub first_mismatch_frame: Option<u32>,
    pub outcome_match: bool,
    pub seed: u64,
    pub params_hash: String,
    pub wasm_binary_hash: String,
    pub diagnostics: Vec<String>,
}

/// Verifier: reads a replay, checks integrity, replays simulation,
/// compares CPSF checkpoints, produces VEB.
pub struct Verifier {
    wasm_binary_hash: [u8; 32],
}

impl Verifier {
    pub fn new(wasm_binary_hash: [u8; 32]) -> Self {
        Self { wasm_binary_hash }
    }

    /// Verify raw replay bytes. Returns VEB.
    /// Per QA-3: binary mismatch must NOT run the simulation.
    pub fn verify_bytes(&self, replay_data: &[u8]) -> VerificationEvidenceBundle {
        // Step 1: Parse replay (checks magic, version, checksum).
        let container = match ReplayContainer::from_bytes(replay_data) {
            Ok(c) => c,
            Err(e) => {
                let (verdict, diag) = match &e {
                    ReplayError::IntegrityViolation => {
                        (VerificationVerdict::IntegrityViolation, e.to_string())
                    }
                    ReplayError::UnsupportedVersion(_) => {
                        (VerificationVerdict::UnsupportedVersion, e.to_string())
                    }
                    ReplayError::FormatError(_) => {
                        (VerificationVerdict::FormatError, e.to_string())
                    }
                    ReplayError::BinaryMismatch { .. } => {
                        (VerificationVerdict::BinaryMismatch, e.to_string())
                    }
                };
                return VerificationEvidenceBundle {
                    verdict: verdict.to_string(),
                    replay_checksum_ok: matches!(verdict, VerificationVerdict::FormatError | VerificationVerdict::UnsupportedVersion),
                    binary_hash_match: false,
                    checkpoints_verified: 0,
                    checkpoints_total: 0,
                    first_mismatch_frame: None,
                    outcome_match: false,
                    seed: 0,
                    params_hash: String::new(),
                    wasm_binary_hash: hex::encode(self.wasm_binary_hash),
                    diagnostics: vec![diag],
                };
            }
        };

        self.verify_container(&container)
    }

    /// Verify a parsed replay container.
    pub fn verify_container(&self, container: &ReplayContainer) -> VerificationEvidenceBundle {
        let header = &container.header;
        let mut diagnostics = Vec::new();

        // Step 2: Check binary hash match (QA-3: must NOT simulate on mismatch).
        if header.wasm_binary_hash != self.wasm_binary_hash {
            return VerificationEvidenceBundle {
                verdict: VerificationVerdict::BinaryMismatch.to_string(),
                replay_checksum_ok: true,
                binary_hash_match: false,
                checkpoints_verified: 0,
                checkpoints_total: container.checkpoints.len() as u32,
                first_mismatch_frame: None,
                outcome_match: false,
                seed: header.seed,
                params_hash: hex::encode(header.params_hash_short.to_le_bytes()),
                wasm_binary_hash: hex::encode(self.wasm_binary_hash),
                diagnostics: vec![format!(
                    "Binary mismatch: expected {}, replay has {}",
                    hex::encode(self.wasm_binary_hash),
                    hex::encode(header.wasm_binary_hash)
                )],
            };
        }

        // Step 3: Deserialize params from header blob.
        let params: SimulationParameters = match bincode::deserialize(&header.params_blob) {
            Ok(p) => p,
            Err(e) => {
                return VerificationEvidenceBundle {
                    verdict: VerificationVerdict::FormatError.to_string(),
                    replay_checksum_ok: true,
                    binary_hash_match: true,
                    checkpoints_verified: 0,
                    checkpoints_total: container.checkpoints.len() as u32,
                    first_mismatch_frame: None,
                    outcome_match: false,
                    seed: header.seed,
                    params_hash: String::new(),
                    wasm_binary_hash: hex::encode(self.wasm_binary_hash),
                    diagnostics: vec![format!("Failed to deserialize params: {}", e)],
                };
            }
        };

        // Step 4: Replay simulation deterministically.
        let mut sim = Simulation::new(params.clone(), header.seed);
        let replay_checkpoints = &container.checkpoints;
        let total_checkpoints = replay_checkpoints.len() as u32;
        let mut verified_count: u32 = 0;
        let mut first_mismatch: Option<u32> = None;

        // Verify frame-0 checkpoint.
        if let Some(cp0) = replay_checkpoints.first() {
            let sim_cp = sim.get_cpsf_checkpoint();
            if cp0.hash == sim_cp.hash {
                verified_count += 1;
            } else {
                first_mismatch = Some(0);
                diagnostics.push(format!(
                    "Checkpoint mismatch at frame 0: expected {}, got {}",
                    hex::encode(cp0.hash),
                    hex::encode(sim_cp.hash)
                ));
            }
        }

        // Run simulation frame by frame, checking checkpoints.
        if first_mismatch.is_none() {
            // Build a lookup: frame -> expected hash.
            let mut checkpoint_map: std::collections::BTreeMap<u32, Vec<[u8; 32]>> =
                std::collections::BTreeMap::new();
            for cp in replay_checkpoints.iter().skip(1) {
                checkpoint_map
                    .entry(cp.frame)
                    .or_default()
                    .push(cp.hash);
            }

            let max_frame = replay_checkpoints
                .last()
                .map(|cp| cp.frame)
                .unwrap_or(params.max_draw_duration_frames);

            for _frame_i in 0..max_frame {
                if sim.is_completed() {
                    break;
                }
                sim.step(1);

                let current_frame = sim.frame();
                if let Some(expected_hashes) = checkpoint_map.remove(&current_frame) {
                    let sim_cp = sim.get_cpsf_checkpoint();
                    for expected_hash in &expected_hashes {
                        if *expected_hash == sim_cp.hash {
                            verified_count += 1;
                        } else {
                            first_mismatch = Some(current_frame);
                            diagnostics.push(format!(
                                "Checkpoint mismatch at frame {}: expected {}, got {}",
                                current_frame,
                                hex::encode(expected_hash),
                                hex::encode(sim_cp.hash)
                            ));
                            break;
                        }
                    }
                    if first_mismatch.is_some() {
                        break;
                    }
                }
            }
        }

        // Step 5: Compare outcome.
        let sim_outcome = sim.get_outcome();
        let outcome_match = sim_outcome == container.outcome;
        if !outcome_match {
            diagnostics.push("DrawOutcome mismatch between replay and re-simulation".into());
        }

        // Step 6: Produce verdict.
        let verdict = if first_mismatch.is_some() {
            VerificationVerdict::CheckpointMismatch
        } else if !outcome_match {
            VerificationVerdict::CheckpointMismatch
        } else {
            VerificationVerdict::Verified
        };

        let params_hash_hex = hex::encode(params.hash());

        VerificationEvidenceBundle {
            verdict: verdict.to_string(),
            replay_checksum_ok: true,
            binary_hash_match: true,
            checkpoints_verified: verified_count,
            checkpoints_total: total_checkpoints,
            first_mismatch_frame: first_mismatch,
            outcome_match,
            seed: header.seed,
            params_hash: params_hash_hex,
            wasm_binary_hash: hex::encode(self.wasm_binary_hash),
            diagnostics,
        }
    }
}

/// Minimal hex encoding (no external crate needed for core).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}
