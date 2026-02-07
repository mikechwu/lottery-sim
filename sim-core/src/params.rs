use serde::{Serialize, Deserialize};

/// SimulationParameters — versioned, deterministic configuration.
/// Per spec: all physics and deadlock parameters are versioned here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationParameters {
    pub ball_count: u32,
    pub required_exits: u32,
    pub ball_radius: f32,
    pub ball_mass: f32, // pinned per v4 §F: default 0.010 kg
    pub drum_radius: f32,
    pub drum_height: f32,
    pub gravity: f32,
    pub restitution: f32,
    pub friction: f32,
    pub fixed_dt: f32, // 1/120 per spec
    pub max_draw_duration_frames: u32,
    // Geometry
    pub chute_exit_normal: [f32; 3],
    pub chute_exit_origin: [f32; 3],
    // Exit detection (v4 §C)
    pub exit_velocity_threshold: f32,
    pub exit_position_threshold: f32,
    pub exit_distance_quantum: f32, // v4 §D: default 0.001
    // Deadlock (v3 §5, v4 §F)
    pub deadlock: DeadlockParams,
    // CPSF
    pub cpsf_checkpoint_interval_frames: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeadlockParams {
    pub drum_stall_mss_threshold: f32,
    pub drum_stall_window_frames: u32,
    pub chute_jam_window_frames: u32,
    pub chute_progress_epsilon: f32,
    pub nudge_force_magnitude: f32,
    pub max_nudges_per_ball: u32,
    pub max_global_nudges: u32,
    pub check_interval_frames: u32,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            ball_count: 50,
            required_exits: 6,
            ball_radius: 0.02,
            ball_mass: 0.010, // 10 grams per v4 §F
            drum_radius: 0.5,
            drum_height: 0.4,
            gravity: 9.81,
            restitution: 0.6,
            friction: 0.3,
            fixed_dt: 1.0 / 120.0,
            max_draw_duration_frames: 36_000,
            chute_exit_normal: [0.0, -1.0, 0.0],
            chute_exit_origin: [0.0, -0.5, 0.0],
            exit_velocity_threshold: 0.01,
            exit_position_threshold: 0.005,
            exit_distance_quantum: 0.001,
            deadlock: DeadlockParams::default(),
            cpsf_checkpoint_interval_frames: 120,
        }
    }
}

impl Default for DeadlockParams {
    fn default() -> Self {
        Self {
            drum_stall_mss_threshold: 0.200,
            drum_stall_window_frames: 240,
            chute_jam_window_frames: 360,
            chute_progress_epsilon: 0.01,
            nudge_force_magnitude: 5.0,
            max_nudges_per_ball: 3,
            max_global_nudges: 20,
            check_interval_frames: 120,
        }
    }
}

impl SimulationParameters {
    pub fn hash(&self) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let bytes = bincode::serialize(self).expect("params serialization");
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }

    pub fn serialize_bincode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("params serialization")
    }
}
