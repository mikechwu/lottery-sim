use wasm_bindgen::prelude::*;
use lottery_sim_core::{
    Simulation, SimulationParameters, InputEvent,
};

/// WASM-exposed simulation handle.
#[wasm_bindgen]
pub struct WasmSimulation {
    inner: Simulation,
    wasm_binary_hash: [u8; 32],
}

#[wasm_bindgen]
impl WasmSimulation {
    /// Initialize simulation with JSON params and seed.
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &str, seed: u64, wasm_hash_hex: &str) -> Result<WasmSimulation, JsValue> {
        let params: SimulationParameters = serde_json::from_str(params_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        let wasm_binary_hash = parse_hex_hash(wasm_hash_hex)
            .map_err(|e| JsValue::from_str(&e))?;
        let sim = Simulation::new(params, seed);
        Ok(Self { inner: sim, wasm_binary_hash })
    }

    /// Step N physics ticks. Returns true if draw is complete.
    pub fn step(&mut self, n_ticks: u32) -> bool {
        self.inner.step(n_ticks)
    }

    /// Apply events (JSON array of InputEvent).
    pub fn apply_events(&mut self, events_json: &str) -> Result<(), JsValue> {
        let events: Vec<InputEvent> = serde_json::from_str(events_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid events: {}", e)))?;
        self.inner.apply_events(&events);
        Ok(())
    }

    /// Export replay as binary bytes.
    pub fn export_replay(&self) -> Vec<u8> {
        let container = self.inner.export_replay(self.wasm_binary_hash);
        container.to_bytes()
    }

    /// Get draw outcome as JSON.
    pub fn get_outcome(&self) -> String {
        let outcome = self.inner.get_outcome();
        serde_json::to_string(&outcome).unwrap_or_default()
    }

    /// Get current CPSF checkpoint as JSON.
    pub fn get_cpsf_checkpoint(&self) -> String {
        let cp = self.inner.get_cpsf_checkpoint();
        let hex_hash: String = cp.hash.iter().map(|b| format!("{:02x}", b)).collect();
        format!("{{\"frame\":{},\"hash\":\"{}\"}}", cp.frame, hex_hash)
    }

    /// Check if simulation is complete.
    pub fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }

    /// Get current frame number.
    pub fn frame(&self) -> u32 {
        self.inner.frame()
    }

    /// Get ball positions as flat Float32Array for Three.js rendering.
    /// Layout: [x, y, z, active, x, y, z, active, ...] (4 floats per ball).
    pub fn get_ball_positions(&self) -> Vec<f32> {
        self.inner.get_ball_positions_flat()
    }

    /// Get drum rotation angle (radians around Y) for rendering.
    pub fn get_drum_angle(&self) -> f32 {
        self.inner.get_drum_angle()
    }

    /// Get simulation parameters as JSON.
    pub fn get_params(&self) -> String {
        serde_json::to_string(self.inner.params()).unwrap_or_default()
    }

    /// Get exit count so far.
    pub fn exit_count(&self) -> u32 {
        self.inner.exits().len() as u32
    }

    /// Get required exit count.
    pub fn required_exits(&self) -> u32 {
        self.inner.params().required_exits
    }

    /// Deterministic screen→drum coordinate mapping.
    /// Uses the authoritative camera parameters from the replay header.
    /// screen_x, screen_y: pixel coordinates
    /// viewport_w, viewport_h: viewport dimensions
    /// Returns [drum_x, drum_y] as f32 values.
    pub fn map_screen_to_drum(
        screen_x: f32, screen_y: f32,
        viewport_w: f32, viewport_h: f32,
        camera_pos_x: f32, camera_pos_y: f32, camera_pos_z: f32,
        camera_target_x: f32, camera_target_y: f32, _camera_target_z: f32,
        fov_rad: f32,
        drum_plane_z: f32,
    ) -> Vec<f32> {
        // NDC coordinates: [-1, 1]
        let ndc_x = (2.0 * screen_x / viewport_w) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / viewport_h);

        // Perspective ray from camera through pixel
        let aspect = viewport_w / viewport_h;
        let half_fov_tan = (fov_rad / 2.0).tan();

        // Ray direction in camera space
        let ray_x = ndc_x * aspect * half_fov_tan;
        let ray_y = ndc_y * half_fov_tan;
        let ray_z: f32 = -1.0; // camera looks along -Z

        // Simple ray-plane intersection with drum_plane_z
        let cam_z = camera_pos_z;
        if (ray_z).abs() < 1e-10 {
            return vec![0.0, 0.0]; // parallel to plane
        }
        let t = (drum_plane_z - cam_z) / ray_z;
        let drum_x = camera_pos_x + ray_x * t - camera_target_x;
        let drum_y = camera_pos_y + ray_y * t - camera_target_y;

        vec![drum_x, drum_y]
    }
}

/// Compute SHA-256 hash of the WASM binary bytes.
/// Called from JS with the loaded .wasm file bytes.
#[wasm_bindgen]
pub fn compute_wasm_hash(bytes: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hash: [u8; 32] = hasher.finalize().into();
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

fn parse_hex_hash(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("Expected 64 hex chars, got {}", hex.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| format!("Invalid hex at byte {}: {}", i, e))?;
    }
    Ok(out)
}
