use wasm_bindgen::prelude::*;
use lottery_sim_core::{
    Simulation, SimulationParameters, InputEvent,
};

/// WASM-exposed simulation handle.
/// Provides the minimal wasm-bindgen interface per dev1v1.md §4.
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

    /// Apply events (JSON array of InputEvent). Phase-1: stub pass-through.
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
