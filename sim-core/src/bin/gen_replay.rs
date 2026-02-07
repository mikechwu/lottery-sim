use lottery_sim_core::{Simulation, SimulationParameters};
use std::io::Write;

fn main() {
    let params = SimulationParameters::default();
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let mut wasm_hash = [0u8; 32];
    for i in 0..32 {
        wasm_hash[i] = (i as u8).wrapping_mul(7).wrapping_add(0xAB);
    }

    let mut sim = Simulation::new(params, seed);
    while !sim.is_completed() {
        sim.step(120);
    }

    let outcome = sim.get_outcome();
    let container = sim.export_replay(wasm_hash);
    let bytes = container.to_bytes();

    let path = std::env::args().nth(1).unwrap_or_else(|| "test.replay".to_string());
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(&bytes).expect("write");

    let hash_hex: String = wasm_hash.iter().map(|b| format!("{:02x}", b)).collect();
    eprintln!("Seed: {}", seed);
    eprintln!("Status: {:?}", outcome.status);
    eprintln!("Exits: {}", outcome.exits.len());
    eprintln!("Frames: {}", sim.frame());
    eprintln!("Checkpoints: {}", sim.checkpoints().len());
    eprintln!("Replay size: {} bytes", bytes.len());
    eprintln!("Wrote: {}", path);
    eprintln!("WASM hash: {}", hash_hex);
}
