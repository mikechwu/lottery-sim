use std::process;
use lottery_sim_core::verify::Verifier;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: verify <replay_file> <wasm_hash_hex>");
        eprintln!("  replay_file: path to .replay binary file");
        eprintln!("  wasm_hash_hex: 64-char hex string of the WASM binary SHA-256");
        process::exit(2);
    }

    let replay_path = &args[1];
    let wasm_hash_hex = &args[2];

    // Parse wasm hash.
    let wasm_hash = match parse_hex_hash(wasm_hash_hex) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Error: invalid wasm hash: {}", e);
            process::exit(2);
        }
    };

    // Read replay file.
    let replay_data = match std::fs::read(replay_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: cannot read {}: {}", replay_path, e);
            process::exit(2);
        }
    };

    // Run verification.
    let verifier = Verifier::new(wasm_hash);
    let veb = verifier.verify_bytes(&replay_data);

    // Output VEB as JSON.
    let json = serde_json::to_string_pretty(&veb).expect("VEB serialization");
    println!("{}", json);

    // Deterministic exit codes per spec.
    let exit_code = match veb.verdict.as_str() {
        "VERIFIED" => 0,
        "INTEGRITY_VIOLATION" => 10,
        "BINARY_MISMATCH" => 11,
        "CHECKPOINT_MISMATCH" => 12,
        "FORMAT_ERROR" => 20,
        "UNSUPPORTED_VERSION" => 21,
        _ => 99,
    };
    process::exit(exit_code);
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
