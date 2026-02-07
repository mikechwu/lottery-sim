# SimulationParameters Schema — v1.0

Defined in `sim-core/src/params.rs`.

## Fields

| Field | Type | Default | Spec Reference |
|-------|------|---------|----------------|
| `ball_count` | u32 | 50 | v1 §3 |
| `required_exits` | u32 | 6 | v1 §3 |
| `ball_radius` | f32 | 0.02 | v1 §3 |
| `ball_mass` | f32 | 0.010 | v4 §F |
| `drum_radius` | f32 | 0.5 | v1 §3 |
| `drum_height` | f32 | 0.4 | v1 §3 |
| `gravity` | f32 | 9.81 | v1 §3 |
| `restitution` | f32 | 0.6 | v1 §3 |
| `friction` | f32 | 0.3 | v1 §3 |
| `fixed_dt` | f32 | 1/120 | v1 §1 |
| `max_draw_duration_frames` | u32 | 36000 | v3 §5 |
| `chute_exit_normal` | [f32;3] | [0,-1,0] | v4 §C |
| `chute_exit_origin` | [f32;3] | [0,-0.5,0] | v4 §C |
| `exit_velocity_threshold` | f32 | 0.01 | v4 §C.2 |
| `exit_position_threshold` | f32 | 0.005 | v4 §C.2 |
| `exit_distance_quantum` | f32 | 0.001 | v4 §D.2 |
| `deadlock` | DeadlockParams | (see below) | v3 §5, v4 §F |
| `cpsf_checkpoint_interval_frames` | u32 | 120 | v4 §B.2 |

## DeadlockParams

| Field | Type | Default | Spec Reference |
|-------|------|---------|----------------|
| `drum_stall_mss_threshold` | f32 | 0.200 | v4 §F |
| `drum_stall_window_frames` | u32 | 240 | v3 §5 |
| `chute_jam_window_frames` | u32 | 360 | v3 §5 |
| `chute_progress_epsilon` | f32 | 0.01 | v3 §5 |
| `nudge_force_magnitude` | f32 | 5.0 | v3 §5 |
| `max_nudges_per_ball` | u32 | 3 | v3 §5 |
| `max_global_nudges` | u32 | 20 | v3 §5 |
| `check_interval_frames` | u32 | 120 | v3 §5 |

## Serialization

- Binary: bincode (deterministic, compact)
- Hash: SHA-256 of bincode bytes
- JSON: serde_json (for WASM interface)
