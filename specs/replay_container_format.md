# Replay Container Format — v1.0

Defined in `sim-core/src/replay.rs`. Per v4 §H.

## Binary Layout

```
+---------------------------+--------+
| Magic                     | 8 B    | "LOTTORPL" (ASCII)
| Format Version            | 4 B LE | u32, currently 1
| Header Length              | 4 B LE | u32
| Header (variable)         | N B    |
| Event Stream              | M B    | (see event_stream.md)
| Checkpoint Count           | 4 B LE | u32
| Checkpoints               | K*36 B | (frame:4 + hash:32 each)
| Draw Outcome              | P B    | (see below)
| SHA-256 Checksum           | 32 B   | Covers all bytes above
+---------------------------+--------+
```

## Header Fields

| Offset | Field | Size | Type |
|--------|-------|------|------|
| 0 | wasm_binary_hash | 32 | [u8;32] |
| 32 | seed | 8 | u64 LE |
| 40 | sim_version_major | 4 | u32 LE |
| 44 | sim_version_minor | 4 | u32 LE |
| 48 | sim_version_patch | 4 | u32 LE |
| 52 | params_hash_short | 4 | u32 LE (first 4 bytes of SHA-256) |
| 56 | params_blob_len | 4 | u32 LE |
| 60 | params_blob | var | bincode bytes |
| 60+N | mapping_version | 2 | u16 LE |
| 62+N | camera_pos | 12 | [f32;3] (bits LE) |
| 74+N | camera_target | 12 | [f32;3] (bits LE) |
| 86+N | camera_fov_rad | 4 | f32 (bits LE) |
| 90+N | viewport_width | 4 | f32 (bits LE) |
| 94+N | viewport_height | 4 | f32 (bits LE) |
| 98+N | drum_plane_z | 4 | f32 (bits LE) |
| 102+N | tan_half_fov | 4 | f32 (bits LE) |

## Draw Outcome

```
[status: u8] [exit_count: u8] [exits: exit_count * 8 bytes]
```

Each exit: entity_id (u32 LE) + exit_frame (u32 LE).

## Checksum

SHA-256 of all bytes preceding the checksum (magic through outcome).

## Verification

1. Check magic = "LOTTORPL"
2. Check format version = 1
3. Verify checksum (last 32 bytes vs SHA-256 of preceding bytes)
4. Parse header, verify wasm_binary_hash matches
5. Replay simulation, compare CPSF checkpoints
6. Compare DrawOutcome

## Error Codes

| Verdict | CLI Exit Code | Description |
|---------|--------------|-------------|
| VERIFIED | 0 | All checks pass |
| INTEGRITY_VIOLATION | 10 | Checksum mismatch |
| BINARY_MISMATCH | 11 | WASM hash mismatch |
| CHECKPOINT_MISMATCH | 12 | CPSF hash mismatch |
| FORMAT_ERROR | 20 | Parse/structure error |
| UNSUPPORTED_VERSION | 21 | Unknown format version |
