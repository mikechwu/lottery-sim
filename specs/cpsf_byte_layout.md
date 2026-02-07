# CPSF (Canonical Physics State Fingerprint) — v1.0 Byte Layout

Defined in `sim-core/src/cpsf.rs`. Per v2 §1.

## Serialization Layout

```
[frame_number: u32 LE]     (4 bytes)
[body_count: u32 LE]       (4 bytes)
[bodies: body_count * 56 bytes]
```

Total: 8 + body_count * 56 bytes.

## BodyState Layout (56 bytes per body)

| Offset | Field | Size | Type |
|--------|-------|------|------|
| 0 | entity_id | 4 | u32 LE |
| 4 | position.x | 4 | f32 (bits LE) |
| 8 | position.y | 4 | f32 (bits LE) |
| 12 | position.z | 4 | f32 (bits LE) |
| 16 | orientation.x | 4 | f32 (bits LE) |
| 20 | orientation.y | 4 | f32 (bits LE) |
| 24 | orientation.z | 4 | f32 (bits LE) |
| 28 | orientation.w | 4 | f32 (bits LE) |
| 32 | linear_velocity.x | 4 | f32 (bits LE) |
| 36 | linear_velocity.y | 4 | f32 (bits LE) |
| 40 | linear_velocity.z | 4 | f32 (bits LE) |
| 44 | angular_velocity.x | 4 | f32 (bits LE) |
| 48 | angular_velocity.y | 4 | f32 (bits LE) |
| 52 | angular_velocity.z | 4 | f32 (bits LE) |
| 54 | is_sleeping | 1 | u8 (0 or 1) |
| 55 | is_active | 1 | u8 (0 or 1) |

## Hash

SHA-256 of the serialized CPSF bytes. Per v2 §1.7.

## Checkpoint

```
[frame: u32 LE] [hash: 32 bytes]
```

36 bytes per checkpoint.

## Ordering

Bodies MUST be sorted by entity_id ascending. In the current implementation, balls are created in entity_id order (1..N) and this order is maintained throughout.

## Frozen Bodies (v4 §C.3)

When a ball exits the chute:
- Position is frozen at last active position
- Velocity is zeroed
- `is_active` is set to false
- Body remains in CPSF with frozen state
