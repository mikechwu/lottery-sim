# EventStream Schema — v1.0

Defined in `sim-core/src/event.rs`.

## Event Types

| Value | Name | Priority | Description |
|-------|------|----------|-------------|
| 0 | SimStart | 0 | Simulation initialized |
| 1 | ParameterChange | 1 | Runtime parameter change |
| 2 | ActuatorEvent | 2 | Drum motor, paddle, etc. |
| 3 | UserTouchStart | 3 | Touch/click begin |
| 4 | UserTouchMove | 4 | Touch/click move |
| 5 | UserTouchEnd | 5 | Touch/click end |
| 6 | BallExit | 6 | Ball exited chute |
| 7 | DrawTimeout | 7 | Max duration reached |
| 8 | SimEnd | 8 | Simulation complete |

## InputEvent Structure

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `event_type` | u8 | 1 | EventType discriminant |
| `frame` | u32 LE | 4 | Physics frame number |
| `priority` | u8 | 1 | Same as event_type value |
| `pointer_id` | u8 | 1 | Multi-touch pointer ID |
| `sequence_no` | u16 LE | 2 | Per-frame sequence number |
| `payload_len` | u16 LE | 2 | Length of payload bytes |
| `payload` | [u8] | var | Event-specific data |

## Sort Key (Canonical Ordering per v2 §2.2)

`(frame, event_priority, pointer_id, sequence_no)` — ascending.

No two events may share an identical sort key (v2 §2.5 invariant).

## Binary Format

```
[event_count: u32 LE]
[events...]
```

Each event: 11 + payload_len bytes.
