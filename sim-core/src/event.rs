use serde::{Serialize, Deserialize};

/// Event types per v2 §2.3 priority table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventType {
    SimStart = 0,
    ParameterChange = 1,
    ActuatorEvent = 2,
    UserTouchStart = 3,
    UserTouchMove = 4,
    UserTouchEnd = 5,
    BallExit = 6,
    DrawTimeout = 7,
    SimEnd = 8,
}

impl EventType {
    pub fn priority(&self) -> u8 {
        *self as u8
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::SimStart),
            1 => Some(Self::ParameterChange),
            2 => Some(Self::ActuatorEvent),
            3 => Some(Self::UserTouchStart),
            4 => Some(Self::UserTouchMove),
            5 => Some(Self::UserTouchEnd),
            6 => Some(Self::BallExit),
            7 => Some(Self::DrawTimeout),
            8 => Some(Self::SimEnd),
            _ => None,
        }
    }
}

/// Per-event payload for user input events (v4 §H.2.4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserInputPayload {
    pub screen_x: i32,
    pub screen_y: i32,
    pub drum_x: f32,
    pub drum_y: f32,
    pub force_magnitude: f32,
    pub direction_x: f32,
    pub direction_y: f32,
}

/// A single event in the event stream.
/// Sort key: (frame, event_priority, pointer_id, sequence_no) per v2 §2.2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputEvent {
    pub event_type: EventType,
    pub frame: u32,
    pub pointer_id: u8,
    pub sequence_no: u16,
    pub payload: Vec<u8>, // event-type-specific payload bytes
}

impl InputEvent {
    pub fn sort_key(&self) -> (u32, u8, u8, u16) {
        (self.frame, self.event_type.priority(), self.pointer_id, self.sequence_no)
    }
}

/// The full event stream for a draw.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventStream {
    pub events: Vec<InputEvent>,
}

impl EventStream {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn push(&mut self, event: InputEvent) {
        self.events.push(event);
    }

    /// Sort events by the canonical ordering per v2 §2.2.
    pub fn sort_canonical(&mut self) {
        self.events.sort_by_key(|e| e.sort_key());
    }

    /// Validate that no two events have identical sort keys (v2 §2.5 invariant).
    pub fn validate_ordering(&self) -> Result<(), String> {
        for i in 1..self.events.len() {
            if self.events[i].sort_key() == self.events[i - 1].sort_key() {
                return Err(format!(
                    "Duplicate sort key at index {}: {:?}",
                    i, self.events[i].sort_key()
                ));
            }
        }
        Ok(())
    }

    /// Serialize events to binary format per v4 §H.2.4.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.events.len() as u32).to_le_bytes());
        for ev in &self.events {
            buf.push(ev.event_type as u8);
            buf.extend_from_slice(&ev.frame.to_le_bytes());
            buf.push(ev.event_type.priority());
            buf.push(ev.pointer_id);
            buf.extend_from_slice(&ev.sequence_no.to_le_bytes());
            buf.extend_from_slice(&(ev.payload.len() as u16).to_le_bytes());
            buf.extend_from_slice(&ev.payload);
        }
        buf
    }

    /// Deserialize events from binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("Event stream too short".into());
        }
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            if offset + 9 > data.len() {
                return Err("Truncated event".into());
            }
            let event_type = EventType::from_u8(data[offset])
                .ok_or_else(|| format!("Unknown event type: {}", data[offset]))?;
            let frame = u32::from_le_bytes(data[offset+1..offset+5].try_into().unwrap());
            // skip priority byte at offset+5
            let pointer_id = data[offset + 6];
            let sequence_no = u16::from_le_bytes(data[offset+7..offset+9].try_into().unwrap());
            let payload_len = u16::from_le_bytes(data[offset+9..offset+11].try_into().unwrap()) as usize;
            offset += 11;
            if offset + payload_len > data.len() {
                return Err("Truncated event payload".into());
            }
            let payload = data[offset..offset + payload_len].to_vec();
            offset += payload_len;
            events.push(InputEvent { event_type, frame, pointer_id, sequence_no, payload });
        }
        Ok(EventStream { events })
    }
}
