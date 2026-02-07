use rand::prelude::*;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;

use crate::cpsf::{BodyState, Cpsf, CpsfCheckpoint};
use crate::event::{EventStream, EventType, InputEvent};
use crate::outcome::{BallExit, DrawOutcome, DrawStatus};
use crate::params::SimulationParameters;

/// Hash function for domain-separated PRNG streams per v2 §4.3.
fn domain_seed(master_seed: u64, domain: &str) -> u64 {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(&master_seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(hash[0..8].try_into().unwrap())
}

/// Ball state for the placeholder physics.
#[derive(Debug, Clone)]
struct Ball {
    entity_id: u32,
    position: [f32; 3],
    velocity: [f32; 3],
    active: bool,
    exit_frame: Option<u32>,
    /// Frozen position for CPSF after exit (v4 §C.3).
    frozen_position: Option<[f32; 3]>,
}

/// Deterministic simulation engine.
/// Uses placeholder physics (gravity + boundary bounce) — adequate for
/// Phase-1 foundation. Real Rapier integration deferred to Phase-2.
pub struct Simulation {
    params: SimulationParameters,
    seed: u64,
    frame: u32,
    balls: Vec<Ball>,
    event_stream: EventStream,
    checkpoints: Vec<CpsfCheckpoint>,
    exits: Vec<BallExit>,
    completed: bool,
    /// BTreeMap for deterministic iteration order (no HashMap per v1 §1.4).
    ball_nudge_counts: BTreeMap<u32, u32>,
    global_nudge_count: u32,
    rng_recovery: Pcg64,
}

impl Simulation {
    /// Initialize simulation per spec: deterministic RNG seeding.
    pub fn new(params: SimulationParameters, seed: u64) -> Self {
        // Domain-separated PRNG streams per v2 §4.3.
        let pos_seed = domain_seed(seed, "positions");
        let vel_seed = domain_seed(seed, "velocities");
        let recovery_seed = domain_seed(seed, "runtime_recovery");

        let mut pos_rng = Pcg64::seed_from_u64(pos_seed);
        let mut vel_rng = Pcg64::seed_from_u64(vel_seed);
        let rng_recovery = Pcg64::seed_from_u64(recovery_seed);

        // Create balls with deterministic initial state.
        // Entity IDs: balls are 1..N, drum would be N+1, paddles N+2..
        let mut balls = Vec::with_capacity(params.ball_count as usize);
        for i in 0..params.ball_count {
            let entity_id = i + 1;
            // Random positions inside drum volume.
            let x = (pos_rng.gen::<f32>() - 0.5) * params.drum_radius;
            let y = (pos_rng.gen::<f32>() - 0.5) * params.drum_height;
            let z = (pos_rng.gen::<f32>() - 0.5) * params.drum_radius;

            // Small random initial velocities.
            let vx = (vel_rng.gen::<f32>() - 0.5) * 0.5;
            let vy = (vel_rng.gen::<f32>() - 0.5) * 0.5;
            let vz = (vel_rng.gen::<f32>() - 0.5) * 0.5;

            balls.push(Ball {
                entity_id,
                position: [x, y, z],
                velocity: [vx, vy, vz],
                active: true,
                exit_frame: None,
                frozen_position: None,
            });
        }

        let mut sim = Self {
            params,
            seed,
            frame: 0,
            balls,
            event_stream: EventStream::new(),
            checkpoints: Vec::new(),
            exits: Vec::new(),
            completed: false,
            ball_nudge_counts: BTreeMap::new(),
            global_nudge_count: 0,
            rng_recovery,
        };

        // Record SimStart event.
        sim.event_stream.push(InputEvent {
            event_type: EventType::SimStart,
            frame: 0,
            pointer_id: 0,
            sequence_no: 0,
            payload: Vec::new(),
        });

        // Checkpoint at frame 0 per v4 §B.2.
        let cpsf = sim.compute_cpsf();
        sim.checkpoints.push(CpsfCheckpoint {
            frame: 0,
            hash: cpsf.hash(),
        });

        sim
    }

    /// Step N physics ticks with fixed dt.
    /// Returns true if the draw is complete.
    pub fn step(&mut self, n_ticks: u32) -> bool {
        for _ in 0..n_ticks {
            if self.completed {
                return true;
            }
            self.step_one();
        }
        self.completed
    }

    fn step_one(&mut self) {
        self.frame += 1;
        let dt = self.params.fixed_dt;

        // Placeholder physics: gravity + boundary bounce.
        // Stable iteration: iterate by entity_id order (balls are pre-sorted).
        for ball in self.balls.iter_mut() {
            if !ball.active {
                continue;
            }

            // Apply gravity.
            ball.velocity[1] -= self.params.gravity * dt;

            // Integrate position.
            for i in 0..3 {
                ball.position[i] += ball.velocity[i] * dt;
            }

            // Boundary bounce (drum walls — simplified cylinder).
            let r = self.params.drum_radius - self.params.ball_radius;
            let dist_xz = (ball.position[0] * ball.position[0]
                + ball.position[2] * ball.position[2])
                .sqrt();
            if dist_xz > r {
                // Reflect radially.
                let nx = ball.position[0] / dist_xz;
                let nz = ball.position[2] / dist_xz;
                let dot = ball.velocity[0] * nx + ball.velocity[2] * nz;
                if dot > 0.0 {
                    ball.velocity[0] -= 2.0 * dot * nx * self.params.restitution;
                    ball.velocity[2] -= 2.0 * dot * nz * self.params.restitution;
                }
                // Push back inside.
                ball.position[0] = nx * r;
                ball.position[2] = nz * r;
            }

            // Floor/ceiling bounce.
            let half_h = self.params.drum_height / 2.0 - self.params.ball_radius;
            if ball.position[1] < -half_h {
                ball.position[1] = -half_h;
                if ball.velocity[1] < 0.0 {
                    ball.velocity[1] = -ball.velocity[1] * self.params.restitution;
                }
            }
            if ball.position[1] > half_h {
                ball.position[1] = half_h;
                if ball.velocity[1] > 0.0 {
                    ball.velocity[1] = -ball.velocity[1] * self.params.restitution;
                }
            }
        }

        // Exit detection (post-step per v3 §2.3 / v4 §C).
        let mut new_exits = Vec::new();
        for ball in self.balls.iter() {
            if !ball.active {
                continue;
            }
            // Check exit conditions per v4 §C.2.
            let normal = self.params.chute_exit_normal;
            let origin = self.params.chute_exit_origin;
            let rel = [
                ball.position[0] - origin[0],
                ball.position[1] - origin[1],
                ball.position[2] - origin[2],
            ];
            let pos_dot = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
            let vel_dot = ball.velocity[0] * normal[0]
                + ball.velocity[1] * normal[1]
                + ball.velocity[2] * normal[2];

            if pos_dot > self.params.exit_position_threshold
                && vel_dot > self.params.exit_velocity_threshold
            {
                // Quantized distance for tie-breaking per v4 §D.2.
                let quantized = (pos_dot / self.params.exit_distance_quantum) as i32;
                new_exits.push((ball.entity_id, quantized, ball.position));
            }
        }

        // Sort same-frame exits: quantized_distance DESC, entity_id ASC (v4 §D.2).
        new_exits.sort_by(|a, b| {
            b.1.cmp(&a.1).then(a.0.cmp(&b.0))
        });

        // Process exits.
        for (entity_id, _, last_pos) in &new_exits {
            if self.exits.len() >= self.params.required_exits as usize {
                break;
            }
            // Record exit.
            self.exits.push(BallExit {
                entity_id: *entity_id,
                exit_frame: self.frame,
            });
            self.event_stream.push(InputEvent {
                event_type: EventType::BallExit,
                frame: self.frame,
                pointer_id: 0,
                sequence_no: self.exits.len() as u16,
                payload: entity_id.to_le_bytes().to_vec(),
            });
            // Mark ball inactive, freeze position (v4 §C.3).
            if let Some(ball) = self.balls.iter_mut().find(|b| b.entity_id == *entity_id) {
                ball.active = false;
                ball.frozen_position = Some(*last_pos);
                ball.exit_frame = Some(self.frame);
                // Zero velocities for CPSF (v4 §C.3).
                ball.velocity = [0.0, 0.0, 0.0];
            }

            // Checkpoint on exit per v4 §B.2.
            let cpsf = self.compute_cpsf();
            self.checkpoints.push(CpsfCheckpoint {
                frame: self.frame,
                hash: cpsf.hash(),
            });
        }

        // Check completion.
        if self.exits.len() >= self.params.required_exits as usize {
            self.completed = true;
            self.event_stream.push(InputEvent {
                event_type: EventType::SimEnd,
                frame: self.frame,
                pointer_id: 0,
                sequence_no: 0,
                payload: Vec::new(),
            });
            // Final checkpoint.
            let cpsf = self.compute_cpsf();
            self.checkpoints.push(CpsfCheckpoint {
                frame: self.frame,
                hash: cpsf.hash(),
            });
            return;
        }

        // Check timeout.
        if self.frame >= self.params.max_draw_duration_frames {
            self.completed = true;
            self.event_stream.push(InputEvent {
                event_type: EventType::DrawTimeout,
                frame: self.frame,
                pointer_id: 0,
                sequence_no: 0,
                payload: Vec::new(),
            });
            self.event_stream.push(InputEvent {
                event_type: EventType::SimEnd,
                frame: self.frame,
                pointer_id: 0,
                sequence_no: 1,
                payload: Vec::new(),
            });
            let cpsf = self.compute_cpsf();
            self.checkpoints.push(CpsfCheckpoint {
                frame: self.frame,
                hash: cpsf.hash(),
            });
            return;
        }

        // Periodic CPSF checkpoint per v4 §B.2.
        if self.frame % self.params.cpsf_checkpoint_interval_frames == 0 {
            let cpsf = self.compute_cpsf();
            self.checkpoints.push(CpsfCheckpoint {
                frame: self.frame,
                hash: cpsf.hash(),
            });
        }
    }

    /// Apply external events (user input). Phase-1: stub, events are recorded.
    pub fn apply_events(&mut self, events: &[InputEvent]) {
        for event in events {
            self.event_stream.push(event.clone());
        }
    }

    /// Compute CPSF at current frame.
    pub fn compute_cpsf(&self) -> Cpsf {
        let mut bodies = Vec::with_capacity(self.balls.len());
        for ball in &self.balls {
            let pos = if let Some(frozen) = ball.frozen_position {
                frozen
            } else {
                ball.position
            };
            bodies.push(BodyState {
                entity_id: ball.entity_id,
                position: pos,
                orientation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
                linear_velocity: ball.velocity,
                angular_velocity: [0.0, 0.0, 0.0],
                is_sleeping: false,
                is_active: ball.active,
            });
        }
        // Bodies sorted by entity_id ascending (already in order by construction).
        Cpsf {
            frame_number: self.frame,
            bodies,
        }
    }

    /// Get current CPSF checkpoint hash.
    pub fn get_cpsf_checkpoint(&self) -> CpsfCheckpoint {
        let cpsf = self.compute_cpsf();
        CpsfCheckpoint {
            frame: self.frame,
            hash: cpsf.hash(),
        }
    }

    /// Get the draw outcome.
    pub fn get_outcome(&self) -> DrawOutcome {
        let status = if self.exits.len() >= self.params.required_exits as usize {
            DrawStatus::DrawComplete
        } else if self.frame >= self.params.max_draw_duration_frames {
            DrawStatus::DrawTimedOut
        } else {
            DrawStatus::DrawComplete // in progress, but report what we have
        };
        DrawOutcome {
            status,
            exits: self.exits.clone(),
        }
    }

    /// Export a replay container.
    pub fn export_replay(&self, wasm_hash: [u8; 32]) -> crate::replay::ReplayContainer {
        let header = crate::replay::ReplayContainer::make_header(&self.params, self.seed, wasm_hash);
        crate::replay::ReplayContainer {
            header,
            event_stream: self.event_stream.clone(),
            checkpoints: self.checkpoints.clone(),
            outcome: self.get_outcome(),
        }
    }

    pub fn frame(&self) -> u32 {
        self.frame
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn params(&self) -> &SimulationParameters {
        &self.params
    }

    pub fn checkpoints(&self) -> &[CpsfCheckpoint] {
        &self.checkpoints
    }
}
