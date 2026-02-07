use rand::prelude::*;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;

use rapier3d::prelude::*;
use glam::Vec3;

use crate::cpsf::{BodyState, Cpsf, CpsfCheckpoint};
use crate::event::{EventStream, EventType, InputEvent, UserInputPayload};
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

/// Tracked ball metadata (entity_id mapping to Rapier handle).
#[derive(Debug, Clone)]
struct BallInfo {
    entity_id: u32,
    body_handle: RigidBodyHandle,
    active: bool,
    exit_frame: Option<u32>,
    frozen_position: Option<[f32; 3]>,
    frozen_orientation: Option<[f32; 4]>,
}

/// Deterministic simulation engine with Rapier 3D physics.
/// Phase-2: real rigid-body physics with drum rotation and chute geometry.
pub struct Simulation {
    params: SimulationParameters,
    seed: u64,
    frame: u32,

    // Rapier physics state.
    pipeline: PhysicsPipeline,
    integration_params: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    gravity: Vec3,

    // Ball tracking (sorted by entity_id for deterministic iteration).
    ball_infos: Vec<BallInfo>,
    drum_body_handle: RigidBodyHandle,

    // Simulation state.
    event_stream: EventStream,
    checkpoints: Vec<CpsfCheckpoint>,
    exits: Vec<BallExit>,
    completed: bool,

    // Deadlock recovery (per v3 §5, v4 §F).
    ball_nudge_counts: BTreeMap<u32, u32>,
    global_nudge_count: u32,
    rng_recovery: Pcg64,
    /// Recent mean-square-speed samples for stall detection.
    recent_mss: Vec<f32>,
}

impl Simulation {
    /// Initialize simulation per spec: deterministic RNG seeding + Rapier physics.
    pub fn new(params: SimulationParameters, seed: u64) -> Self {
        // Domain-separated PRNG streams per v2 §4.3.
        let pos_seed = domain_seed(seed, "positions");
        let vel_seed = domain_seed(seed, "velocities");
        let recovery_seed = domain_seed(seed, "runtime_recovery");

        let mut pos_rng = Pcg64::seed_from_u64(pos_seed);
        let mut vel_rng = Pcg64::seed_from_u64(vel_seed);
        let rng_recovery = Pcg64::seed_from_u64(recovery_seed);

        // Initialize Rapier physics.
        let mut integration_params = IntegrationParameters::default();
        integration_params.dt = params.fixed_dt;

        let pipeline = PhysicsPipeline::new();
        let island_manager = IslandManager::new();
        let broad_phase = DefaultBroadPhase::new();
        let narrow_phase = NarrowPhase::new();
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();
        let ccd_solver = CCDSolver::new();
        let gravity = Vec3::new(0.0, -params.gravity, 0.0);

        // --- Build drum geometry ---
        // The drum is a kinematic body that rotates around Y axis.
        let drum_rb = RigidBodyBuilder::kinematic_velocity_based()
            .translation(Vec3::ZERO)
            // Slow rotation around Y axis for tumbling.
            .angvel(Vec3::new(0.0, 2.0, 0.0))
            .build();
        let drum_body_handle = bodies.insert(drum_rb);

        // Drum walls: approximate cylinder with 12 wall segments.
        let r = params.drum_radius;
        let half_h = params.drum_height / 2.0;
        let n_segments = 12;
        let wall_thickness = 0.02;
        for i in 0..n_segments {
            let angle = std::f32::consts::TAU * (i as f32) / (n_segments as f32);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let seg_width = 2.0 * r * (std::f32::consts::PI / n_segments as f32).sin();

            let wall = ColliderBuilder::cuboid(seg_width / 2.0, half_h, wall_thickness / 2.0)
                .translation(Vec3::new(cos_a * r, 0.0, sin_a * r))
                .rotation(Vec3::new(0.0, -angle, 0.0))
                .restitution(params.restitution)
                .friction(params.friction)
                .build();
            colliders.insert_with_parent(wall, drum_body_handle, &mut bodies);
        }

        // Drum floor — with a chute opening (hole) at one edge.
        // We model the floor as two half-moon pieces with a gap for the chute.
        let floor_thickness = 0.02;
        // Left half of floor.
        let floor_left = ColliderBuilder::cuboid(r * 0.4, floor_thickness, r * 0.9)
            .translation(Vec3::new(-r * 0.5, -half_h, 0.0))
            .restitution(params.restitution)
            .friction(params.friction)
            .build();
        colliders.insert_with_parent(floor_left, drum_body_handle, &mut bodies);
        // Right half of floor.
        let floor_right = ColliderBuilder::cuboid(r * 0.4, floor_thickness, r * 0.9)
            .translation(Vec3::new(r * 0.5, -half_h, 0.0))
            .restitution(params.restitution)
            .friction(params.friction)
            .build();
        colliders.insert_with_parent(floor_right, drum_body_handle, &mut bodies);
        // Back portion of floor.
        let floor_back = ColliderBuilder::cuboid(r * 0.15, floor_thickness, r * 0.5)
            .translation(Vec3::new(0.0, -half_h, -r * 0.4))
            .restitution(params.restitution)
            .friction(params.friction)
            .build();
        colliders.insert_with_parent(floor_back, drum_body_handle, &mut bodies);

        // Drum ceiling.
        let ceiling = ColliderBuilder::cuboid(r, floor_thickness, r)
            .translation(Vec3::new(0.0, half_h, 0.0))
            .restitution(params.restitution)
            .friction(params.friction)
            .build();
        colliders.insert_with_parent(ceiling, drum_body_handle, &mut bodies);

        // --- Chute / funnel geometry (static) ---
        // The chute is a static body below the drum opening.
        // Funnel: two angled ramps guiding balls downward.
        let chute_left = ColliderBuilder::cuboid(0.01, 0.15, r * 0.3)
            .translation(Vec3::new(-0.08, -half_h - 0.12, r * 0.3))
            .rotation(Vec3::new(0.0, 0.0, 0.3)) // angled inward
            .restitution(0.3)
            .friction(0.2)
            .build();
        colliders.insert(chute_left);

        let chute_right = ColliderBuilder::cuboid(0.01, 0.15, r * 0.3)
            .translation(Vec3::new(0.08, -half_h - 0.12, r * 0.3))
            .rotation(Vec3::new(0.0, 0.0, -0.3)) // angled inward
            .restitution(0.3)
            .friction(0.2)
            .build();
        colliders.insert(chute_right);

        // Chute back wall (prevents balls from falling backward).
        let chute_back = ColliderBuilder::cuboid(0.12, 0.15, 0.01)
            .translation(Vec3::new(0.0, -half_h - 0.12, 0.01))
            .restitution(0.3)
            .friction(0.2)
            .build();
        colliders.insert(chute_back);

        // --- Create balls ---
        let mut ball_infos = Vec::with_capacity(params.ball_count as usize);
        for i in 0..params.ball_count {
            let entity_id = i + 1;

            // Random positions inside drum volume (within 70% of radius to avoid spawning in walls).
            let spawn_r = r * 0.7;
            let x = (pos_rng.gen::<f32>() - 0.5) * spawn_r;
            let y = (pos_rng.gen::<f32>() - 0.5) * params.drum_height * 0.6;
            let z = (pos_rng.gen::<f32>() - 0.5) * spawn_r;

            // Small random initial velocities.
            let vx = (vel_rng.gen::<f32>() - 0.5) * 0.5;
            let vy = (vel_rng.gen::<f32>() - 0.5) * 0.5;
            let vz = (vel_rng.gen::<f32>() - 0.5) * 0.5;

            let ball_rb = RigidBodyBuilder::dynamic()
                .translation(Vec3::new(x, y, z))
                .linvel(Vec3::new(vx, vy, vz))
                .ccd_enabled(true)
                .build();
            let ball_handle = bodies.insert(ball_rb);

            let ball_collider = ColliderBuilder::ball(params.ball_radius)
                .restitution(params.restitution)
                .friction(params.friction)
                .density(params.ball_mass / (4.0 / 3.0 * std::f32::consts::PI * params.ball_radius.powi(3)))
                .build();
            colliders.insert_with_parent(ball_collider, ball_handle, &mut bodies);

            ball_infos.push(BallInfo {
                entity_id,
                body_handle: ball_handle,
                active: true,
                exit_frame: None,
                frozen_position: None,
                frozen_orientation: None,
            });
        }

        let mut sim = Self {
            params,
            seed,
            frame: 0,
            pipeline,
            integration_params,
            island_manager,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            ccd_solver,
            gravity,
            ball_infos,
            drum_body_handle,
            event_stream: EventStream::new(),
            checkpoints: Vec::new(),
            exits: Vec::new(),
            completed: false,
            ball_nudge_counts: BTreeMap::new(),
            global_nudge_count: 0,
            rng_recovery,
            recent_mss: Vec::new(),
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

        // Step Rapier physics pipeline (deterministic with enhanced-determinism).
        self.pipeline.step(
            self.gravity,
            &self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );

        // --- Exit detection (post-step per v3 §2.3 / v4 §C) ---
        let mut new_exits = Vec::new();
        for info in &self.ball_infos {
            if !info.active {
                continue;
            }
            let body = &self.bodies[info.body_handle];
            let pos = body.translation();
            let vel = body.linvel();

            let normal = self.params.chute_exit_normal;
            let origin = self.params.chute_exit_origin;
            let rel = [
                pos.x - origin[0],
                pos.y - origin[1],
                pos.z - origin[2],
            ];
            let pos_dot = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
            let vel_dot = vel.x * normal[0] + vel.y * normal[1] + vel.z * normal[2];

            if pos_dot > self.params.exit_position_threshold
                && vel_dot > self.params.exit_velocity_threshold
            {
                let quantized = (pos_dot / self.params.exit_distance_quantum) as i32;
                new_exits.push((info.entity_id, quantized, [pos.x, pos.y, pos.z]));
            }
        }

        // Sort same-frame exits: quantized_distance DESC, entity_id ASC (v4 §D.2).
        new_exits.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // Process exits.
        for (entity_id, _, last_pos) in &new_exits {
            if self.exits.len() >= self.params.required_exits as usize {
                break;
            }
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

            // Freeze ball (v4 §C.3): set kinematic, zero velocity, freeze position.
            if let Some(info) = self.ball_infos.iter_mut().find(|b| b.entity_id == *entity_id) {
                info.active = false;
                info.frozen_position = Some(*last_pos);
                info.exit_frame = Some(self.frame);

                // Read orientation before freezing.
                let body = &self.bodies[info.body_handle];
                let rot = body.rotation();
                info.frozen_orientation = Some([rot.x, rot.y, rot.z, rot.w]);

                // Disable the body by setting it to kinematic with zero velocity.
                let body = &mut self.bodies[info.body_handle];
                body.set_body_type(RigidBodyType::KinematicVelocityBased, true);
                body.set_linvel(Vec3::ZERO, true);
                body.set_angvel(Vec3::ZERO, true);
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
            let cpsf = self.compute_cpsf();
            self.checkpoints.push(CpsfCheckpoint {
                frame: self.frame,
                hash: cpsf.hash(),
            });
            return;
        }

        // --- Deadlock recovery (v3 §5, v4 §F) ---
        if self.frame % self.params.deadlock.check_interval_frames == 0 {
            self.check_deadlock();
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

    /// Deadlock detection and recovery per v3 §5, v4 §F.
    fn check_deadlock(&mut self) {
        // Compute mean-square-speed (MSS) of active balls.
        let mut sum_v2: f32 = 0.0;
        let mut count: u32 = 0;
        for info in &self.ball_infos {
            if !info.active {
                continue;
            }
            let body = &self.bodies[info.body_handle];
            let v = body.linvel();
            sum_v2 += v.x * v.x + v.y * v.y + v.z * v.z;
            count += 1;
        }
        if count == 0 {
            return;
        }
        let mss = sum_v2 / count as f32;
        self.recent_mss.push(mss);

        // Check if stalled: MSS below threshold for sufficient samples.
        let window_checks = (self.params.deadlock.drum_stall_window_frames
            / self.params.deadlock.check_interval_frames) as usize;
        if self.recent_mss.len() < window_checks {
            return;
        }

        // Keep only the recent window.
        while self.recent_mss.len() > window_checks {
            self.recent_mss.remove(0);
        }

        let all_stalled = self.recent_mss.iter().all(|&m| m < self.params.deadlock.drum_stall_mss_threshold);
        if !all_stalled {
            return;
        }

        // Stall detected — apply nudge to a random active ball.
        if self.global_nudge_count >= self.params.deadlock.max_global_nudges {
            return; // Hit global limit.
        }

        // Find an eligible ball (hasn't exceeded per-ball limit).
        let eligible: Vec<u32> = self.ball_infos.iter()
            .filter(|b| b.active)
            .filter(|b| {
                let count = self.ball_nudge_counts.get(&b.entity_id).copied().unwrap_or(0);
                count < self.params.deadlock.max_nudges_per_ball
            })
            .map(|b| b.entity_id)
            .collect();

        if eligible.is_empty() {
            return;
        }

        // Pick a ball deterministically using rng_recovery.
        let idx = self.rng_recovery.gen_range(0..eligible.len());
        let target_id = eligible[idx];

        // Generate deterministic nudge direction.
        let nx: f32 = self.rng_recovery.gen::<f32>() - 0.5;
        let ny: f32 = self.rng_recovery.gen::<f32>().abs() + 0.5; // bias upward
        let nz: f32 = self.rng_recovery.gen::<f32>() - 0.5;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let force_mag = self.params.deadlock.nudge_force_magnitude;
        let impulse = Vec3::new(
            nx / len * force_mag * self.params.fixed_dt,
            ny / len * force_mag * self.params.fixed_dt,
            nz / len * force_mag * self.params.fixed_dt,
        );

        // Apply impulse to the target ball.
        if let Some(info) = self.ball_infos.iter().find(|b| b.entity_id == target_id) {
            let body = &mut self.bodies[info.body_handle];
            body.apply_impulse(impulse, true);
        }

        // Record nudge counts.
        *self.ball_nudge_counts.entry(target_id).or_insert(0) += 1;
        self.global_nudge_count += 1;

        // Clear MSS history after nudge to re-evaluate.
        self.recent_mss.clear();
    }

    /// Apply external events (user input forces).
    /// Parses UserInputPayload from events and applies forces to nearest ball.
    pub fn apply_events(&mut self, events: &[InputEvent]) {
        for event in events {
            self.event_stream.push(event.clone());

            // Apply force for touch events.
            match event.event_type {
                EventType::UserTouchStart | EventType::UserTouchMove => {
                    if event.payload.len() >= 28 {
                        // Parse UserInputPayload from bytes.
                        let payload = parse_input_payload(&event.payload);
                        if let Some(p) = payload {
                            self.apply_user_force(p.drum_x, p.drum_y, p.force_magnitude, p.direction_x, p.direction_y);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Apply a user interaction force at drum coordinates.
    fn apply_user_force(&mut self, drum_x: f32, drum_y: f32, force_mag: f32, dir_x: f32, dir_y: f32) {
        // Find the nearest active ball to the touch point in drum space.
        let touch_pos = Vec3::new(drum_x, drum_y, 0.0);
        let mut best_dist = f32::MAX;
        let mut best_handle: Option<RigidBodyHandle> = None;

        for info in &self.ball_infos {
            if !info.active {
                continue;
            }
            let body = &self.bodies[info.body_handle];
            let pos = body.translation();
            let dx = pos.x - touch_pos.x;
            let dy = pos.y - touch_pos.y;
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_handle = Some(info.body_handle);
            }
        }

        // Apply force if within reasonable range (2x ball radius).
        let max_dist = (self.params.ball_radius * 4.0).powi(2);
        if let Some(handle) = best_handle {
            if best_dist < max_dist {
                let impulse = Vec3::new(
                    dir_x * force_mag * self.params.fixed_dt,
                    dir_y * force_mag * self.params.fixed_dt,
                    0.0,
                );
                let body = &mut self.bodies[handle];
                body.apply_impulse(impulse, true);
            }
        }
    }

    /// Compute CPSF at current frame.
    pub fn compute_cpsf(&self) -> Cpsf {
        let mut bodies_state = Vec::with_capacity(self.ball_infos.len());
        for info in &self.ball_infos {
            let (pos, rot, linvel, angvel, sleeping) = if let Some(frozen_pos) = info.frozen_position {
                let frozen_rot = info.frozen_orientation.unwrap_or([0.0, 0.0, 0.0, 1.0]);
                (frozen_pos, frozen_rot, [0.0f32, 0.0, 0.0], [0.0f32, 0.0, 0.0], false)
            } else {
                let body = &self.bodies[info.body_handle];
                let t = body.translation();
                let r = body.rotation();
                let lv = body.linvel();
                let av = body.angvel();
                (
                    [t.x, t.y, t.z],
                    [r.x, r.y, r.z, r.w],
                    [lv.x, lv.y, lv.z],
                    [av.x, av.y, av.z],
                    body.is_sleeping(),
                )
            };

            bodies_state.push(BodyState {
                entity_id: info.entity_id,
                position: pos,
                orientation: rot,
                linear_velocity: linvel,
                angular_velocity: angvel,
                is_sleeping: sleeping,
                is_active: info.active,
            });
        }
        Cpsf {
            frame_number: self.frame,
            bodies: bodies_state,
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
            DrawStatus::DrawComplete // in progress
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

    pub fn exits(&self) -> &[BallExit] {
        &self.exits
    }

    /// Get ball render data as flat f32 array: [x, y, z, active, x, y, z, active, ...]
    /// 4 floats per ball. active = 1.0 if still in play, 0.0 if exited.
    pub fn get_ball_positions_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.ball_infos.len() * 4);
        for info in &self.ball_infos {
            if let Some(fp) = info.frozen_position {
                out.push(fp[0]);
                out.push(fp[1]);
                out.push(fp[2]);
                out.push(0.0);
            } else {
                let body = &self.bodies[info.body_handle];
                let t = body.translation();
                out.push(t.x);
                out.push(t.y);
                out.push(t.z);
                out.push(if info.active { 1.0 } else { 0.0 });
            }
        }
        out
    }

    /// Get drum rotation angle (radians around Y) for rendering.
    pub fn get_drum_angle(&self) -> f32 {
        let body = &self.bodies[self.drum_body_handle];
        let rot = body.rotation();
        // Extract Y rotation from quaternion (simplified for Y-only rotation).
        2.0 * rot.y.atan2(rot.w)
    }
}

/// Parse UserInputPayload from raw bytes.
fn parse_input_payload(data: &[u8]) -> Option<UserInputPayload> {
    if data.len() < 28 {
        return None;
    }
    Some(UserInputPayload {
        screen_x: i32::from_le_bytes(data[0..4].try_into().ok()?),
        screen_y: i32::from_le_bytes(data[4..8].try_into().ok()?),
        drum_x: f32::from_bits(u32::from_le_bytes(data[8..12].try_into().ok()?)),
        drum_y: f32::from_bits(u32::from_le_bytes(data[12..16].try_into().ok()?)),
        force_magnitude: f32::from_bits(u32::from_le_bytes(data[16..20].try_into().ok()?)),
        direction_x: f32::from_bits(u32::from_le_bytes(data[20..24].try_into().ok()?)),
        direction_y: f32::from_bits(u32::from_le_bytes(data[24..28].try_into().ok()?)),
    })
}
