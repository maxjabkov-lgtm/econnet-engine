// Module 3 — dynamics integrator (core; the Web Worker in src/worker wraps this).
//
// PRIMARY dynamics: projection (gradient) dynamics  ẋ = Π_{T_x}( F(x) ).
//   In the Euclidean metric this is the flow that ascends the potential Φ on the
//   potential part, so "converges to argmax Φ" is a statement about THIS flow.
//
// OPTIONAL: replicator dynamics  ẋ_i = x_i ( F_i − F̄ ) per population.
//   Note: the replicator flow's Lyapunov relationship with Φ holds in the
//   SHAHSHAHANI metric, not the Euclidean one (math-model §2.2). We keep it as a
//   selectable protocol, but the potential-based statements target projection.
//
// Fixed seed, configurable dt and step count. Deterministic given (game, x0, cfg).

import { type Vec } from "./linalg.js";
import { makeRng } from "./rng.js";
import {
  barycenter,
  blockRanges,
  type PopulationStructure,
  projectTangent,
  projectToSimplex,
  tangentConeProject,
} from "./simplex.js";
import type { DynamicsKind, Game } from "./types.js";

export interface SimConfig {
  dynamics: DynamicsKind;
  dt: number;
  steps: number;
  /** Record a sample every `record` steps (default 1). */
  record?: number;
}

export interface Trajectory {
  dynamics: DynamicsKind;
  dt: number;
  /** Simulated time at each recorded sample. */
  times: number[];
  /** Recorded states (each on the product simplex). */
  states: Vec[];
}

/** Replicator field, applied per population block. */
export function replicatorField(
  blocks: PopulationStructure,
  x: Vec,
  F: Vec,
): Vec {
  const v = new Array(x.length).fill(0);
  for (const [start, end] of blockRanges(blocks)) {
    let mean = 0;
    for (let i = start; i < end; i++) mean += x[i] * F[i];
    for (let i = start; i < end; i++) v[i] = x[i] * (F[i] - mean);
  }
  return v;
}

/** The chosen mean-dynamic vector field at state x. */
export function vectorField(game: Game, x: Vec, kind: DynamicsKind): Vec {
  const F = game.F(x);
  if (kind === "replicator") return replicatorField(game.blocks, x, F);
  // projection dynamics: project the payoff field onto the tangent cone
  return tangentConeProject(game.blocks, x, F);
}

/** One explicit RK4 step. */
export function rk4Step(
  game: Game,
  x: Vec,
  dt: number,
  kind: DynamicsKind,
): Vec {
  const f = (s: Vec) => vectorField(game, s, kind);
  const n = x.length;
  const k1 = f(x);
  const x2 = new Array(n);
  for (let i = 0; i < n; i++) x2[i] = x[i] + (dt / 2) * k1[i];
  const k2 = f(x2);
  const x3 = new Array(n);
  for (let i = 0; i < n; i++) x3[i] = x[i] + (dt / 2) * k2[i];
  const k3 = f(x3);
  const x4 = new Array(n);
  for (let i = 0; i < n; i++) x4[i] = x[i] + dt * k3[i];
  const k4 = f(x4);
  const out = new Array(n);
  for (let i = 0; i < n; i++)
    out[i] = x[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  // Keep the iterate on the product simplex (cleans O(dt⁵) drift and any
  // boundary excursion). Interior points are returned unchanged.
  return projectToSimplex(game.blocks, out);
}

/** Integrate a trajectory. Deterministic given (game, x0, cfg). */
export function integrate(game: Game, x0: Vec, cfg: SimConfig): Trajectory {
  const record = Math.max(1, cfg.record ?? 1);
  let x = projectToSimplex(game.blocks, x0.slice());
  const times: number[] = [0];
  const states: Vec[] = [x.slice()];
  for (let step = 1; step <= cfg.steps; step++) {
    x = rk4Step(game, x, cfg.dt, cfg.dynamics);
    if (step % record === 0) {
      times.push(step * cfg.dt);
      states.push(x.slice());
    }
  }
  return { dynamics: cfg.dynamics, dt: cfg.dt, times, states };
}

/**
 * Reproducible initial condition: barycenter plus a small tangent perturbation,
 * scaled to `amplitude`, projected onto the simplex. A fixed seed → fixed x0.
 */
export function initialCondition(
  blocks: PopulationStructure,
  amplitude: number,
  seed: number,
): Vec {
  const rng = makeRng(seed);
  const x0 = barycenter(blocks);
  const noise = x0.map(() => rng.normal());
  const t = projectTangent(blocks, noise);
  let norm = 0;
  for (const c of t) norm += c * c;
  norm = Math.sqrt(norm) || 1;
  const x = x0.map((v, i) => v + (amplitude / norm) * t[i]);
  return projectToSimplex(blocks, x);
}
