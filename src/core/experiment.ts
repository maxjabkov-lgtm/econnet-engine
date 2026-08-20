// One place to run a spec end-to-end, synchronously — used by the validation
// harness and the exporter. The Web Worker mirrors this loop for the live view;
// this version is deterministic and returns aligned h(t) and speed(t) series so
// paper figures can be built from data, not screenshots.

import { makeAdaptiveSystem } from "./adaptive.js";
import { helmholtz } from "./decomposition.js";
import { type Trajectory, initialCondition, rk4Step, vectorField } from "./dynamics.js";
import { norm2 } from "./linalg.js";
import { makeRng } from "./rng.js";
import { buildGame, type GameSpec } from "./registry.js";
import { barycenter, projectToSimplex } from "./simplex.js";
import type { DynamicsKind, Game } from "./types.js";

export interface ExperimentConfig {
  dynamics: DynamicsKind;
  dt: number;
  steps: number;
  /** Record every `record` steps (default 1). */
  record?: number;
  /** Explicit initial condition; else a seeded perturbation of the barycenter. */
  x0?: number[];
  x0Amplitude?: number;
  seed?: number;
  /** Adaptive-operator drift (only meaningful for spec.id === "adaptive"). */
  adaptive?: { updateEvery: number; shock: number; shockSeed: number };
}

export interface ExperimentResult {
  spec: GameSpec;
  /** Final operator (post-adaptation for adaptive runs). */
  game: Game;
  x0: number[];
  trajectory: Trajectory;
  /** Harmonic fraction at each recorded sample (constant unless adaptive). */
  hSeries: number[];
  /** ‖ẋ‖ at each recorded sample. */
  speedSeries: number[];
}

export function resolveInitialCondition(
  blocks: number[],
  cfg: ExperimentConfig,
): number[] {
  if (cfg.x0) return projectToSimplex(blocks, cfg.x0.slice());
  const amp = cfg.x0Amplitude ?? 0.15;
  return initialCondition(blocks, amp, cfg.seed ?? 1);
}

export function runExperiment(
  spec: GameSpec,
  cfg: ExperimentConfig,
): ExperimentResult {
  const record = Math.max(1, cfg.record ?? 1);
  const adaptiveSys =
    spec.id === "adaptive" ? makeAdaptiveSystem(spec.params) : null;
  const game: Game = adaptiveSys ? adaptiveSys.game : buildGame(spec);

  const x0 = resolveInitialCondition(game.blocks, cfg);
  let x = x0.slice();
  const shockRng =
    cfg.adaptive && cfg.adaptive.shock > 0
      ? makeRng(cfg.adaptive.shockSeed)
      : null;

  const times: number[] = [0];
  const states: number[][] = [x.slice()];
  const speedSeries: number[] = [norm2(vectorField(game, x, cfg.dynamics))];
  const hSeries: number[] = [
    helmholtz(game.blocks, game.DF(barycenter(game.blocks))).h,
  ];

  for (let step = 1; step <= cfg.steps; step++) {
    x = rk4Step(game, x, cfg.dt, cfg.dynamics);
    if (adaptiveSys && cfg.adaptive) {
      if (shockRng) {
        const s = cfg.adaptive.shock * Math.sqrt(cfg.dt);
        x = projectToSimplex(
          game.blocks,
          x.map((v) => v + s * shockRng.normal()),
        );
      }
      if (step % cfg.adaptive.updateEvery === 0) adaptiveSys.evolve(x);
    }
    if (step % record === 0) {
      times.push(step * cfg.dt);
      states.push(x.slice());
      speedSeries.push(norm2(vectorField(game, x, cfg.dynamics)));
      hSeries.push(
        adaptiveSys
          ? helmholtz(game.blocks, game.DF(barycenter(game.blocks))).h
          : hSeries[0],
      );
    }
  }

  const trajectory: Trajectory = {
    dynamics: cfg.dynamics,
    dt: cfg.dt,
    times,
    states,
  };
  return { spec, game, x0, trajectory, hSeries, speedSeries };
}
