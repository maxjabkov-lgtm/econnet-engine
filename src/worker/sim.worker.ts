// Module 3 — the simulation runs here, in a Web Worker. The main thread sends
// control messages and reads back samples; it never steps the dynamics itself.

import { helmholtz } from "../core/decomposition.js";
import { rk4Step, vectorField } from "../core/dynamics.js";
import { eigenvalues } from "../core/linalg.js";
import { makeRng, type Rng } from "../core/rng.js";
import { buildGame, type GameSpec } from "../core/registry.js";
import {
  barycenter,
  buildTangentBasis,
  projectToSimplex,
  reduceOperator,
} from "../core/simplex.js";
import { makeAdaptiveSystem, type AdaptiveSystem } from "../core/adaptive.js";
import type { DynamicsKind, Game } from "../core/types.js";
import type { InitMsg, MainToWorker, WorkerToMain } from "./protocol.js";

const ctx = self as unknown as {
  postMessage: (m: WorkerToMain) => void;
  onmessage: ((e: MessageEvent<MainToWorker>) => void) | null;
};

interface RunState {
  spec: GameSpec;
  game: Game;
  adaptive: AdaptiveSystem | null;
  x: number[];
  x0: number[];
  step: number;
  dynamics: DynamicsKind;
  dt: number;
  totalSteps: number;
  emitEvery: number;
  adaptiveCfg: InitMsg["adaptive"];
  shockRng: Rng | null;
  running: boolean;
}

let run: RunState | null = null;

function currentH(g: Game): number {
  return helmholtz(g.blocks, g.DF(barycenter(g.blocks))).h;
}

function emitMeta(spec: GameSpec, game: Game): void {
  const DF = game.DF(barycenter(game.blocks));
  const decomp = helmholtz(game.blocks, DF);
  const Q = buildTangentBasis(game.blocks);
  const ev = eigenvalues(reduceOperator(Q, DF));
  ctx.postMessage({
    type: "meta",
    name: game.name,
    blocks: game.blocks,
    dim: game.blocks.reduce((a, b) => a + b, 0),
    h: decomp.h,
    hFull: decomp.hFull,
    eigenReal: ev.map((l) => l.re),
    eigenImag: ev.map((l) => l.im),
  });
  void spec;
}

function emitSample(r: RunState): void {
  const speed = Math.hypot(...vectorField(r.game, r.x, r.dynamics));
  ctx.postMessage({
    type: "sample",
    step: r.step,
    time: r.step * r.dt,
    x: r.x.slice(),
    speed,
    h: currentH(r.game),
  });
}

function tick(): void {
  if (!run || !run.running) return;
  const r = run;
  const chunk = Math.min(r.emitEvery, r.totalSteps - r.step);
  for (let i = 0; i < chunk; i++) {
    r.x = rk4Step(r.game, r.x, r.dt, r.dynamics);
    r.step++;
    if (r.adaptive && r.adaptiveCfg) {
      if (r.shockRng && r.adaptiveCfg.shock > 0) {
        const s = r.adaptiveCfg.shock * Math.sqrt(r.dt);
        r.x = projectToSimplex(
          r.game.blocks,
          r.x.map((v) => v + s * r.shockRng!.normal()),
        );
      }
      if (r.step % r.adaptiveCfg.updateEvery === 0) r.adaptive.evolve(r.x);
    }
  }
  emitSample(r);
  if (r.step >= r.totalSteps) {
    r.running = false;
    ctx.postMessage({ type: "done", step: r.step });
    return;
  }
  // yield so control messages (pause/reset) are processed between chunks
  setTimeout(tick, 0);
}

ctx.onmessage = (e: MessageEvent<MainToWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "init": {
      let game: Game;
      let adaptive: AdaptiveSystem | null = null;
      if (msg.spec.id === "adaptive") {
        adaptive = makeAdaptiveSystem(msg.spec.params);
        game = adaptive.game;
      } else {
        game = buildGame(msg.spec);
      }
      const x0 = projectToSimplex(game.blocks, msg.x0.slice());
      run = {
        spec: msg.spec,
        game,
        adaptive,
        x: x0.slice(),
        x0,
        step: 0,
        dynamics: msg.dynamics,
        dt: msg.dt,
        totalSteps: msg.totalSteps,
        emitEvery: msg.emitEvery,
        adaptiveCfg: msg.adaptive,
        shockRng: msg.adaptive ? makeRng(msg.adaptive.shockSeed) : null,
        running: false,
      };
      emitMeta(msg.spec, game);
      emitSample(run);
      break;
    }
    case "start":
      if (run && !run.running && run.step < run.totalSteps) {
        run.running = true;
        tick();
      }
      break;
    case "pause":
      if (run) run.running = false;
      break;
    case "reset":
      if (run) {
        run.running = false;
        run.x = run.x0.slice();
        run.step = 0;
        // adaptive runs mutate the operator; rebuild it so reset is a true reset
        if (run.spec.id === "adaptive") {
          run.adaptive = makeAdaptiveSystem(run.spec.params);
          run.game = run.adaptive.game;
        }
        emitSample(run);
      }
      break;
  }
};
