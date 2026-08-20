// Message protocol between the main thread and the simulation Web Worker.
// The worker owns the integration loop; the main thread only sends control
// messages and reads back state samples (it never runs the dynamics).

import type { GameSpec } from "../core/registry.js";
import type { DynamicsKind } from "../core/types.js";

export interface InitMsg {
  type: "init";
  spec: GameSpec;
  x0: number[];
  dynamics: DynamicsKind;
  dt: number;
  totalSteps: number;
  /** Post a sample to the main thread every `emitEvery` steps. */
  emitEvery: number;
  /** If present, evolve the operator (adaptive mode) during the run. */
  adaptive?: { updateEvery: number; shock: number; shockSeed: number };
}

export type MainToWorker =
  | InitMsg
  | { type: "start" }
  | { type: "pause" }
  | { type: "reset" };

export interface MetaMsg {
  type: "meta";
  name: string;
  blocks: number[];
  dim: number;
  h: number;
  hFull: number;
  eigenReal: number[];
  eigenImag: number[];
}

export interface SampleMsg {
  type: "sample";
  step: number;
  time: number;
  x: number[];
  /** ‖ẋ‖ at this state (≈0 at a rest point). */
  speed: number;
  /** Harmonic fraction of the current operator (varies in adaptive mode). */
  h: number;
}

export interface DoneMsg {
  type: "done";
  step: number;
}

export type WorkerToMain = MetaMsg | SampleMsg | DoneMsg;
