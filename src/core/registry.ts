// A serializable description of a game, so the main thread can ask the Web
// Worker to build it (functions can't cross postMessage). Also the single place
// that maps an id + params → a Game, used by the worker, harness, and exporter.

import {
  coordination,
  generalizedRps,
  matchingPennies,
  prisonersDilemma,
  randomGame,
  rockPaperScissors,
} from "./games.js";
import { adaptiveGame, type AdaptiveParams } from "./adaptive.js";
import type { Game } from "./types.js";

export type GameSpec =
  | { id: "prisoners_dilemma"; T?: number; R?: number; P?: number; S?: number }
  | { id: "coordination"; a?: number; b?: number }
  | { id: "rock_paper_scissors"; win?: number; loss?: number; selfWeight?: number }
  | { id: "generalized_rps"; selfWeight: number }
  | { id: "matching_pennies" }
  | { id: "random_game"; seed: number; n?: number; sigma?: number }
  | { id: "adaptive"; params?: Partial<AdaptiveParams> };

export function buildGame(spec: GameSpec): Game {
  switch (spec.id) {
    case "prisoners_dilemma":
      return prisonersDilemma(spec.T, spec.R, spec.P, spec.S);
    case "coordination":
      return coordination(spec.a, spec.b);
    case "rock_paper_scissors":
      return rockPaperScissors(spec.win, spec.loss, spec.selfWeight);
    case "generalized_rps":
      return generalizedRps(spec.selfWeight);
    case "matching_pennies":
      return matchingPennies();
    case "random_game":
      return randomGame(spec.seed, spec.n, spec.sigma);
    case "adaptive":
      return adaptiveGame(spec.params);
  }
}
