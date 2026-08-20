import type { Mat, Vec } from "./linalg.js";
import type { PopulationStructure } from "./simplex.js";

/** The three math-model failure regimes, plus "marginal" (spectrum too close to
 *  neutral to classify at the given horizon — a readout, not a fourth regime). */
export type Regime = "I" | "II" | "III" | "marginal";

export const REGIME_LABEL: Record<Regime, string> = {
  I: "I — landscape trap (potential exists, suboptimal critical point)",
  II: "II — goal misalignment (potential exists, argmax Φ ≠ argmax W)",
  III: "III — rotational (no global potential, persistent cycle)",
  marginal: "marginal — spectrum near-neutral, extend horizon",
};

export type DynamicsKind = "projection" | "replicator";

/**
 * An interaction operator = a population game.
 * F(x)   payoff field:      F_i(x) is the payoff to strategy i at state x.
 * DF(x)  its Jacobian:      the interaction operator whose symmetry decides
 *                           whether a landscape (potential) exists.
 * Games with a theoretically known regime carry `expectedRegime` so the
 * validation harness can assert engine output against theory.
 */
export interface Game {
  name: string;
  blocks: PopulationStructure;
  F(x: Vec): Vec;
  /** Analytic Jacobian. Cross-checked against a numeric one in tests. */
  DF(x: Vec): Mat;
  /** Social welfare W(x) where defined (Σ x_i F_i for a single population). */
  welfare?: (x: Vec) => number;
  /** Theoretical prediction, for the validation harness. */
  expectedRegime?: Regime;
  params?: Record<string, number | string>;
  notes?: string;
}
