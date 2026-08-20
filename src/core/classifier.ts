// Module 4 — regime classifier.
//
// The regime is a READOUT, never a switch. Procedure (math-model §4 taxonomy):
//   Step 0  find an equilibrium x* (Newton on the tangent-projected field).
//   Step 1  spectrum of DF(x*)|_TΔ; set horizon T ≥ 15/min|Re λ|.
//   Step 2  envelope r(t) peaks → slope b (decay / plateau / growth).
//   Step 3  Poincaré section → d_k (→0 spiral-in, →d*>0 cycle).
// Verdict III = (unstable-or-neutral focus) ∧ (b ≳ 0) ∧ (d_k → d* > 0).
// Otherwise the trajectory settles → regime I or II, told apart by comparing x*
// to argmax Φ and by the welfare gap Δ = max W − W(x*).
//
// Guard rail (README): a disagreement between the spectrum and the trajectory is
// a bug or a too-short horizon, NOT a new regime. We surface it as a warning.

import { helmholtz } from "./decomposition.js";
import type { Trajectory } from "./dynamics.js";
import { findEquilibrium, type Spectrum, tangentSpectrum } from "./equilibrium.js";
import { type Complex, norm2 } from "./linalg.js";
import { analyzeObjectives, type ObjectiveAnalysis } from "./potential.js";
import { barycenter, tangentConeProject } from "./simplex.js";
import {
  dominantDirection,
  envelopeInfo,
  type EnvelopeInfo,
  poincareInfo,
  type PoincareInfo,
  settleInfo,
  type SettleInfo,
} from "./trajectory.js";
import type { Game, Regime } from "./types.js";

export interface ClassifyResult {
  regime: Regime;
  /** Finer reading within the regime (e.g. "neutral center", "landscape trap"). */
  subLabel: string;
  harmonicFraction: number; // h on the tangent space (the headline value)
  /** Raw full-matrix harmonic fraction — transparency only. */
  harmonicFractionFull: number;
  spectrum: Spectrum;
  envelope: EnvelopeInfo;
  poincare: PoincareInfo | null;
  settle: SettleInfo;
  objectives: ObjectiveAnalysis;
  equilibrium: { x: number[]; residual: number; interior: boolean };
  /** Δ = max W − W(x*) (regime-II gap), or null if no welfare defined. */
  welfareGap: number | null;
  /** Φ(argmax) − Φ(x*): >0 ⇒ trapped below the global potential optimum. */
  potentialGap: number;
  /** Spectrum/trajectory disagreement — investigate horizon or a bug. */
  warnings: string[];
  /** Everything the spec asks to log. */
  log: {
    eigenvalues: Complex[];
    envelopeSlopeB: number;
    dStar: number;
    horizonT: number;
    label: Regime;
  };
}

export interface ClassifyOptions {
  /** slope magnitude below which the envelope is a "plateau". */
  slopeTol?: number;
  /** tail radius below which the trajectory counts as settled to a point. */
  settleTol?: number;
  seed?: number;
}

export function classify(
  game: Game,
  traj: Trajectory,
  opts: ClassifyOptions = {},
): ClassifyResult {
  const slopeTol = opts.slopeTol ?? 0.02;
  const settleTol = opts.settleTol ?? 1e-3;
  const seed = opts.seed ?? 777;
  const warnings: string[] = [];

  // h on the tangent space (evaluate the operator at the barycenter).
  const decomp = helmholtz(game.blocks, game.DF(barycenter(game.blocks)));

  // Step 0 — equilibrium.
  //  • settled trajectory (regime I/II): x* is the limit the motion reached,
  //    which may sit on the simplex boundary (a vertex). Newton on the interior
  //    tangent field would run off the simplex there, so we take the tail
  //    centroid and measure the residual with the tangent-CONE projection (which
  //    correctly vanishes at a boundary rest point where the flow points inward).
  //  • oscillating trajectory (regime III): the relevant equilibrium is the
  //    interior focus, found by Newton from the barycenter.
  const settle = settleInfo(traj.states, traj.dt, 0.3, settleTol);
  const eqInterior = findEquilibrium(game); // barycenter start → interior root
  let xStar: number[];
  let residual: number;
  let interior: boolean;
  if (settle.settled) {
    xStar = settle.tailCentroid;
    residual = norm2(tangentConeProject(game.blocks, xStar, game.F(xStar)));
    interior = xStar.every((v) => v > 1e-6);
  } else {
    xStar = eqInterior.x;
    residual = eqInterior.residual;
    interior = eqInterior.interior;
  }

  // Step 1 — spectrum of DF|_TΔ. For affine operators this is constant; evaluate
  // at the interior equilibrium when available, else the barycenter.
  const specPoint =
    eqInterior.converged && eqInterior.interior
      ? eqInterior.x
      : barycenter(game.blocks);
  const spectrum = tangentSpectrum(game, specPoint);

  // Step 2 — envelope slope.
  const envelope = envelopeInfo(traj.states);

  // Step 3 — Poincaré (only meaningful when there is rotation to sample).
  // Section is taken through the interior equilibrium (the orbit's center).
  const normal = dominantDirection(traj.states);
  const poincare = poincareInfo(traj.states, specPoint, normal);

  const objectives = analyzeObjectives(game, xStar, seed);
  const potentialGap = objectives.phiMax - objectives.phiAtStar;

  // ---- verdict -----------------------------------------------------------
  const oscillating =
    !settle.settled && poincare.crossings >= 2 && poincare.dStar > settleTol;
  const envelopeNonDecaying = envelope.monotone
    ? false // monotone motion is not sustained oscillation
    : envelope.slope > -slopeTol;

  let regime: Regime;
  let subLabel: string;

  if (oscillating && envelopeNonDecaying) {
    // Regime III — persistent rotation, no convergence to a point.
    regime = "III";
    if (spectrum.marginal) {
      subLabel = "neutral center (conservative cycle, Re λ ≈ 0)";
    } else if (spectrum.maxRe > slopeTol) {
      subLabel = "unstable focus → cycle (Re λ > 0)";
    } else {
      subLabel = "rotational (landscape undefined)";
    }
    if (spectrum.allStable) {
      warnings.push(
        "Spectrum says all-stable but the trajectory oscillates persistently — extend the horizon or check for a bug; not a new regime.",
      );
    }
  } else if (settle.settled) {
    // Converged to a point → regime I or II. Told apart by welfare alignment.
    const potExists = decomp.potentialGame;
    const gap = objectives.welfareGap ?? 0;
    const aligned = objectives.phiAlignsWelfare ?? true;
    if (potentialGap > 10 * slopeTol) {
      regime = "I";
      subLabel = "landscape trap (settled below the global potential optimum)";
    } else if (gap > 1e-3 && !aligned) {
      regime = "II";
      subLabel = "goal misalignment (argmax Φ ≠ argmax W; Goodhart)";
    } else {
      regime = "I";
      subLabel = "landscape regime (reached the global, welfare-aligned optimum)";
    }
    if (!potExists) {
      warnings.push(
        `Trajectory settled but tangent operator is not symmetric (h=${decomp.h.toFixed(3)}); reported Φ is the best-fit quadratic, and this is a damped-rotational spiral sink rather than a genuine potential landscape.`,
      );
    }
    // A genuine spectrum/trajectory discrepancy is only when the trajectory
    // settles at an INTERIOR point the spectrum calls unstable. An unstable
    // interior equilibrium with a boundary attractor is the normal bistable
    // (regime-I) picture, not a discrepancy.
    if (interior && spectrum.maxRe > 10 * slopeTol) {
      warnings.push(
        "Spectrum calls the reached interior point unstable, yet the trajectory settled there — check horizon or a bug; not a new regime.",
      );
    }
  } else {
    // Neither clearly settled nor clearly cycling.
    regime = "marginal";
    subLabel = spectrum.marginal
      ? "near-neutral spectrum; extend horizon"
      : "indeterminate over this horizon; extend steps";
    warnings.push(
      "Motion neither settled nor a clean cycle over this horizon — extend steps or dt.",
    );
  }

  return {
    regime,
    subLabel,
    harmonicFraction: decomp.h,
    harmonicFractionFull: decomp.hFull,
    spectrum,
    envelope,
    poincare,
    settle,
    objectives,
    equilibrium: { x: xStar, residual, interior },
    welfareGap: objectives.welfareGap,
    potentialGap,
    warnings,
    log: {
      eigenvalues: spectrum.eigenvalues,
      envelopeSlopeB: envelope.slope,
      dStar: poincare.dStar,
      horizonT: spectrum.horizon,
      label: regime,
    },
  };
}
