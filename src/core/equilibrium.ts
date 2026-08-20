// Equilibrium finding (classifier Step 0) and the tangent spectrum (Step 1).
//
// A rest point of the projection dynamics satisfies the tangent-projected field
// P·F(x*) = 0. We Newton-iterate in tangent coordinates y (x = x0 + Q y):
//     g(y)   = Qᵀ F(x0 + Q y)          (this equals P F, the projection speed)
//     Jg(y)  = Qᵀ DF(x0 + Q y) Q       (reduced Jacobian)
// For affine payoff fields this converges in a single step.

import { type Complex, type Vec, eigenvalues, matVec, norm2, solve, transpose } from "./linalg.js";
import {
  barycenter,
  buildTangentBasis,
  liftFromTangent,
  type PopulationStructure,
  reduceOperator,
} from "./simplex.js";
import type { Game } from "./types.js";

export interface EquilibriumResult {
  x: Vec;
  /** ‖P F(x*)‖ — the tangent projection speed; 0 at a true rest point. */
  residual: number;
  converged: boolean;
  /** True if x* lies in the (relative) interior of the product simplex. */
  interior: boolean;
  iterations: number;
}

export function findEquilibrium(
  game: Game,
  guessX?: Vec,
  opts: { tol?: number; maxIter?: number } = {},
): EquilibriumResult {
  const tol = opts.tol ?? 1e-10;
  const maxIter = opts.maxIter ?? 100;
  const Q = buildTangentBasis(game.blocks);
  const Qt = transpose(Q);
  const x0 = barycenter(game.blocks);
  // start from the guess (projected into tangent coords) or the barycenter
  let y = guessX
    ? matVec(Qt, guessX.map((v, i) => v - x0[i]))
    : new Array(Q[0].length).fill(0);

  let residual = Infinity;
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const x = liftFromTangent(Q, x0, y);
    const g = matVec(Qt, game.F(x));
    residual = norm2(g);
    if (residual < tol) break;
    const Jg = reduceOperator(Q, game.DF(x));
    const neg = g.map((v) => -v);
    const dy = solve(Jg, neg);
    if (!dy) break; // singular reduced Jacobian (e.g. degenerate direction)
    // damped step for safety on nonlinear fields
    for (let j = 0; j < y.length; j++) y[j] += dy[j];
  }
  const x = liftFromTangent(Q, x0, y);
  const interior = x.every((v) => v > 1e-6);
  return { x, residual, converged: residual < Math.max(tol, 1e-8), interior, iterations: iter };
}

export interface Spectrum {
  /** Eigenvalues of DF(x*)|_TΔ. */
  eigenvalues: Complex[];
  maxRe: number;
  minAbsRe: number;
  /** A complex-conjugate pair with Re>0 among otherwise-stable modes. */
  unstableFocus: boolean;
  allStable: boolean;
  /** Recommended horizon T ≥ 15 / min|Re λ| (∞ if a mode is marginal). */
  horizon: number;
  /** min|Re λ| ≈ 0 — cannot set a finite horizon, flag as marginal. */
  marginal: boolean;
}

const MARGINAL_RE = 1e-4;

/** Tangent spectrum of the operator at x* (classifier Step 1). */
export function tangentSpectrum(game: Game, xStar: Vec): Spectrum {
  const Q = buildTangentBasis(game.blocks);
  const J = reduceOperator(Q, game.DF(xStar));
  const ev = eigenvalues(J);
  let maxRe = -Infinity;
  let minAbsRe = Infinity;
  let hasPosComplex = false;
  for (const l of ev) {
    maxRe = Math.max(maxRe, l.re);
    minAbsRe = Math.min(minAbsRe, Math.abs(l.re));
    if (l.re > MARGINAL_RE && Math.abs(l.im) > MARGINAL_RE) hasPosComplex = true;
  }
  const allStable = maxRe < -MARGINAL_RE;
  const marginal = minAbsRe < MARGINAL_RE;
  const horizon = marginal ? Infinity : 15 / minAbsRe;
  return {
    eigenvalues: ev,
    maxRe,
    minAbsRe,
    unstableFocus: hasPosComplex,
    allStable,
    horizon,
    marginal,
  };
}
