// Module 7 (optional) — adaptive interaction operator.
//
// Economically-motivated reappraisal of counterparties, NOT a Hebbian rule:
// each agent i withdraws a fraction of its exposure W_ij to counterparties j
// that have fallen below a default threshold, and redistributes exactly that
// withdrawn weight across its healthy counterparties. Two design points matter:
//
//  • NON-HEBBIAN. We never do W += η x xᵀ. The update is a reallocation of a
//    fixed per-agent relationship budget (row sum conserved) — a meaningful
//    resource constraint, not a clip. Nothing is clamped to a box.
//
//  • ASYMMETRIC ON THE TANGENT SPACE. A naive rule ΔW_ij = g(x_j) (depending on
//    the counterparty only) is column-structured; on the tangent space it is a
//    non-strategic component (Candogan's 𝒩) and would NOT move h. To grow the
//    *strategic* asymmetry we make the update depend on i's own exposures W_ij,
//    so W_ij and W_ji drift apart: i can flee j while j still leans on i. That
//    directed asymmetry is exactly what raises the harmonic fraction h and can
//    carry the system out of the potential regime into rotation (regime III).
//
// The system is switchable: build it once and either freeze W (static, for the
// clean validations) or call evolve() alongside integration (adaptive, to show
// the drift). h(t) rising is the observable.

import {
  type Mat,
  type Vec,
  matVec,
  symmetricEigen,
  zerosMat,
} from "./linalg.js";
import { makeRng } from "./rng.js";
import { buildTangentBasis, reduceOperator } from "./simplex.js";
import type { Game } from "./types.js";

export interface AdaptiveParams {
  n: number;
  /** Adaptation rate (fraction of exposure withdrawn per evolve step). */
  eta: number;
  /** Default threshold on node state; below it a counterparty is "defaulted". */
  theta: number;
  /** Symmetric baseline damping −damp·I on the tangent (starts as potential). */
  damp: number;
  seed: number;
}

export const DEFAULT_ADAPTIVE: AdaptiveParams = {
  n: 4,
  eta: 0.03,
  theta: 1 / 4,
  damp: 0.15,
  seed: 11,
};

export interface AdaptiveSystem {
  params: AdaptiveParams;
  /** Current operator (mutable); F(x)=Wx, DF=W. */
  W: Mat;
  /** Snapshot game reading the current W. */
  game: Game;
  /** One reappraisal step given the current node state. Mutates W in place. */
  evolve: (x: Vec) => void;
}

export function makeAdaptiveSystem(
  overrides: Partial<AdaptiveParams> = {},
): AdaptiveSystem {
  const params = { ...DEFAULT_ADAPTIVE, ...overrides };
  const { n, theta } = params;
  const rng = makeRng(params.seed);
  // Baseline: a SYMMETRIC but non-uniform relationship network (rows genuinely
  // differ — this is what lets per-row reappraisal diverge and build strategic
  // asymmetry). Positive off-diagonal exposures give the reappraisal something
  // to reallocate. The diagonal is then shifted so the tangent operator is
  // stable (top tangent eigenvalue = −damp): the system STARTS as a convergent
  // potential game, h ≈ 0, and only drifts once reappraisal runs.
  const W: Mat = zerosMat(n, n);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const w = 0.5 + rng.next(); // exposure in [0.5, 1.5], symmetric
      W[i][j] = w;
      W[j][i] = w;
    }
  const Q = buildTangentBasis([n]);
  const tangentEig = symmetricEigen(reduceOperator(Q, W));
  const shift = tangentEig.values[0] + params.damp; // make top tangent λ = −damp
  for (let i = 0; i < n; i++) W[i][i] -= shift;

  const game: Game = {
    name: "adaptive",
    blocks: [n],
    F: (x: Vec) => matVec(W, x),
    DF: () => W,
    welfare: (x: Vec) => {
      // mean payoff Σ x_i F_i
      const f = matVec(W, x);
      let s = 0;
      for (let i = 0; i < n; i++) s += x[i] * f[i];
      return s;
    },
    params: { n, eta: params.eta, theta, damp: params.damp, seed: params.seed },
    notes:
      "Adaptive operator: counterparty reappraisal reallocates a fixed relationship budget; asymmetry (hence h) grows over time.",
  };

  const evolve = (x: Vec): void => {
    const eta = params.eta;
    for (let i = 0; i < n; i++) {
      // withdraw a fraction of exposure to defaulted counterparties
      let withdrawn = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        if (x[j] < theta) {
          const dw = eta * W[i][j];
          W[i][j] -= dw;
          withdrawn += dw;
        }
      }
      if (withdrawn === 0) continue;
      // redistribute across healthy counterparties ∝ (x_k − θ)_+
      let wsum = 0;
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        if (x[k] > theta) wsum += x[k] - theta;
      }
      if (wsum > 1e-12) {
        for (let k = 0; k < n; k++) {
          if (k === i) continue;
          if (x[k] > theta) W[i][k] += withdrawn * ((x[k] - theta) / wsum);
        }
      } else {
        // no healthy counterparty: return the budget to self (conserve resource)
        W[i][i] += withdrawn;
      }
    }
  };

  return { params, W, game, evolve };
}

/** A static t=0 snapshot as a plain Game (used by the registry). */
export function adaptiveGame(overrides: Partial<AdaptiveParams> = {}): Game {
  return makeAdaptiveSystem(overrides).game;
}
