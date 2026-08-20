// Module 1 — the interaction-operator library.
//
// Each game returns the payoff field F(x), its analytic Jacobian DF(x) (the
// interaction operator), and a social-welfare function W where one is defined.
// The potential Φ is NOT stored per game — it is reconstructed generically from
// the operator in welfare.ts, so "does a landscape exist" is answered by the
// engine, not asserted by hand.
//
// Games with a theoretically known regime carry `expectedRegime`; the validation
// harness asserts the engine's readout against it. A mismatch is an engine bug,
// not a theory bug.

import { type Mat, type Vec, dot, matVec } from "./linalg.js";
import { makeRng } from "./rng.js";
import type { PopulationStructure } from "./simplex.js";
import type { Game, Regime } from "./types.js";

interface AffineOpts {
  welfare?: (x: Vec) => number;
  expectedRegime?: Regime;
  params?: Record<string, number | string>;
  notes?: string;
}

/**
 * All standards here have a payoff field linear in the state, F(x) = A x, so the
 * Jacobian DF = A is constant and the Helmholtz split is exact (no linearization
 * error). Default welfare is the population's mean payoff Σ x_i F_i = xᵀ A x.
 */
function affineGame(
  name: string,
  blocks: PopulationStructure,
  A: Mat,
  opts: AffineOpts = {},
): Game {
  const welfare = opts.welfare ?? ((x: Vec) => dot(x, matVec(A, x)));
  return {
    name,
    blocks,
    F: (x: Vec) => matVec(A, x),
    DF: () => A,
    welfare,
    expectedRegime: opts.expectedRegime,
    params: opts.params,
    notes: opts.notes,
  };
}

// ---------------------------------------------------------------------------
// Prisoner's Dilemma — potential game, expect regime II (goal misalignment).
// Strategies: 0 = C (cooperate), 1 = D (defect). Row = own action, col = opponent.
//   A = [[R, S], [T, P]],  with T > R > P > S.
// D strictly dominates C, so dynamics converge to (D,D) = argmax Φ, while social
// welfare peaks at (C,C). That gap — not any geometry — is the failure.
// ---------------------------------------------------------------------------
export function prisonersDilemma(
  T = 5,
  R = 3,
  P = 1,
  S = 0,
): Game {
  const A: Mat = [
    [R, S],
    [T, P],
  ];
  return affineGame("prisoners_dilemma", [2], A, {
    expectedRegime: "II",
    params: { T, R, P, S },
    notes:
      "Exact potential game; argmax Φ = (D,D) ≠ (C,C) = argmax W. Failure is the welfare gap Δ, not a local minimum.",
  });
}

// ---------------------------------------------------------------------------
// Coordination — potential game, expect regime I (landscape trap).
// A = [[a, 0], [0, b]] with a > b > 0: two pure equilibria, one strictly better.
// From basins below the interior threshold the dynamics settle into the WORSE
// optimum — the only regime where "stuck in a local optimum" is literally correct.
// ---------------------------------------------------------------------------
export function coordination(a = 2, b = 1): Game {
  const A: Mat = [
    [a, 0],
    [0, b],
  ];
  return affineGame("coordination", [2], A, {
    expectedRegime: "I",
    params: { a, b },
    notes:
      "Bistable potential game. Global optimum is 'coordinate on 1'; basin below p*=b/(a+b) traps into the suboptimal vertex.",
  });
}

// ---------------------------------------------------------------------------
// Rock–Paper–Scissors — harmonic game, expect regime III (rotational).
// selfWeight w tunes the interior equilibrium under PROJECTION dynamics:
//   w = 0  antisymmetric operator, h = 1, neutral center → closed orbits;
//   w > 0  unstable focus (Re λ = w > 0) → spirals out, persistent rotation;
//   w < 0  stable focus → spirals into the barycenter.
// The reduced tangent Jacobian has eigenvalues w ± iω, so w is exactly Re λ.
// ---------------------------------------------------------------------------
export function rockPaperScissors(win = 1, loss = 1, selfWeight = 0): Game {
  const w = selfWeight;
  const A: Mat = [
    [w, -loss, win],
    [win, w, -loss],
    [-loss, win, w],
  ];
  return affineGame("rock_paper_scissors", [3], A, {
    expectedRegime: "III",
    params: { win, loss, selfWeight },
    notes:
      "Zero-sum (w=0) is purely harmonic: h=1, welfare ≡ 0, ω-limit is a closed orbit, not a point.",
  });
}

/** Generalized RPS exposing the self-weight knob explicitly (Hopf-like family). */
export function generalizedRps(selfWeight: number): Game {
  const g = rockPaperScissors(1, 1, selfWeight);
  return {
    ...g,
    name: "generalized_rps",
    expectedRegime: selfWeight > 0 ? "III" : selfWeight < 0 ? "I" : "III",
    notes:
      "Projection dynamics: Re λ = selfWeight. w>0 unstable focus (spiral out, regime III); w<0 stable focus (settles to interior point).",
  };
}

// ---------------------------------------------------------------------------
// Matching Pennies — two-population harmonic game, expect regime III.
// blocks = [2, 2]. Population 1 (Matcher) wants to match; population 2
// (Mismatcher) wants to differ. State x = (p1H, p1T, p2H, p2T).
// The operator is antisymmetric → h = 1 → closed orbits about (½,½,½,½).
// This is the genuinely 2-D case where Poincaré–Bendixson actually applies.
// ---------------------------------------------------------------------------
export function matchingPennies(): Game {
  // F1H = p2H − p2T, F1T = p2T − p2H, F2H = p1T − p1H, F2T = p1H − p1T
  const A: Mat = [
    [0, 0, 1, -1],
    [0, 0, -1, 1],
    [-1, 1, 0, 0],
    [1, -1, 0, 0],
  ];
  return affineGame("matching_pennies", [2, 2], A, {
    expectedRegime: "III",
    // zero-sum between the two players: total welfare ≡ 0
    welfare: () => 0,
    notes:
      "Two populations, antisymmetric operator, h=1. Genuinely 2-D → Poincaré–Bendixson valid; closed orbits.",
  });
}

// ---------------------------------------------------------------------------
// Random game — arbitrary operator, for measuring the harmonic-fraction
// distribution. No expected regime: it is a probe, not a standard.
// ---------------------------------------------------------------------------
export function randomGame(seed: number, n = 3, sigma = 1): Game {
  const rng = makeRng(seed);
  const A: Mat = Array.from({ length: n }, () =>
    Array.from({ length: n }, () => sigma * rng.normal()),
  );
  return affineGame(`random_game(seed=${seed},n=${n})`, [n], A, {
    params: { seed, n, sigma },
    notes: "Arbitrary payoff operator; used to sample the harmonic fraction h.",
  });
}

/** The catalog of theory-anchored standards used by the validation harness. */
export function standardGames(): Game[] {
  return [
    prisonersDilemma(),
    coordination(),
    rockPaperScissors(),
    matchingPennies(),
  ];
}
