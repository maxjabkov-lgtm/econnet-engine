// Potential Φ, social welfare W, and the argmax machinery that separates
// regime I from regime II.
//
// For projection dynamics in tangent coordinates y (x = x0 + Q y):
//     ẏ = Ĵ y + ĝ,   Ĵ = Qᵀ DF Q,   ĝ = Qᵀ F(x0).
// When the operator is a potential game (Ĵ symmetric) this is exactly gradient
// ascent of
//     Φ(x) = ½ yᵀ Ŝ y + ĝᵀ y,   Ŝ = sym(Ĵ),
// so the flow climbs Φ and settles at argmax Φ over the simplex. Φ is the EXACT
// potential for an affine payoff field; elsewhere it is the barycenter quadratic
// model (the harmonic residual, reported alongside, says how far off that is).

import {
  type Vec,
  antiPart,
  dot,
  frobNorm,
  matVec,
  symPart,
  transpose,
} from "./linalg.js";
import { makeRng } from "./rng.js";
import {
  barycenter,
  blockRanges,
  buildTangentBasis,
  type PopulationStructure,
  projectToSimplex,
  reduceOperator,
  toTangentCoords,
} from "./simplex.js";
import type { Game } from "./types.js";

export interface PotentialModel {
  phi: (x: Vec) => number;
  /** True if the tangent operator is (near-)symmetric — a real landscape. */
  exists: boolean;
  symNorm: number;
  antiNorm: number;
}

export function buildPotential(game: Game): PotentialModel {
  const Q = buildTangentBasis(game.blocks);
  const Qt = transpose(Q);
  const x0 = barycenter(game.blocks);
  const k = Q[0].length;
  const Jhat = reduceOperator(Q, game.DF(x0)); // Ĵ = Qᵀ DF Q
  const S = symPart(Jhat);
  const A = antiPart(Jhat);
  const symNorm = frobNorm(S);
  const antiNorm = frobNorm(A);
  const ghat = matVec(Qt, game.F(x0));
  const exists = antiNorm < 1e-7 * (symNorm + 1e-12) || antiNorm < 1e-9;

  const phi = (x: Vec): number => {
    const y = toTangentCoords(Q, x0, x);
    // ½ yᵀ S y + ĝᵀ y
    let quad = 0;
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++) quad += y[i] * S[i][j] * y[j];
    return 0.5 * quad + dot(ghat, y);
  };
  return { phi, exists, symNorm, antiNorm };
}

/** Pure-strategy vertices of the product simplex (one per population block). */
export function productVertices(blocks: PopulationStructure): Vec[] {
  const ranges = blockRanges(blocks);
  const dim = ranges[ranges.length - 1][1];
  let acc: Vec[] = [new Array(dim).fill(0)];
  for (const [start, end] of ranges) {
    const next: Vec[] = [];
    for (const base of acc) {
      for (let i = start; i < end; i++) {
        const v = base.slice();
        v[i] = 1;
        next.push(v);
      }
    }
    acc = next;
  }
  return acc;
}

export interface Argmax {
  x: Vec;
  value: number;
}

/**
 * Maximize f over the product simplex. Vertices are enumerated exactly (the
 * standards' optima are vertices), then a seeded random sample plus local
 * pattern-search refinement covers interior optima.
 */
export function argmaxOnSimplex(
  f: (x: Vec) => number,
  blocks: PopulationStructure,
  opts: { seed?: number; samples?: number } = {},
): Argmax {
  const seed = opts.seed ?? 12345;
  const samples = opts.samples ?? 2000;
  const rng = makeRng(seed);
  const candidates: Vec[] = [...productVertices(blocks), barycenter(blocks)];
  for (let s = 0; s < samples; s++) {
    const v: Vec = [];
    for (const b of blocks) {
      const block: number[] = [];
      let tot = 0;
      for (let i = 0; i < b; i++) {
        const e = -Math.log(1 - rng.next()); // Exp(1) → Dirichlet(1,…,1)
        block.push(e);
        tot += e;
      }
      for (const e of block) v.push(e / tot);
    }
    candidates.push(v);
  }
  let best = candidates[0];
  let bestVal = f(best);
  for (const c of candidates) {
    const val = f(c);
    if (val > bestVal) {
      bestVal = val;
      best = c;
    }
  }
  // local pattern search along tangent basis directions
  const Q = buildTangentBasis(blocks);
  const k = Q[0].length;
  for (let radius = 0.1; radius > 1e-4; radius *= 0.5) {
    let improved = true;
    while (improved) {
      improved = false;
      for (let d = 0; d < k; d++) {
        for (const sgn of [1, -1]) {
          const trial = best.slice();
          for (let i = 0; i < trial.length; i++) trial[i] += sgn * radius * Q[i][d];
          const proj = projectToSimplex(blocks, trial);
          const val = f(proj);
          if (val > bestVal + 1e-15) {
            bestVal = val;
            best = proj;
            improved = true;
          }
        }
      }
    }
  }
  return { x: best, value: bestVal };
}

export interface ObjectiveAnalysis {
  phiArgmax: Vec;
  phiMax: number;
  phiAtStar: number;
  welfareArgmax: Vec | null;
  welfareMax: number | null;
  welfareAtStar: number | null;
  /** Δ = max W − W(x*): the regime-II welfare gap. */
  welfareGap: number | null;
  /** argmax Φ ≈ argmax W ? If false with a positive gap ⇒ regime-II signature. */
  phiAlignsWelfare: boolean | null;
}

/** Compare where the dynamics go (argmax Φ) with what society wants (argmax W). */
export function analyzeObjectives(
  game: Game,
  xStar: Vec,
  seed = 777,
): ObjectiveAnalysis {
  const pot = buildPotential(game);
  const phiArg = argmaxOnSimplex(pot.phi, game.blocks, { seed });
  const res: ObjectiveAnalysis = {
    phiArgmax: phiArg.x,
    phiMax: phiArg.value,
    phiAtStar: pot.phi(xStar),
    welfareArgmax: null,
    welfareMax: null,
    welfareAtStar: null,
    welfareGap: null,
    phiAlignsWelfare: null,
  };
  if (game.welfare) {
    const wArg = argmaxOnSimplex(game.welfare, game.blocks, { seed: seed + 1 });
    const wStar = game.welfare(xStar);
    res.welfareArgmax = wArg.x;
    res.welfareMax = wArg.value;
    res.welfareAtStar = wStar;
    res.welfareGap = wArg.value - wStar;
    let dist = 0;
    for (let i = 0; i < xStar.length; i++)
      dist += (phiArg.x[i] - wArg.x[i]) ** 2;
    res.phiAlignsWelfare = Math.sqrt(dist) < 1e-2;
  }
  return res;
}

/** Population mean payoff Σ x_i F_i — a convenience welfare when none is given. */
export function meanPayoff(game: Game, x: Vec): number {
  return dot(x, game.F(x));
}
