// Module 2 — Helmholtz decomposition of the interaction operator and the
// harmonic fraction h.
//
// We split the Jacobian into a symmetric (gradient / potential) part and an
// antisymmetric (rotational) part:
//
//     DF = S + A,   S = (DF + DFᵀ)/2,   A = (DF − DFᵀ)/2
//
// and report the harmonic fraction
//
//     h = ‖A‖_F / (‖S‖_F + ‖A‖_F) ∈ [0, 1].
//
// h → 0 : operator (near-)symmetric → a landscape (potential Φ) exists.
// h → 1 : operator (near-)antisymmetric → purely rotational, no landscape.
//
// ── SCOPE / HONESTY (read before citing h as a Hodge coefficient) ────────────
// 1. The split is done on the TANGENT-restricted operator Qᵀ·DF·Q, not on the
//    raw n×n Jacobian. Only the tangent space carries the dynamics; the raw
//    matrix's antisymmetry includes the non-strategic component (Candogan's 𝒩)
//    that acts trivially on best-response dynamics. On the raw matrix the
//    2-strategy Prisoner's Dilemma looks rotational (h≈0.7); on the tangent
//    space it is correctly h=0 (an exact potential game). We report the raw
//    value too, as `hFull`, purely for transparency.
// 2. This is a decomposition at the level of the JACOBIAN. It is exact for an
//    affine/quadratic payoff field (our matrix standards), and a linearization
//    elsewhere. The exact discrete analogue for finite games is the Hodge
//    decomposition of the game via the response-graph Laplacian (Candogan,
//    Menache, Ozdaglar, Parrilo 2011). h is a faithful, computable PROXY for
//    the size of the harmonic component — it is NOT the full Hodge projection,
//    and must not be presented as one.

import { type Mat, antiPart, frobNorm, symPart } from "./linalg.js";
import { buildTangentBasis, type PopulationStructure, reduceOperator } from "./simplex.js";

export interface Decomposition {
  /** Reduced operator on the tangent space, Qᵀ·DF·Q. */
  reduced: Mat;
  /** Symmetric (potential/gradient) part on the tangent space. */
  symmetric: Mat;
  /** Antisymmetric (rotational) part on the tangent space. */
  antisymmetric: Mat;
  normS: number;
  normA: number;
  /** Canonical harmonic fraction, computed on the tangent space. */
  h: number;
  /** Raw full-matrix harmonic fraction — transparency only, do not use for
   *  classification (double-counts the non-strategic component). */
  hFull: number;
  /** h below this threshold ⇒ effectively a potential game (landscape exists). */
  potentialGame: boolean;
}

const POTENTIAL_TOL = 1e-7;

/**
 * Helmholtz split of an interaction operator evaluated at a point.
 * @param blocks population structure (defines the tangent space)
 * @param DF     the Jacobian DF(x) at the point of interest (barycenter or x*)
 */
export function helmholtz(blocks: PopulationStructure, DF: Mat): Decomposition {
  const Q = buildTangentBasis(blocks);
  const reduced = reduceOperator(Q, DF);
  const S = symPart(reduced);
  const A = antiPart(reduced);
  const normS = frobNorm(S);
  const normA = frobNorm(A);
  const denom = normS + normA;
  const h = denom < 1e-15 ? 0 : normA / denom;

  const Sf = symPart(DF);
  const Af = antiPart(DF);
  const nsf = frobNorm(Sf);
  const naf = frobNorm(Af);
  const denomF = nsf + naf;
  const hFull = denomF < 1e-15 ? 0 : naf / denomF;

  return {
    reduced,
    symmetric: S,
    antisymmetric: A,
    normS,
    normA,
    h,
    hFull,
    potentialGame: h < POTENTIAL_TOL,
  };
}
