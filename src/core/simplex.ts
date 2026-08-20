// Population / simplex geometry.
//
// The state lives on a product of probability simplices — one per population.
//   single-population RPS      -> blocks = [3]
//   two-population matching P.  -> blocks = [2, 2]
//
// The DYNAMICS and every operator property that matters live on the TANGENT
// space TΔ = { z : each population block sums to 0 }, not on all of Rⁿ. This is
// the crux of the whole engine: the harmonic fraction, the potential, and the
// spectrum must all be computed on TΔ. Computing them on the raw nⓧn Jacobian
// double-counts the non-strategic component (Candogan's 𝒩), which acts trivially
// on the dynamics — e.g. it would make the 2-strategy Prisoner's Dilemma look
// rotational when it is in fact an exact potential game.

import {
  type Mat,
  type Vec,
  matMat,
  transpose,
  zerosMat,
} from "./linalg.js";

export type PopulationStructure = number[];

export function totalDim(blocks: PopulationStructure): number {
  let s = 0;
  for (const b of blocks) s += b;
  return s;
}

/** Reduced (tangent) dimension: Σ (n_b − 1). */
export function tangentDim(blocks: PopulationStructure): number {
  let s = 0;
  for (const b of blocks) s += b - 1;
  return s;
}

/** Barycenter of the product simplex: uniform within each block. */
export function barycenter(blocks: PopulationStructure): Vec {
  const x: Vec = [];
  for (const b of blocks) for (let i = 0; i < b; i++) x.push(1 / b);
  return x;
}

/** Per-block index ranges [start, end). */
export function blockRanges(blocks: PopulationStructure): [number, number][] {
  const ranges: [number, number][] = [];
  let off = 0;
  for (const b of blocks) {
    ranges.push([off, off + b]);
    off += b;
  }
  return ranges;
}

/**
 * Orthonormal basis Q (dim × k) of the product tangent space, using Helmert
 * contrasts per block. Columns are orthonormal and each is sum-zero within its
 * block, so Qᵀ Q = I_k and the columns span TΔ exactly.
 */
export function buildTangentBasis(blocks: PopulationStructure): Mat {
  const dim = totalDim(blocks);
  const k = tangentDim(blocks);
  const Q = zerosMat(dim, k);
  let rowOff = 0;
  let colOff = 0;
  for (const m of blocks) {
    for (let c = 1; c <= m - 1; c++) {
      // Helmert contrast c: first c entries = 1, entry c = −c, rest 0.
      const norm = Math.sqrt(c * (c + 1));
      for (let i = 0; i < c; i++) Q[rowOff + i][colOff + c - 1] = 1 / norm;
      Q[rowOff + c][colOff + c - 1] = -c / norm;
    }
    rowOff += m;
    colOff += m - 1;
  }
  return Q;
}

/** Reduce a full operator to tangent coordinates: Qᵀ M Q  (k × k). */
export function reduceOperator(Q: Mat, M: Mat): Mat {
  return matMat(transpose(Q), matMat(M, Q));
}

/** Full-space vector of tangent coords: x0 + Q y. */
export function liftFromTangent(Q: Mat, x0: Vec, y: Vec): Vec {
  const dim = Q.length;
  const x = x0.slice();
  for (let i = 0; i < dim; i++) {
    let s = 0;
    for (let j = 0; j < y.length; j++) s += Q[i][j] * y[j];
    x[i] += s;
  }
  return x;
}

/** Tangent coords of a state: Qᵀ (x − x0). */
export function toTangentCoords(Q: Mat, x0: Vec, x: Vec): Vec {
  const k = Q[0].length;
  const y = new Array(k).fill(0);
  for (let j = 0; j < k; j++) {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += Q[i][j] * (x[i] - x0[i]);
    y[j] = s;
  }
  return y;
}

/** Project a velocity onto the tangent hyperplane (subtract per-block mean). */
export function projectTangent(blocks: PopulationStructure, v: Vec): Vec {
  const out = v.slice();
  let off = 0;
  for (const b of blocks) {
    let mean = 0;
    for (let i = 0; i < b; i++) mean += v[off + i];
    mean /= b;
    for (let i = 0; i < b; i++) out[off + i] -= mean;
    off += b;
  }
  return out;
}

/**
 * Project a desired velocity g onto the TANGENT CONE of the product simplex at
 * x: { z : block-sums 0, z_i ≥ 0 wherever x_i = 0 }. This is the exact velocity
 * of Sandholm's projection dynamics. Interior points reduce to projectTangent.
 * Active-set solve per block.
 */
export function tangentConeProject(
  blocks: PopulationStructure,
  x: Vec,
  g: Vec,
  boundaryTol = 1e-9,
): Vec {
  const out = g.slice();
  let off = 0;
  for (const b of blocks) {
    // active constraints: coordinates pinned at the boundary
    const bound = new Array<boolean>(b).fill(false); // forced z_i = 0
    for (let iter = 0; iter < b; iter++) {
      // μ = mean of g over the currently free coordinates
      let mean = 0;
      let free = 0;
      for (let i = 0; i < b; i++) {
        if (!bound[i]) {
          mean += g[off + i];
          free++;
        }
      }
      if (free === 0) break;
      mean /= free;
      // find the worst active-but-free coordinate whose flow points outward
      // of the feasible cone (x_i≈0 and z_i = g_i − μ < 0)
      let worst = -1;
      let worstVal = -boundaryTol;
      for (let i = 0; i < b; i++) {
        if (bound[i]) continue;
        if (x[off + i] > boundaryTol) continue; // interior coord, never pinned
        const zi = g[off + i] - mean;
        if (zi < worstVal) {
          worstVal = zi;
          worst = i;
        }
      }
      if (worst < 0) {
        // done: write z
        for (let i = 0; i < b; i++) out[off + i] = bound[i] ? 0 : g[off + i] - mean;
        break;
      }
      bound[worst] = true;
    }
    off += b;
  }
  return out;
}

/** Euclidean projection of x onto the product of probability simplices. */
export function projectToSimplex(blocks: PopulationStructure, x: Vec): Vec {
  const out = x.slice();
  let off = 0;
  for (const b of blocks) {
    projectBlockToSimplex(out, off, b);
    off += b;
  }
  return out;
}

// Projection of one block onto { p ≥ 0, Σ p = 1 } (Held–Wolfe / Wang & C.-P.).
function projectBlockToSimplex(x: Vec, off: number, n: number): void {
  const u = new Array(n);
  for (let i = 0; i < n; i++) u[i] = x[off + i];
  const sorted = u.slice().sort((a, b) => b - a);
  let cssv = 0;
  let rho = 0;
  let theta = 0;
  for (let i = 0; i < n; i++) {
    cssv += sorted[i];
    const t = (cssv - 1) / (i + 1);
    if (sorted[i] - t > 0) {
      rho = i + 1;
      theta = t;
    }
  }
  for (let i = 0; i < n; i++) x[off + i] = Math.max(u[i] - theta, 0);
}
