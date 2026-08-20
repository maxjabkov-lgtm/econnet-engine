// Trajectory readouts for the classifier: settling test, envelope slope
// (Step 2), and Poincaré returns (Step 3). All operate on a recorded trajectory
// and know nothing about the game — the regime must be read out of the motion.

import { type Mat, type Vec, dot, norm2, symmetricEigen, zerosMat } from "./linalg.js";

/** Centered sliding mean of a state series (per coordinate), half-window w.
 *  O(n·dim) via prefix sums (the window cost is independent of w). */
export function slidingMean(states: Vec[], w: number): Vec[] {
  const n = states.length;
  const dim = states[0].length;
  // prefix[i] = Σ states[0..i-1]  (length n+1)
  const prefix: Vec[] = new Array(n + 1);
  prefix[0] = new Array(dim).fill(0);
  for (let i = 0; i < n; i++) {
    const p = prefix[i];
    const s = states[i];
    const next = new Array(dim);
    for (let d = 0; d < dim; d++) next[d] = p[d] + s[d];
    prefix[i + 1] = next;
  }
  const out: Vec[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - w);
    const hi = Math.min(n - 1, i + w);
    const cnt = hi - lo + 1;
    const acc = new Array(dim);
    for (let d = 0; d < dim; d++) acc[d] = (prefix[hi + 1][d] - prefix[lo][d]) / cnt;
    out[i] = acc;
  }
  return out;
}

export interface SettleInfo {
  /** Radius of the trajectory's tail about its own centroid ≈ d* for a cycle. */
  tailRadius: number;
  /** True if the tail collapses to a point (converged equilibrium). */
  settled: boolean;
  /** Speed ‖ẋ‖ estimated from the last step (≈0 at a rest point). */
  finalSpeed: number;
  tailCentroid: Vec;
}

/** Does the trajectory settle to a point, and how big is its limit set? */
export function settleInfo(
  states: Vec[],
  dt: number,
  tailFrac = 0.3,
  settleTol = 1e-3,
): SettleInfo {
  const n = states.length;
  const start = Math.max(0, Math.floor(n * (1 - tailFrac)));
  const dim = states[0].length;
  const centroid = new Array(dim).fill(0);
  let cnt = 0;
  for (let i = start; i < n; i++) {
    for (let d = 0; d < dim; d++) centroid[d] += states[i][d];
    cnt++;
  }
  for (let d = 0; d < dim; d++) centroid[d] /= cnt;
  let tailRadius = 0;
  for (let i = start; i < n; i++) {
    let r = 0;
    for (let d = 0; d < dim; d++) r += (states[i][d] - centroid[d]) ** 2;
    tailRadius = Math.max(tailRadius, Math.sqrt(r));
  }
  const last = states[n - 1];
  const prev = states[n - 2] ?? last;
  const finalSpeed = norm2(last.map((v, i) => (v - prev[i]) / dt));
  return {
    tailRadius,
    settled: tailRadius < settleTol,
    finalSpeed,
    tailCentroid: centroid,
  };
}

export interface EnvelopeInfo {
  /** Slope b of log r_k vs peak index k. b<−δ decay, |b|<δ plateau, b>δ growth. */
  slope: number;
  numPeaks: number;
  peaks: number[];
  /** True if too few oscillation peaks to fit (monotone / overdamped motion). */
  monotone: boolean;
}

/**
 * Envelope r(t) = ‖x − sliding_mean(x)‖, its peaks r_1..r_k, and the slope b of
 * log r_k ~ a + b·k. Fewer than `minPeaks` peaks ⇒ no oscillation to fit.
 */
export function envelopeInfo(
  states: Vec[],
  opts: { minPeaks?: number; window?: number } = {},
): EnvelopeInfo {
  const n = states.length;
  const minPeaks = opts.minPeaks ?? 8;
  const w = opts.window ?? Math.max(3, Math.floor(n / 12));
  const center = slidingMean(states, w);
  const r: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let d = 0; d < states[i].length; d++)
      s += (states[i][d] - center[i][d]) ** 2;
    r[i] = Math.sqrt(s);
  }
  // local maxima with a small prominence relative to the series scale
  let scale = 0;
  for (const v of r) scale = Math.max(scale, v);
  const prom = 1e-6 * (scale + 1e-12);
  const peaks: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (r[i] > r[i - 1] && r[i] >= r[i + 1] && r[i] > prom) peaks.push(r[i]);
  }
  if (peaks.length < minPeaks) {
    return { slope: 0, numPeaks: peaks.length, peaks, monotone: true };
  }
  // least-squares slope of log(peak) vs index
  const k = peaks.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < k; i++) {
    const x = i;
    const y = Math.log(peaks[i] + 1e-300);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const denom = k * sxx - sx * sx;
  const slope = denom === 0 ? 0 : (k * sxy - sx * sy) / denom;
  return { slope, numPeaks: k, peaks, monotone: false };
}

/** Dominant oscillation direction of the tail (PCA), used as the section normal. */
export function dominantDirection(states: Vec[], tailFrac = 0.5): Vec {
  const n = states.length;
  const start = Math.max(0, Math.floor(n * (1 - tailFrac)));
  const dim = states[0].length;
  const mean = new Array(dim).fill(0);
  let cnt = 0;
  for (let i = start; i < n; i++) {
    for (let d = 0; d < dim; d++) mean[d] += states[i][d];
    cnt++;
  }
  for (let d = 0; d < dim; d++) mean[d] /= cnt;
  const cov: Mat = zerosMat(dim, dim);
  for (let i = start; i < n; i++) {
    const c = states[i].map((v, d) => v - mean[d]);
    for (let a = 0; a < dim; a++)
      for (let b = 0; b < dim; b++) cov[a][b] += c[a] * c[b];
  }
  const { vectors } = symmetricEigen(cov);
  return vectors.map((row) => row[0]); // top eigenvector (largest variance)
}

export interface PoincareInfo {
  /** Distances d_k = ‖x(t_k) − x*‖ at successive section crossings. */
  distances: number[];
  /** Estimated limit distance d* (mean of the last few crossings). */
  dStar: number;
  /** |d_{k+1} − d*| / |d_k − d*| near the end: <1 contracting, ≈1 neutral. */
  contraction: number;
  crossings: number;
}

/**
 * Poincaré section through x* with the given normal; record d_k at upward
 * crossings. d_k→0 ⇒ spiral into the point; d_k→d*>0 ⇒ cycle.
 */
export function poincareInfo(
  states: Vec[],
  xStar: Vec,
  normal: Vec,
): PoincareInfo {
  const s = states.map((x) => dot(x.map((v, i) => v - xStar[i]), normal));
  const distances: number[] = [];
  for (let i = 0; i < states.length - 1; i++) {
    if (s[i] < 0 && s[i + 1] >= 0) {
      // linear interpolation to the crossing
      const t = s[i] / (s[i] - s[i + 1]);
      const xc = states[i].map((v, d) => v + t * (states[i + 1][d] - v));
      let d = 0;
      for (let k = 0; k < xc.length; k++) d += (xc[k] - xStar[k]) ** 2;
      distances.push(Math.sqrt(d));
    }
  }
  const m = distances.length;
  if (m === 0) return { distances, dStar: 0, contraction: NaN, crossings: 0 };
  const tail = distances.slice(Math.max(0, m - 5));
  const dStar = tail.reduce((a, b) => a + b, 0) / tail.length;
  let contraction = NaN;
  if (m >= 3) {
    const dk = distances[m - 2];
    const dk1 = distances[m - 1];
    const num = Math.abs(dk1 - dStar);
    const den = Math.abs(dk - dStar);
    contraction = den < 1e-12 ? 1 : num / den;
  }
  return { distances, dStar, contraction, crossings: m };
}
