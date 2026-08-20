import type { Mat, Vec } from "./linalg.js";

/**
 * Numeric Jacobian by central differences. Used to (a) cross-check every game's
 * analytic DF and (b) provide a Jacobian for any game that only supplies F.
 */
export function numericJacobian(
  F: (x: Vec) => Vec,
  x: Vec,
  eps = 1e-6,
): Mat {
  const n = x.length;
  const f0 = F(x);
  const m = f0.length;
  const J: Mat = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    const xp = x.slice();
    const xm = x.slice();
    xp[j] += eps;
    xm[j] -= eps;
    const fp = F(xp);
    const fm = F(xm);
    for (let i = 0; i < m; i++) J[i][j] = (fp[i] - fm[i]) / (2 * eps);
  }
  return J;
}

/** Max abs difference between two matrices — for validating analytic DF. */
export function maxMatDiff(a: Mat, b: Mat): number {
  let d = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[i].length; j++)
      d = Math.max(d, Math.abs(a[i][j] - b[i][j]));
  return d;
}
