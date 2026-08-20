// Small, dependency-free linear algebra for the engine. Everything here operates
// on plain number[] / number[][] so it runs unchanged in Node and the browser.
// Matrices are row-major: M[i][j] is row i, column j.

export type Vec = number[];
export type Mat = number[][];

export interface Complex {
  re: number;
  im: number;
}

// ---------------------------------------------------------------------------
// vectors
// ---------------------------------------------------------------------------

export const zeros = (n: number): Vec => new Array(n).fill(0);

export function add(a: Vec, b: Vec): Vec {
  const r = new Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] + b[i];
  return r;
}

export function sub(a: Vec, b: Vec): Vec {
  const r = new Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i];
  return r;
}

export function scale(a: Vec, s: number): Vec {
  const r = new Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] * s;
  return r;
}

/** y <- y + s*x, returns a new vector. */
export function axpy(s: number, x: Vec, y: Vec): Vec {
  const r = new Array(x.length);
  for (let i = 0; i < x.length; i++) r[i] = y[i] + s * x[i];
  return r;
}

export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const norm2 = (a: Vec): number => Math.sqrt(dot(a, a));

export function normInf(a: Vec): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
  return m;
}

export const cloneVec = (a: Vec): Vec => a.slice();

export function sum(a: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

// ---------------------------------------------------------------------------
// matrices
// ---------------------------------------------------------------------------

export function zerosMat(rows: number, cols: number): Mat {
  const m: Mat = new Array(rows);
  for (let i = 0; i < rows; i++) m[i] = new Array(cols).fill(0);
  return m;
}

export function identity(n: number): Mat {
  const m = zerosMat(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}

export const cloneMat = (a: Mat): Mat => a.map((row) => row.slice());

export function matVec(m: Mat, x: Vec): Vec {
  const rows = m.length;
  const r = new Array(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0;
    const row = m[i];
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    r[i] = s;
  }
  return r;
}

export function matMat(a: Mat, b: Mat): Mat {
  const n = a.length;
  const k = b.length;
  const p = b[0].length;
  const r = zerosMat(n, p);
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    const ri = r[i];
    for (let t = 0; t < k; t++) {
      const ait = ai[t];
      const bt = b[t];
      for (let j = 0; j < p; j++) ri[j] += ait * bt[j];
    }
  }
  return r;
}

export function transpose(a: Mat): Mat {
  const rows = a.length;
  const cols = a[0].length;
  const r = zerosMat(cols, rows);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) r[j][i] = a[i][j];
  return r;
}

export function matAdd(a: Mat, b: Mat): Mat {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

export function matScale(a: Mat, s: number): Mat {
  return a.map((row) => row.map((v) => v * s));
}

/** Symmetric part (A + Aᵀ)/2. */
export function symPart(a: Mat): Mat {
  const n = a.length;
  const r = zerosMat(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) r[i][j] = 0.5 * (a[i][j] + a[j][i]);
  return r;
}

/** Antisymmetric part (A − Aᵀ)/2. */
export function antiPart(a: Mat): Mat {
  const n = a.length;
  const r = zerosMat(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) r[i][j] = 0.5 * (a[i][j] - a[j][i]);
  return r;
}

export function frobNorm(a: Mat): number {
  let s = 0;
  for (const row of a) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

export function outer(a: Vec, b: Vec): Mat {
  const r = zerosMat(a.length, b.length);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++) r[i][j] = a[i] * b[j];
  return r;
}

// ---------------------------------------------------------------------------
// linear solve (Gaussian elimination with partial pivoting)
// ---------------------------------------------------------------------------

/** Solve A x = b. Returns null if A is (numerically) singular. */
export function solve(A: Mat, b: Vec): Vec | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // partial pivot
    let piv = col;
    let best = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-14) return null;
    if (piv !== col) {
      const tmp = M[piv];
      M[piv] = M[col];
      M[col] = tmp;
    }
    const pivRow = M[col];
    const pv = pivRow[col];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / pv;
      if (f === 0) continue;
      for (let j = col; j <= n; j++) M[r][j] -= f * pivRow[j];
    }
  }
  const x = new Array(n);
  for (let i = 0; i < n; i++) x[i] = M[i][n] / M[i][i];
  return x;
}

// ---------------------------------------------------------------------------
// symmetric eigenproblem (cyclic Jacobi) — used for trajectory PCA and for
// inspecting the symmetric (potential) part of an operator.
// ---------------------------------------------------------------------------

export interface SymEigen {
  values: number[]; // descending
  vectors: Mat; // columns are eigenvectors, aligned with values
}

export function symmetricEigen(input: Mat, maxSweeps = 100): SymEigen {
  const n = input.length;
  const a = cloneMat(input);
  const v = identity(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // off-diagonal magnitude
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-28) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const app = a[p][p];
        const aqq = a[q][q];
        const apq = a[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort(
    (i, j) => a[j][j] - a[i][i],
  );
  const values = idx.map((i) => a[i][i]);
  const vectors = zerosMat(n, n);
  for (let col = 0; col < n; col++)
    for (let row = 0; row < n; row++) vectors[row][col] = v[row][idx[col]];
  return { values, vectors };
}

// ---------------------------------------------------------------------------
// general (nonsymmetric) eigenvalues, complex-aware.
// Small matrices only (the reduced tangent operators here are ≤ ~8×8).
// Exact analytic paths for n≤2; general path = Faddeev–LeVerrier char poly
// + Durand–Kerner root finding.
// ---------------------------------------------------------------------------

const cAdd = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });
const cSub = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });
const cMul = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});
export const cAbs = (a: Complex): number => Math.hypot(a.re, a.im);
function cDiv(a: Complex, b: Complex): Complex {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
}

/** Characteristic polynomial coefficients of A, monic, highest degree first:
 *  returns [1, c_{n-1}, …, c_1, c_0] for det(λI − A). Faddeev–LeVerrier. */
export function charPoly(A: Mat): number[] {
  const n = A.length;
  const coeffs = new Array<number>(n + 1);
  coeffs[0] = 1;
  let M = identity(n); // M_1
  for (let k = 1; k <= n; k++) {
    const AM = matMat(A, M);
    let tr = 0;
    for (let i = 0; i < n; i++) tr += AM[i][i];
    const ck = -tr / k;
    coeffs[k] = ck;
    // M_{k+1} = A·M_k + c_k I
    M = AM;
    for (let i = 0; i < n; i++) M[i][i] += ck;
  }
  return coeffs;
}

/** Roots of a monic real polynomial (highest degree first) via Durand–Kerner. */
export function polyRoots(coeffs: number[]): Complex[] {
  const n = coeffs.length - 1;
  if (n === 0) return [];
  // normalize to monic just in case
  const a = coeffs.map((c) => c / coeffs[0]);
  const evalP = (z: Complex): Complex => {
    let r: Complex = { re: a[0], im: 0 };
    for (let i = 1; i <= n; i++) r = cAdd(cMul(r, z), { re: a[i], im: 0 });
    return r;
  };
  // seed with a spiral of non-real, non-unit points
  const roots: Complex[] = [];
  const seed: Complex = { re: 0.4, im: 0.9 };
  let cur: Complex = { re: 1, im: 0 };
  for (let i = 0; i < n; i++) {
    cur = cMul(cur, seed);
    roots.push({ re: cur.re, im: cur.im });
  }
  for (let iter = 0; iter < 500; iter++) {
    let maxDelta = 0;
    for (let i = 0; i < n; i++) {
      const pi = evalP(roots[i]);
      let denom: Complex = { re: 1, im: 0 };
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        denom = cMul(denom, cSub(roots[i], roots[j]));
      }
      const delta = cDiv(pi, denom);
      roots[i] = cSub(roots[i], delta);
      maxDelta = Math.max(maxDelta, cAbs(delta));
    }
    if (maxDelta < 1e-13) break;
  }
  // clean tiny imaginary parts (real polynomial → conjugate pairs)
  for (const r of roots) if (Math.abs(r.im) < 1e-9) r.im = 0;
  roots.sort((x, y) => y.re - x.re || y.im - x.im);
  return roots;
}

/** Eigenvalues of a small real matrix, complex-aware. */
export function eigenvalues(A: Mat): Complex[] {
  const n = A.length;
  if (n === 0) return [];
  if (n === 1) return [{ re: A[0][0], im: 0 }];
  if (n === 2) {
    const a = A[0][0];
    const b = A[0][1];
    const c = A[1][0];
    const d = A[1][1];
    const tr = a + d;
    const det = a * d - b * c;
    const disc = tr * tr - 4 * det;
    if (disc >= 0) {
      const s = Math.sqrt(disc);
      return [
        { re: (tr + s) / 2, im: 0 },
        { re: (tr - s) / 2, im: 0 },
      ].sort((x, y) => y.re - x.re);
    }
    const s = Math.sqrt(-disc) / 2;
    return [
      { re: tr / 2, im: s },
      { re: tr / 2, im: -s },
    ];
  }
  return polyRoots(charPoly(A));
}
