// Seeded, deterministic RNG. Reproducibility is a hard requirement: every
// random_game, every trajectory, every export must be regenerable bit-for-bit
// from its seed. We use mulberry32 (fast, good enough statistical quality for
// generating payoff matrices and initial conditions; NOT for cryptography).

export interface Rng {
  /** Uniform in [0, 1). */
  next(): number;
  /** Uniform in [lo, hi). */
  uniform(lo: number, hi: number): number;
  /** Standard normal via Box–Muller. */
  normal(): number;
  /** The seed this generator was constructed with (for logging/export). */
  readonly seed: number;
}

export function makeRng(seed: number): Rng {
  // Fold the seed into a 32-bit state; keep it nonzero.
  let a = (seed >>> 0) || 0x9e3779b9;
  const next = (): number => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | null = null;
  return {
    seed,
    next,
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    normal(): number {
      if (spare !== null) {
        const s = spare;
        spare = null;
        return s;
      }
      // Box–Muller.
      let u = 0;
      let v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      const mag = Math.sqrt(-2 * Math.log(u));
      spare = mag * Math.sin(2 * Math.PI * v);
      return mag * Math.cos(2 * Math.PI * v);
    },
  };
}
