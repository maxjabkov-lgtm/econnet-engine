// Module 5 — validation harness. Run the theory-anchored standards and assert
// the engine's readout against what the math model predicts. If the engine does
// not reproduce the predicted regime, that is an ENGINE bug, not a theory bug —
// the harness exits non-zero so `npm run validate` is a real gate.
//
// Also cross-checks every game's analytic Jacobian against a numeric one
// (Module 1), and prints the summary table  game / h / predicted / observed.

import { classify, type ClassifyResult } from "../core/classifier.js";
import { helmholtz } from "../core/decomposition.js";
import { runExperiment, type ExperimentConfig } from "../core/experiment.js";
import {
  coordination,
  matchingPennies,
  prisonersDilemma,
  randomGame,
  rockPaperScissors,
} from "../core/games.js";
import { maxMatDiff, numericJacobian } from "../core/numeric.js";
import type { GameSpec } from "../core/registry.js";
import { barycenter } from "../core/simplex.js";
import type { Game, Regime } from "../core/types.js";

let failures = 0;
let checks = 0;
function assert(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.log(`   ✗ FAIL: ${msg}`);
  } else {
    console.log(`   ✓ ${msg}`);
  }
}

function fmtVec(v: number[] | null): string {
  return v ? `[${v.map((x) => x.toFixed(2)).join(", ")}]` : "—";
}

// ---- analytic vs numeric Jacobian (Module 1) ------------------------------
function checkJacobians(): void {
  console.log("\n## Analytic Jacobian vs numeric (central differences)");
  const games: Game[] = [
    prisonersDilemma(),
    coordination(),
    rockPaperScissors(),
    matchingPennies(),
    randomGame(1, 4),
  ];
  for (const g of games) {
    const x = barycenter(g.blocks).map((v, i) => v + 0.01 * Math.sin(i + 1));
    const d = maxMatDiff(g.DF(x), numericJacobian(g.F, x));
    assert(d < 1e-5, `${g.name}: |DF_analytic − DF_numeric| = ${d.toExponential(1)} < 1e-5`);
  }
}

// ---- regime standards -----------------------------------------------------
interface Case {
  label: string;
  spec: GameSpec;
  cfg: ExperimentConfig;
  expected: Regime;
  extra?: (c: ClassifyResult) => void;
}

const D = [0, 1]; // (D,D) vertex for PD (strategy index 1 = defect)
const C = [1, 0]; // (C,C) vertex

const cases: Case[] = [
  {
    label: "Prisoner's Dilemma → II (Goodhart: argmax Φ ≠ argmax W)",
    spec: { id: "prisoners_dilemma" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 6000, x0: [0.5, 0.5] },
    expected: "II",
    extra: (c) => {
      assert(c.harmonicFraction < 1e-6, `h ≈ 0 (potential game), got ${c.harmonicFraction.toFixed(4)}`);
      assert((c.welfareGap ?? 0) > 0.5, `welfare gap Δ > 0, got ${c.welfareGap?.toFixed(3)}`);
      assert(dist(c.objectives.phiArgmax, D) < 1e-2, `argmax Φ = (D,D)=${fmtVec(D)}, got ${fmtVec(c.objectives.phiArgmax)}`);
      assert(dist(c.objectives.welfareArgmax!, C) < 1e-2, `argmax W = (C,C)=${fmtVec(C)}, got ${fmtVec(c.objectives.welfareArgmax)}`);
      assert(c.objectives.phiAlignsWelfare === false, "argmax Φ ≠ argmax W (the regime-II signature)");
    },
  },
  {
    label: "Coordination, low basin → I (landscape trap)",
    spec: { id: "coordination" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 6000, x0: [0.2, 0.8] },
    expected: "I",
    extra: (c) => {
      assert(c.harmonicFraction < 1e-6, `h ≈ 0 (potential game), got ${c.harmonicFraction.toFixed(4)}`);
      assert(c.potentialGap > 0.1, `settled below global optimum: Φ-gap = ${c.potentialGap.toFixed(3)} > 0`);
    },
  },
  {
    label: "Coordination, high basin → global optimum (bistability check)",
    spec: { id: "coordination" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 6000, x0: [0.6, 0.4] },
    expected: "I",
    extra: (c) => {
      assert(c.potentialGap < 1e-2, `reached the global optimum: Φ-gap ≈ 0, got ${c.potentialGap.toFixed(3)}`);
    },
  },
  {
    label: "Rock–Paper–Scissors → III (harmonic cycle)",
    spec: { id: "rock_paper_scissors" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 30000, record: 2, x0: undefined, x0Amplitude: 0.15, seed: 42 },
    expected: "III",
    extra: (c) => {
      assert(c.harmonicFraction > 0.9, `h high (harmonic), got ${c.harmonicFraction.toFixed(4)}`);
      assert(!c.settle.settled, "trajectory does NOT settle to a point (ω-limit is a cycle)");
      assert((c.poincare?.dStar ?? 0) > 1e-2, `Poincaré d* > 0 (bounded away from equilibrium), got ${c.poincare?.dStar.toFixed(4)}`);
    },
  },
  {
    label: "Matching Pennies (2-pop) → III (harmonic cycle)",
    spec: { id: "matching_pennies" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 30000, record: 2, x0Amplitude: 0.15, seed: 7 },
    expected: "III",
    extra: (c) => {
      assert(c.harmonicFraction > 0.9, `h high (harmonic), got ${c.harmonicFraction.toFixed(4)}`);
      assert(!c.settle.settled, "trajectory does NOT settle to a point");
    },
  },
];

function dist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

interface Row {
  game: string;
  h: number;
  predicted: Regime;
  observed: Regime;
  ok: boolean;
}

function runCases(): Row[] {
  const rows: Row[] = [];
  for (const cs of cases) {
    console.log(`\n## ${cs.label}`);
    const exp = runExperiment(cs.spec, cs.cfg);
    const c = classify(exp.game, exp.trajectory);
    console.log(
      `   regime: predicted ${cs.expected}, observed ${c.regime}  — ${c.subLabel}`,
    );
    console.log(
      `   h=${c.harmonicFraction.toFixed(4)}  eig=[${c.spectrum.eigenvalues.map((l) => `${l.re.toFixed(2)}${l.im >= 0 ? "+" : ""}${l.im.toFixed(2)}i`).join(", ")}]  b=${c.envelope.slope.toFixed(4)}  d*=${(c.poincare?.dStar ?? 0).toFixed(4)}  T=${c.spectrum.horizon === Infinity ? "∞" : c.spectrum.horizon.toFixed(1)}`,
    );
    assert(c.regime === cs.expected, `regime = ${cs.expected}`);
    cs.extra?.(c);
    for (const w of c.warnings) console.log(`   · note: ${w}`);
    rows.push({
      game: exp.game.name,
      h: c.harmonicFraction,
      predicted: cs.expected,
      observed: c.regime,
      ok: c.regime === cs.expected,
    });
  }
  return rows;
}

function printTable(rows: Row[]): void {
  console.log("\n" + "=".repeat(72));
  console.log("SUMMARY: game / h / predicted regime / observed");
  console.log("=".repeat(72));
  const pad = (s: string, n: number) => s.padEnd(n);
  console.log(
    `${pad("game", 22)}${pad("h", 9)}${pad("predicted", 11)}${pad("observed", 10)}match`,
  );
  console.log("-".repeat(72));
  for (const r of rows) {
    console.log(
      `${pad(r.game, 22)}${pad(r.h.toFixed(4), 9)}${pad(r.predicted, 11)}${pad(r.observed, 10)}${r.ok ? "✓" : "✗ MISMATCH"}`,
    );
  }
}

// ---- harmonic-fraction probe over random games ----------------------------
function randomHProbe(): void {
  console.log("\n## Harmonic fraction over random games (probe, not asserted)");
  console.log(`${"seed/n".padEnd(12)}${"h (tangent)".padEnd(14)}hFull`);
  for (const [seed, n] of [[1, 3], [2, 3], [3, 4], [4, 4], [5, 5], [6, 5]] as const) {
    const g = randomGame(seed, n);
    const d = helmholtz(g.blocks, g.DF(barycenter(g.blocks)));
    console.log(
      `${`s=${seed} n=${n}`.padEnd(12)}${d.h.toFixed(4).padEnd(14)}${d.hFull.toFixed(4)}`,
    );
  }
}

console.log("EconNet engine — validation harness");
console.log("Regimes are READ OUT of the dynamics; the table asserts them against theory.");
checkJacobians();
const rows = runCases();
randomHProbe();
printTable(rows);

console.log("\n" + "=".repeat(72));
console.log(`${checks} checks, ${failures} failure(s).`);
if (failures > 0) {
  console.log("VALIDATION FAILED — engine does not reproduce the predicted regime(s).");
  process.exit(1);
} else {
  console.log("VALIDATION PASSED — engine reproduces the theory on every standard.");
}
