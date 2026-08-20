// Module 6 CLI — regenerate the reference data set under out/reference/.
// Each run writes a full JSON record and a CSV of the state time-series, so the
// paper's figures are built from these files, not from screenshots. Seeds and
// parameters travel inside every record; the timestamp is fixed so regenerating
// the set is byte-stable in git.

import { mkdirSync, writeFileSync } from "node:fs";
import { classify } from "../core/classifier.js";
import { buildRunRecord, toCSV, toJSON } from "../core/export.js";
import { runExperiment, type ExperimentConfig } from "../core/experiment.js";
import type { GameSpec } from "../core/registry.js";

const OUT = "out/reference";
const GENERATED_AT = "2026-08-20T00:00:00.000Z"; // fixed → stable reference files

interface RunDef {
  file: string;
  spec: GameSpec;
  cfg: ExperimentConfig;
}

const runs: RunDef[] = [
  {
    file: "prisoners_dilemma",
    spec: { id: "prisoners_dilemma" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 6000, record: 5, x0: [0.5, 0.5] },
  },
  {
    file: "coordination_low_basin",
    spec: { id: "coordination" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 6000, record: 5, x0: [0.2, 0.8] },
  },
  {
    file: "coordination_high_basin",
    spec: { id: "coordination" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 6000, record: 5, x0: [0.6, 0.4] },
  },
  {
    file: "rock_paper_scissors",
    spec: { id: "rock_paper_scissors" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 30000, record: 10, x0Amplitude: 0.15, seed: 42 },
  },
  {
    file: "generalized_rps_unstable_focus",
    spec: { id: "generalized_rps", selfWeight: 0.3 },
    cfg: { dynamics: "projection", dt: 0.01, steps: 20000, record: 10, x0Amplitude: 0.05, seed: 42 },
  },
  {
    file: "matching_pennies",
    spec: { id: "matching_pennies" },
    cfg: { dynamics: "projection", dt: 0.01, steps: 30000, record: 10, x0Amplitude: 0.15, seed: 7 },
  },
  {
    file: "adaptive_drift",
    spec: { id: "adaptive", params: { n: 4, eta: 0.02, damp: 0.2, seed: 3 } },
    cfg: {
      dynamics: "projection",
      dt: 0.02,
      steps: 8000,
      record: 20,
      x0Amplitude: 0.35,
      seed: 5,
      adaptive: { updateEvery: 5, shock: 0.015, shockSeed: 99 },
    },
  },
];

mkdirSync(OUT, { recursive: true });

interface ManifestRow {
  file: string;
  game: string;
  regime: string;
  subLabel: string;
  h: number;
  welfareGap: number | null;
  potentialGap: number;
}

const manifest: ManifestRow[] = [];

for (const r of runs) {
  const exp = runExperiment(r.spec, r.cfg);
  const cls = classify(exp.game, exp.trajectory);
  const record = buildRunRecord(exp, cls, GENERATED_AT);
  writeFileSync(`${OUT}/${r.file}.json`, toJSON(record));
  writeFileSync(`${OUT}/${r.file}.csv`, toCSV(record));
  manifest.push({
    file: r.file,
    game: exp.game.name,
    regime: cls.regime,
    subLabel: cls.subLabel,
    h: cls.harmonicFraction,
    welfareGap: cls.welfareGap,
    potentialGap: cls.potentialGap,
  });
  console.log(
    `wrote ${r.file}: ${exp.game.name} → regime ${cls.regime} (h=${cls.harmonicFraction.toFixed(3)}, Δ=${cls.welfareGap?.toFixed(3) ?? "—"})`,
  );
}

writeFileSync(
  `${OUT}/manifest.json`,
  JSON.stringify({ engine: "econnet", generatedAt: GENERATED_AT, runs: manifest }, null, 2),
);
console.log(`\nwrote ${OUT}/manifest.json (${manifest.length} runs)`);
