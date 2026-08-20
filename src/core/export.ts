// Module 6 — export for the paper. A run becomes a JSON record (full detail)
// plus a CSV of the state time-series, so figures are built from data with the
// seed, parameters, operator invariants (h, Δ, spectrum) and regime label all
// travelling alongside the numbers.

import { classify, type ClassifyResult } from "./classifier.js";
import type { ExperimentResult } from "./experiment.js";

export const ENGINE_VERSION = "0.1.0";

export interface RunRecord {
  engineVersion: string;
  generatedAt: string;
  meta: {
    game: string;
    spec: unknown;
    blocks: number[];
    dynamics: string;
    dt: number;
    steps: number;
    recordedSamples: number;
    x0: number[];
  };
  operator: {
    h: number;
    hFull: number;
    potentialGame: boolean;
    eigenvalues: { re: number; im: number }[];
  };
  classification: {
    regime: string;
    subLabel: string;
    welfareGap: number | null;
    potentialGap: number;
    argmaxPhi: number[];
    argmaxWelfare: number[] | null;
    equilibrium: number[];
    equilibriumResidual: number;
    horizonT: number;
    envelopeSlopeB: number;
    dStar: number;
    warnings: string[];
  };
  series: {
    times: number[];
    states: number[][];
    speed: number[];
    h: number[];
  };
}

export function buildRunRecord(
  exp: ExperimentResult,
  cls?: ClassifyResult,
  generatedAt: string = new Date().toISOString(),
): RunRecord {
  const c = cls ?? classify(exp.game, exp.trajectory);
  return {
    engineVersion: ENGINE_VERSION,
    generatedAt,
    meta: {
      game: exp.game.name,
      spec: exp.spec,
      blocks: exp.game.blocks,
      dynamics: exp.trajectory.dynamics,
      dt: exp.trajectory.dt,
      steps: exp.trajectory.times.length ? exp.trajectory.times.length - 1 : 0,
      recordedSamples: exp.trajectory.states.length,
      x0: exp.x0,
    },
    operator: {
      h: c.harmonicFraction,
      hFull: c.harmonicFractionFull,
      potentialGame: c.harmonicFraction < 1e-7,
      eigenvalues: c.spectrum.eigenvalues.map((l) => ({ re: l.re, im: l.im })),
    },
    classification: {
      regime: c.regime,
      subLabel: c.subLabel,
      welfareGap: c.welfareGap,
      potentialGap: c.potentialGap,
      argmaxPhi: c.objectives.phiArgmax,
      argmaxWelfare: c.objectives.welfareArgmax,
      equilibrium: c.equilibrium.x,
      equilibriumResidual: c.equilibrium.residual,
      horizonT: c.spectrum.horizon,
      envelopeSlopeB: c.envelope.slope,
      dStar: c.poincare?.dStar ?? 0,
      warnings: c.warnings,
    },
    series: {
      times: exp.trajectory.times,
      states: exp.trajectory.states,
      speed: exp.speedSeries,
      h: exp.hSeries,
    },
  };
}

export function toJSON(record: RunRecord): string {
  return JSON.stringify(record, null, 2);
}

/** CSV of the state time-series: time, x0..x{n-1}, speed, h. */
export function toCSV(record: RunRecord): string {
  const dim = record.meta.blocks.reduce((a, b) => a + b, 0);
  const header = [
    "time",
    ...Array.from({ length: dim }, (_, i) => `x${i}`),
    "speed",
    "h",
  ].join(",");
  const rows = record.series.times.map((t, i) => {
    const xs = record.series.states[i];
    return [
      t,
      ...xs,
      record.series.speed[i] ?? "",
      record.series.h[i] ?? "",
    ].join(",");
  });
  return [header, ...rows].join("\n");
}
