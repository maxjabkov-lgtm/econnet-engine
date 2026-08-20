// Thin live view. The Web Worker owns the simulation; this main thread only
// sends control messages, reads back state samples, and renders. The regime is
// still a READOUT — it is classified from the buffered trajectory, never chosen.

import "./style.css";
import { classify } from "../core/classifier.js";
import type { Trajectory } from "../core/dynamics.js";
import { buildGame, type GameSpec } from "../core/registry.js";
import {
  barycenter,
  buildTangentBasis,
  toTangentCoords,
} from "../core/simplex.js";
import { productVertices } from "../core/potential.js";
import type { DynamicsKind } from "../core/types.js";
import type { MainToWorker, WorkerToMain } from "../worker/protocol.js";

interface Preset { label: string; spec: GameSpec; dt: number; steps: number; amp: number; }

const PRESETS: Preset[] = [
  { label: "Prisoner's Dilemma (→ II)", spec: { id: "prisoners_dilemma" }, dt: 0.01, steps: 6000, amp: 0.4 },
  { label: "Coordination — low basin (→ I trap)", spec: { id: "coordination" }, dt: 0.01, steps: 6000, amp: 0.3 },
  { label: "Rock–Paper–Scissors (→ III cycle)", spec: { id: "rock_paper_scissors" }, dt: 0.01, steps: 40000, amp: 0.15 },
  { label: "Generalized RPS, w=+0.3 (→ III unstable focus)", spec: { id: "generalized_rps", selfWeight: 0.3 }, dt: 0.01, steps: 30000, amp: 0.05 },
  { label: "Matching Pennies, 2-pop (→ III cycle)", spec: { id: "matching_pennies" }, dt: 0.01, steps: 40000, amp: 0.15 },
  { label: "Adaptive operator (drift → III)", spec: { id: "adaptive", params: { n: 4, eta: 0.02, damp: 0.2, seed: 3 } }, dt: 0.02, steps: 12000, amp: 0.35 },
];

let worker: Worker | null = null;
let spec: GameSpec = PRESETS[0].spec;
let dynamics: DynamicsKind = "projection";
let blocks: number[] = [2];
let running = false;

const times: number[] = [];
const states: number[][] = [];
let lastH = 0;
let lastSpeed = 0;
let lastMeta: Extract<WorkerToMain, { type: "meta" }> | null = null;

// ---------------------------------------------------------------- DOM
const app = document.getElementById("app")!;
app.innerHTML = `
  <div class="sidebar">
    <h1>EconNet — regime readout</h1>
    <div class="subtitle">the failure regime (I / II / III) is computed from the operator and the trajectory, never toggled</div>
    <div class="field">
      <label>standard</label>
      <select id="preset"></select>
    </div>
    <div class="row">
      <div class="field"><label>dynamics</label>
        <select id="dyn"><option value="projection">projection (gradient)</option><option value="replicator">replicator</option></select>
      </div>
    </div>
    <div class="row">
      <div class="field"><label>dt</label><input id="dt" type="number" step="0.005" value="0.01" /></div>
      <div class="field"><label>steps</label><input id="steps" type="number" step="1000" value="6000" /></div>
    </div>
    <div class="field"><label>initial amplitude</label><input id="amp" type="number" step="0.05" value="0.4" /></div>
    <div class="buttons">
      <button id="run">Run</button>
      <button id="pause" class="secondary">Pause</button>
      <button id="reset" class="secondary">Reset</button>
    </div>
    <div style="margin-top:16px" class="note" id="operator"></div>
  </div>
  <div class="stage">
    <div class="readout" id="readout"><span class="note">pick a standard and press Run.</span></div>
    <canvas id="portrait" width="720" height="520"></canvas>
    <div class="note">Phase portrait in tangent coordinates. Regime I/II → the point settles; regime III → it never stops circling (no landscape; the vertical axis of a "loss surface" is undefined — the honest coordinate is phase/time).</div>
  </div>
`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const presetSel = $<HTMLSelectElement>("preset");
PRESETS.forEach((p, i) => {
  const o = document.createElement("option");
  o.value = String(i);
  o.textContent = p.label;
  presetSel.appendChild(o);
});

presetSel.onchange = () => applyPreset(PRESETS[Number(presetSel.value)]);
$<HTMLSelectElement>("dyn").onchange = (e) => { dynamics = (e.target as HTMLSelectElement).value as DynamicsKind; };
$<HTMLButtonElement>("run").onclick = start;
$<HTMLButtonElement>("pause").onclick = () => { post({ type: "pause" }); running = false; };
$<HTMLButtonElement>("reset").onclick = () => { post({ type: "reset" }); running = false; resetBuffer(); };

function applyPreset(p: Preset): void {
  spec = p.spec;
  $<HTMLInputElement>("dt").value = String(p.dt);
  $<HTMLInputElement>("steps").value = String(p.steps);
  $<HTMLInputElement>("amp").value = String(p.amp);
}
applyPreset(PRESETS[0]);

function resetBuffer(): void { times.length = 0; states.length = 0; }

function post(m: MainToWorker): void { worker?.postMessage(m); }

function start(): void {
  worker?.terminate();
  resetBuffer();
  worker = new Worker(new URL("../worker/sim.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e: MessageEvent<WorkerToMain>) => onMessage(e.data);
  const dt = Number($<HTMLInputElement>("dt").value);
  const steps = Number($<HTMLInputElement>("steps").value);
  const amp = Number($<HTMLInputElement>("amp").value);
  const b = specBlocks(spec);
  blocks = b;
  const x0 = seededInitial(b, amp, 1);
  post({ type: "init", spec, x0, dynamics, dt, totalSteps: steps, emitEvery: 20,
    adaptive: spec.id === "adaptive" ? { updateEvery: 5, shock: 0.015, shockSeed: 99 } : undefined });
  post({ type: "start" });
  running = true;
}

function onMessage(m: WorkerToMain): void {
  if (m.type === "meta") {
    lastMeta = m;
    blocks = m.blocks;
    renderOperator(m);
  } else if (m.type === "sample") {
    times.push(m.time);
    states.push(m.x);
    lastH = m.h;
    lastSpeed = m.speed;
  } else if (m.type === "done") {
    running = false;
  }
}

// ---------------------------------------------------------------- operator panel
function renderOperator(m: Extract<WorkerToMain, { type: "meta" }>): void {
  const eig = m.eigenReal.map((re, i) => `${re.toFixed(2)}${m.eigenImag[i] >= 0 ? "+" : ""}${m.eigenImag[i].toFixed(2)}i`).join(", ");
  $("operator").innerHTML = `
    <div><strong>${m.name}</strong> · dim ${m.dim} · blocks [${m.blocks.join(",")}]</div>
    <div style="margin:8px 0 4px">harmonic fraction h = <code>${m.h.toFixed(3)}</code> <span class="note">(hFull ${m.hFull.toFixed(3)})</span></div>
    <div class="hbar"><span style="width:${(m.h * 100).toFixed(0)}%"></span></div>
    <div style="margin-top:8px">spectrum DF|_TΔ: <code>${eig}</code></div>
    <div class="note" style="margin-top:6px">h→0 potential (landscape exists) · h→1 rotational (no landscape)</div>`;
}

// ---------------------------------------------------------------- live classify
let lastClassify = 0;
function maybeClassify(now: number): void {
  if (now - lastClassify < 500 || states.length < 40) return;
  lastClassify = now;
  const readout = $("readout");
  if (spec.id === "adaptive") {
    readout.innerHTML = `
      <div>operator is <strong>drifting</strong> (adaptive reappraisal).</div>
      <div style="margin-top:8px">live h = <code>${lastH.toFixed(3)}</code> · speed ‖ẋ‖ = <code>${lastSpeed.toExponential(1)}</code></div>
      <div class="note" style="margin-top:6px">As asymmetry grows, h rises and the interior equilibrium loses stability — watch the point stop settling.</div>`;
    return;
  }
  try {
    const game = buildGame(spec);
    const traj: Trajectory = { dynamics, dt: times[1] - times[0] || 0.01, times: times.slice(), states: states.slice() };
    const c = classify(game, traj);
    const cls = `regime-${c.regime}`;
    readout.innerHTML = `
      <div><span class="regime-badge ${cls}">Regime ${c.regime}</span> <span class="note">${c.subLabel}</span></div>
      <div class="metrics">
        <div class="metric"><div class="k">harmonic h</div><div class="v">${c.harmonicFraction.toFixed(3)}</div></div>
        <div class="metric"><div class="k">welfare gap Δ</div><div class="v">${c.welfareGap === null ? "—" : c.welfareGap.toFixed(3)}</div></div>
        <div class="metric"><div class="k">potential gap</div><div class="v">${c.potentialGap.toFixed(3)}</div></div>
        <div class="metric"><div class="k">Poincaré d*</div><div class="v">${(c.poincare?.dStar ?? 0).toFixed(3)}</div></div>
        <div class="metric"><div class="k">argmax Φ</div><div class="v">[${c.objectives.phiArgmax.map((x) => x.toFixed(2)).join(", ")}]</div></div>
        <div class="metric"><div class="k">argmax W</div><div class="v">${c.objectives.welfareArgmax ? "[" + c.objectives.welfareArgmax.map((x) => x.toFixed(2)).join(", ") + "]" : "—"}</div></div>
      </div>
      ${c.warnings.map((w) => `<div class="warn" style="margin-top:8px">⚠ ${w}</div>`).join("")}`;
  } catch {
    /* buffer not ready */
  }
}

// ---------------------------------------------------------------- phase portrait
const canvas = $<HTMLCanvasElement>("portrait");
const ctx = canvas.getContext("2d")!;
function sizeCanvas(): void {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 520;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", sizeCanvas);

function draw(): void {
  const W = canvas.clientWidth;
  const H = 520;
  ctx.clearRect(0, 0, W, H);
  if (states.length < 2) { requestAnimationFrame(loop); return; }

  const Q = buildTangentBasis(blocks);
  const x0 = barycenter(blocks);
  const td = Q[0].length;
  const proj = (x: number[]): [number, number] => {
    const y = toTangentCoords(Q, x0, x);
    return td >= 2 ? [y[0], y[1]] : [y[0], 0];
  };

  // scale from the projected vertices (the simplex outline)
  const verts = productVertices(blocks).map(proj);
  let R = 0.01;
  for (const [a, b] of verts) R = Math.max(R, Math.hypot(a, b));
  for (const s of states) { const [a, b] = proj(s); R = Math.max(R, Math.hypot(a, b)); }
  const pad = 40;
  const scale = (Math.min(W, H) / 2 - pad) / R;
  const cx = W / 2;
  const cy = H / 2;
  const toPx = (p: [number, number]): [number, number] => [cx + p[0] * scale, cy - p[1] * scale];

  // simplex boundary (angle-sorted hull of projected vertices)
  if (verts.length >= 2) {
    const cxv = verts.reduce((a, v) => a + v[0], 0) / verts.length;
    const cyv = verts.reduce((a, v) => a + v[1], 0) / verts.length;
    const ordered = verts.slice().sort((p, q) => Math.atan2(p[1] - cyv, p[0] - cxv) - Math.atan2(q[1] - cyv, q[0] - cxv));
    ctx.beginPath();
    ordered.forEach((v, i) => { const [px, py] = toPx(v); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    if (ordered.length > 2) ctx.closePath();
    ctx.strokeStyle = "#2a3038";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // equilibrium marker (barycenter → origin in tangent coords)
  const [ex, ey] = toPx([0, 0]);
  ctx.fillStyle = "#8b949e";
  ctx.beginPath(); ctx.arc(ex, ey, 3, 0, 7); ctx.fill();

  // trajectory trail (fade older points)
  const n = states.length;
  for (let i = 1; i < n; i++) {
    const [ax, ay] = toPx(proj(states[i - 1]));
    const [bx, by] = toPx(proj(states[i]));
    const age = i / n;
    ctx.strokeStyle = `rgba(88,166,255,${0.15 + 0.75 * age})`;
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  }
  // head
  const [hx, hy] = toPx(proj(states[n - 1]));
  ctx.fillStyle = "#58a6ff";
  ctx.beginPath(); ctx.arc(hx, hy, 4.5, 0, 7); ctx.fill();
}

function loop(t?: number): void {
  draw();
  maybeClassify(t ?? performance.now());
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- helpers (mirror core, DOM-side)
function specBlocks(s: GameSpec): number[] {
  switch (s.id) {
    case "matching_pennies": return [2, 2];
    case "rock_paper_scissors":
    case "generalized_rps": return [3];
    case "random_game": return [s.n ?? 3];
    case "adaptive": return [s.params?.n ?? 4];
    default: return [2];
  }
}
function seededInitial(b: number[], amp: number, seed: number): number[] {
  // deterministic small perturbation of the barycenter, projected to the simplex
  let a = (seed >>> 0) || 1;
  const rnd = () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const x0 = barycenter(b);
  let off = 0;
  for (const m of b) {
    const noise = Array.from({ length: m }, () => rnd() - 0.5);
    const mean = noise.reduce((s, v) => s + v, 0) / m;
    let norm = 0;
    for (let i = 0; i < m; i++) { noise[i] -= mean; norm += noise[i] ** 2; }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < m; i++) x0[off + i] = Math.max(1e-4, x0[off + i] + (amp / norm) * noise[i]);
    off += m;
  }
  // renormalize each block
  off = 0;
  for (const m of b) { let s = 0; for (let i = 0; i < m; i++) s += x0[off + i]; for (let i = 0; i < m; i++) x0[off + i] /= s; off += m; }
  return x0;
}

sizeCanvas();
requestAnimationFrame(loop);
