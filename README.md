# EconNet — numerical-experiment engine

Companion engine for the paper *"Markets as neural networks"*. The central claim
of the paper is that a market and a neural network are two instances of one object
— a dynamical system on a graph driven by an **interaction operator** — and that
"is there a landscape?" reduces, for both, to the **symmetry of that operator**.

This engine makes the taxonomy *checkable*: the failure regime is **computed from
the operator and the trajectory**, never set by a toggle.

> Terminology and the three regimes follow `econnet-math-model.md` (v1.0). This
> engine is a separate, headless numerical core; the 3-D visualization app lives
> in a different repository and is untouched here.

## The three regimes (read out, not chosen)

| Regime | Operator condition | Potential | Failure | "local minimum"? |
|---|---|---|---|---|
| **I** landscape trap | symmetric | exists | settles at a *suboptimal* critical point of Φ | yes — correct |
| **II** goal misalignment | symmetric | exists, but Φ ≠ 𝒲 | argmax Φ ≠ argmax 𝒲 (Goodhart) | no |
| **III** rotational | asymmetric | none | persistent cycle, no convergence | no — no landscape at all |

Only regime I is a "ball in a bowl". Conflating the three is the error the paper
corrects, and this engine refuses to paper over it: a Prisoner's Dilemma is shown
converging to (D,D) = argmax Φ with (C,C) = argmax 𝒲 marked **separately** — the
failure is the welfare *gap*, not any geometry.

## Module map

| Module | Files | What it does |
|---|---|---|
| 1 · operator library | `src/core/games.ts` | F(x), analytic DF (numeric-checked), welfare 𝒲. Standards with a known regime. |
| 2 · Helmholtz split | `src/core/decomposition.ts` | DF = S + A on the tangent space; harmonic fraction `h = ‖A‖/(‖S‖+‖A‖)`. |
| 3 · integrator | `src/core/dynamics.ts`, `src/worker/*` | RK4 projection (primary) & replicator dynamics; runs in a Web Worker. |
| 4 · classifier | `src/core/classifier.ts`, `equilibrium.ts`, `trajectory.ts`, `potential.ts` | spectrum → envelope → Poincaré → regime. |
| 5 · validation | `src/harness/validate.ts` | asserts each standard against theory; exits non-zero on mismatch. |
| 6 · export | `src/core/export.ts`, `src/harness/exportRun.ts` | JSON + CSV per run for paper figures. |
| 7 · adaptive (optional) | `src/core/adaptive.ts` | counterparty reappraisal → h rises → drift out of the potential regime. |

Everything in `src/core` is DOM-free and runs identically under Node (harness,
export) and in the browser Worker.

## Run it

```bash
npm install
npm run validate     # Module 5 — run the standards, assert regimes vs theory
npm run export       # Module 6 — regenerate out/reference/*.{json,csv}
npm run dev          # live view: pick a standard, watch the regime get read out
npm run build        # typecheck + production bundle
```

`npm run validate` is a real gate (23 checks). Current output:

```
game                  h        predicted  observed  match
prisoners_dilemma     0.0000   II         II        ✓
coordination          0.0000   I          I         ✓   (low basin → trap)
coordination          0.0000   I          I         ✓   (high basin → global)
rock_paper_scissors   1.0000   III        III       ✓
matching_pennies      1.0000   III        III       ✓
```

## Classifier procedure (Module 4)

The regime is a readout of a strict procedure, not a label:

- **Step 0** — equilibrium x*: Newton on the tangent-projected field P·F(x*)=0.
  Boundary attractors (regime I/II vertices) are taken from the trajectory's
  limit with the residual measured by the tangent-**cone** projection.
- **Step 1** — spectrum of `DF(x*)|_TΔ`; horizon `T ≥ 15/min|Re λ|`. If
  `min|Re λ| ≈ 0` the run is flagged *marginal* (extend the horizon), not classified.
- **Step 2** — envelope `r(t) = ‖x − sliding_mean‖`, peaks `r₁..r_k`, slope `b`
  of `log r_k ~ a + b·k`. `b < −δ` decay → fixed point; `|b| < δ` plateau → Step 3.
- **Step 3** — Poincaré section through x*: `d_k = ‖x(t_k) − x*‖`. `d_k → 0`
  spiral-in; `d_k → d* > 0` cycle.

**Verdict III** = (unstable-or-neutral focus) ∧ (envelope not decaying) ∧ (d_k → d* > 0).
Otherwise the trajectory settles → **I vs II** by comparing x* to argmax Φ and the
welfare gap `Δ = max 𝒲 − 𝒲(x*)`. Everything logged: eigenvalues(x*), b, d*, T, label.

## Caveats — read before citing any number

These are load-bearing. The engine is honest about what it does and does not compute.

1. **h is a proxy, on the tangent space, not a Hodge decomposition.** The split
   `DF = S + A` is done on the tangent-restricted operator `Qᵀ·DF·Q`, because only
   the tangent space `TΔ` carries the dynamics. The raw n×n antisymmetry includes
   the **non-strategic component** (Candogan's 𝒩) that acts trivially on
   best-response dynamics — on the raw matrix a 2-strategy Prisoner's Dilemma looks
   ~43 % rotational (`hFull ≈ 0.43`); on the tangent space it is correctly `h = 0`,
   an exact potential game. We report both. This is a decomposition at the level of
   the **Jacobian** — exact for an affine/quadratic field, a linearization
   elsewhere. The exact discrete analogue for finite games is the Hodge
   decomposition via the response-graph Laplacian (Candogan, Menache, Ozdaglar,
   Parrilo 2011). **`h` is a faithful, computable proxy for the harmonic component
   — not the full Hodge projection. Do not present it as one.**

2. **Poincaré–Bendixson only holds in 2-D.** Matching Pennies (two populations,
   tangent dim 2) is the genuinely planar case where "closed orbit ⇒ cycle" is a
   theorem. In dimension > 2 the "cycle" verdict is **empirical**, read from `d_k`;
   it could equally be a torus or chaos. The engine reports the readout, not a proof.

3. **Projection vs replicator — metric matters.** Projection (gradient) dynamics is
   the primary flow because in the Euclidean metric it ascends the potential Φ on the
   potential part, so "converges to argmax Φ" is a statement about *this* flow.
   Replicator dynamics is provided as an option, but its Lyapunov relationship with Φ
   holds in the **Shahshahani** metric, not the Euclidean one (math-model §2.2). Don't
   mix the two.

4. **Spectrum/trajectory disagreement is a bug or a short horizon — not a new
   regime.** If the spectrum says all-stable but the trajectory keeps circling, or
   the reached interior point is called unstable yet it settled, the classifier emits
   a warning and declines to invent a fourth regime.

5. **Adaptive operator (Module 7) is non-Hebbian and resource-conserving.** We never
   use `W += η·xxᵀ`. The rule reallocates a *fixed per-agent relationship budget*
   (row sum conserved) — no clipping, no box. Crucially, an update that depends only
   on the counterparty's state, `ΔW_ij = g(x_j)`, is column-structured and therefore
   **non-strategic** (the 𝒩 component); it would not move the tangent `h`. To grow
   *strategic* asymmetry the update is coupled to i's own exposures `W_ij`, so `W_ij`
   and `W_ji` drift apart — that directed asymmetry is what raises `h` and can carry
   the interior equilibrium across a Hopf-type boundary (`Re λ` crosses 0) out of the
   potential regime. It is a switchable mode: freeze W for clean validations, evolve
   it to demonstrate the drift.

## Exported data (Module 6)

`out/reference/<run>.json` — full record: seed, params, spec, operator invariants
(`h`, `hFull`, spectrum), classification (regime, Δ, potential gap, argmax Φ / 𝒲,
horizon, envelope slope, d*), and the state time-series. `<run>.csv` — `time, x0…,
speed, h` for direct plotting. `manifest.json` indexes the set. The timestamp is
fixed so regeneration is byte-stable.

## Reproducibility

Deterministic given (spec, x0, config). All randomness (random games, initial
conditions, adaptive shocks) flows through a seeded generator; the same seed
regenerates the same run bit-for-bit.
