# Experiment 4 — Process-Aware Rewards and Reward Hacking
### A Validated Replication of §6 (Figures 7–8) of Zhang, Neubig & Yue (2025)

**Paper:** *On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language
Models* (arXiv 2512.07783, ICML 2026 Spotlight).
**Run:** PACE ICE, 1× H100 (job 6589709), 1 h 20 m wall, 640,000 sampled solutions.
**Status:** complete; every number below is reproduced from the released checkpoints and
validated by four independent audits (data integrity, clean-room recomputation,
qualitative re-scoring, paired statistics).

---

## 1. Question

Compared with an **outcome-only** reward (pay for the right final answer), do **mixed**
or **strict process** rewards (also pay for *verified reasoning steps*) (a) raise strict
pass@k and (b) lower the **reward-hacking rate** — the share of correct answers reached
through invalid reasoning?

## 2. Systems compared (five checkpoints)

One 100M-parameter base model (pre-trained 99.9% `crazy_zootopia` at op 2–20 + 0.1%
`teachers_in_school` at op 2–20 — the paper's appendix A.6.1 "shared primitives"
setting), and four GRPO runs from it on **identical** data (90% zoo / 10% teachers,
op 2–20, 204,800 samples ≈ 200 steps), differing **only in the reward**:

| Label | Reward (verified from the released training configs) |
|---|---|
| `base` | none (pre-training only) |
| `outcome` | `R = 1[answer correct]` — pure answer match (`compute_score`) |
| `0.5/0.5` | `R = 1[answer correct]·(0.5 + 0.5·R_pv)` (weights assumed; yaml absent) |
| `0.2/0.8` | `R = 1[answer correct]·(0.2 + 0.8·R_pv)` (verified) |
| `strict` | `R = 1[answer correct]·1[every step verified]` (verified) |

`R_pv` = *process accuracy*, the fraction of gold reasoning-graph nodes the solution
reproduces with correct parents and values. **Note:** the released code applies an
*answer gate* (reward zeroed when the final answer is wrong), so the mixes are
conditional on a correct answer — this diverges from the paper's literal Eq. 3, which
was never trained in the released family.

## 3. Method

| Choice | Value |
|---|---|
| Eval data | `crazy_zootopia` + `teachers_in_school` val, op ∈ {12,14,16,18,20}, 200 problems each |
| Samples | k = 64/problem, temperature 0.7, top-p 1.0, ≤1024 new tokens |
| Budget | 5 ckpts × 2 contexts × 1,000 problems × 64 = **640,000 samples** |
| Scoring | authors' graph pipeline → strict (process-verified) pass@k + answer-only pass@k |
| Metric: pass@k | problem counts at k if any of its first k samples is correct (first-success-rank estimator) |
| Metric: hacking rate | `H = Σ max(0, a_i − s_i) / Σ a_i` = P(invalid reasoning \| correct answer); `a_i` = answer-correct count, `s_i` = strictly-correct count |
| Uncertainty | bootstrap over problems (B=1,000); **paired** bootstrap for all contrasts (B=5,000, same resampled problems applied to both systems); Bonferroni α = 0.05/9 |
| Bands | edge = op 12–14, hard = op 16–20 |

## 4. Results

### 4.1 Headline table — both contexts, overall (percent; CIs are 95%)

| Context | Reward | strict p@1 | strict p@64 | answer p@1 | answer p@64 | **hacking rate** |
|---|---|---|---|---|---|---|
| **teachers** | base | 20.3 | 62.7 | 42.7 | 95.2 | **52.3** [49.1, 55.7] |
| | outcome | 36.1 | 68.8 | 66.9 | 94.4 | **45.8** [42.7, 48.8] |
| | 0.5/0.5 | 40.0 | 70.3 | 69.5 | 93.9 | **42.3** [39.3, 45.7] |
| | 0.2/0.8 | 39.2 | 68.0 | 68.6 | 93.4 | **42.2** [38.7, 45.4] |
| | **strict** | 38.7 | 68.8 | 64.4 | 94.4 | **40.5** [37.3, 43.5] |
| **zoo** | base | 60.3 | 87.0 | 89.5 | 99.6 | 33.6 [30.9, 36.6] |
| | outcome | 64.3 | 85.1 | 92.8 | 98.7 | 32.6 [29.8, 35.3] |
| | 0.5/0.5 | 62.3 | 86.4 | 92.7 | 98.7 | 32.3 [29.6, 34.9] |
| | 0.2/0.8 | 62.6 | 86.2 | 93.0 | 98.6 | 32.2 [29.5, 34.9] |
| | strict | 63.0 | 85.6 | 92.5 | 98.9 | 32.0 [29.2, 34.8] |

### 4.2 The reward-hacking ladder (teachers) — the strongest result

Hacking falls monotonically with process-supervision strength:
**base 52.3% → outcome 45.8% → mixes ≈42.3% → strict 40.5%.** Because all five
checkpoints answered *identical* problems, the right test is the **paired** bootstrap.
Every step of the ladder is significant and survives Bonferroni correction (α = 0.05/9):

| Paired contrast (teachers, overall) | Δ hacking (pp) | 95% CI | survives Bonferroni |
|---|---|---|---|
| base − outcome | +6.51 | [+4.74, +8.25] | ✓ |
| 0.5/0.5 − outcome | −3.48 | [−4.35, −2.62] | ✓ |
| 0.2/0.8 − outcome | −3.59 | [−4.46, −2.73] | ✓ |
| **strict − outcome** | **−5.31** | **[−6.37, −4.27]** | ✓ |

Under the released scorer, strict is also significantly below both mixes (≈ −1.8 pp,
p < 10⁻³). On the well-covered zoo context the same ordering holds but the effect is
small (33.6% → 32.0%).

### 4.3 Strict pass@k by difficulty band

| Context | Band | metric | base | outcome | 0.5/0.5 | 0.2/0.8 | strict |
|---|---|---|---|---|---|---|---|
| teachers | edge | p@1 | 33.5 | 44.8 | **51.2** | 47.8 | 48.0 |
| teachers | edge | p@64 | 80.8 | 81.0 | 81.5 | 80.5 | **83.0** |
| teachers | hard | p@1 | 11.5 | 30.3 | 32.5 | **33.5** | 32.5 |
| teachers | hard | p@64 | 50.7 | 60.7 | **62.8** | 59.7 | 59.3 |
| zoo | edge | p@1 | 68.8 | 71.0 | 67.8 | 68.2 | 69.8 |
| zoo | edge | p@64 | **94.5** | 89.8 | 92.8 | 92.2 | 93.2 |
| zoo | hard | p@1 | 54.7 | 59.8 | 58.7 | 58.8 | 58.5 |
| zoo | hard | p@64 | 82.0 | 82.0 | **82.2** | 82.2 | 80.5 |

Key paired contrasts:
- **RL vs base (teachers, unambiguous):** hard p@1 **+18.8 pp** [+15.2, +22.5]; hard
  p@64 **+10.0 pp** [+6.7, +13.3].
- **Process vs outcome, edge band:** 0.5/0.5 − outcome = **+6.50 pp** [+2.75, +10.25]
  (significant, Bonferroni-robust).
- **Process vs outcome, hard band:** all three process variants ahead (+2.2 to +3.2 pp),
  consistent in sign across all 5 op counts (sign test p = 0.031), but **no individual
  difference excludes zero** (best p = 0.056). Directional support, not confirmation.
- **Zoo edge p@64:** base − outcome = **+4.75 pp** [+2.25, +7.25] — RL *narrows* the
  sampled solution set where the base model was already strong (sharpening, not new
  ability).

### 4.4 Per-op strict pass@1 (teachers)

| Reward | op 12 | op 14 | op 16 | op 18 | op 20 |
|---|---|---|---|---|---|
| base | 43.5 | 23.5 | 12.0 | 14.0 | 8.5 |
| outcome | 51.5 | 38.0 | 34.5 | 30.5 | 26.0 |
| 0.5/0.5 | 57.0 | 45.5 | 34.5 | 36.5 | 26.5 |
| 0.2/0.8 | 52.0 | 43.5 | 35.5 | 37.0 | 28.0 |
| strict | 53.5 | 42.5 | 37.5 | 33.0 | 27.0 |

(Process variants beat outcome-only in 14 of 15 op × variant cells, tie once — the
sign-consistency behind the hard-band sign test.)

### 4.5 Error breakdown (teachers, share of all samples %)

**Hard band (op 16–20):**

| Reward | correct | wrong dependency | missing steps | wrong value | wrong answer |
|---|---|---|---|---|---|
| base | 11.7 | 86.5 | 46.6 | 79.3 | 69.2 |
| outcome | 29.5 | 68.1 | 30.1 | 47.8 | 40.5 |
| 0.5/0.5 | 32.0 | 66.7 | **24.8** | 47.8 | 40.0 |
| 0.2/0.8 | 32.0 | 66.8 | 27.6 | 48.2 | 40.5 |
| strict | 30.9 | 67.9 | *34.7* | 52.0 | 44.5 |

**Edge band (op 12–14):**

| Reward | correct | wrong dependency | missing steps | wrong value | wrong answer |
|---|---|---|---|---|---|
| base | 32.3 | 63.8 | 20.4 | 48.4 | 41.7 |
| outcome | 46.3 | 50.8 | 16.1 | 27.7 | 22.3 |
| 0.5/0.5 | 49.6 | 48.3 | **11.9** | 26.5 | 20.7 |
| 0.2/0.8 | 48.9 | 49.2 | 13.5 | 27.2 | 21.4 |
| strict | 49.2 | 48.8 | 14.6 | 27.9 | 22.6 |

Process rewards shift errors away from structural failures (wrong dependencies,
missing steps), the dense mixes most clearly. **Nuance:** strict gating *increases*
missing-step errors on the hard band (34.7% vs outcome's 30.1%) while dense mixes
decrease them (→24.8%) — an all-or-nothing trace reward favors short, conservative
derivations, whereas a dense per-step reward pays for completing each step.

## 5. Verdicts vs the paper's three predictions

1. *"Any process signal raises strict pass@1 by ~4–5 pp over outcome-only on the
   hardest problems."* — **Directionally supported** on the hard band (+2.2 to +3.2 pp,
   sign-consistent across all ops, individually unresolved); **unambiguous on the edge
   band** (+6.5 pp [+2.8, +10.3]).
2. *"0.2/0.8 and strict are best; strict has the lowest hacking rate."* — **Confirmed.**
   0.2/0.8 is the best hard-band pass@1; the hacking ladder base > outcome > mixes >
   strict is Bonferroni-robust at every step.
3. *"Process rewards shift errors away from wrong dependencies and missing steps."* —
   **Supported in aggregate**, with the strict-gating missing-step reversal as a
   refinement.

## 6. Validation (why these numbers are trustworthy)

Four independent audits:
- **Data integrity:** all 10 runs = exactly 64,000 records, zero truncation at the
  token cap, zero duplicate samples, problems byte-identical across checkpoints within
  a context, contexts cleanly separated.
- **Clean-room recomputation:** an independently written implementation reproduced
  every number above to ~10⁻¹⁶ and matched the authors' own aggregator exactly.
- **Qualitative audit:** 240 random samples re-scored from scratch, 240/240 agreement;
  genuine reward hacking confirmed by reading solutions.
- **Paired statistics:** every headline contrast recomputed with the paired bootstrap
  (which shrinks CIs ≈3× vs independent CIs by cancelling shared problem difficulty).

## 7. Limitations

- **One training run per reward variant** — the CIs quantify evaluation uncertainty
  only; seed variance cannot be excluded (the paper shares this).
- **Context-setting substitution** — the paper ran §6 on extrapolation; the released
  reward variants exist only in the context setting, so results are direction-comparable,
  not number-comparable.
- **k = 64, not the paper's 128** — to fit a single 3-hour GPU job; the paper's own
  curves are near-flat between 64 and 128.
- **0.5/0.5 weights assumed** (its config is absent from the release).
- **Scorer artifact** (see the companion extension N0.4): the authors' strict scorer
  has a letter-collision bug that inflates *absolute* hacking rates and deflates
  *absolute* strict pass@k for all checkpoints equally; the **paired differences and
  orderings reported here are robust**, but absolute levels should not be compared
  literally against the paper.

## 8. Reproducibility

One command on PACE: `sbatch scripts/run_all_pace.sbatch` (pinned environment:
vLLM 0.10.1.1, transformers 4.55.2, torch 2.7.1). Pipeline: HF downloads → smoke test
→ vLLM generation → authors' graph scoring → analysis (`src/analyze_e4.py`, bootstrap
seed 0; paired bootstrap seed 12345) → figures (`src/plots_e4.py`). Raw per-sample
outputs in `results/`; metrics, CSVs, figures, and the four-audit validation in
`analysis/`. Total compute: 1 h 20 m on one H100 (≈140 samples/s).

**Figures** (`analysis/`): `fig7_bars_{teachers,zoo}.png` (strict pass@k by reward and
band), `fig_hacking_rate.png` (hacking rate, overall + per op), `fig_error_breakdown_
{teachers,zoo}.png` (error categories), `fig_passk_per_op.png`, `fig_passk_curves.png`.
