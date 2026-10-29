# Experiment 4 — Process-Aware Rewards and Reward Hacking

**Replication of §6 (Figs. 7–8) of Zhang, Neubig & Yue (2025), "On the Interplay of
Pre-Training, Mid-Training, and RL on Reasoning Language Models" (arXiv 2512.07783),
in the contextual-generalization setting, using the authors' released checkpoints.**

*Course: DLT — Final Project, Experiment 4. Run: PACE ICE, 1× H100, job 6589709
(2026-10-26, 1 h 20 m wall time). All code, data manifests, and validation artifacts:
`experiment4/` (this directory).*

---

## Abstract

Outcome-based RL rewards are vulnerable to *reward hacking*: models learn to produce
correct final answers through invalid reasoning. The paper proposes mixing a
process-verification signal into the reward — `R = α·R_out + (1−α)·R_pv`, plus a
strict variant that grants the outcome reward only when every intermediate step is
verified — and reports that process-aware rewards raise strict accuracy and reduce
hacking. We replicate this claim on the released checkpoints: one shared base model
(context A "zoo" at op 2–20 + 0.1% context B "teachers") and four RL variants
differing only in reward (outcome-only, 0.5/0.5 mix, 0.2/0.8 mix, strict), evaluated
with 64 samples per problem on 2,000 problems (ops 12–20, both contexts; 640k
samples). **Result:** the paper's core claim is confirmed, and more decisively than
the paper's own presentation: the reward-hacking rate falls monotonically with
process-supervision strength (base 52.3% → outcome 45.8% → mixes ≈42.3% → strict
40.5% on the long-tail context), with every step of the ladder significant under
paired bootstrap after Bonferroni correction. Strict pass@1 gains from process
rewards are decisive on edge-difficulty problems (+6.5 pp [+2.8, +10.3] for the
0.5/0.5 mix over outcome-only) and directional but individually unresolved on the
hardest band (+2.2 to +3.2 pp, sign-consistent across all five op counts). In the
course of validating the pipeline we additionally discovered a **letter-collision
artifact in the authors' released strict scorer** that inflates absolute hacking
rates (up to ~6–8×) and deflates absolute strict pass@k (12–22 pp) for *all*
checkpoints equally. The headline process-vs-outcome contrast survives correction at
full strength (−4.5 pp hacking, p<10⁻³); the finer ranking is verifier-regime-
dependent — under corrected scoring the three process variants tie on hacking and
the dense mixes dominate strict on accuracy. Absolute numbers from the released
scorer should not be taken literally.

---

## 1. Background and question

The paper's §6 asks whether incorporating process-level verification into the RL
reward mitigates reward hacking while preserving generalization. Its reward designs
(Eq. 3):

- **Outcome-only**: `R = R_out` (1 iff the final answer is correct);
- **Mixed**: `R = α·R_out + (1−α)·R_pv`, with `R_pv` the step-level process accuracy
  (the fraction of gold dependency-graph nodes the solution reproduces with correct
  dependencies and values), at α = 0.5 and α = 0.2;
- **Strict**: `R = R_out` if the process is fully correct, else 0.

The paper ran this on the extrapolation setting (RL on op 11–14); those reward-variant
checkpoints were not released. The released `context_pretrain` repo contains the same
four-way reward comparison in the **context setting**, which we use. Results are
therefore comparable to the paper in *direction*, not exact numbers.

**Questions.** (Q1) Do mixed/strict process rewards raise strict pass@k over
outcome-only? (Q2) Do they lower the share of right answers reached through wrong
steps (the reward-hacking rate)? (Q3) Do they shift the error distribution away from
structural errors (wrong dependencies, missing steps)?

## 2. Setup

**Checkpoints** (HF `Interplay-LM-Reasoning/context_pretrain`, folder
`idzoo_0.999zoo_0.001teacher/`): one 100M-parameter Qwen2.5-style base model
pre-trained on 99.9% context A (crazy_zootopia, op 2–20) + 0.1% context B
(teachers_in_school, also spanning op 2–20 — this release is the paper's *appendix* A.6.1
"shared primitives" setting; the main-text §4 family with its op-2-only seed is the separate
`context_pretrain_2` release); four GRPO runs from it on identical data
(90% A / 10% B) differing only in reward:

| Label | Subfolder | Reward (verified from the released training configs) |
|---|---|---|
| base | `base` | none (pre-training only) |
| outcome | `rl/contextzoo_0.9zoo_0.1teacher` | answer-only (`compute_score`; **verified**, resolving the guide's open question) |
| 0.5/0.5 | `rl/…_process0.5` | 0.5 answer + 0.5 step score (no yaml in the release; weights assumed) |
| 0.2/0.8 | `rl/…_process` | 0.2 answer + 0.8 step score (`answer_weight: 0.2, step_weight: 0.8`) |
| strict | `rl/…_process_strict` | answer reward gated on perfect process (`zero_on_process_mismatch: true`) |

**Implementation detail that diverges from the paper's Eq. 3.** The released reward
code (`verl/reward_fn.py`) applies an *answer gate* to the mixed rewards: after
computing `α·R_out + (1−α)·R_pv` it zeroes the reward whenever the final answer is
wrong. As trained, the mixes are therefore
`R = 1[answer correct] · (α + (1−α)·R_pv)` — the process term modulates reward
magnitude *among answer-correct rollouts*; wrong-answer rollouts never receive the
dense process signal the paper's Eq. 3 implies. The literal ungated Eq. 3 was never
trained in the released family. A second consequence: training-time process rewards
were computed with the same artifact-affected scorer discussed in §3.4, so the
released process-reward runs were themselves trained under a systematically noisy
verifier.

**Evaluation.** `Interplay-LM-Reasoning/context` val splits, ops {12, 14, 16, 18, 20},
200 problems per op, both contexts (identical problems across all five checkpoints
within a context; different numeric instantiations across contexts). Sampling:
temperature 0.7, top-p 1.0, ≤1,024 new tokens, k = 64 samples/problem via the
authors' vLLM evaluator → 5 × 2 × 1,000 × 64 = **640,000 samples**. Scoring: the
authors' graph pipeline (gold dependency graphs → per-sample strict verdicts →
pass@k), plus our analysis layer (answer-only pass@k, reward-hacking rate, error
breakdown, bootstrap CIs over problems). Difficulty bands: *edge* = op 12–14,
*hard* = op 16–20.

**Metrics.** *Strict pass@k*: a problem counts at k if any of the first k samples
reproduces every gold step (dependencies + values) *and* the final answer
(paper's A.4 criterion). *Answer-only pass@k*: final answer only. *Reward-hacking
rate*: among samples with the correct final answer, the share whose reasoning is not
strictly correct — computed per problem as max(0, n_answer_correct − n_strict) and
aggregated as a ratio of sums.

**Integrity (summary of the validation audits; details in
`analysis/validation/VALIDATION.md`).** All 10 runs complete (64,000/64,000 records,
zero parse failures, zero truncation at the token cap, zero duplicate samples);
problems byte-identical across checkpoints within each context; all published
numbers reproduced to machine precision by an independent clean-room
reimplementation and matched the authors' own aggregator exactly.

## 3. Results

### 3.1 Headline: process supervision reduces reward hacking, monotonically (Q2) — the strongest result

Reward-hacking rate on context B (teachers), overall, with 95% bootstrap CIs over
problems:

| Reward | Hacking rate | 95% CI |
|---|---|---|
| base | 52.3% | [49.1, 55.7] |
| outcome | 45.8% | [42.7, 48.8] |
| 0.5/0.5 | 42.3% | [39.3, 45.7] |
| 0.2/0.8 | 42.2% | [38.7, 45.4] |
| **strict** | **40.5%** | [37.3, 43.5] |

Because all five checkpoints answered *identical* problems, paired differences are
the right test. Under paired bootstrap (B = 5,000, problem-level resampling), **every
step of the ladder is significant and survives Bonferroni correction** (α = 0.05/9):
strict − outcome = **−5.31 pp [−6.37, −4.27]**; mixes − outcome ≈ −3.5 pp each; and
strict is additionally below both mixes (≈ −1.8 pp, p < 0.001). The same ordering
holds per difficulty band (hard band: base 62.1% → strict 44.3%) and, attenuated, on
the well-covered zoo context (33.6% → 32.0%, individually small). **The paper's
prediction that strict process gating yields the lowest hacking rate is confirmed
decisively.**

### 3.2 Strict pass@k (Q1): decisive on edge, directional on hard

Strict pass@1 / pass@64 (%), teachers:

| Reward | p@1 edge | p@1 hard | p@64 edge | p@64 hard |
|---|---|---|---|---|
| base | 33.5 | 11.5 | 80.8 | 50.7 |
| outcome | 44.8 | 30.3 | 81.0 | 60.7 |
| 0.5/0.5 | **51.2** | 32.5 | 81.5 | **62.8** |
| 0.2/0.8 | 47.8 | **33.5** | 80.5 | 59.7 |
| strict | 48.0 | 32.5 | **83.0** | 59.3 |

- RL vs base is unambiguous: hard pass@1 +18.8 pp [+15.2, +22.5], hard pass@64
  +10.0 pp [+6.7, +13.3] (paired, Bonferroni-robust) — RL transfers reasoning to the
  long-tail context given its 0.1% pre-training seed (consistent with the paper's §4).
- Process vs outcome on the **edge band** is significant: 0.5/0.5 − outcome =
  **+6.50 pp [+2.75, +10.25]** at pass@1.
- On the **hard band**, all three process variants beat outcome-only at pass@1
  (+2.2 to +3.2 pp), with a consistent sign across all 5 op counts (sign test
  p = 0.031), but no individual difference excludes zero under the paired test (best:
  0.2/0.8, p = 0.056). The paper predicted ~4–5 pp here; we observe the same
  direction at somewhat smaller magnitude, below per-comparison significance. The
  best hard-band variant being 0.2/0.8 matches the paper's Fig. 7.
- On zoo (well-covered context A), RL raises pass@1 (60.3 → ~63%) but *lowers*
  edge pass@64 (94.5 → 89.8 for outcome; paired base − outcome = +4.75 pp
  [+2.25, +7.25]) — RL sharpens existing ability and measurably *narrows* the
  sampled solution set where the base model was already strong, echoing the paper's
  §3 claim from an independent angle.

### 3.3 Error breakdown (Q3)

Share of all samples, teachers hard band:

| Reward | correct | wrong dependency | missing steps | wrong value | wrong answer |
|---|---|---|---|---|---|
| base | 11.7 | 86.5 | 46.6 | 79.3 | 69.2 |
| outcome | 29.5 | 68.1 | 30.1 | 47.8 | 40.5 |
| 0.5/0.5 | 32.0 | 66.7 | **24.8** | 47.8 | 40.0 |
| 0.2/0.8 | 32.0 | 66.8 | 27.6 | 48.2 | 40.5 |
| strict | 30.9 | 67.9 | 34.7 | 52.0 | 44.5 |

Process rewards move errors away from structural failures, as predicted — the dense
mixes most clearly (missing steps 30.1% → 24.8%). One nuance the paper's framing
misses: **strict gating slightly increases missing-step errors relative to
outcome-only** (34.7% vs 30.1%), plausibly because an all-or-nothing trace reward
encourages shorter, more conservative solutions, whereas *dense* per-step rewards
directly pay for completing every step. Qualitative reading of matched hacked
samples supports a second nuance: among *genuinely* flawed solutions, base and
strict error compositions are similar — the process-reward benefit is mostly
*fewer* genuine errors rather than *different kinds*.

### 3.4 A measurement artifact in the released strict scorer

While validating per-sample verdicts by re-scoring solutions from scratch and
reading them, we found a systematic defect in the authors' `SolutionParser`: the
dependency extractor captures **every single letter** in an assignment's right-hand
side, whose regex capture runs through trailing English prose. In the dominant
`forwardreverse` problem mode, the final step reads like "… = 16 and we solve it
through the following steps: Simplifying …" — so letters from *words* ("s", "o",
"p", …) become phantom dependencies whenever the model (or the gold solution)
happened to assign those letters to earlier quantities. "Perfect" therefore partly
requires the model's random variable-letter choices to collide with gold's phantom
pattern.

We verified the mechanism (structurally identical solutions flip from
`dependency_mismatch` to perfect when the prose tail is stripped from both sides)
and re-scored the full op-12 + op-20 population (25,600 samples/run) with the tail
stripped:

| Run | Hacking: original → corrected | Strict p@1: original → corrected |
|---|---|---|
| zoo/base | 29.8% → 4.6% | 64.8 → 85.0 |
| zoo/outcome | 29.0% → 4.2% | 68.5 → 90.5 |
| zoo/strict | 28.2% → 3.7% | 66.5 → 89.5 |
| teachers/base | 46.3% → 30.6% | 26.0 → 33.3 |
| teachers/outcome | 42.5% → 23.7% | 38.8 → 51.0 |
| teachers/strict | 37.5% → 19.6% | 40.3 → 53.5 |

Three take-aways (updated after re-scoring the **full** population of all 10 runs
with the corrected instrument; `extensions/n0.4_corrected_scorer/`). (i)
**Absolute** hacking rates from this scorer are inflated (~6–8× on zoo, ~1.5× on
teachers) and absolute strict pass@k deflated by 12–22 pp. (ii) Under corrected
scoring, the decisive finding survives at full strength: **any process signal
reduces hacking by ≈4.5 pp over outcome-only** (paired, p<10⁻³, teachers, all three
variants). Two finer orderings, however, are **revised**: strict's position as
uniquely lowest-hacking does not survive — the three process variants statistically
tie (all pairwise CIs cross zero) — and the dense mixes dominate strict on corrected
pass@1 (+3.1 to +3.5 pp, p<0.02) while strict's pass@1 advantage over outcome-only
vanishes (+0.7 pp, n.s.). The zoo hacking ladder is entirely artifact. Because the
training-time reward used this same defective verifier, a natural hypothesis —
testable with the released training stack — is that the strict gate, an
all-or-nothing function of the noisy verifier, was the variant most distorted by
training-time verifier noise. (iii) The paper used this same scorer, so its absolute
Fig. 7–8 numbers likely carry the same artifact, and its strict/0.2–0.8 ranking is
exactly the kind of conclusion our re-analysis shows to be verifier-regime-dependent;
comparisons against the paper should be limited to the process-vs-outcome contrast.
Genuine reward hacking remains real and abundant: manual audit of corrected-hacked
samples shows internally consistent but factually wrong derivations that land the
right final answer, and in 0/900 audited cases did the scorer claim a step was
missing when it was actually present in the text.

## 4. Conclusions vs the paper's predictions

1. *"Any process signal raises strict pass@1 by ~4–5 points over outcome-only on the
   hardest problems."* — **Directionally supported** (+2.2 to +3.2 pp on hard,
   sign-consistent across all ops but individually unresolved); **unambiguous on the
   edge band** (+6.5 pp [+2.8, +10.3]).
2. *"0.2/0.8 and strict are best; strict has the lowest hacking rate."* —
   **Confirmed under the released scorer** (0.2/0.8 best hard-band pass@1; the
   hacking ladder base > outcome > mixes > strict Bonferroni-robust at every step).
   Under the **corrected** scorer the process-vs-outcome reduction survives in full
   (−4.5 pp, p<10⁻³) but strict's uniqueness does not: the three process variants
   tie on hacking and the dense mixes dominate strict on accuracy — the reward
   ranking is verifier-regime-dependent (§3.4).
3. *"Process rewards shift errors away from wrong dependencies and missing steps."* —
   **Supported in aggregate**, with the nuance that dense mixes reduce missing-step
   errors while strict gating slightly increases them, and that the mechanism is
   mostly fewer genuine errors rather than changed error types.

## 5. Limitations

- **One training run per reward variant** (the released checkpoints): CIs quantify
  evaluation (problem-sampling) uncertainty only; seed variance cannot be excluded.
  The paper shares this limitation.
- **Context-setting substitution**: the paper ran §6 on extrapolation; the released
  reward variants exist only in the context setting. Directionally comparable only.
- **Scorer artifact** (§3.4): absolute strict metrics are biased for all systems;
  we report orderings and paired differences, which are robust. Corrected numbers
  cover ops 12 + 20; the mechanism is op-independent.
- The `outcome` checkpoint's reward is answer-only — verified from the released
  training config (`custom_reward_function: compute_score`); the 0.5/0.5
  checkpoint's exact weights remain assumed (its yaml is absent from the release).
- k = 64 (not the paper's 128) to fit the 3-hour single-GPU budget; the paper's own
  curves are near-flat between k = 64 and 128.

## 6. Reproducibility

One command on PACE: `sbatch scripts/run_all_pace.sbatch` (pinned environment:
vllm 0.10.1.1, transformers 4.55.2, torch 2.7.1; see `requirements.txt`). Pipeline:
HF downloads → smoke test → vLLM generation → authors' graph scoring → analysis
(`src/analyze_e4.py`, bootstrap seed 0) → figures (`src/plots_e4.py`). Raw
per-sample outputs: `results/`; metrics and figures: `analysis/`; independent
validation (clean-room recomputation, paired statistics `paired_stats.py` seed
12345, 48 annotated qualitative cases, artifact re-scoring): `analysis/validation/`.
Total compute: 1 h 20 m on one H100 (generation ≈ 76 min for 640k samples, ~140
samples/s).

### Figures (in `analysis/`)

- `fig7_bars_{zoo,teachers}.png` — strict pass@1/pass@64 by reward and band
  (paper-Fig. 7 analogue)
- `fig_hacking_rate.png` — hacking rate by reward, overall and per op
- `fig_error_breakdown_{zoo,teachers}.png` — error categories by reward and band
  (paper-Fig. 8 analogue)
- `fig_passk_per_op.png`, `fig_passk_curves.png` — full pass@k surfaces with CIs
