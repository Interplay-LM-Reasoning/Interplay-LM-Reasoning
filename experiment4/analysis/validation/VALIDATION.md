# E4 Validity Assessment (four independent audits, 2026-10-26)

Four independent audits of the completed E4 run (job 6589709): data integrity,
clean-room metric recomputation, qualitative solution audit, and paired
statistical review. Scripts and evidence files in this directory reproduce
every number below from the raw per-sample data in `../../results/`.

## Verdict in one paragraph

The experiment is **valid**: data is complete and uncorrupted, all published
numbers reproduce to machine precision under independently written code and
match the authors' own aggregator exactly, and the headline conclusions hold —
with two wording corrections from the paired statistical analysis and one
substantive discovery: the authors' strict scorer has a **letter-collision
artifact** that inflates *absolute* hacking rates (and deflates absolute strict
pass@k) for all checkpoints equally; every *relative* (variant-ordering)
conclusion survives, and most strengthen, after correcting for it.

## Audit 1 — Data integrity: PASS (all checks)

All 10 runs: exactly 64,000 records, 1,000 prompts (200 × ops {12,14,16,18,20}),
64 samples each, zero unparseable lines, zero duplicates. All five checkpoints
evaluated on byte-identical problems (reward comparisons are problem-paired).
Context separation verified per-record (no eval-folder mixup). Truncation at the
1,024-token cap: 0.00%; near-empty outputs: 0.00%; within-prompt duplicate
solutions: 0.00%. Evaluator metrics reconcile with raw data to 4 decimals.
Note for the report: zoo and teachers use different numeric instantiations, so
cross-context deltas are distribution-level, not problem-paired.

## Audit 2 — Clean-room recomputation: PASS (exact agreement)

Independent reimplementation of strict/answer-only pass@k and hacking rate from
raw files: max discrepancy vs `analysis/*.csv` ≈ 1e-16 (float noise); vs the
authors' `results.json` aggregates: exactly 0.0 for every op and k, all 10 runs.
All 64,000 samples scored per run (zero parse failures), so the three rank
conventions provably coincide. (`recompute.py`, `cleanroom_metrics.json`)

## Audit 3 — Statistical review (paired bootstrap, B=5000): two corrections

Pairing on shared problems shrinks CI widths ~3× vs the independent CIs in the
CSVs. Key results (percentage points, 95% paired CIs):

**Decisive (survive Bonferroni, α=0.05/9):**
- Teachers hacking rate: strict − outcome = **−5.31 [−6.37, −4.27]**; also
  mixes − outcome ≈ −3.5 (both), and strict beats both mixes (−1.8, p<0.001).
  The full ladder base > outcome > mixes > strict is significant at every step.
  **Strongest confirmed result of the experiment.**
- Teachers RL vs base: pass@1 +18.8 [+15.2, +22.5]; pass@64 +10.0 [+6.7, +13.3].
- Teachers edge pass@1: 0.5/0.5 − outcome = **+6.50 [+2.75, +10.25]**.
- Zoo edge pass@64: base − outcome = +4.75 [+2.25, +7.25] (RL narrows the
  solution set on well-covered data).

**Not individually significant (report as directional only):**
- Teachers *hard* pass@1, process vs outcome: +2.2 to +3.2, all CIs cross 0
  (best p=0.056 for 0.2/0.8). Sign-consistent across all 5 ops (sign test
  p=0.031), so report as "consistent in sign, individually unresolved".
- Zoo overall pass@64 decline: +1.9, p=0.051 — keep hedged.

**Mandatory caveat:** one training run per reward variant; CIs quantify
evaluation (problem-sampling) uncertainty only, not seed variance. Same
limitation as the paper. (`paired_stats.py`, seed 12345)

## Audit 4 — Qualitative audit: pipeline faithful; scorer artifact found

- Re-scoring 240 random samples from scratch reproduced stored verdicts
  **240/240**; 12/12 deep-read "perfect" samples truly match gold structure;
  genuine reward hacking confirmed by reading (internally consistent but
  factually wrong derivations landing the right answer; 0/900 "missing" steps
  were actually present in text).
- **Artifact:** the authors' `SolutionParser` dependency extractor captures
  every letter in an assignment's tail — which, in `forwardreverse` problems,
  runs through English prose ("...= 16 and we solve it through the following
  steps..."), creating phantom dependencies keyed to the model's *random letter
  choices*. "Perfect" therefore partly requires reproducing gold's phantom
  letter pattern. Corrected full-population numbers (ops 12+20):
  zoo hacking 29.8→4.6% (base), 28.2→3.7% (strict); teachers 46.3→30.6% (base),
  37.5→19.6% (strict); strict pass@1 rises 12–22 points for all checkpoints.
- **Variant ordering survives correction everywhere** and process-reward gaps
  grow; uncorrected base-vs-RL comparisons were conservative. The paper used
  the same scorer, so its absolute rates likely carry the same artifact —
  compare orderings, not absolute levels, against the paper.
- Nuance for Fig-8 framing: among *genuine* hacked samples, base and strict
  error compositions are similar — process rewards mostly cause *fewer*
  genuine errors rather than different kinds.
  (`audit.py`, `corrected.py`, `cases.json` 48 annotated examples,
  `artifact_stats.json`, `corrected_metrics.json`)

## Claim-by-claim wording for the report

1. Paper prediction "process signal raises strict pass@1 on hardest problems":
   directional support on hard (+2.2..+3.2, sign-consistent, not individually
   significant); unambiguous on edge (+6.5 [+2.8, +10.3]).
2. "0.2/0.8 and strict best; strict lowest hacking": **confirmed decisively** —
   the hacking-rate ladder is Bonferroni-robust at every step; lead with this.
3. "Process rewards shift errors away from wrong deps / missing steps":
   supported in aggregate shares; qualitative nuance — the mechanism is mostly
   fewer genuine errors, not changed error types.
4. Bonus (§3 cross-ref): RL sharpens-not-expands on zoo — real on the edge band.
5. Always carry: single-run caveat + scorer-artifact caveat (orderings robust,
   absolute levels inflated/deflated).
