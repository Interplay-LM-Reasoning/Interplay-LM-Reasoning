# Beyond E4: Research Extensions

What the Interplay paper leaves open, and a concrete experimental program that could
grow this course project into a standalone paper. Ordered by cost; every Tier-0 item
reuses the 640k samples already on disk.

## The gap in the field this project is positioned to attack

The paper — and the broader process-reward-model literature — treats the process
verifier as **ground truth**. Our E4 validation falsified that assumption in the
paper's *own* testbed: the released strict scorer has a letter-collision artifact,
so even a synthetic, programmatically-generated gold verifier was noisy — inflating
absolute hacking rates up to ~6–8× while (fortunately) preserving variant orderings.
That is an accident waiting to be studied on purpose: **what happens to process-aware
RL when the verifier itself is imperfect?** Real-world process rewards (LLM judges,
heuristic step checkers, PRMs) are far noisier than this synthetic one. Nobody has a
controlled answer, and this testbed — tiny models, fully synthetic tasks, explicit
gold graphs, released training stack — is the ideal instrument. That is the flagship
(Experiment N1 below). Everything else feeds it or stands alone as a solid result.

---

## Tier 0 — Free: new science from the existing 640k samples (CPU only)

### N0.1 — Does RL trade solution diversity for precision? (mechanism of "sharpening")
**Gap.** The paper reports pass@k curves but never measures *why* pass@128 stalls or
drops. We observed zoo edge pass@64 drop under RL (−4.75 pp, significant).
**Method.** For each prompt's 64 samples, compute: number of distinct predicted
dependency-graph structures (hash of the parsed graph), token-level self-similarity,
and answer entropy. Compare base vs each reward variant, per band.
**Novelty.** First direct measurement of exploration collapse in this controlled
setting; tests whether process rewards (which constrain the *path*) collapse
diversity more or less than outcome rewards (which constrain only the destination).
Our corrected numbers hint strict preserves *more* useful diversity (highest
corrected edge pass@64) — counterintuitive and worth nailing down.

### N0.2 — What makes a problem hackable?
**Gap.** The paper reports aggregate hacking rates; it never characterizes *which*
problems get hacked.
**Method.** Per-problem hacking rate (already computed) regressed on gold-graph
structure: op count, depth, max fan-in, share of implicit edges, forward vs reverse
mode, answer magnitude. Logistic/beta regression + feature importance; compare the
hackability profile across reward variants.
**Novelty.** A predictive theory of reward hacking ("hacking concentrates where
dependency chains are shallow-but-wide / where implicit edges dominate") would be a
quotable contribution and directly informs RL data curation — the paper's own
"edge of competence" guidance, extended to integrity.

### N0.3 — The self-consistency dividend of process rewards
**Gap.** The paper never connects reward design to test-time compute strategies.
**Hypothesis.** Process-trained models are right *for consistent reasons*, so
majority-vote@k should gain more over pass@1 for them than for outcome-trained ones.
**Method.** Majority vote over the existing 64 answers per prompt; report vote@k vs
pass@1 vs pass@k per reward variant. Zero new compute.
**Novelty.** Links reward design to inference-time scaling — currently a hot, open
question (does RL training style change the value of sampling more?).

### N0.4 — Corrected-scorer re-analysis + upstream fix (community contribution)
Patch `SolutionParser` (strip the prose tail before dependency extraction; match
whole-token variables), re-score all 640k samples on CPU, publish the corrected E4
tables alongside the original, and open a PR + issue against the authors' repo with
the minimal reproduction we already have. Low effort, high credibility; also makes
every other experiment's absolute numbers trustworthy going forward.

---

## Tier 1 — One 3-hour GPU job each (evaluation only, existing checkpoints)

### N1.1 — Reward design × contextual transfer (crossing E4 with X2)
**Gap.** The paper studies reward design (§6) and contextual transfer (§4) as
separate axes; it never asks whether process supervision changes *what transfers*.
**Method.** Evaluate all five E4 checkpoints on `movie_festival_awards` (a context
never seen in pre-training or RL), ops 4–20, k=64 (~512k samples, one job).
**Predictions that would each be interesting:** (a) process variants transfer
reasoning fidelity better (lower hacking on the unseen context) → process rewards
teach *context-invariant* step discipline; (b) no difference → step discipline is
context-bound, a real limitation of the paper's Takeaway 4; (c) outcome-only
transfers answers but not steps → hacking explodes OOD, the most dramatic outcome.
**Novelty.** First evidence on whether reasoning *integrity* (not just accuracy)
generalizes across surface forms.

### N1.2 — Is there an "anti-hacking direction" in weight space? (X4 on the reward axis)
**Gap.** The paper's interpolation experiment (X4 in our guide) runs base↔RL; nobody
interpolates *between reward variants*.
**Method.** θ(α) = θ_outcome + α(θ_strict − θ_outcome) for α ∈ {0.25, 0.5, 0.75,
1.0, 1.5}; both models share the base init and data, differing only in reward, so
the difference vector isolates the reward's contribution. Evaluate strict pass@k and
hacking rate along the line (merging is a few lines of PyTorch; one eval job).
**Why disruptive.** If hacking decreases smoothly along α — and especially if
α = 1.5 reduces it *further* without pass@1 collapse — reasoning integrity is a
*steerable linear direction* obtainable post-hoc by weight arithmetic, no RL
retraining. That is a headline result for the model-merging and safety communities.
If instead the hacking drop is non-linear/emergent, that constrains how process
rewards reshape the policy. Either outcome publishes.

### N1.3 — Does outcome-only catch up at very large k? (X1 with corrected scoring)
**Method.** Ops 12/16/20, first 50 problems, k = 1,024, the five E4 checkpoints,
corrected scorer. Question: at large k, does outcome-only's *strict* pass@k converge
to the process variants' (process rewards only re-rank faithful modes) or stay below
(process rewards grow the set of faithful solutions the model can produce at all)?
**Novelty.** Applies Yue et al.'s large-k critique to the *integrity* dimension,
which no one has done — pass@k crossover analyses to date only track answers.

---

## Tier 2 — Training runs (the flagship; needs the authors' verl stack, ~1–4 GPUs/run)

### N2.1 — FLAGSHIP: Process rewards under imperfect verifiers
**Question.** How much verifier noise can process-aware RL tolerate before it stops
helping — or starts teaching the model to hack the *verifier*?
**Design.** The repo releases the full GRPO stack and reward functions. Wrap
`compute_score_with_step_process` with controlled corruption and train a grid from
the same base model (200 steps each, matching the released runs):
- **False-negative noise** (valid steps randomly rejected at rate ρ_FN ∈ {5, 15,
  30%}) — models the letter-collision artifact class;
- **False-positive noise** (invalid steps accepted at ρ_FP) — models weak LLM
  judges;
- **Systematic bias** (e.g., verifier always accepts steps that mention gold
  variable names regardless of values) — models gameable judges.
Evaluate everything with the *clean corrected* verifier plus held-out hacking rate.
**Predictions.** FN noise → conservative truncation (the strict missing-step
pathology we already observed, now induced causally); FP noise → verifier-hacking:
strict pass measured by the *noisy* verifier rises while clean-verifier pass falls —
a quantified "Goodhart curve" for process rewards; a tolerance threshold ρ* below
which process rewards still beat outcome-only.
**Why this is a paper.** It converts our accidental discovery into the first
controlled study of verifier-noise robustness for process rewards, in a setting
where ground truth is *actually known* — the one thing LLM-judge studies can never
have. Title sketch: *"Goodharting the Grader: How Verifier Noise Reshapes
Process-Reward RL."*

### N2.2 — Dense vs gated: why strict gating causes missing steps
**Gap/observation.** Our E4 found strict gating *increases* missing-step errors vs
outcome-only while dense mixes decrease them — unexplained by the paper.
**Design.** Train matched variants isolating the mechanism: (a) dense per-step
reward only (no gate), (b) gated all-or-nothing (strict), (c) gated + length bonus,
(d) dense + answer gate. If (b) under-produces steps and (c) repairs it, the
pathology is an exploration/credit-assignment artifact of sparse gating — practical
guidance the paper's "use strict" takeaway misses.

### N2.3 — Annealed process weight (reward curricula)
The paper tests only *static* α. Train schedules: outcome-first → process-later
(exploit then discipline) vs process-first → outcome-later (scaffold then optimize)
vs static 0.2/0.8, at matched compute. Connects §5's budget-split insight
(mid-training vs RL) to the reward-design axis — two halves of the paper that never
meet.

---

## Suggested packaging

- **Course-project+:** Tier 0 (all four) + N1.1 or N1.2 → "E4 replication + three
  original analyses + one original experiment," already beyond the assignment.
- **Workshop paper:** N0.4 + N1.2 + N1.3 → *"Reasoning integrity is measurable,
  steerable, and scorer-fragile: a re-examination of process rewards in controlled
  reasoning RL."*
- **Full paper:** N2.1 flagship, with Tier 0/1 results as the motivation and
  evaluation infrastructure (the corrected verifier from N0.4 becomes the paper's
  measurement instrument).

## Risks / honest caveats

- Single-seed training in the released checkpoints carries over to Tier 1
  conclusions; Tier 2 should train 2–3 seeds per cell (cheap at 100M params).
- N1.2's merge requires the two RL runs to live in a connected low-loss basin
  (likely, since both start from the same base and train briefly — but verify with
  the α=0/1 smoke test first).
- N2.x needs the verl training stack working on PACE (multi-GPU node; the repo
  pins verl + LLaMA-Factory versions; budget ~1 day of setup).
- Tokenizer/caveat from the guide still applies: results speak about this synthetic
  family; the claim is about *mechanisms*, with transfer to real LMs as motivated
  conjecture.
