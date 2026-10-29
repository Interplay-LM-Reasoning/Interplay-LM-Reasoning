# ============================================================================
# E4 — Process-aware rewards and reward hacking: shared configuration
#
# Source this file from every script:   source "$(dirname "$0")/../config/e4_env.sh"
# Override any variable by exporting it before calling a script, e.g.
#   SCRATCH=/storage/scratch1/0/$USER bash scripts/01_download.sh
# ============================================================================

# --- Locations --------------------------------------------------------------
# Root of this experiment codebase (directory containing config/, scripts/, src/)
E4_ROOT="${E4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Scratch space for big downloads (checkpoints + data). On PACE keep this on
# scratch, NOT home. Example on Phoenix: /storage/scratch1/<x>/<user>
SCRATCH="${SCRATCH:-$HOME/scratch}"

# Where the authors' code repo is cloned (scripts/eval_checkpoints.py etc.)
INTERPLAY_REPO="${INTERPLAY_REPO:-$SCRATCH/interplay/Interplay-LM-Reasoning}"

# Download targets
MODELS_DIR="${MODELS_DIR:-$SCRATCH/interplay/models/context_pretrain}"
DATA_DIR="${DATA_DIR:-$SCRATCH/interplay/data/context}"

# Experiment working dirs (small files — fine inside the codebase)
EVAL_DIR="${EVAL_DIR:-$E4_ROOT/eval}"            # symlinked op files per context
GOLD_DIR="${GOLD_DIR:-$E4_ROOT/gold}"            # gold dependency graphs
GEN_DIR="${GEN_DIR:-$E4_ROOT/results/generations}"   # *_generations.jsonl + *_metrics.json
GRAPH_DIR="${GRAPH_DIR:-$E4_ROOT/results/graph_results}" # strict per-sample scores
ANALYSIS_DIR="${ANALYSIS_DIR:-$E4_ROOT/analysis}"

# --- HF repos ----------------------------------------------------------------
HF_MODEL_REPO="Interplay-LM-Reasoning/context_pretrain"
HF_DATA_REPO="Interplay-LM-Reasoning/context"
CKPT_PARENT="idzoo_0.999zoo_0.001teacher"

# --- Checkpoint registry (keep in sync with src/e4_registry.py) -------------
# label -> subfolder inside $HF_MODEL_REPO
E4_LABELS=(base outcome mix_0.5_0.5 mix_0.2_0.8 strict)
e4_subfolder() {
  case "$1" in
    base)         echo "$CKPT_PARENT/base" ;;
    outcome)      echo "$CKPT_PARENT/rl/contextzoo_0.9zoo_0.1teacher" ;;
    mix_0.5_0.5)  echo "$CKPT_PARENT/rl/contextzoo_0.9zoo_0.1teacher_process0.5" ;;
    mix_0.2_0.8)  echo "$CKPT_PARENT/rl/contextzoo_0.9zoo_0.1teacher_process" ;;
    strict)       echo "$CKPT_PARENT/rl/contextzoo_0.9zoo_0.1teacher_process_strict" ;;
    *) echo "unknown label: $1" >&2; return 1 ;;
  esac
}

# --- Eval settings (fixed by the experiment spec) ----------------------------
E4_CONTEXTS=(zoo teachers)
e4_template() {  # context key -> dataset template folder
  case "$1" in
    zoo)      echo "crazy_zootopia" ;;
    teachers) echo "teachers_in_school" ;;
    *) echo "unknown context: $1" >&2; return 1 ;;
  esac
}
E4_OPS=(12 14 16 18 20)
SAMPLE_K="${SAMPLE_K:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
ID_OP_THRESHOLD="${ID_OP_THRESHOLD:-10}"   # ops 12-20 are all OOD -> *_ood_generations.jsonl

export PYTHONPATH="$INTERPLAY_REPO:${PYTHONPATH:-}"
