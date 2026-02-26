#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="./venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

# Defaults (override with env vars if needed)
SYSTEM="${SYSTEM:-Rober}"
NODES="${NODES:-uniform}"
A_MIN="${A_MIN:-0.01}"
A_MAX="${A_MAX:-1.0}"
N_X_VALUES="${N_X_VALUES:-3,5,7}"
COARSE_FACTORS="${COARSE_FACTORS:-10,20,50,100,200}"
NUM_TRIALS="${NUM_TRIALS:-5}"
SEED_BASE="${SEED_BASE:-1234}"

# W&B configuration: set USE_WANDB=1 to enable.
USE_WANDB="${USE_WANDB:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-dadeslam}"
WANDB_PROJECT="${WANDB_PROJECT:-RPNN}"

ab_inits=("uniform" "centred")

for ab_init in "${ab_inits[@]}"; do
  echo "Running sweep: system=${SYSTEM}, nodes=${NODES}, ab_init=${ab_init}, a_min=${A_MIN}, a_max=${A_MAX}, n_x_values=${N_X_VALUES}, coarse_factors=${COARSE_FACTORS}, num_trials=${NUM_TRIALS}"

  cmd=(
    "$PYTHON_BIN" "RPNN/main_sweep_nx_coarse.py"
    "--system" "$SYSTEM"
    "--nodes" "$NODES"
    "--ab_init" "$ab_init"
    "--a_min" "$A_MIN"
    "--a_max" "$A_MAX"
    "--n_x_values" "$N_X_VALUES"
    "--coarse_factors" "$COARSE_FACTORS"
    "--num_trials" "$NUM_TRIALS"
    "--seed_base" "$SEED_BASE"
  )

  if [[ "$USE_WANDB" == "1" ]]; then
    cmd+=(
      "--use_wandb"
      "--wandb_entity" "$WANDB_ENTITY"
      "--wandb_project" "$WANDB_PROJECT"
    )
  fi

  MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "${cmd[@]}"
done

echo "Completed both uniform and centred sweeps."
