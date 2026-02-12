#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="./venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

A_MIN="0.01"
A_MAX="1.0"

systems=("SIR" "Brusselator" "Arenstorf" "Lorenz")
ab_inits=("uniform" "centred")

for ab_init in "${ab_inits[@]}"; do
  for system in "${systems[@]}"; do
    if [[ "$system" == "Lorenz" ]]; then
      nodes_list=("uniform" "lobatto")
    else
      nodes_list=("uniform")
    fi

    for nodes in "${nodes_list[@]}"; do
      echo "Running diagnostics: system=${system}, nodes=${nodes}, ab_init=${ab_init}, a_min=${A_MIN}, a_max=${A_MAX}"
      MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp "$PYTHON_BIN" RPNN/main_diagnostics.py \
        --system "$system" \
        --nodes "$nodes" \
        --ab_init "$ab_init" \
        --a_min "$A_MIN" \
        --a_max "$A_MAX"
    done
  done
done
