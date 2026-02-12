#!/usr/bin/env bash
set -euo pipefail

python RPNN/main_repeated_experiments.py --system_name all_but_burgers --ab_init uniform
python RPNN/main_repeated_experiments.py --system_name all_but_burgers --ab_init centred
