#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
args=(train --quick)
if [[ -f data/kddcup.data_10_percent.gz ]]; then
  args+=(--data data/kddcup.data_10_percent.gz)
fi
.venv/bin/python ids.py "${args[@]}"
.venv/bin/python ids.py predict results-quick/detector.joblib results-quick/example-connections.csv
