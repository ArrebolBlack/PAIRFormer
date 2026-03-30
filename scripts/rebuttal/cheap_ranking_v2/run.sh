#!/usr/bin/env bash
# run.sh — Execute cheap_ranking_v2 analysis
# Usage: cd PAIRFormer && bash scripts/rebuttal/cheap_ranking_v2/run.sh
set -euo pipefail

echo "=== Cheap Encoder Ranking Quality v2 ==="
echo "Working dir: $(pwd)"
echo "Device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo ""

# Set PYTHONPATH to include PAIRFormer root
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

python3 scripts/rebuttal/cheap_ranking_v2/cheap_ranking_v2.py 2>&1 | tee scripts/rebuttal/cheap_ranking_v2/run.log

echo ""
echo "=== Results ==="
ls -la scripts/rebuttal/cheap_ranking_v2/results/
echo ""
echo "Log saved to: scripts/rebuttal/cheap_ranking_v2/run.log"
