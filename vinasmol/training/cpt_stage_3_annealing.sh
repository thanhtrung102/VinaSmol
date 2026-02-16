#!/bin/bash
# Stage 3: Annealing
#
# Runs the annealing phase of continued pretraining.  Uses a data mixture
# that includes high-quality annealing data (CCVJ academic papers, curated
# Vietnamese text) alongside the standard Vietnamese-English-Code split.
#
# Prerequisites:
#   - EEVE Stage 7 checkpoint at checkpoints/VinaSmol/cpt/eeve_stage_7/final
#   - Tokenized datasets including annealing split
#
# Outputs:
#   - Final pretrained checkpoint in data/pretrain/vinasmol_stage_3/
#
# Usage:
#   bash cpt_stage_3_annealing.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/cpt_stage_3_annealing.yml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

echo "============================================"
echo "  Stage 3: Annealing"
echo "  Config: $CONFIG"
echo "============================================"

litgpt pretrain --config "$CONFIG"

echo ""
echo "Annealing stage complete."
