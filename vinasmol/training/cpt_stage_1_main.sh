#!/bin/bash
# Stage 1: Main Continued Pretraining
#
# Runs the initial continued pretraining on the Vietnamese-English-Code
# data mixture using LitGPT.  This is a standard (non-EEVE) pretraining
# run that produces the base VinaSmol checkpoint.
#
# Prerequisites:
#   - Extended model checkpoint from Stage 2 (vocab extension)
#   - Tokenized datasets in dataset/data/tokenized/
#
# Outputs:
#   - Checkpoint: checkpoints/VinaSmol/cpt/stage_1/final
#
# Usage:
#   bash cpt_stage_1_main.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/cpt_stage_1_main.yml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

echo "============================================"
echo "  Stage 1: Main Continued Pretraining"
echo "  Config: $CONFIG"
echo "============================================"

litgpt pretrain --config "$CONFIG"

echo ""
echo "Stage 1 pretraining complete."
