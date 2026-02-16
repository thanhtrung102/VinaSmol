#!/bin/bash
# EEVE Multi-Stage Training Pipeline for VinaSmol
#
# Runs the 4 EEVE stages sequentially (stages 3, 4, 6, 7).
# Each stage initializes from the previous stage's final checkpoint.
#
# Usage:
#   bash cpt_eeve_stages.sh [--start-stage STAGE]
#
# Options:
#   --start-stage STAGE   Resume from a specific stage (3, 4, 6, or 7).
#                         Default: 3 (start from the beginning).

set -euo pipefail

START_STAGE=${1:-3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_stage() {
    local stage=$1
    local config="$SCRIPT_DIR/cpt_eeve_stage_${stage}.yml"

    echo "============================================"
    echo "  EEVE Stage $stage"
    echo "  Config: $config"
    echo "============================================"

    python -m vinasmol.training.run_eeve --config "$config" --eeve-stage "$stage"

    echo "Stage $stage complete."
    echo ""
}

STAGES=(3 4 6 7)
STARTED=false

for stage in "${STAGES[@]}"; do
    if [ "$stage" -ge "$START_STAGE" ]; then
        STARTED=true
    fi
    if [ "$STARTED" = true ]; then
        run_stage "$stage"
    fi
done

echo "All EEVE stages complete."
echo "Final checkpoint: checkpoints/VinaSmol/cpt/eeve_stage_7/final"
