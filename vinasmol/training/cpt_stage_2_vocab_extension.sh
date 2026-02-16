#!/bin/bash
# Stage 2: Vocabulary Extension
#
# Trains a Vietnamese tokenizer on the prepared corpora, merges it with
# the base SmolLM2 tokenizer, and initialises new token embeddings.
#
# Prerequisites:
#   - Prepared Vietnamese corpus in vinasmol/training/dataset/data/deduped/vi-all
#   - HuggingFace access to SmolLM2-360M-Instruct
#
# Outputs:
#   - Merged tokenizer: vinasmol/tokenization/checkpoints/merged_tokenizer
#   - Extended model:   vinasmol/tokenization/checkpoints/SmolLM2-360M_extended
#
# Usage:
#   bash cpt_stage_2_vocab_extension.sh [DATA_DIR...]
#
# The DATA_DIR arguments should point to the directories containing the
# prepared Vietnamese corpora (Parquet files).  Defaults to the standard
# pipeline output directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZER_DIR="${SCRIPT_DIR}/../tokenization"
CHECKPOINT_DIR="${TOKENIZER_DIR}/checkpoints"

# Default data directories (override by passing args)
DATA_DIRS=("${@:-${SCRIPT_DIR}/dataset/data/deduped/vi-all}")

MERGED_TOKENIZER_DIR="${CHECKPOINT_DIR}/merged_tokenizer"
EXTENDED_MODEL_DIR="${CHECKPOINT_DIR}/SmolLM2-360M_extended"

echo "============================================"
echo "  Stage 2a: Train & Merge Vietnamese Tokenizer"
echo "============================================"
python -m vinasmol.tokenization.training \
    "${DATA_DIRS[@]}" \
    --tokenizer-out-dir "${CHECKPOINT_DIR}"

echo ""
echo "============================================"
echo "  Stage 2b: Initialize New Token Embeddings"
echo "============================================"
python -m vinasmol.tokenization.vocab_extension \
    "${MERGED_TOKENIZER_DIR}" \
    "${EXTENDED_MODEL_DIR}"

echo ""
echo "Vocabulary extension complete."
echo "Merged tokenizer: ${MERGED_TOKENIZER_DIR}"
echo "Extended model:   ${EXTENDED_MODEL_DIR}"
