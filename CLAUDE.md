# CLAUDE.md - VinaSmol-Unaite

## Project Overview

VinaSmol extends the SmolLM2-360M English language model with Vietnamese language capabilities through efficient continued pretraining on <10B Vietnamese tokens. This is a research/academic project demonstrating multilingual LLM development with limited compute.

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: uv (Astral)
- **Training Framework**: LitGPT (Lightning AI)
- **ML Libraries**: PyTorch, Transformers, Accelerate, Datatrove
- **Model Merging**: MergeKit
- **Data Processing**: LitData, Datatrove

## Repository Structure

- `vinasmol/` — Main package
  - `tokenization/` — Vocabulary extension (SmolLM2 49K → 56K tokens) using EEVE method
  - `training/` — Continued pretraining configs and scripts (multi-stage: main + annealing)
  - `training/dataset/` — Data pipeline: download, filter, deduplicate, prepare
  - `finetuning/` — Instruction tuning configs (LoRA) and model merging (YAML configs)
  - `evaluation/` — Benchmark evaluation (SEA-HELM, M3Exam, VMLU)
- `ccvj/` — CreativeCommons Vietnamese Journals dataset tool (uv workspace member)
- `vinasmol_datamodule/` — Custom LitGPT data loader (uv workspace member)
- `scripts/` — Conversion utilities (LitGPT ↔ HuggingFace, pth → safetensors)

## Common Commands

```bash
# Install dependencies
uv sync --all-packages

# Run tokenizer training
uv run python -m vinasmol.tokenization.training

# Run data preparation
uv run python -m vinasmol.training.dataset.preparation

# Launch standard continued pretraining
bash vinasmol/training/cpt_stage_1_main.sh

# Launch EEVE multi-stage training (stages 3→4→6→7)
bash vinasmol/training/cpt_eeve_stages.sh

# Launch annealing
bash vinasmol/training/cpt_stage_3_annealing.sh

# Convert CCVJ PDFs to training data
uv run python -m ccvj.convert

# Run evaluation benchmarks
uv run python -m vinasmol.evaluation.run_benchmarks --model-path <path>
uv run python -m vinasmol.evaluation.run_english_benchmarks --model-path <path>

# Convert checkpoints
bash scripts/convert_lit_ft_to_hf.sh

# Run tests
uv run pytest tests/ -v
```

## Key Architecture Decisions

- **EEVE Method** for vocabulary extension (embedding initialization from aligned tokens)
- **Multi-stage EEVE training** (stages 3,4,6,7): progressive unfreezing of embeddings→all→transformers
- **Annealing phase**: high-quality data mixture including CCVJ academic papers
- **LoRA** for parameter-efficient finetuning
- **Multi-stage model merging** to combine different finetuning objectives
- SmolLM2 has **tied embeddings** (lm_head shares weights with token embeddings)

## Data Pipeline

1. Download → Preprocess → Unicode normalization
2. Quality filtering (KenLM language models, custom Vietnamese filters)
3. Deduplication (document-level + substring via Google Research submodule)
4. Tokenization with extended vocabulary
5. Optimization with LitData for training

## Datasets

- **Vietnamese**: Wikipedia, CulturaX, FineWeb2-HQ, MADLAD-400, news corpus
- **English** (replay to prevent catastrophic forgetting): Cosmopedia v2, FineWebEdu, Gutenberg, Wikipedia
- **Code**: Starcoderdata Python subset
- **Finetuning**: Vietnamese Alpaca, LIMA, MURI-IT, multi-turn chat

## Rules

- Before answering data-dependent questions (SQL results, API outputs, computed values), always run the actual query or computation first. Never fabricate or estimate answers.
- When modifying training configs (YAML files), preserve existing hyperparameter comments and documentation.
- This project uses `uv` — do not use `pip` directly. Use `uv sync`, `uv run`, `uv add`.
