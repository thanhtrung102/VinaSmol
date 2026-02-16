# Evaluation

We evaluate VinaSmol on English and Vietnamese benchmarks to measure:
1. **Vietnamese language capabilities** gained from continued pretraining.
2. **English language retention** to verify minimal catastrophic forgetting.

## English Benchmarks

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for English benchmarks:

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| HellaSwag | Commonsense reasoning | acc_norm |
| ARC-Easy | Grade-school science questions | acc_norm |
| ARC-Challenge | Harder science questions | acc_norm |
| PIQA | Physical intuition QA | acc_norm |
| WinoGrande | Coreference resolution | acc |

```bash
uv run python -m vinasmol.evaluation.run_english_benchmarks \
    --model-path checkpoints/VinaSmol/VinaSmol_stage_1 \
    --output-dir results/english
```

## SEA Benchmarks

We use the individual Vietnamese scores of existing SEA languages benchmarks.

- [SEA-HELM](https://arxiv.org/pdf/2502.14301) ([leaderboard](https://leaderboard.sea-lion.ai/))
- [M3Exam](https://arxiv.org/abs/2306.05179) ([GitHub](https://github.com/DAMO-NLP-SG/M3Exam))

## Vietnamese Benchmarks

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| VMLU | Vietnamese multitask understanding | acc |
| M3Exam (vi) | Multilingual exam questions | acc |
| ViLLM-Eval | Vietnamese LLM evaluation suite | varies |

```bash
uv run python -m vinasmol.evaluation.run_benchmarks \
    --model-path checkpoints/VinaSmol/VinaSmol_stage_1 \
    --output-dir results/vietnamese
```

Resources:
- [ViLLM-Eval](https://arxiv.org/abs/2404.11086) ([HuggingFace](https://huggingface.co/datasets/vlsp-2023-vllm/ViLLM-Eval), [website](https://ai.stanford.edu/~sttruong/villm/))
- [VMLU](https://vmlu.ai/docs/VMLU_Report_2024.pdf) ([website](https://vmlu.ai/), [leaderboard](https://vmlu.ai/leaderboard))
- [ViGPTQA](https://aclanthology.org/2023.emnlp-industry.70/) ([GitHub](https://github.com/DopikAI-Labs/ViGPT))

## Results

Results will be reported after training is complete. Expected format:

| Model | HellaSwag | ARC-E | ARC-C | PIQA | VMLU | M3Exam |
|-------|-----------|-------|-------|------|------|--------|
| SmolLM2-360M (baseline) | - | - | - | - | - | - |
| VinaSmol-360M (standard CPT) | - | - | - | - | - | - |
| VinaSmol-360M (EEVE) | - | - | - | - | - | - |

## Manual Evaluation

If possible, gather feedback from native Vietnamese speakers using [OpenWebUI](https://docs.openwebui.com/features/evaluation/) with Arena evaluation (anonymous A/B comparisons, ELO scoring).
