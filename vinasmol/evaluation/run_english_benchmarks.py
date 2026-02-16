"""Run English benchmarks to measure catastrophic forgetting.

Uses lm-evaluation-harness to evaluate on standard English benchmarks.
Compares VinaSmol against the SmolLM2-360M baseline.

Usage:
    python -m vinasmol.evaluation.run_english_benchmarks \
        --model-path checkpoints/VinaSmol/VinaSmol_stage_1 \
        --output-dir results/english
"""

import json
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(pretty_exceptions_show_locals=False)

ENGLISH_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
]


@app.command()
def main(
    model_path: Path = typer.Option(..., help="Path to the HuggingFace model checkpoint."),
    output_dir: Path = typer.Option("results/english", help="Directory to save results."),
    batch_size: int = typer.Option(8, help="Batch size for evaluation."),
    tasks: str = typer.Option(",".join(ENGLISH_TASKS), help="Comma-separated list of tasks."),
):
    """Run English benchmarks on a VinaSmol checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    task_list = tasks.split(",")
    logger.info("Evaluating model: {}", model_path)
    logger.info("Tasks: {}", task_list)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", ",".join(task_list),
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir),
        "--log_samples",
    ]

    logger.info("Running: {}", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error("Evaluation failed with return code: {}", result.returncode)
        raise typer.Exit(result.returncode)

    logger.info("Results saved to: {}", output_dir)


if __name__ == "__main__":
    app()
