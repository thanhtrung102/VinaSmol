"""Run Vietnamese benchmarks to evaluate language capabilities.

Uses lm-evaluation-harness with Vietnamese-specific tasks.
For VMLU and M3Exam, custom task configurations are used.

Usage:
    python -m vinasmol.evaluation.run_benchmarks \
        --model-path checkpoints/VinaSmol/VinaSmol_stage_1 \
        --output-dir results/vietnamese
"""

import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(pretty_exceptions_show_locals=False)

# Vietnamese tasks available in lm-evaluation-harness
VIETNAMESE_TASKS = [
    "m3exam_vi",       # M3Exam Vietnamese subset
    "xcopa_vi",        # Cross-lingual Choice of Plausible Alternatives
    "xnli_vi",         # Cross-lingual Natural Language Inference
    "xstorycloze_vi",  # Cross-lingual StoryCloze
]


@app.command()
def main(
    model_path: Path = typer.Option(..., help="Path to the HuggingFace model checkpoint."),
    output_dir: Path = typer.Option("results/vietnamese", help="Directory to save results."),
    batch_size: int = typer.Option(8, help="Batch size for evaluation."),
    tasks: str = typer.Option(",".join(VIETNAMESE_TASKS), help="Comma-separated list of tasks."),
    vmlu: bool = typer.Option(False, help="Include VMLU benchmark (requires custom config)."),
):
    """Run Vietnamese benchmarks on a VinaSmol checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    task_list = tasks.split(",")

    if vmlu:
        # VMLU requires a custom task config pointing to the VMLU dataset
        vmlu_config = output_dir / "vmlu_config.yaml"
        if not vmlu_config.exists():
            logger.warning(
                "VMLU config not found at {}. "
                "Download from https://vmlu.ai/ and create a task config. "
                "Skipping VMLU.",
                vmlu_config,
            )
        else:
            task_list.append("vmlu")

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
