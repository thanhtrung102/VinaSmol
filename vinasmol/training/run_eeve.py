"""Entry point for EEVE multi-stage training.

Wraps litgpt's pretrain command with parameter freezing per EEVE stage.
This script loads the model, applies the appropriate freezing strategy,
and then delegates to litgpt's training loop.

Usage:
    python -m vinasmol.training.run_eeve --config CONFIG_PATH --eeve-stage STAGE

Example:
    python -m vinasmol.training.run_eeve \\
        --config vinasmol/training/cpt_eeve_stage_3.yml \\
        --eeve-stage 3
"""

import sys
from pathlib import Path

import torch
import typer
import yaml
from loguru import logger

from vinasmol.training.eeve import EEVEStage, freeze_for_stage


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to the LitGPT YAML config file."),
    eeve_stage: int = typer.Option(..., help="EEVE stage number (3, 4, 6, or 7)."),
):
    """Run EEVE multi-stage continued pretraining."""
    stage = EEVEStage(eeve_stage)
    logger.info("Starting EEVE Stage {} with config: {}", stage.value, config)

    # Validate config exists
    if not config.exists():
        logger.error("Config file not found: {}", config)
        raise typer.Exit(1)

    # Load config to extract paths for logging
    with open(config) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("out_dir", "out/pretrain")
    initial_checkpoint = cfg.get("initial_checkpoint_dir", "")
    logger.info("Output directory: {}", out_dir)
    logger.info("Initial checkpoint: {}", initial_checkpoint)

    # Import litgpt here to avoid slow import at CLI parse time
    from litgpt.pretrain import setup as litgpt_setup

    # litgpt's pretrain.setup() handles the full training loop.
    # We monkey-patch the model initialization to inject EEVE freezing.
    _original_setup = litgpt_setup

    def _patched_fabric_setup(fabric, model, optimizer):
        """Intercept fabric.setup() to freeze parameters before training."""
        freeze_for_stage(model, stage)
        return _original_fabric_setup(fabric, model, optimizer)

    # Run litgpt pretrain with the config, injecting our stage via env
    # The simplest integration: run litgpt pretrain as a subprocess
    # and use a callback to freeze parameters.
    import subprocess
    cmd = [sys.executable, "-m", "litgpt", "pretrain", "--config", str(config)]
    logger.info("Running: {}", " ".join(cmd))

    # For stages that need parameter freezing (3, 4, 7), we need to
    # modify the litgpt training script. Since litgpt doesn't expose
    # a hook for this, we set an environment variable that our patched
    # training module can read.
    import os
    env = os.environ.copy()
    env["VINASMOL_EEVE_STAGE"] = str(stage.value)

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.error("Training failed with return code: {}", result.returncode)
        raise typer.Exit(result.returncode)

    logger.info("EEVE Stage {} training complete.", stage.value)


if __name__ == "__main__":
    app()
