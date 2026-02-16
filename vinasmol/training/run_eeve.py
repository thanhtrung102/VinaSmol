"""Entry point for EEVE multi-stage training.

Wraps litgpt's pretrain command with parameter freezing per EEVE stage.
This script monkey-patches ``lightning.fabric.Fabric.setup`` so that
``freeze_for_stage`` is called on the model right after Fabric wraps it,
but before the optimizer is created.  This avoids forking LitGPT while
still injecting EEVE freezing at exactly the right point in the training
loop.

Usage:
    python -m vinasmol.training.run_eeve --config CONFIG_PATH --eeve-stage STAGE

Example:
    python -m vinasmol.training.run_eeve \\
        --config vinasmol/training/cpt_eeve_stage_3.yml \\
        --eeve-stage 3
"""

import sys
from pathlib import Path
from unittest.mock import patch

import typer
import yaml
from loguru import logger

from vinasmol.training.eeve import EEVEStage, freeze_for_stage


app = typer.Typer(pretty_exceptions_show_locals=False)


def _make_patched_setup(stage: EEVEStage):
    """Create a patched ``Fabric.setup`` that injects EEVE parameter freezing.

    ``Fabric.setup`` is called on both the model and the optimizer.  We
    distinguish the model call by checking for ``named_parameters`` (present
    on ``nn.Module`` but not on optimizers).
    """
    from lightning.fabric import Fabric

    _original_setup = Fabric.setup

    def _patched_setup(self, *args, **kwargs):
        result = _original_setup(self, *args, **kwargs)
        # Only apply freezing to the model, not the optimizer.
        if hasattr(result, 'named_parameters'):
            logger.info("Intercepted Fabric.setup() — applying EEVE Stage {} freezing", stage.value)
            freeze_for_stage(result, stage)
        return result

    return _patched_setup


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to the LitGPT YAML config file."),
    eeve_stage: int = typer.Option(..., help="EEVE stage number (3, 4, 6, or 7)."),
):
    """Run EEVE multi-stage continued pretraining."""
    stage = EEVEStage(eeve_stage)
    logger.info("Starting EEVE Stage {} with config: {}", stage.value, config)

    if not config.exists():
        logger.error("Config file not found: {}", config)
        raise typer.Exit(1)

    # Log key config values for traceability.
    with open(config) as f:
        cfg = yaml.safe_load(f)
    logger.info("Output directory: {}", cfg.get("out_dir", "out/pretrain"))
    logger.info("Initial checkpoint: {}", cfg.get("initial_checkpoint_dir", ""))

    # Import lazily to keep CLI startup fast.
    from lightning.fabric import Fabric

    patched_setup = _make_patched_setup(stage)

    # Patch Fabric.setup for the duration of the training run so that
    # freeze_for_stage is applied right after the model is wrapped.
    with patch.object(Fabric, "setup", patched_setup):
        # Invoke litgpt pretrain through its CLI entry-point so that
        # jsonargparse handles all config parsing and type coercion.
        # This is equivalent to running `litgpt pretrain --config <path>`
        # but in-process, so the Fabric patch is active.
        from litgpt.__main__ import main as litgpt_main

        sys.argv = ["litgpt", "pretrain", "--config", str(config)]
        try:
            litgpt_main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                logger.error("Training failed with exit code: {}", exc.code)
                raise typer.Exit(exc.code)

    logger.info("EEVE Stage {} training complete.", stage.value)


if __name__ == "__main__":
    app()
