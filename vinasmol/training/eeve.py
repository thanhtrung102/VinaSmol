"""EEVE (Efficient and Effective Vocabulary Extension) multi-stage training.

Implements the parameter freezing strategy from Kim et al. (2024),
adapted for SmolLM2-360M's tied embeddings (stages 3, 4, 6, 7 only).

Reference: https://arxiv.org/abs/2402.14714v1

Stage 3: Train only new Vietnamese token embeddings (freeze all else).
Stage 4: Train all embedding weights (freeze transformer layers).
Stage 6: Train all parameters (no freezing).
Stage 7: Train only transformer layers (freeze all embeddings).
"""

from enum import IntEnum

from loguru import logger


# Original SmolLM2-360M vocabulary size before Vietnamese extension
ORIGINAL_VOCAB_SIZE = 49152


class EEVEStage(IntEnum):
    """EEVE training stages (numbered per original paper)."""
    STAGE_3 = 3  # New token embeddings only
    STAGE_4 = 4  # All embeddings
    STAGE_6 = 6  # All parameters
    STAGE_7 = 7  # Transformer layers only


def freeze_for_stage(model, stage: EEVEStage, original_vocab_size: int = ORIGINAL_VOCAB_SIZE):
    """Freeze/unfreeze model parameters according to the EEVE stage.

    In LitGPT's GPT model, the parameter names follow this structure:
        - transformer.wte.weight  (token embeddings)
        - transformer.h.{i}.*    (transformer layers)
        - lm_head.weight         (output head, tied with wte in SmolLM2-360M)

    Args:
        model: The LitGPT model instance.
        stage: Which EEVE stage to configure.
        original_vocab_size: Vocab size of the base model before extension.
    """
    stage = EEVEStage(stage)

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    total_count = sum(p.numel() for p in model.parameters())

    if stage == EEVEStage.STAGE_3:
        # Train only the NEW Vietnamese token embeddings
        # The new tokens are at indices [original_vocab_size:]
        for name, param in model.named_parameters():
            if "wte" in name or "lm_head" in name:
                # We can't partially freeze an embedding layer's requires_grad,
                # so we unfreeze the whole embedding and use a hook to zero out
                # gradients for the original tokens after backward.
                param.requires_grad = True
                trainable_count += param.numel()

        # Register hook to zero out gradients for original token embeddings
        _register_embedding_mask_hook(model, original_vocab_size)
        logger.info(
            "EEVE Stage 3: Training new token embeddings only "
            "(tokens {} onwards). Gradient masking enabled.",
            original_vocab_size,
        )

    elif stage == EEVEStage.STAGE_4:
        # Train all embedding weights (wte + lm_head), freeze transformer layers
        for name, param in model.named_parameters():
            if "wte" in name or "lm_head" in name:
                param.requires_grad = True
                trainable_count += param.numel()
        logger.info("EEVE Stage 4: Training all embedding weights.")

    elif stage == EEVEStage.STAGE_6:
        # Train everything
        for param in model.parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        logger.info("EEVE Stage 6: Training all parameters.")

    elif stage == EEVEStage.STAGE_7:
        # Train only transformer layers, freeze embeddings
        for name, param in model.named_parameters():
            if "transformer.h." in name:
                param.requires_grad = True
                trainable_count += param.numel()
        logger.info("EEVE Stage 7: Training transformer layers only.")

    frozen_count = total_count - trainable_count
    logger.info(
        "Parameters: {:,} trainable, {:,} frozen ({:.1f}% trainable)",
        trainable_count,
        frozen_count,
        100 * trainable_count / total_count if total_count > 0 else 0,
    )

    return model


def _register_embedding_mask_hook(model, original_vocab_size: int):
    """Register a backward hook that zeros gradients for original token embeddings.

    This allows Stage 3 to only update the NEW Vietnamese tokens while keeping
    the original SmolLM2 token embeddings frozen.
    """
    def _mask_embedding_grad(grad):
        grad[:original_vocab_size] = 0
        return grad

    for name, param in model.named_parameters():
        if "wte" in name or "lm_head" in name:
            if param.requires_grad:
                param.register_hook(_mask_embedding_grad)
                logger.debug("Registered gradient mask hook on: {}", name)
