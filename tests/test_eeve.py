"""Tests for EEVE multi-stage parameter freezing."""

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from vinasmol.training.eeve import EEVEStage, freeze_for_stage


class MockGPTModel(nn.Module):
    """Minimal mock of LitGPT's GPT model structure."""

    def __init__(self, vocab_size=55936, n_embd=64, n_layer=2):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "h": nn.ModuleList([
                nn.Linear(n_embd, n_embd) for _ in range(n_layer)
            ]),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        return x


@pytest.fixture
def model():
    return MockGPTModel()


def _trainable_param_names(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def _frozen_param_names(model):
    return {name for name, p in model.named_parameters() if not p.requires_grad}


class TestEEVEStage3:
    def test_only_embeddings_trainable(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_3)
        trainable = _trainable_param_names(model)
        # Only embedding-related params should be trainable
        for name in trainable:
            assert "wte" in name or "lm_head" in name

    def test_transformer_layers_frozen(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_3)
        frozen = _frozen_param_names(model)
        has_frozen_transformer = any("transformer.h" in name for name in frozen)
        assert has_frozen_transformer


class TestEEVEStage4:
    def test_all_embeddings_trainable(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_4)
        trainable = _trainable_param_names(model)
        assert any("wte" in name for name in trainable)
        assert any("lm_head" in name for name in trainable)

    def test_transformer_layers_frozen(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_4)
        frozen = _frozen_param_names(model)
        has_frozen_transformer = any("transformer.h" in name for name in frozen)
        assert has_frozen_transformer


class TestEEVEStage6:
    def test_all_params_trainable(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_6)
        frozen = _frozen_param_names(model)
        assert len(frozen) == 0


class TestEEVEStage7:
    def test_only_transformer_trainable(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_7)
        trainable = _trainable_param_names(model)
        for name in trainable:
            assert "transformer.h" in name

    def test_embeddings_frozen(self, model):
        freeze_for_stage(model, EEVEStage.STAGE_7)
        frozen = _frozen_param_names(model)
        assert any("wte" in name for name in frozen)
        assert any("lm_head" in name for name in frozen)


class TestEEVEStageEnum:
    def test_valid_stages(self):
        assert EEVEStage.STAGE_3 == 3
        assert EEVEStage.STAGE_4 == 4
        assert EEVEStage.STAGE_6 == 6
        assert EEVEStage.STAGE_7 == 7

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            EEVEStage(5)


class TestFabricPatchIntegration:
    """Test that _make_patched_setup correctly intercepts Fabric.setup."""

    def test_patched_setup_calls_freeze_for_stage_on_model(self):
        """Verify freeze_for_stage is invoked when Fabric.setup receives a model."""
        from vinasmol.training.run_eeve import _make_patched_setup

        model = MockGPTModel()
        assert all(p.requires_grad for p in model.parameters())

        # _make_patched_setup imports Fabric internally and captures its
        # original .setup.  We mock Fabric.setup to simply return the model.
        with patch("lightning.fabric.Fabric.setup", return_value=model):
            patched_fn = _make_patched_setup(EEVEStage.STAGE_7)

        # Invoke the patched function with a dummy self (the Fabric instance)
        result = patched_fn(MagicMock(), model)

        # After Stage 7: embeddings frozen, transformer layers trainable
        frozen = {n for n, p in result.named_parameters() if not p.requires_grad}
        trainable = {n for n, p in result.named_parameters() if p.requires_grad}
        assert any("wte" in n for n in frozen), "wte should be frozen in Stage 7"
        assert any("lm_head" in n for n in frozen), "lm_head should be frozen in Stage 7"
        assert any("transformer.h" in n for n in trainable), "transformer.h should be trainable"

    def test_patched_setup_skips_non_model_objects(self):
        """Verify freeze_for_stage is NOT called on non-model objects like optimizers."""
        from vinasmol.training.run_eeve import _make_patched_setup

        optimizer_mock = MagicMock(spec=[])  # No named_parameters

        with patch("lightning.fabric.Fabric.setup", return_value=optimizer_mock):
            patched_fn = _make_patched_setup(EEVEStage.STAGE_3)

        result = patched_fn(MagicMock(), optimizer_mock)
        assert result is optimizer_mock
