"""Tests for the exact-substring deduplication uint32 monkey-patch."""
import struct

import numpy as np

from vinasmol.training.dataset.deduplication import (
    SEPARATOR_BYTES_UINT32,
    _prepare_doc_uint32,
    _read_bytes_uint32,
    apply_uint32_patch,
    revert_uint32_patch,
)


class FakeTokenizer:
    """Minimal tokenizer mock that returns pre-defined token IDs."""

    def __init__(self, token_ids: list[int]):
        self._ids = token_ids

    def encode(self, text):
        return type("Enc", (), {"ids": self._ids})()

    def decode(self, ids, **kwargs):
        return f"decoded:{ids}"


class TestPrepareDocUint32:
    def test_encodes_tokens_as_uint32(self):
        tok = FakeTokenizer([100, 200, 60000])
        result = _prepare_doc_uint32(tok, "hello", rank=0, doc_id=0)

        # Separator: 4 bytes (0xFFFFFFFF) + 4 bytes doc_id + 4 bytes (0xFFFFFFFF) + 4 bytes rank = 16
        assert len(result) == SEPARATOR_BYTES_UINT32 + 3 * 4  # 3 tokens * 4 bytes each

    def test_separator_is_4_bytes(self):
        tok = FakeTokenizer([1])
        result = _prepare_doc_uint32(tok, "x", rank=0, doc_id=0)
        assert result[:4] == b"\xff\xff\xff\xff"
        assert result[8:12] == b"\xff\xff\xff\xff"

    def test_doc_id_and_rank_encoded(self):
        tok = FakeTokenizer([42])
        result = _prepare_doc_uint32(tok, "x", rank=7, doc_id=99)
        doc_id = struct.unpack("<I", result[4:8])[0]
        rank = struct.unpack("<I", result[12:16])[0]
        assert doc_id == 99
        assert rank == 7

    def test_high_token_ids_preserved(self):
        """Token IDs > 65535 should survive uint32 encoding (would overflow uint16)."""
        high_ids = [70000, 100000, 55936]
        tok = FakeTokenizer(high_ids)
        result = _prepare_doc_uint32(tok, "x", rank=0, doc_id=0)
        decoded = np.frombuffer(result[SEPARATOR_BYTES_UINT32:], dtype=np.uint32).tolist()
        assert decoded == high_ids


class TestReadBytesUint32:
    def test_roundtrip(self):
        """prepare_doc → read_bytes should recover original token IDs."""
        original_ids = [100, 55000, 42]
        tok = FakeTokenizer(original_ids)
        encoded = _prepare_doc_uint32(tok, "x", rank=0, doc_id=0)
        decoded = _read_bytes_uint32(encoded)
        assert decoded == original_ids

    def test_high_ids_roundtrip(self):
        """IDs > 65535 survive the roundtrip."""
        original_ids = [70000, 131072]
        tok = FakeTokenizer(original_ids)
        encoded = _prepare_doc_uint32(tok, "x", rank=0, doc_id=0)
        decoded = _read_bytes_uint32(encoded)
        assert decoded == original_ids


class TestApplyRevertPatch:
    def test_patch_modifies_module_functions(self):
        import datatrove.pipeline.dedup.exact_substrings as es_mod

        original_prepare = es_mod.prepare_doc
        original_read = es_mod.read_bytes

        apply_uint32_patch()
        try:
            assert es_mod.prepare_doc is _prepare_doc_uint32
            assert es_mod.read_bytes is _read_bytes_uint32
            assert es_mod.SEPARATOR_BYTES == SEPARATOR_BYTES_UINT32
        finally:
            revert_uint32_patch()

        assert es_mod.prepare_doc is original_prepare
        assert es_mod.read_bytes is original_read

    def test_double_apply_is_noop(self):
        apply_uint32_patch()
        try:
            apply_uint32_patch()  # should not raise or double-wrap
        finally:
            revert_uint32_patch()

    def test_revert_without_apply_is_noop(self):
        revert_uint32_patch()  # should not raise


class TestESFeatureFlag:
    def test_flag_reflects_cargo_availability(self):
        import shutil
        try:
            from vinasmol.training.dataset.filtering.vietnamese import ENABLE_EXACT_SUBSTRING_DEDUP
        except ImportError:
            # Filtering module has heavy deps (ftfy, spacy, etc.) that may not be installed
            import pytest
            pytest.skip("filtering module dependencies not installed")

        has_cargo = shutil.which("cargo") is not None
        assert ENABLE_EXACT_SUBSTRING_DEDUP == has_cargo
