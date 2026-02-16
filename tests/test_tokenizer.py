"""Tests for tokenizer merge functions."""

import json

import pytest
from tokenizers.implementations import SentencePieceBPETokenizer
from transformers import AutoTokenizer

from vinasmol.tokenization.training import merge_tokenizers_bpe


@pytest.fixture
def base_tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")


@pytest.fixture
def vietnamese_sp_tokenizer():
    """Small SentencePiece BPE tokenizer trained on Vietnamese-like text."""
    sp = SentencePieceBPETokenizer(
        replacement="\u0120",  # Ġ
        add_prefix_space=True,
        fuse_unk=False,
    )
    corpus = [
        "Viet Nam la mot quoc gia nam o Dong Nam A",
        "Thu do Ha Noi thanh pho Ho Chi Minh",
        "Cong hoa xa hoi chu nghia Viet Nam",
        "giao duc khoa hoc cong nghe van hoa",
    ] * 500
    sp.train_from_iterator(
        corpus,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
        min_frequency=2,
    )
    return sp


class TestMergeTokenizersBpe:
    def test_adds_new_tokens(self, base_tokenizer, vietnamese_sp_tokenizer):
        original_size = len(base_tokenizer)
        n_added = merge_tokenizers_bpe(base_tokenizer, vietnamese_sp_tokenizer)
        assert n_added > 0
        assert len(base_tokenizer) == original_size + n_added

    def test_preserves_english_tokenization(self, base_tokenizer, vietnamese_sp_tokenizer):
        text = "The quick brown fox jumps over the lazy dog"
        original = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        original_ids = original(text)["input_ids"]

        merge_tokenizers_bpe(base_tokenizer, vietnamese_sp_tokenizer)
        merged_ids = base_tokenizer(text)["input_ids"]

        assert original_ids == merged_ids

    def test_adds_merge_rules_not_added_tokens(self, base_tokenizer, vietnamese_sp_tokenizer):
        merge_tokenizers_bpe(base_tokenizer, vietnamese_sp_tokenizer)
        merged_json = json.loads(base_tokenizer.backend_tokenizer.to_str())
        original = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        original_json = json.loads(original.backend_tokenizer.to_str())

        # New merges should be appended to the merge list
        assert len(merged_json["model"]["merges"]) > len(original_json["model"]["merges"])
        # New vocab entries in the BPE model itself (not in added_tokens)
        assert len(merged_json["model"]["vocab"]) > len(original_json["model"]["vocab"])

    def test_merged_tokenizer_roundtrips(self, base_tokenizer, vietnamese_sp_tokenizer):
        merge_tokenizers_bpe(base_tokenizer, vietnamese_sp_tokenizer)
        texts = [
            "Viet Nam la mot quoc gia",
            "The capital of Vietnam is Hanoi",
            "giao duc khoa hoc cong nghe",
        ]
        for text in texts:
            ids = base_tokenizer(text)["input_ids"]
            decoded = base_tokenizer.decode(ids)
            # Decoded text should contain the original content (modulo whitespace)
            assert text.replace(" ", "") in decoded.replace(" ", "")

    def test_no_duplicate_merges(self, base_tokenizer, vietnamese_sp_tokenizer):
        merge_tokenizers_bpe(base_tokenizer, vietnamese_sp_tokenizer)
        merged_json = json.loads(base_tokenizer.backend_tokenizer.to_str())
        merges = merged_json["model"]["merges"]
        merge_tuples = [(m[0], m[1]) for m in merges]
        assert len(merge_tuples) == len(set(merge_tuples))
