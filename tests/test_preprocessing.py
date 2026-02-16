"""Tests for data preprocessing and normalization."""

import pytest

from vinasmol.training.dataset.preprocessing import (
    convert_en_wiki_to_md,
    convert_vi_wiki_to_md,
    convert_mediawiki_to_md,
    replace_md_links_with_text,
    format_olmocr_pes2o,
    NormalizeCols,
    DatasetNames,
)


class TestReplaceMarkdownLinks:
    def test_simple_link(self):
        result = replace_md_links_with_text("[click here](https://example.com)")
        assert result == "click here"

    def test_no_links(self):
        result = replace_md_links_with_text("no links here")
        assert result == "no links here"

    def test_multiple_links(self):
        result = replace_md_links_with_text("[a](url1) and [b](url2)")
        assert result == "a and b"


class TestFormatOlmocrPes2o:
    def test_removes_references(self):
        text = "Title\nAbstract\nSome content\nReferences\n[1] Some ref"
        result = format_olmocr_pes2o(text)
        assert "References" not in result
        assert "Some content" in result

    def test_keeps_abstract(self):
        text = "Title\nAbstract\nThe main findings...\nReferences\n[1] ref"
        result = format_olmocr_pes2o(text)
        assert "Abstract" in result

    def test_no_abstract(self):
        text = "Title\nDirect content"
        result = format_olmocr_pes2o(text)
        assert "Direct content" in result


class TestConvertMediawikiToMd:
    def test_fallback_regex_cleaning(self):
        row = {
            'title': 'Test',
            'text': 'Hello {{template}} world\nTham khảo\n[1] ref',
        }
        result = convert_mediawiki_to_md(row, lang='vi')
        assert "{{template}}" not in result['text']
        assert "Tham khảo" not in result['text']
        assert "Hello" in result['text']


class TestNormalizeCols:
    @pytest.fixture(autouse=True)
    def init_dataset_names(self):
        DatasetNames._init_cls_vars()

    def test_ccvj_normalizer(self):
        row = {
            'text': 'Academic paper content',
            'url': 'https://vjol.info.vn/paper/123',
            'title': 'Research Title',
            'journal_id': 'test_journal',
        }
        result = NormalizeCols.ccvj(row)
        assert result['text'] == 'Academic paper content'
        assert result['metadata']['url'] == 'https://vjol.info.vn/paper/123'
        assert result['metadata']['title'] == 'Research Title'
        assert result['metadata']['journal_id'] == 'test_journal'
        assert 'id' in result
        assert 'origin' in result['metadata']

    def test_format_prompt_response(self):
        result = NormalizeCols.format_prompt_response("Q?", "A.")
        assert "Q?" in result
        assert "A." in result
