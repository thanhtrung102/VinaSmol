"""Tests for CCVJ PDF conversion and post-processing."""

import pytest

from ccvj.convert import PaperProcessor


class TestPaperProcessorPostprocessMd:
    @pytest.fixture
    def processor(self):
        return PaperProcessor()

    def test_removes_references_section(self, processor):
        md = "# Introduction\nContent here.\n## References\n[1] Some paper."
        result = processor.postprocess_md(md)
        assert "References" not in result
        assert "Content here." in result

    def test_removes_vietnamese_references(self, processor):
        md = "# Giới thiệu\nNội dung.\n## Tài liệu tham khảo\n[1] Bài báo."
        result = processor.postprocess_md(md)
        assert "Tài liệu tham khảo" not in result
        assert "Nội dung." in result

    def test_removes_acknowledgments(self, processor):
        md = "# Methods\nContent.\n## Acknowledgments\nThanks to...\n# Results\nMore content."
        result = processor.postprocess_md(md)
        assert "Acknowledgments" not in result

    def test_removes_header_footer_artifacts(self, processor):
        md = "Vol. 5, No. 2\n# Title\nContent.\nISSN 1234-5678"
        result = processor.postprocess_md(md)
        assert "Vol." not in result
        assert "ISSN" not in result
        assert "Content." in result

    def test_removes_standalone_page_numbers(self, processor):
        md = "Content before.\n  42  \nContent after."
        result = processor.postprocess_md(md)
        assert "Content before." in result
        assert "Content after." in result

    def test_collapses_excessive_newlines(self, processor):
        md = "Line 1.\n\n\n\n\n\nLine 2."
        result = processor.postprocess_md(md)
        assert "\n\n\n\n" not in result
        assert "Line 1." in result
        assert "Line 2." in result

    def test_strips_whitespace(self, processor):
        md = "  \n\nContent.\n\n  "
        result = processor.postprocess_md(md)
        assert result == "Content."

    def test_empty_input(self, processor):
        assert processor.postprocess_md("") == ""
