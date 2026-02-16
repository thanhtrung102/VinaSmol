import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetBuilder, load_dataset_builder
from loguru import logger
from tqdm import tqdm
import typer

from . import DATA_DIR, PDF_DOWNLOAD_DIR, JournalId, RecordMetadata
from .resource import ListResource

OUT_DATASET_DIR = DATA_DIR / "ccvj" / "parquet"
REPO_ID = "ccvj"

# Patterns for cleaning academic paper markdown
_REFERENCES_RE = re.compile(
    r"^(?:#+\s*)?(?:References|Tài liệu tham khảo|TÀI LIỆU THAM KHẢO).*",
    flags=re.DOTALL | re.MULTILINE,
)
_ACKNOWLEDGMENTS_RE = re.compile(
    r"^(?:#+\s*)?(?:Acknowledgm?ents?|Lời cảm ơn).*?(?=^#|\Z)",
    flags=re.DOTALL | re.MULTILINE,
)
_HEADER_FOOTER_RE = re.compile(
    r"^(?:Vol\.|Tập|Số|No\.|ISSN|DOI:|http[s]?://doi).*$",
    flags=re.MULTILINE,
)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$", flags=re.MULTILINE)
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{4,}")


@dataclass
class ConversionResult:
    input_pdf_file: Path
    output_md_file: Path
    success: bool


class PaperProcessor:
    def __init__(self):
        self.results: list[ConversionResult] = []
        self._converter = None

    def _get_converter(self):
        """Lazily initialize the Docling converter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    def postprocess_md(self, md_content: str) -> str:
        """Clean the markdown content to remove superfluous items.

        Removes:
        - References/bibliography section
        - Acknowledgments section
        - Journal headers/footers (volume, ISSN, DOI lines)
        - Standalone page numbers
        - Excessive blank lines
        """
        md_content = _REFERENCES_RE.sub("", md_content)
        md_content = _ACKNOWLEDGMENTS_RE.sub("", md_content)
        md_content = _HEADER_FOOTER_RE.sub("", md_content)
        md_content = _PAGE_NUMBER_RE.sub("", md_content)
        md_content = _EXCESSIVE_NEWLINES_RE.sub("\n\n", md_content)
        return md_content.strip()

    async def convert_pdf(
            self,
            pdf_file: str | Path,
            output_md_file: str | Path,
        ) -> ConversionResult:
        """Convert a PDF paper to Markdown using Docling.

        Args:
            pdf_file: the path to the downloaded paper as a PDF file.
            output_md_file: the file path to write Markdown content.
        Returns:
            ConversionResult: whether the PDF file was successfully converted.
        """
        pdf_file = Path(pdf_file)
        output_md_file = Path(output_md_file)

        try:
            converter = self._get_converter()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: converter.convert(str(pdf_file)),
            )

            md_content = result.document.export_to_markdown()
            md_content = self.postprocess_md(md_content)

            if not md_content or len(md_content) < 100:
                logger.warning("Conversion produced too little text for: {}", pdf_file.name)
                return ConversionResult(
                    input_pdf_file=pdf_file,
                    output_md_file=output_md_file,
                    success=False,
                )

            output_md_file.parent.mkdir(parents=True, exist_ok=True)
            output_md_file.write_text(md_content, encoding="utf-8")

            return ConversionResult(
                input_pdf_file=pdf_file,
                output_md_file=output_md_file,
                success=True,
            )
        except Exception as e:
            logger.error("Failed to convert {}: {}", pdf_file.name, e)
            return ConversionResult(
                input_pdf_file=pdf_file,
                output_md_file=output_md_file,
                success=False,
            )

    async def convert_papers(self, download_dir: str | Path) -> list[ConversionResult]:
        """Process PDF papers in bulk using Docling.

        Args:
            download_dir: the directory that contains the downloaded PDFs.

        Returns:
            results: the conversion results.
        """
        download_dir = Path(download_dir)

        tasks = []
        for pdf_file in sorted(download_dir.glob("**/*.pdf")):
            md_file = pdf_file.with_suffix(".md")
            if md_file.exists():
                self.results.append(ConversionResult(
                    input_pdf_file=pdf_file,
                    output_md_file=md_file,
                    success=True,
                ))
                continue
            task = self.convert_pdf(pdf_file, md_file)
            tasks.append(task)

        if tasks:
            results: list[ConversionResult] = await asyncio.gather(*tasks)
            self.results.extend(results)

        return self.results


def compile_dataset(
        records: dict[JournalId, ListResource[RecordMetadata]],
        markdown_files: dict[Path, Path],
        dataset_dir: Path,
    ) -> DatasetBuilder:
    """Compile a Parquet dataset into a directory.

    Args:
        records: journal records with metadata.
        markdown_files: mapping from PDF files to Markdown files.
        dataset_dir: the output directory for the Parquet files.

    Returns:
        DatasetBuilder: a builder of the compiled Parquet dataset files.
    """

    def gen_journal_records(journal: JournalId, journal_records: list[RecordMetadata]):
        for record in journal_records:
            if 'local_pdf_file' not in record:
                continue
            local_pdf = record['local_pdf_file']
            if local_pdf not in markdown_files:
                continue
            md_file = markdown_files[local_pdf]
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read {}: {}", md_file, e)
                continue
            if len(text) < 100:
                continue
            yield dict(
                text=text,
                journal_id=journal,
                title=record.get('title', ''),
                url=record.get('url', ''),
            )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for journal, journal_records in tqdm(records.items(), desc="Compile journals"):
        journal_ds = Dataset.from_generator(
            gen_journal_records,
            gen_kwargs=dict(journal=journal, journal_records=journal_records),
        )
        if len(journal_ds) > 0:
            journal_ds.to_parquet(dataset_dir / f"{journal}.parquet")

    return load_dataset_builder(
        'parquet',
        data_files=str(OUT_DATASET_DIR / "*.parquet"),
    )


def process_papers(processor: PaperProcessor, pdf_dir: Path) -> dict[Path, Path]:
    results = asyncio.run(processor.convert_papers(pdf_dir))

    pdf_to_md = {}
    successes = []
    failures = []
    for result in results:
        if result.success:
            pdf_to_md[result.input_pdf_file] = result.output_md_file
            successes.append(result)
        else:
            failures.append(result)

    logger.info("{} PDFs successfully converted to Markdown.", len(successes))
    if failures:
        logger.warning("{} PDFs couldn't be converted.", len(failures))
    return pdf_to_md


def compile_and_load_ccvj(
        records: dict[JournalId, ListResource[RecordMetadata]],
        pdf_to_md: dict[Path, Path],
        dataset_dir: Path = OUT_DATASET_DIR,
        streaming: bool = True,
    ):
    builder = compile_dataset(records, pdf_to_md, dataset_dir=dataset_dir)
    if streaming:
        return builder.as_streaming_dataset(split='train')
    else:
        return builder.as_dataset(split='train')


app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        pdf_dir: Path = PDF_DOWNLOAD_DIR,
    ):
    """Convert CCVJ PDFs to Markdown using Docling."""
    logger.info("Converting PDFs from: {}", pdf_dir)

    processor = PaperProcessor()
    pdf_to_md = process_papers(processor, pdf_dir)

    logger.info("Conversion complete. {} files converted.", len(pdf_to_md))


if __name__ == "__main__":
    app()
