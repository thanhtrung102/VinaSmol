import glob
from pathlib import Path
import struct
import subprocess
from subprocess import CalledProcessError
import tempfile

import numpy as np
from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
from datatrove.pipeline.dedup.exact_substrings import (
    ESRangeRemover,
    SEPARATOR_BYTES,
)
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import ExtensionHelperES as EH

from rensa import RMinHash, RMinHashLSH


# ---------------------------------------------------------------------------
# Monkey-patch datatrove exact-substring dedup to use uint32 token encoding.
#
# Upstream datatrove (<=0.6) encodes token IDs as np.uint16 (max 65535).
# VinaSmol's merged vocab is 55936 tokens — this technically fits but the
# separator bytes 0xFFFF (=65535 in uint16) can collide with valid token IDs,
# and any future vocab expansion would overflow.  Using uint32 avoids both
# problems.
#
# The patch replaces `prepare_doc` and `read_bytes` and must be applied
# before any ES pipeline stage runs.
#
# See: https://github.com/huggingface/datatrove/issues/121
# ---------------------------------------------------------------------------

_UINT32_PATCHED = False
_ORIGINAL_PREPARE_DOC = None
_ORIGINAL_READ_BYTES = None


def _prepare_doc_uint32(tokenizer, doc: str, rank: int, doc_id: int) -> bytes:
    """Encode a document's tokens as uint32 instead of uint16."""
    tokens = tokenizer.encode(doc).ids
    tokens = np.fromiter(tokens, dtype=np.uint32, count=len(tokens))
    b_doc = (
        b"\xff\xff\xff\xff"  # 4-byte separator (cannot collide with valid uint32 token)
        + struct.pack("<I", doc_id)
        + b"\xff\xff\xff\xff"
        + struct.pack("<I", rank)
        + tokens.tobytes()
    )
    return b_doc


# Separator size changes: original = 12 bytes (2+4+2+4), patched = 16 bytes (4+4+4+4)
SEPARATOR_BYTES_UINT32 = 16


def _read_bytes_uint32(x: bytes) -> list[int]:
    """Decode token bytes encoded with uint32."""
    return np.frombuffer(x[SEPARATOR_BYTES_UINT32:], dtype=np.uint32).tolist()


def apply_uint32_patch():
    """Monkey-patch datatrove's ES dedup to use uint32 token encoding.

    This patches the module-level functions ``prepare_doc`` and ``read_bytes``
    in ``datatrove.pipeline.dedup.exact_substrings``.  It also patches
    ``ESRangeRemover.remove_duplicate`` to decode with uint32.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _UINT32_PATCHED, _ORIGINAL_PREPARE_DOC, _ORIGINAL_READ_BYTES
    if _UINT32_PATCHED:
        return

    import datatrove.pipeline.dedup.exact_substrings as es_mod

    _ORIGINAL_PREPARE_DOC = es_mod.prepare_doc
    _ORIGINAL_READ_BYTES = es_mod.read_bytes

    es_mod.prepare_doc = _prepare_doc_uint32
    es_mod.read_bytes = _read_bytes_uint32
    es_mod.SEPARATOR_BYTES = SEPARATOR_BYTES_UINT32

    # Patch ESRangeRemover.remove_duplicate to use uint32 for decoding
    _original_remove_duplicate = ESRangeRemover.remove_duplicate

    def _remove_duplicate_uint32(self, doc, bytes_content):
        n_bytes = len(bytes_content)
        duplicates_ranges = self.get_duplicate_range(n_bytes)
        duplicates = []
        for byte_a, byte_b in duplicates_ranges:
            dup_sentence = self.tokenizer.decode(
                np.frombuffer(bytes_content[byte_a:byte_b], dtype=np.uint32).tolist()
            )
            duplicates.append(dup_sentence)

        if duplicates:
            text = doc.text
            for d in duplicates:
                text = text.replace(d, "")
            doc.text = text

        self.bytes_counter += len(bytes_content)

        if len(self.word_tokenizer.word_tokenize(doc.text)) < self.min_doc_words:
            return False
        return True

    ESRangeRemover.remove_duplicate = _remove_duplicate_uint32

    _UINT32_PATCHED = True
    logger.info("Applied uint32 monkey-patch to datatrove exact-substring dedup")


def revert_uint32_patch():
    """Revert the uint32 monkey-patch (useful for testing)."""
    global _UINT32_PATCHED, _ORIGINAL_PREPARE_DOC, _ORIGINAL_READ_BYTES
    if not _UINT32_PATCHED:
        return

    import datatrove.pipeline.dedup.exact_substrings as es_mod

    es_mod.prepare_doc = _ORIGINAL_PREPARE_DOC
    es_mod.read_bytes = _ORIGINAL_READ_BYTES
    es_mod.SEPARATOR_BYTES = SEPARATOR_BYTES  # Restore original 12

    _UINT32_PATCHED = False


# TODO: Possibly apply stronger rules for deduplication, such as:
# - Linewise filtering: Read more, sign-in, items in cart... (see RefinedWeb G.2)

# Hashing is massively parallelizable but we don't need to implement merging
# since Rensa is efficient enough for ~10 GB datasets.
class RensaBuildIndex(PipelineStep):
    """Build a MinHashLSH index with [Rensa](https://github.com/beowolx/rensa) as a backend.
    
    This pipeline step is single-threaded and should be run with one worker only.
    It eagerly processes its input documents all at once, has no side effect on the documents
    and returns `None`. This means you should write the result of the previous pipeline steps
    to disk, and read it back just after.

    # Example

    ```python
    rensa_index = RensaBuildIndex()
    pipeline = [
        JsonlReader(data_dir),
        rensa_index,            # Process all of the documents eagerly and discard them
        JsonlReader(data_dir),  # Stream all of the documents back into the pipeline
        ...
    ]
    LocalPipelineExecutor(pipeline, workers=1).run()
    # `rensa_index` can be reused by other pipeline steps or even from different pipelines
    ```
    """

    name = "🗂️ Rensa deduplication stage 1"

    def __init__(
            self,
            num_perm=128,
            seed=20250801,
            lsh_threshold: float = 0.8,
            num_bands: int = 16,
            final_jaccard_threshold: float = 0.85
        ):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_bands ({num_bands}) must divide num_perm ({num_perm}).")
    
        super().__init__()
        self.num_perm = num_perm
        self.seed = seed
        self.lsh_threshold = lsh_threshold
        self.num_bands = num_bands
        self.final_jaccard_threshold = final_jaccard_threshold
        self._minhashes = {}
        self._lsh_index = RMinHashLSH(
            threshold=lsh_threshold,
            num_perm=num_perm,
            num_bands=num_bands,
        )
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> None:
        if rank != 0:
            raise ValueError("Cannot run RensaBuildIndex with multiple workers")
        if not data:
            raise ValueError("Missing documents to index")
        for document in data:
            self.index(document)
    
    def index(self, document: Document):
        rminhash_obj = self.generate_minhash_signature(document.text)
        if document.id in self._minhashes:
            self.stat_update('bad_duplicated_doc_id')
        else:
            self._minhashes[document.id] = rminhash_obj
            self._lsh_index.insert(document.id, rminhash_obj)
    
    # Define a function to generate MinHash (works for RMinHash, CMinHash)
    def generate_minhash_signature(self, text: str):
        m = RMinHash(num_perm=self.num_perm, seed=self.seed)
        m.update(text.split())
        return m


# TODO: rehydration (might be less efficient since we're doing that on a subset...)
# https://github.com/huggingface/fineweb-2/tree/main/misc/rehydration
class RensaDeduplicate(BaseFilter):
    """Deduplicate documents with a MinHashLSH index implemented by [Rensa](https://github.com/beowolx/rensa).

    This pipeline step is single-threaded and needs `RensaBuildIndex` to be fully built
    before it processes any documents.

    # Example

    ```python
    rensa_index = RensaBuildIndex()
    pipeline = [
        JsonlReader(data_dir),
        rensa_index,
        JsonlReader(data_dir),
        RensaDeduplicate(rensa_index=rensa_index),
        ...
    ]
    ```

    # Performance

    Deduplication does not need to be distributed for ~10 GB datasets since Rensa is optimized
    enough for single-thread performance. See real-world benchmarks
    [here](https://github.com/beowolx/rensa?tab=readme-ov-file#large-scale-benchmark-salesforcewikitext-18-million-rows).
    """

    type = "🫂 - DEDUP"
    name = "🦀 Rensa deduplication stage 2"

    def __init__(self, rensa_index: RensaBuildIndex, exclusion_writer = None):
        """Initialize the pipeline step.

        Args:
            rensa_index (RensaBuildIndex): Must be the previous pipeline step.
            exclusion_writer (DiskWriter, optional): Exclusion writer.
        """
        super().__init__(exclusion_writer, batch_size=1)
        self.rensa_index = rensa_index
        self._to_remove = set()
        self._num_visited = 0
    
    def clear(self):
        logger.debug("Clearing Rensa index to free {} minhashes", self._num_visited)
        if not hasattr(self, '_to_remove'):
            raise RuntimeError("RensaBuildIndex did not eagerly process all of the documents")
        del self._to_remove
        del self.rensa_index._lsh_index
        del self.rensa_index._minhashes
    
    def increment_counter(self):
        """Free memory if the pipeline step is finished."""
        self._num_visited += 1
        #if self._num_visited == len(self.rensa_index._minhashes):
        #    self.clear()

    def filter(self, document: Document) -> bool | tuple[bool, str]:
        if document.id in self._to_remove:
            self.increment_counter()
            return False, "duplicate"

        query_minhash = self.rensa_index._minhashes[document.id]
        candidate_ids = self.rensa_index._lsh_index.query(query_minhash)

        for candidate_id in candidate_ids:
            if candidate_id == document.id or candidate_id in self._to_remove:
                continue
            
            candidate_minhash = self.rensa_index._minhashes[candidate_id]
            actual_jaccard = query_minhash.jaccard(candidate_minhash)

            if actual_jaccard >= self.rensa_index.final_jaccard_threshold:
                # Keep the item with the smaller original index
                # TODO: if possible, prioritize older document
                if document.id < candidate_id:
                    self._to_remove.add(candidate_id)
                else:
                    self._to_remove.add(document.id)
                    self.increment_counter()
                    return False, "duplicate"

        self.increment_counter()
        return True


# NOTE: uint16 encoding issue addressed by apply_uint32_patch() above.
# See: https://github.com/huggingface/datatrove/issues/121
class ESComputeRangesExternal(PipelineStep):
    """STAGE 2.5 of exact substring deduplication

    Runs the external scripts from https://github.com/google-research/deduplicate-text-datasets
    submodule. Requires Rust to be installed.

    Before using this stage, call ``apply_uint32_patch()`` to fix datatrove's
    uint16 token encoding limitation.
    """

    type = "🫂 - DEDUP"
    name = "🦀 - exact-substrings stage 2.5 (compute ranges in Rust)"

    def __init__(
            self,
            length_threshold: int = 100,
            data_folder: DataFolderLike = None,
            tmp_dir: Path = None,
            google_repo_path: Path = None,
            num_threads: int = 8,
            release_build: bool = True,
        ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        if google_repo_path is None:
            google_repo_path = Path(__file__).parent / "deduplicate-text-datasets"
        self.google_repo_path = google_repo_path
        self._tmp_dir = tmp_dir

        self.length_threshold = length_threshold
        self.num_threads = num_threads
        self.release_build = release_build

    def _fix_cargo_lock_edition(self):
        """Fix Cargo.lock resolver edition for older google-research repos.

        The deduplicate-text-datasets repo uses an old Cargo edition that may
        produce a lockfile incompatible with newer Rust toolchains.  If
        Cargo.toml exists but lacks ``edition``, we add ``edition = "2021"``.
        """
        cargo_toml = self.google_repo_path / "Cargo.toml"
        if not cargo_toml.exists():
            return
        content = cargo_toml.read_text()
        if 'edition' not in content:
            content = content.replace(
                '[package]',
                '[package]\nedition = "2021"',
                1,
            )
            cargo_toml.write_text(content)
            logger.info("Patched Cargo.toml to add edition = \"2021\"")
        # Remove stale Cargo.lock so it gets regenerated
        cargo_lock = self.google_repo_path / "Cargo.lock"
        if cargo_lock.exists():
            cargo_lock.unlink()
            logger.info("Removed stale Cargo.lock for regeneration")

    def setup(self):
        self.tmp_dir = tempfile.TemporaryDirectory(
            prefix="deduplicate-text-datasets",
            dir=self._tmp_dir,
        )
        self._fix_cargo_lock_edition()

        cargo_cmd = ["cargo", "build"]
        if self.release_build:
            cargo_cmd.append("--release")

        try:
            with self.track_time(unit="cargo_build"):
                cargo_build = subprocess.run(
                    cargo_cmd,
                    capture_output=True,
                    check=True,
                    cwd=self.google_repo_path,
                )
            logger.info(f"Successfully built {self.google_repo_path}")
            logger.info(f"Got cargo output:\n{cargo_build.stdout.decode()}")
        except CalledProcessError as e:
            logger.critical(
                "cargo command might not be installed or accessible, "
                "or cargo build failed.\n{}",
                e.stderr.decode()
            )
            raise

        subprocess.run(
            ["ln", "-s", "-f", self.tmp_dir.name, "./tmp"],
            capture_output=True,
            check=True,
            cwd=self.google_repo_path,
        )

    def run(self, data, rank = 0, world_size = 1):
        if rank != 0:
            raise ValueError(
                "deduplicate-text-datasets is already parallelized and doesn't need "
                "to be run with multiple Python workers"
            )
        self.setup()

        dataset_file = f"dataset{EH.stage_2_big_sequence}"
        byterange_file = dataset_file.replace(EH.stage_2_big_sequence, EH.stage_3_bytes_ranges)

        if not self.data_folder.exists(dataset_file):
            raise RuntimeError(
                f"Missing {self.data_folder.path}/{dataset_file}. "
                "Did you run the `ESMergeSequences` block in the pipeline before?"
            )
        dataset_file = str(Path(self.data_folder.path) / dataset_file)

        with self.track_time(unit="make_suffix_array"):
            try:
                make_suffix_array = subprocess.run(
                    ["python", "scripts/make_suffix_array.py", dataset_file],
                    capture_output=True,
                    check=True,
                    cwd=self.google_repo_path,
                )
                logger.info(
                    "Got output for make_suffix_array:\n"
                    f"{make_suffix_array.stdout.decode()}"
                )
            except CalledProcessError as e:
                logger.critical("make_suffix_array.py failed\n{}", e.stderr.decode())
                raise e

        # Cleaning up manually because the previous script fails
        _temp_big_sequences = glob.glob("out.table.bin.*", root_dir=self.tmp_dir.name)
        #for file in _temp_big_sequences:
        #    Path(file).unlink(missing_ok=True)
        
        cargo_run = ["cargo", "run"]
        if self.release_build:
            cargo_run.extend(["--release", "--"])

        try:
            with self.track_time(unit="self_similar"):
                self_similar = subprocess.run(
                    [
                        *cargo_run, "self-similar",
                        "--data-file", dataset_file,
                        "--length-threshold", str(self.length_threshold),
                        "--cache-dir", self.tmp_dir.name,
                        "--num-threads", str(self.num_threads),
                    ],
                    capture_output=True,
                    check=True,
                    cwd=self.google_repo_path,
                )
                logger.info(
                    "Got output for self-similar:\n"
                    f"{self_similar.stdout.decode()}"
                )
            with self.track_time(unit="collect"):
                with self.data_folder.open(byterange_file, "wb") as f:
                    subprocess.run(
                        [
                            *cargo_run, "collect",
                            "--data-file", dataset_file,
                            "--length-threshold", str(self.length_threshold),
                            "--cache-dir", self.tmp_dir.name,
                        ],
                        check=True,
                        cwd=self.google_repo_path,
                        stdout=f
                    )
                    logger.info("Done collecting self-similarity")
        except CalledProcessError as e:
            logger.critical("self-similar failed\n{}", e.stderr.decode())
            raise
        
        self.data_folder.move(f"00000{EH.stage_1_sequence}", f"dataset{EH.stage_1_sequence}")
        self.data_folder.move(f"00000{EH.stage_1_sequence_size}", f"dataset{EH.stage_1_sequence_size}")

        # Clean up intermediate files produced by suffix array construction
        for pattern in [
            f"*.{EH.stage_1_sequence}*",
            f"dataset{EH.stage_2_big_sequence}.*",
        ]:
            for file in glob.glob(pattern, root_dir=self.data_folder.path):
                (Path(self.data_folder.path) / file).unlink(missing_ok=True)

        tmp_symlink = Path(self.google_repo_path) / "tmp"
        if tmp_symlink.is_symlink():
            tmp_symlink.unlink()
        self.tmp_dir.cleanup()
        logger.info("Done")
