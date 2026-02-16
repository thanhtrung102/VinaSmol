from pathlib import Path

from datasets import Dataset, IterableDataset, enable_progress_bars, load_dataset
from loguru import logger
from tqdm import tqdm

from ...hfmodel import BASE_MODEL, SMOLLM2, LUCIE
from . import DATA_DIR, DATA_DIR_CODE, DATA_DIR_EN, DATA_DIR_EN_ANNEALING, DATA_DIR_VI
from .preprocessing import (
    convert_mediawiki_to_md, format_vbpl_md,
    gutenberg_is_license_acceptable, clean_gutenberg_text,
    NormalizeCols,
)

LOAD_KWARGS = dict(split="train", streaming=True)
# TODO: download the sampled subset as a Dataset and then map in parallel
NUM_PROC = 16
SEED = 20250801
SHARD_SIZE = 100_000_000

MIN_PYTHON_EDU_SCORE = 3

def estimate_dataset_size(dataset: Dataset | IterableDataset, text_column: str = 'text') -> int:
    """Estimate the number of bytes to store the text column as compressed Parquet."""
    n_chars = 0.0
    # TODO: https://github.com/huggingface/datasets/pull/5533#issuecomment-2498180088
    batches = dataset.select_columns(text_column).iter(batch_size=1_000)
    n_batches = None
    if hasattr(dataset, '__len__'):
        n_batches = (len(dataset) - 1) // 1_000 + 1
    for batch in tqdm(batches, desc="Estimate dataset size", total=n_batches):
        texts = batch[text_column]
        n_chars += sum(len(t) for t in texts)

    # 1 char = 2 bytes in general
    return int(0.5 * n_chars)

def to_sharded_parquet(
        dataset: Dataset | IterableDataset,
        dir: str | Path,
        shard_size_bytes: int = SHARD_SIZE,
        main_column: str = 'text',
    ) -> list[Path]:
    """Save a dataset sharded into Parquet files.

    Args:
        dataset (Dataset): The dataset.
        dir (Path): The directory to write the Parquet files.
        shard_size_bytes (int, optional): The approximate number of bytes for each shard.
            Defaults to `SHARD_SIZE`.

    Returns:
        files (list[Path]): The list of shards paths.
    """
    if isinstance(dataset, IterableDataset):
        # FIXME: this is very memory-inefficient
        logger.warning("Converting IterableDataset to list")
        dataset = Dataset.from_list(list(dataset), features=dataset.features)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, estimate_dataset_size(dataset, text_column=main_column) // shard_size_bytes)

    files = []
    for i in range(num_shards):
        shard = dataset.shard(num_shards, i)
        file = dir / f"part-{i:04}.parquet"
        shard.to_parquet(file)
        files.append(file)
    return files

def load_sharded_parquet(name: str, data_dir: str | Path = DATA_DIR) -> Dataset:
    data_dir = Path(data_dir)
    parquet_files = [str(f) for f in data_dir.glob(f"**/{name}/*.parquet")]
    return load_dataset('parquet', split='train', data_files=parquet_files)


# FIXME: datasets shards are too large and target data is smaller than one shard
# TODO: possibly perform map after shuffle/take/Dataset.from_generator?

def download_english_datasets(data_dir: Path):
    if BASE_MODEL == SMOLLM2:
        # 28B tokens, 39M rows
        cosmopedia_v2 = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", **LOAD_KWARGS)
        cosmopedia_v2 = (cosmopedia_v2
            .map(NormalizeCols.cosmopedia_v2, remove_columns=cosmopedia_v2.column_names)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(1_000_000)
        )
        to_sharded_parquet(cosmopedia_v2, data_dir / "cosmopedia_v2")

        # ~400 GB, 190M rows
        fineweb_edu_dedup = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", **LOAD_KWARGS)
        fineweb_edu_dedup = (fineweb_edu_dedup
            .map(NormalizeCols.fineweb_edu_dedup, remove_columns=fineweb_edu_dedup.column_names)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(1_000_000)
        )
        to_sharded_parquet(fineweb_edu_dedup, data_dir / "fineweb_edu")

        # ~20GB, 7.03M rows
        wikipedia_en = load_dataset("omarkamali/wikipedia-monthly", "20250702.en", **LOAD_KWARGS)
        wikipedia_en = (wikipedia_en
            .map(convert_mediawiki_to_md, fn_kwargs=dict(lang='en'))
            .map(NormalizeCols.wikipedia_en, remove_columns=wikipedia_en.column_names)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(200_000)
        )
        to_sharded_parquet(wikipedia_en, data_dir / "wikipedia_en")

        # Datasets used in the pretraining phase of Lucie (mostly orthogonal with SmolLM-Corpus)

        # 1B tokens, 1M rows, a bit low quality...
        #claire_en = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Claire-en", revision="v1.2", **LOAD_KWARGS)

        # 5.5B tokens, 56k rows
        gutenberg_en = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Gutenberg-en", revision="v1.2", **LOAD_KWARGS)
        gutenberg_en = (gutenberg_en
            .map(clean_gutenberg_text)
            .filter(gutenberg_is_license_acceptable)
            .map(NormalizeCols.gutenberg_en, remove_columns=gutenberg_en.column_names)
            .shuffle(seed=SEED, buffer_size=1_000)
            .take(5_000)
        )
        to_sharded_parquet(gutenberg_en, data_dir / "gutenberg_en")

        # 70M tokens, 10k rows
        #europarl_en = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Europarl-en", revision="v1.2", **LOAD_KWARGS)
        
        # Avoid training a very small model on too many programming languages
        #lucie_training_code = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code", revision="v1.2", **load_kwargs)

    elif BASE_MODEL == LUCIE:
        #config_names = list(load_dataset_builder("OpenLLM-France/Lucie-Training-Dataset").builder_configs)
        # Downsample RedPajamaV2
        #lucie_training = load_dataset("OpenLLM-France/Lucie-Training-Dataset", revision="v1.2", **LOAD_KWARGS)
        raise NotImplementedError(BASE_MODEL)
    else:
        raise NotImplementedError(BASE_MODEL)

def download_english_annealing_datasets(data_dir: Path):
    # 10B tokens, 7M rows
    finemath_4plus = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", **LOAD_KWARGS)
    finemath_4plus = (finemath_4plus
        .map(NormalizeCols.finemath_4plus, remove_columns=finemath_4plus.column_names)
        .shuffle(seed=SEED, buffer_size=1000)
        .take(100_000)
    )
    to_sharded_parquet(finemath_4plus, data_dir / "finemath_4plus")

    # 42B tokens, 40M rows
    # Higher-quality and up-to-date pes2o dataset with proper formatting
    olmocr_pes2o = load_dataset("allenai/olmOCR-pes2o-0225", **LOAD_KWARGS)
    olmocr_pes2o = (olmocr_pes2o
        .map(NormalizeCols.olmocr_pes2o, remove_columns=olmocr_pes2o.column_names)
        .shuffle(seed=SEED, buffer_size=1000)
        .take(500_000)
    )
    to_sharded_parquet(olmocr_pes2o, data_dir / "olmocr_pes2o")

    # ~200MB, 200k rows
    stackmathqa = load_dataset("math-ai/StackMathQA", "stackmathqa200k", split="train")
    stackmathqa = (stackmathqa
        .map(NormalizeCols.stackmathqa, remove_columns=stackmathqa.column_names)
        .shuffle(seed=SEED)
    )
    to_sharded_parquet(stackmathqa, data_dir / "stackmathqa")

    # ~400GB, ~300M rows
    # Temporarily excluded because of the large download
    #flan_v2 = load_dataset("SirNeural/flan_v2", **LOAD_KWARGS)
    #(flan_v2
    #    .map(NormalizeCols.flan_v2, remove_columns=flan_v2.column_names)
    #    .shuffle(seed=SEED, buffer_size=1000)
    #    .take(500_000)
    #    .to_parquet(data_dir / "flan_v2.parquet")
    #)

def download_code_datasets(data_dir: Path):
    # 5 GB, 3M rows
    # Alternative: "meryyllebr543/stack-edu-huggingface", "python" (25M rows), one shard
    starcoder_python_edu = load_dataset(
        "JanSchTech/starcoderdata-python-edu-lang-score",
        **LOAD_KWARGS,
        columns=[
            'max_stars_repo_path',
            'max_stars_repo_name',
            'id',
            'language', # TODO: use language=='vi' for finetuning (around 100 examples)
            'content_cleaned',
            'edu_score',
        ]
    )
    starcoder_python_edu = (starcoder_python_edu
        .filter(lambda row: round(row['edu_score']) >= MIN_PYTHON_EDU_SCORE)
        .map(NormalizeCols.starcoder_python_edu, remove_columns=starcoder_python_edu.column_names)
        .shuffle(seed=SEED, buffer_size=5_000)
        .take(500_000)
    )
    to_sharded_parquet(starcoder_python_edu, data_dir / "starcoder_python_edu")

    #lucie_python = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code-python", revision="v1.2", split='train', streaming=True)


def download_vietnamese_datasets(data_dir: Path):
    # ~ 1 GB, 1.3M rows
    wikipedia_vi = load_dataset("omarkamali/wikipedia-monthly", "20250702.vi", split="train")
    wikipedia_vi = (wikipedia_vi
        .map(convert_mediawiki_to_md, fn_kwargs=dict(lang='vi'))
        .map(NormalizeCols.wikipedia_vi, remove_columns=wikipedia_vi.column_names, num_proc=NUM_PROC)
        .shuffle(seed=SEED)
    )
    to_sharded_parquet(wikipedia_vi, data_dir / "wikipedia_vi")

    # 59 GB (uncompressed?), 4M rows
    fineweb2_hq = load_dataset("epfml/FineWeb2-HQ", "vie_Latn", **LOAD_KWARGS)
    fineweb2_hq = (fineweb2_hq
        .map(NormalizeCols.fineweb2_hq, remove_columns=fineweb2_hq.column_names)
        .shuffle(seed=SEED, buffer_size=1000)
        .take(1_000_000)
    )
    to_sharded_parquet(fineweb2_hq, data_dir / "fineweb2_hq")

    # 55 B tokens, 58M rows
    # Possibly add https://huggingface.co/datasets/ontocord/CulturaY (data from Internet Archive)
    cultura_x = load_dataset("uonlp/CulturaX", "vi", token=True, **LOAD_KWARGS)
    cultura_x = (cultura_x
        .map(NormalizeCols.culturax, remove_columns=cultura_x.column_names)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
    )
    to_sharded_parquet(cultura_x, data_dir / "cultura_x")
    # 49 B tokens, 93M rows
    madlad400 = load_dataset("Symato/madlad-400_vi", split="train", streaming=True)
    madlad400 = (madlad400
        .map(NormalizeCols.madlad400, remove_columns=madlad400.column_names)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
    )
    to_sharded_parquet(madlad400, data_dir / "madlad400")

    # 4 GB, 19M rows
    binhvq_news = load_dataset("bigscience-data/roots_vi_binhvq_news_corpus", token=True, **LOAD_KWARGS)
    binhvq_news = (binhvq_news
        .map(NormalizeCols.binhvq_news, remove_columns=binhvq_news.column_names)
        .shuffle(seed=SEED, buffer_size=100_000)
        .take(10_000_000)
    )
    to_sharded_parquet(binhvq_news, data_dir / "binhvq_news_corpus")

    # 800 MB, 8M rows
    # Can slightly overlap with vbpl and vjol
    mtet = load_dataset("phongmt184172/mtet", split="train")
    mtet = (mtet
        .map(NormalizeCols.mtet, remove_columns=mtet.column_names, num_proc=NUM_PROC)
        .shuffle(seed=SEED)
    )
    to_sharded_parquet(mtet, data_dir / "mtet")

    # Muennighoff/flores200
    # ~ 100k words
    # Aligned professional translation data (useful if the base model is multilingual)


    # Official law texts (version: June 2025)
    # ~ 300 MB, 62k rows, mostly clean
    # Don't use streaming mode in order to avoid Arrow column type inference errors
    vbpl = load_dataset("doanhieung/vbpl", split="train")
    vbpl = (vbpl
        .map(format_vbpl_md, num_proc=NUM_PROC)
        .map(NormalizeCols.vbpl, remove_columns=vbpl.column_names)
        .shuffle(seed=SEED)
    )
    to_sharded_parquet(vbpl, data_dir / "vbpl")

def download_vietnamese_annealing_datasets(data_dir: Path):
    """Download and prepare Vietnamese annealing datasets including CCVJ.

    CCVJ (CreativeCommons Vietnamese Journals) must be pre-converted to
    Parquet format using the ccvj package before running this function.
    See ccvj/src/ccvj/README.md for instructions.
    """
    ccvj_parquet_dir = Path("ccvj/src/ccvj/data/ccvj/parquet")
    if not ccvj_parquet_dir.exists() or not list(ccvj_parquet_dir.glob("*.parquet")):
        logger.warning(
            "CCVJ Parquet files not found at {}. "
            "Run 'uv run python -m ccvj.convert' first. Skipping CCVJ.",
            ccvj_parquet_dir,
        )
        return

    ccvj_ds = load_dataset(
        "parquet",
        data_files=str(ccvj_parquet_dir / "*.parquet"),
        split="train",
        streaming=True,
    )
    ccvj_ds = ccvj_ds.map(NormalizeCols.ccvj, remove_columns=ccvj_ds.column_names)
    to_sharded_parquet(ccvj_ds, data_dir / "ccvj")


if __name__ == "__main__":
    enable_progress_bars()
    DATA_DIR_VI.mkdir(parents=True, exist_ok=True)
    DATA_DIR_EN.mkdir(parents=True, exist_ok=True)
    DATA_DIR_CODE.mkdir(parents=True, exist_ok=True)
    DATA_DIR_EN_ANNEALING.mkdir(parents=True, exist_ok=True)

    download_vietnamese_datasets(DATA_DIR_VI)
    download_english_datasets(DATA_DIR_EN)
    download_code_datasets(DATA_DIR_CODE)
    download_english_annealing_datasets(DATA_DIR_EN_ANNEALING)
