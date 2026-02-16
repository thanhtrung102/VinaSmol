from functools import partial
from pathlib import Path

from datasets import IterableDataset, load_dataset, interleave_datasets
from loguru import logger
import numpy as np
import typer

from . import DATA_DIR, DATA_DIR_EN_ANNEALING
from .datasets import SEED, estimate_dataset_size, to_sharded_parquet
from .preprocessing import DatasetNames

ANNEALING_OUT_DIR = DATA_DIR / "deduped" / "annealing-all"

def subset_from_filtered_data(data_dir: Path, subset_name: str) -> IterableDataset:
    """Retrieve a specific subset from the filtered and deduped corpus.

    Args:
        data_dir (Path): The directory of the corpus.
        subset_name (str): The name of the subset, i.e. `document['metadata']['origin']`.

    Returns:
        IterableDataset: the processed subset.
    """
    corpus = load_dataset(path=f'{data_dir}', split='train', streaming=True)
    return corpus.filter(
        lambda metadata: metadata['origin'] == subset_name,
        input_columns='metadata',
    )

def strip_metadata_to(row: dict, fields_to_keep: list[str]) -> dict:
    """Retain only specific metadata fields in order to normalize the row format.

    Args:
        row (dict): A dataset row.
        fields_to_keep (list[str], optional): The list of metadata fields to keep.
    """
    metadata = {
        key: row['metadata'].get(key)
        for key in fields_to_keep
    }
    return dict(
        id=row['id'],
        text=row['text'],
        metadata=metadata,
    )

def main(
    out_dir: Path = ANNEALING_OUT_DIR,
):
    vi_dir = DATA_DIR / "deduped" / "vi-all"
    en_dir = DATA_DIR / "deduped" / "en-all"


    wikipedia_en = subset_from_filtered_data(en_dir, DatasetNames.wikipedia_en)
    gutenberg_en = subset_from_filtered_data(en_dir, DatasetNames.gutenberg_en)
    wikipedia_vi = subset_from_filtered_data(vi_dir, DatasetNames.wikipedia_vi)
    binhvq_news_corpus = subset_from_filtered_data(vi_dir, DatasetNames.binhvq_news_corpus)

    # We don't filter those datasets as they are already high-quality and rather clean
    finemath_4plus = load_dataset(
        f"{DATA_DIR_EN_ANNEALING}/finemath_4plus",
        split='train',
        streaming=True,
    )
    stackmathqa = load_dataset(
        f"{DATA_DIR_EN_ANNEALING}/stackmathqa",
        split='train',
        streaming=True,
    )

    # CCVJ (CreativeCommons Vietnamese Journals) - high-quality academic papers
    ccvj_dir = DATA_DIR / "preprocessed" / "ccvj"
    ccvj = None
    if ccvj_dir.exists() and list(ccvj_dir.glob("*.parquet")):
        ccvj = load_dataset(
            "parquet",
            data_files=str(ccvj_dir / "*.parquet"),
            split='train',
            streaming=True,
        )
        logger.info("Loaded CCVJ dataset for annealing.")
    else:
        logger.warning("CCVJ dataset not found at {}. Skipping.", ccvj_dir)

    # Build the annealing mixture
    subsets = [
        wikipedia_en,
        gutenberg_en,
        wikipedia_vi,
        binhvq_news_corpus,
        finemath_4plus,
        stackmathqa,
    ]
    proportions = [
        0.1,   # wikipedia_en
        0.1,   # gutenberg_en
        0.15,  # wikipedia_vi
        0.2,   # binhvq_news_corpus
        0.2,   # finemath_4plus
        0.1,   # stackmathqa
    ]

    if ccvj is not None:
        subsets.append(ccvj)
        proportions.append(0.15)  # ccvj

    # Normalize proportions to sum to 1.0
    total = sum(proportions)
    proportions = [p / total for p in proportions]

    all_subsets: list[IterableDataset] = [
        subset.map(
            partial(strip_metadata_to, fields_to_keep=['origin', 'url']),
            remove_columns=list(set(subset.column_names) - {'id', 'text'})
        )
        for subset in subsets
    ]

    probabilities = np.zeros(len(proportions))
    for i in range(len(all_subsets)):
        probabilities[i] = proportions[i] / estimate_dataset_size(all_subsets[i])
    probabilities /= probabilities.sum()

    annealing_mix: IterableDataset = interleave_datasets(
        all_subsets,
        probabilities=probabilities,
        seed=SEED,
        stopping_strategy='all_exhausted',
    )
    # FIXME: only to cache dataset, triplicates disk usage
    cache_file = DATA_DIR / "annealing_mix.parquet"
    logger.info("Caching annealing mix to {}", cache_file)
    annealing_mix.take(20_000).to_parquet(f"{cache_file}")
    annealing_mix = load_dataset('parquet', data_files=f"{cache_file}")
    logger.info("Saving sharded annealing mix to {}", cache_file)
    to_sharded_parquet(annealing_mix, out_dir)


if __name__ == '__main__':
    typer.run(main)