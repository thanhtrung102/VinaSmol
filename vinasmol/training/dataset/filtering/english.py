
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.stats import (
    DocStats, LineStats, ParagraphStats,
    WordStats, SentenceStats, TokenStats, 
    CCNetPerplexityStats,
    TopKConfig, StatsMerger
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils.text import Languages


from vinasmol.training.dataset.preprocessing import DatasetNames
from ..deduplication import RensaBuildIndex, RensaDeduplicate
from ..normalization import Formatter

from . import (
    DATA_DIR_EN, MAIN_OUTPUT_DIR, FILTERING_OUTPUT_DIR, FILTERING_REMOVED_DIR,
    ES_DIR, LOGGING_DIR, STATS_DIR, SEED,
)
from .common import (
    JsonlShard, RetainMetadata, URLFilterWithWhitelist, LanguageFilterWithWhitelist,
    FlaggedWordsThresholdFilter,
)

top_k_config = TopKConfig(top_k_groups=["fqdn"], top_k=1_000)



CORPUS = "en-all"

DOMAIN_WHITELIST = [
    "wikipedia.org",
    "wikihow.com",
]
DOMAIN_WHITELIST += DatasetNames.PLACEHOLDER_URLS

CCNET_PPL_DATASET = "wikipedia" # oscar KenLM model is gigantic (34 GB for en.arpa.bin)
PPL_STAT_NAME = f"ccnet_perplexity_{CCNET_PPL_DATASET}_en"
PPL_STAT_DIR = f"{STATS_DIR}/{CORPUS}"
PPL_STAT_TOPK_CONFIG = TopKConfig(top_k_groups=["histogram"], top_k=10_000)
PPL_WHITELIST = [
    DatasetNames.gutenberg_en.placeholder_domain,
    DatasetNames.stackmathqa.placeholder_domain,
    DatasetNames.open_web_math.placeholder_domain,
    DatasetNames.mathpile_commercial.placeholder_domain,
    DatasetNames.olmocr_pes2o.placeholder_domain,
]

output_intermediate_1 = f"{FILTERING_OUTPUT_DIR}/output_1/{CORPUS}"
output_intermediate_2 = f"{FILTERING_OUTPUT_DIR}/output_2/{CORPUS}"
output_intermediate_3 = f"{FILTERING_OUTPUT_DIR}/output_3/{CORPUS}"
es_dir_en = f"{ES_DIR}/{CORPUS}"
final_output_dir = f"{MAIN_OUTPUT_DIR}/{CORPUS}/deduped"

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            str(DATA_DIR_EN),
            recursive=True,
            default_metadata={"dump": CORPUS},
        ),
        URLFilterWithWhitelist(
            extra_domains=None,
            extra_urls=None,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/1_url/{CORPUS}"),
            domain_whitelist=DOMAIN_WHITELIST,
            allow_no_url=True,
        ),
        Formatter(
            strip_whitespace=False, # NOTE: kept False to preserve code formatting in mixed content
            normalize_punctuation=False,
        ),
        LanguageFilterWithWhitelist(
            [Languages.english__latn], # Compatible with GlotLID, not with FT176LID
            language_threshold=0.65,
            backend="glotlid",
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/2_non_english/{CORPUS}"),
            domain_whitelist=[DatasetNames.mtet.placeholder_domain],
            allow_no_url=True,
        ),
        GopherRepetitionFilter(
            # default parameters
            dup_line_frac=0.3,
            dup_para_frac=0.3,
            dup_line_char_frac=0.2,
            dup_para_char_frac=0.2,
            top_n_grams=((2, 0.2), (3, 0.18), (4, 0.16)),
            dup_n_grams=((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.1)),
            language=Languages.english,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/3_gopher_rep/{CORPUS}"),
        ),
        GopherQualityFilter(
            min_doc_words=50,
            max_doc_words=100_000,
            min_avg_word_length=3,
            max_avg_word_length=10,
            max_bullet_lines_ratio=0.9,
            max_ellipsis_lines_ratio=0.3,
            min_stop_words=2,
            language=Languages.english,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/4_gopher_qual/{CORPUS}"),
        ),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            min_num_sentences=3,
            min_words_per_line=-1,
            filter_javascript=False,
            filter_curly_bracket=False,
            filter_policy=True,
            remove_citations=True,
            language=Languages.english,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/5_c4/{CORPUS}"),
        ),
        FineWebQualityFilter(
            line_punct_thr=0.1,
            line_punct_exclude_zero=False,
            short_line_thr=0.67,
            short_line_length=30,
            char_duplicates_ratio=0.1,
            new_line_ratio=0.3,
            language=Languages.english,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/6_fineweb_qual/{CORPUS}")
        ),
        # NOTE: flagged words filter may affect medical/scientific content; review if recall drops
        FlaggedWordsThresholdFilter(
            default_language='en',
            flagged_thr=0.01,
            keep_fraction=0.1,
            seed=SEED,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/7_c4_badwords/{CORPUS}")
        ),
        CCNetPerplexityStats(
            PPL_STAT_DIR,
            model_dataset=CCNET_PPL_DATASET,
            language='en', # ISO639-1
            histogram_round_digits=1,
            top_k_config=PPL_STAT_TOPK_CONFIG,
        ),
        JsonlWriter(output_intermediate_1),
    ],
    tasks=48,
    workers=16,
    logging_dir=f"{LOGGING_DIR}/base_processing/{CORPUS}",
)


rensa_index = RensaBuildIndex(
    num_perm=128,
    seed=SEED,
    lsh_threshold=0.8,
    num_bands=16,
    final_jaccard_threshold=0.85,
)

document_dedup_stage = LocalPipelineExecutor(
    pipeline=[
        StatsMerger(
            input_folder=PPL_STAT_DIR,
            output_folder=PPL_STAT_DIR,
            top_k_config=PPL_STAT_TOPK_CONFIG,
        ),
        JsonlReader(output_intermediate_1),
        rensa_index,
        JsonlReader(output_intermediate_1),
        RensaDeduplicate(
            rensa_index=rensa_index,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/8_dedup/{CORPUS}")
        ),
        # PerplexityFilterWithWhitelist(
        #     stats_dir=PPL_STAT_DIR,
        #     stat_name=PPL_STAT_NAME,
        #     quantiles=(0.2, 0.8),
        #     keep_fraction=0.1,
        #     seed=20250825,
        #     domain_whitelist=PPL_WHITELIST,
        #     exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/9_perplexity/{CORPUS}"),
        # ),
        RetainMetadata(fields_to_keep=[
            'origin',
            'url',
        ]),
        JsonlWriter(output_intermediate_2),
    ],
    logging_dir=f"{LOGGING_DIR}/minhash/{CORPUS}",
    depends=main_processing_executor,
)

reshard_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlShard(
            input_folder=output_intermediate_2,
            output_folder=output_intermediate_3,
            num_shards=48,
        ),
    ],
    depends=document_dedup_stage,
)

final_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_2),
        DocStats(
            f"{STATS_DIR}/docs",
            top_k_config=top_k_config,
        ),
        LineStats(
            f"{STATS_DIR}/lines",
            top_k_config=top_k_config,
        ),
        ParagraphStats(
            f"{STATS_DIR}/paragraphs",
            top_k_config=top_k_config,
        ),
        WordStats(
            f"{STATS_DIR}/words",
            language=Languages.english,
            top_k_config=top_k_config,
        ),
        SentenceStats(
            f"{STATS_DIR}/sentences",
            language=Languages.english,
            top_k_config=top_k_config,
        ),
        TokenStats(
            f"{STATS_DIR}/tokens",
            top_k_config=top_k_config,
        ),
        PIIFormatter(),
        ParquetWriter(final_output_dir),
        # TODO: shard each of them into their original datasets
    ],
    tasks=48,
    workers=16,
    logging_dir=f"{LOGGING_DIR}/final/{CORPUS}",
    depends=document_dedup_stage,
)



def main():
    main_processing_executor.run()
    document_dedup_stage.run()
    final_stage.run()

if __name__ == "__main__":
    main()
