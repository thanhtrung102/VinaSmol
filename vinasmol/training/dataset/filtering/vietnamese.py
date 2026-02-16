
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
from vinasmol.hfmodel import VINALLAMA_7B
from ..constants import (
    STOP_WORDS,
    FLAGGED_WORDS_SAILCRAFT,
)
from ..deduplication import RensaBuildIndex, RensaDeduplicate
from ..normalization import Formatter

from . import (
    DATA_DIR_VI, MAIN_OUTPUT_DIR, FILTERING_OUTPUT_DIR, FILTERING_REMOVED_DIR,
    ES_DIR, LOGGING_DIR, STATS_DIR as GLOBAL_STATS_DIR, SEED,
)
from .common import (
    JsonlShard, RetainMetadata, URLFilterWithWhitelist, LanguageFilterWithWhitelist,
    FlaggedWordsThresholdFilter
)

VIETNAMESE_TOKENIZER = VINALLAMA_7B.tokenizer

top_k_config = TopKConfig(top_k_groups=["fqdn"], top_k=1_000)



CORPUS = "vi-all"

DOMAIN_WHITELIST = [
    "wikipedia.org",
    "wikihow.com",
]
DOMAIN_WHITELIST += DatasetNames.PLACEHOLDER_URLS

# NOTE: oscar chosen as perplexity reference; wikipedia may have different bias profile
CCNET_PPL_DATASET = "oscar"
PPL_STAT_NAME = f"ccnet_perplexity_{CCNET_PPL_DATASET}_vi"
STATS_DIR = f"{GLOBAL_STATS_DIR}/{CORPUS}"
PPL_STAT_DIR = f"{STATS_DIR}/perplexity"
PPL_STAT_TOPK_CONFIG = TopKConfig(top_k_groups=["histogram"], top_k=10_000)
PPL_WHITELIST = [
    "vbpl.vn",
    DatasetNames.mtet.placeholder_domain,
    DatasetNames.ccvj.placeholder_domain, # NOTE: CCVJ uses a single placeholder domain for all papers
]

output_intermediate_1 = f"{FILTERING_OUTPUT_DIR}/output_1/{CORPUS}"
output_intermediate_2 = f"{FILTERING_OUTPUT_DIR}/output_2/{CORPUS}"
output_intermediate_3 = f"{FILTERING_OUTPUT_DIR}/output_3/{CORPUS}"
es_dir_vi = f"{ES_DIR}/{CORPUS}"
final_output_dir = f"{MAIN_OUTPUT_DIR}/deduped/{CORPUS}"

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            str(DATA_DIR_VI),
            recursive=True,
            default_metadata={"dump": CORPUS},
        ),
        # URL filtering for malicious, toxic and adult websites
        # Mainly for CulturaX, MADLAD-400 (NOTE: no upstream source list for MADLAD-400)
        # The default list includes 361 .vn / 4510795 banned websites, which is fairly incomplete
        # CulturaX has already undergone such filtering. Possibly remove MADLAD-400
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
            # It's okay to have English data in the corpus if it's about Vietnam.
            # Since it's a minority, it won't affect much of the proportions.
            # However, for consistency with the next filters, we restrict to only one language
            # as English.
            # NOTE: mtet is whitelisted below; VJOL language split deferred to future work
            [Languages.vietnamese__latn], # Compatible with GlotLID, not with FT176LID
            language_threshold=0.65,
            backend="glotlid", # NOTE: GlotLID model loaded per-worker (stateless, safe for parallelism)
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/2_non_vietnamese/{CORPUS}"),
            domain_whitelist=[DatasetNames.mtet.placeholder_domain],
            allow_no_url=True,
        ),

        # NOTE: tokenization is repeated across filters below; caching token spans would
        # improve throughput but requires datatrove pipeline changes

        GopherRepetitionFilter(
            # default parameters
            dup_line_frac=0.3,
            dup_para_frac=0.3,
            dup_line_char_frac=0.2,
            dup_para_char_frac=0.2,
            top_n_grams=((2, 0.2), (3, 0.18), (4, 0.16)),
            dup_n_grams=((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.1)),
            language=Languages.vietnamese,
            # NOTE: Vietnamese poems may be over-filtered by ngram thresholds;
            # coefficients may need language-specific tuning
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/3_gopher_rep/{CORPUS}"),
        ),
        GopherQualityFilter(
            min_doc_words=50, # Tokens counted by spaCy's Vietnamese tokenizer 
            max_doc_words=200_000, # Around 50 PDF pages
            min_avg_word_length=2,
            max_avg_word_length=10, # High for mixed language document (e.g. parallel translation)
            max_bullet_lines_ratio=0.9,
            max_ellipsis_lines_ratio=0.3,
            min_stop_words=2,
            stop_words=STOP_WORDS, # Some words in this list have two Spacy tokens. This is ok.
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/4_gopher_qual/{CORPUS}"),
        ),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            min_num_sentences=3,
            min_words_per_line=-1,
            # NOTE: JS filtering disabled to preserve programming blog content
            filter_javascript=False,
            filter_curly_bracket=False,
            filter_policy=True,
            remove_citations=True,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/5_c4/{CORPUS}"),
        ),
        FineWebQualityFilter(
            # Keep the lower bound low in order to tolerate many newlines in paragraph formatting
            line_punct_thr=0.1,
            line_punct_exclude_zero=False,
            # Improvement: merge short adjacent examples
            short_line_thr=0.67,
            short_line_length=20, # NOTE: may over-filter poems with short lines
            char_duplicates_ratio=0.1, # NOTE: empirically tuned; may need adjustment per-corpus
            new_line_ratio=0.3,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/6_fineweb_qual/{CORPUS}")
        ),
        # NOTE: flagged words filter may affect medical/scientific content; review if recall drops
        FlaggedWordsThresholdFilter(
            default_language='vi',
            language_flagged_words_override=FLAGGED_WORDS_SAILCRAFT,
            flagged_thr=0.01,
            keep_fraction=0.1,
            seed=SEED,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/7_c4_badwords/{CORPUS}")
        ),
        # Compute perplexity for filtering
        # NOTE: single perplexity model; multi-model clustering (cf. The Pile 6.2) deferred
        # NOTE: ccnet KenLM model may bias against long documents or domain-specific text
        CCNetPerplexityStats(
            PPL_STAT_DIR,
            model_dataset=CCNET_PPL_DATASET,
            language='vi', # ISO639-1
            histogram_round_digits=1,
            top_k_config=PPL_STAT_TOPK_CONFIG,
        ),

        # Other filters: SamplerFilter, datatrove.pipeline.decont.NGramsDecontFilter

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
        #PerplexityFilterWithWhitelist(
        #    stats_dir=PPL_STAT_DIR,
        #    stat_name=PPL_STAT_NAME,
        #    quantiles=(0.2, 0.8),
        #    keep_fraction=0.1,
        #    seed=20250825,
        #    domain_whitelist=PPL_WHITELIST,
        #    exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/9_perplexity/{CORPUS}"),
        #),
        RetainMetadata(fields_to_keep=[
            'origin',
            'url',
        ]),
        JsonlWriter(output_intermediate_2),
    ],
    logging_dir=f"{LOGGING_DIR}/minhash/{CORPUS}",
    depends=main_processing_executor,
)

tasks_sequence_dedup = 16

# sequence_dedup_stage_1 = LocalPipelineExecutor(
#     pipeline=[
#         JsonlReader(output_intermediate_2),
#         ESDatasetToSequence(
#             output_folder=es_dir_vi,
#             tokenizer_name_or_path=VIETNAMESE_TOKENIZER,
#         ),
#     ],
#     workers=tasks_sequence_dedup,
#     tasks=tasks_sequence_dedup,
#     logging_dir=f"{LOGGING_DIR}/es/1/{CORPUS}",
#     depends=document_dedup_stage,
# )

# sequence_dedup_stage_2 = LocalPipelineExecutor(
#     pipeline=[
#         ESMergeSequences(
#             data_folder=es_dir_vi,
#             tasks_stage_1=tasks_sequence_dedup,
#         ),
#     ],
#     logging_dir=f"{LOGGING_DIR}/es/2/{CORPUS}",
#     depends=sequence_dedup_stage_1,
# )

# external_dedup_stage_3 = LocalPipelineExecutor(
#     pipeline=[
#         ESComputeRangesExternal(
#             length_threshold=100,
#             data_folder=es_dir_vi,
#             num_threads=16,
#         ),
#     ],
#     logging_dir=f"{LOGGING_DIR}/es/3/{CORPUS}",
#     depends=sequence_dedup_stage_2,
# )

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
        JsonlReader(output_intermediate_3),
        # ESRangeRemover(
        #     min_doc_words=50,
        #     sequence_folder=es_dir_vi,
        #     tokenizer_name_or_path=VIETNAMESE_TOKENIZER,
        #     language=Languages.vietnamese,
        # ),

        # NOTE: datatrove's PIIFormatter uses regex-based PII removal; scrubadub is
        # more thorough but significantly slower — acceptable for this corpus size
        PIIFormatter(),
        ParquetWriter(final_output_dir),

        # TODO: shard each of them into their original datasets (for finegrained mixture)
        # TODO: compute stats in a separate stage

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
        # NOTE: word length stats are approximate for Vietnamese (spaces within words);
        # syllable-level stats would be more meaningful but require VnCoreNLP or similar
        WordStats(
            f"{STATS_DIR}/words",
            stop_words=set(STOP_WORDS),
            language=Languages.vietnamese,
            top_k_config=top_k_config,
        ),
        SentenceStats(
            f"{STATS_DIR}/sentences",
            language=Languages.vietnamese,
            top_k_config=top_k_config,
        ),
        TokenStats(
            f"{STATS_DIR}/tokens",
            tokenizer_name_or_path=VIETNAMESE_TOKENIZER,
            top_k_config=top_k_config,
        ),
    ],
    tasks=48,
    workers=16,
    logging_dir=f"{LOGGING_DIR}/final/{CORPUS}",
    depends=reshard_stage,
)

def main():
    main_processing_executor.run()
    document_dedup_stage.run()
    # sequence_dedup_stage_1.run()
    # sequence_dedup_stage_2.run()
    # external_dedup_stage_3.run()
    reshard_stage.run()
    final_stage.run()

if __name__ == "__main__":
    main()
