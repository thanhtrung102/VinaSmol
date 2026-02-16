import json
from pathlib import Path
import re

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.filters import (
    C4BadWordsFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.stats.merger import MetricStatsDict
from datasets import load_dataset
from loguru import logger
import numpy as np
from numpy.random import default_rng
from tldextract import TLDExtract


DEFAULT_FLAGGING_SCORE = 3

class JsonlShard(PipelineStep):
    type = "💽 - WRITER"

    name = "Shard JSONL file"

    def __init__(self, input_folder: str, output_folder: str, num_shards: int):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_shards = num_shards
    
    def run(self, data, rank = 0, world_size = 1):
        with self.track_time("shard_jsonl"):
            ds = load_dataset('json', data_dir=self.input_folder, split='train')
            for i in range(self.num_shards):
                shard = ds.shard(self.num_shards, i)
                file = Path(self.output_folder) / f"part-{i:04}.jsonl.gz"
                shard.to_json(f"{file}")

class RetainMetadata(PipelineStep):
    type = "✂️ - FORMAT"

    name = "Retain metadata"

    def __init__(self, fields_to_keep: list[str]):
        super().__init__()
        self.fields_to_keep = fields_to_keep

    def run(self, data, rank = 0, world_size = 1):
        for document in data:
            document.metadata = {
                key: document.metadata.get(key)
                for key in self.fields_to_keep
            }
            yield document

class DomainWhitelistMixin(BaseFilter):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = f"{cls.name} (with 📋 whitelist)"

    def __init__(
            self,
            *args,
            domain_whitelist: list[str] = None,
            allow_no_url: bool = True,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.domain_whitelist = domain_whitelist or []
        # TODO: allow domain suffix wildcard
        self.allow_no_url = allow_no_url

        if not hasattr(self, 'tldextractor'):
            self.tldextractor = TLDExtract()
    
    def filter(self, document: Document):
        url = document.metadata.get("url")
        if not url:
            if self.allow_no_url:
                # Accept sources which have no URL by design
                return True
            raise ValueError("Missing document 'url' field")

        url_info = self.tldextractor(url)
        if url_info.top_domain_under_public_suffix in self.domain_whitelist:
            return True

        return super().filter(document)

class URLFilterWithWhitelist(DomainWhitelistMixin, URLFilter):
    """Perform filtering on URLs, whith a domain whitelist."""


class LanguageFilterWithWhitelist(DomainWhitelistMixin, LanguageFilter):
    """Perform filtering on URLs, whith a domain whitelist."""

class FlaggedWordsThresholdFilter(C4BadWordsFilter):
    """Filter documents that contain too many flagged words.
    
    Contrary to the C4 one, this filter tolerates a proportion of flagged words.
    """

    name = "🚩 Flagged Words Threshold"

    def __init__(
        self,
        default_language: str,
        language_flagged_words_override: list[str] | dict[str, int] = None,
        flagged_thr: float = 0.1,
        keep_fraction: float = 0.1,
        seed: int = None,
        exclusion_writer = None,
    ):
        """Initialize the filter.

        Args:
            default_language (str): the default language for the badwords filter.
            language_flagged_words_override (list[str] | dict [str, int], optional):
                Flagged words for the specified language.
            flagged_thr (float, optional): Maximum proportion of flagged words.
                The ratio is computed by counting the number of syllables.
            keep_fraction (float, optional): Proportion of filtered documents to keep.
        """
        super().__init__(
            keep_fraction=keep_fraction,
            seed=seed,
            default_language=default_language,
            exclusion_writer=exclusion_writer,
        )
        flagged = language_flagged_words_override
        if flagged is not None:
            if default_language in self._badwords_regex:
                logger.warning(f"Overriding badwords for language {default_language}")

            if isinstance(flagged, list):
                scores = {w: DEFAULT_FLAGGING_SCORE for w in flagged}
            elif isinstance(flagged, dict):
                scores = flagged
            else:
                raise TypeError(f"Expected list or dict for flagged words, got {type(flagged)}")
            escaped_words = [re.escape(w.lower()) for w in flagged]
            # Must span over complete syllables
            flagged_re = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(escaped_words)))
            self._badwords_regex[default_language] = flagged_re
            self._flagged_words_scores = scores
        else:
            self._flagged_words_scores = {}

        self.flagged_thr = flagged_thr

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lang: str = doc.metadata.get("language", self.default_language)
        
        badwords_regex = self._get_badwords(lang[:2]) # ISO 639-1
        if badwords_regex is None:
            self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
            return True
        
        flagged = badwords_regex.findall(doc.text.lower())
        scores = [
            self._flagged_words_scores.get(word) or DEFAULT_FLAGGING_SCORE
            for word in flagged
        ]
        # Exponential weight scale in (0, 1]
        weights = [2 ** (x - 5) for x in scores]
        total_flagged_weight = sum(
            weight * word.count(' ')
            for weight, word in zip(weights, flagged)
        )
        num_syllables = len(doc.text.split())
        ratio = total_flagged_weight / num_syllables
        if ratio > self.flagged_thr:
            self.stat_update(f"over_flagged_words_threshold_{lang}")
            if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                return True
            return False, f"too_many_flagged_words_{lang}"
        return True

class StatQuantileFilter(BaseFilter):
    """Filter documents with statistics outside a quantile.

    # Example

    ```python
    stage_1 = LocalPipelineExecutor(
        pipeline=[
            ...,
            CCNetPerplexityStats(
                output_folder="stats/perplexity",
                model_dataset="oscar",
                histogram_round_digits=1,
                top_k_config=TopKConfig(top_k_groups=["histogram"]),
            )
            ...
        ]
    )
    stage_2 = LocalPipelineExecutor(
        pipeline = [
            ...,
            StatQuantileFilter(
                stats_dir="stats/perplexity",
                stat_name="ccnet_perplexity_oscar_vi",
                quantile=0.5,
                is_better='lower',
            )
        ]
    )
    ```
    """

    name = "🔣 Stat quantile"

    def __init__(
            self,
            stats_dir: str,
            stat_name: str,
            quantiles: tuple[float, float] = (0.2, 0.8),
            keep_fraction: float = 0.1,
            seed = 20250825,
            exclusion_writer = None,
        ):
        assert 0.0 <= quantiles[0] < quantiles[1] <= 1.0, f"Invalid quantiles: {quantiles}"

        super().__init__(exclusion_writer, batch_size=1)
        self.stats_dir = stats_dir
        self.stat_name = stat_name
        self._metric_stats: MetricStatsDict = None
        self._min_thr: float = None
        self._max_thr: float = None
        self.quantiles = quantiles
        self.keep_fraction = keep_fraction
        self.uniform = default_rng(seed).uniform
    
    @property
    def metric_stats(self) -> MetricStatsDict:
        if self._metric_stats is None:
            metric_file = f"{self.stats_dir}/histogram/{self.stat_name}/metric.json"
            self._metric_stats = MetricStatsDict.from_dict(json.loads(Path(metric_file).read_text()))
            values = []
            for stat_value_key, metric_stat in self._metric_stats.items():
                for _ in range(metric_stat.total):
                    values.append(float(stat_value_key))
            self._min_thr = np.quantile(values, self.quantiles[0])
            self._max_thr = np.quantile(values, self.quantiles[1])
        return self._metric_stats

    def filter(self, doc: Document):
        _ = self.metric_stats
        try:
            metric = doc.metadata[self.stat_name]
        except KeyError as e:
            raise RuntimeError(f"Metric {self.stat_name} has not been computed") from e

        condition = self._min_thr <= metric <= self._max_thr
        return condition or self.uniform() <= self.keep_fraction


class PerplexityFilterWithWhitelist(DomainWhitelistMixin, StatQuantileFilter):
    """Perform perplexity quantile-based filtering, whith a domain whitelist."""

    name = "🤔 Perplexity filter"
