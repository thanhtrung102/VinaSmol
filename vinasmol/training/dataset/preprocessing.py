from enum import StrEnum
import json
import random
import re
from threading import Lock

from datatrove.utils.text import Languages
import pypandoc
from vinasmol.hfmodel import LUCIE, SMOLLM2


def filter_keys(d: dict, keys: list[str]) -> dict:
    return {k: d[k] for k in keys}

_WIKI_INFOBOX_RE = re.compile(r"\{\{Infobox[^}]*\}\}", flags=re.DOTALL)
_WIKI_CATEGORY_RE = re.compile(r"\[\[Category:[^\]]+\]\]")
_WIKI_FILE_RE = re.compile(r"\[\[(?:File|Image|Tập tin|Hình):[^\]]+\]\]")
_WIKI_EN_END_SECTIONS_RE = re.compile(
    r"^(==\s*(?:See also|References|External links|Further reading|Notes|Bibliography)\s*==).*",
    flags=re.DOTALL | re.MULTILINE,
)
_WIKI_VI_END_SECTIONS_RE = re.compile(
    r"^(==\s*(?:Liên kết ngoài|Tham khảo|Thể loại|Xem thêm|Đọc thêm|Chú thích)\s*==).*",
    flags=re.DOTALL | re.MULTILINE,
)

def _clean_mediawiki(mediawiki: str, end_sections_re: re.Pattern) -> str:
    """Common MediaWiki cleanup before conversion."""
    # Remove infoboxes, file embeds, categories
    mediawiki = _WIKI_INFOBOX_RE.sub("", mediawiki)
    mediawiki = _WIKI_FILE_RE.sub("", mediawiki)
    mediawiki = _WIKI_CATEGORY_RE.sub("", mediawiki)
    # Remove trailing sections (references, see also, etc.)
    mediawiki = end_sections_re.sub("", mediawiki)
    return mediawiki


def convert_en_wiki_to_md(title: str, mediawiki: str) -> str:
    """Format an English Wikipedia page to Markdown.

    Args:
        title: the article title.
        mediawiki: the page content, in MediaWiki format.

    Returns:
        str: The formatted content as Github-Flavored Markdown.
    """
    mediawiki = _clean_mediawiki(mediawiki, _WIKI_EN_END_SECTIONS_RE)
    md = pypandoc.convert_text(mediawiki, "gfm", format="mediawiki")
    md = f"# {title}\n\n{md}"
    return md.strip()


def convert_vi_wiki_to_md(title: str, mediawiki: str) -> str:
    """Format a Vietnamese Wikipedia page to Markdown.

    Args:
        title: the article title.
        mediawiki: the page content, in MediaWiki format.

    Returns:
        str: The formatted content as Github-Flavored Markdown.
    """
    mediawiki = _clean_mediawiki(mediawiki, _WIKI_VI_END_SECTIONS_RE)
    md = pypandoc.convert_text(mediawiki, "gfm", format="mediawiki")
    md = f"# {title}\n\n{md}"
    return md.strip()

_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^\}]+\}\}", flags=re.MULTILINE)
_WIKI_END_SECTIONS_TO_REMOVE_RE = re.compile(
    r"^(Liên kết ngoài|Tham khảo|Thể loại|Xem thêm|Đọc thêm).*",
    flags = re.DOTALL | re.MULTILINE,
)
_WIKI_TEMPLATE_ARTIFACTS_RE = re.compile(r"^\s+(?:\{\{|\|).+\n", flags=re.MULTILINE)
_WIKI_EMPTY_PARENS_RE = re.compile(r"\([ ,;]*\)")

def convert_mediawiki_to_md(row: dict, lang: str = 'en') -> dict:
    """Replace `'text'` by converting the `'raw_mediawiki'` to Markdown.

    Uses pypandoc for full MediaWiki→Markdown conversion when 'raw_mediawiki'
    is available, otherwise falls back to regex-based cleaning of 'text'.
    """
    if 'raw_mediawiki' in row and row['raw_mediawiki']:
        match lang:
            case 'en':
                row['text'] = convert_en_wiki_to_md(row['title'], row['raw_mediawiki'])
            case 'vi':
                row['text'] = convert_vi_wiki_to_md(row['title'], row['raw_mediawiki'])
            case _:
                row['text'] = convert_en_wiki_to_md(row['title'], row['raw_mediawiki'])
    else:
        # Fallback: regex-based cleaning when raw MediaWiki is not available
        for pat in [
            _WIKI_TEMPLATE_RE,
            _WIKI_TEMPLATE_ARTIFACTS_RE,
            _WIKI_EMPTY_PARENS_RE,
            _WIKI_END_SECTIONS_TO_REMOVE_RE,
        ]:
            row['text'] = pat.sub("", row['text'])
    return row

_MD_LINK_RE = re.compile(r"\[([^\[\]]+)\]\(([^)]+)\)")

_ABSTRACT_RE = re.compile(r"^Abstract", flags=re.MULTILINE)
_REFERENCES_RE = re.compile(r"^References|^\*\*References", flags=re.MULTILINE)

def replace_md_links_with_text(
        markdown_content: str,
        pattern: re.Pattern = _MD_LINK_RE,
    ) -> str:
    """Replace all Markdown links with their text content.
    
    Args:
        markdown_content (str): the Markdown content.
        pattern (re.Pattern): a regex pattern that captures the Markdown link text.
    """
    return pattern.sub(lambda m: m.groups()[0], markdown_content)    

def format_olmocr_pes2o(text: str) -> str:
    """Format olmOCR-pes2o-0225 to Markdown and remove superflous content if possible.
    
    - Author list
    - Reference list
    - Appendix
    """
    title = text.splitlines()[0]
    
    abstract_idx = len(title) + 1
    abstract_match = _ABSTRACT_RE.search(text)
    if abstract_match is not None:
        abstract_idx = abstract_match.start()
    
    references_idx = len(text)
    references_match = _REFERENCES_RE.search(text)
    if references_match is not None:
        references_idx = references_match.start()

    # TODO: also remove inline references (hard)

    # This intentionally excludes the appendix, which avoids too long documents
    return title + "\n" + text[abstract_idx:references_idx]

def gutenberg_is_license_acceptable(row: dict) -> bool:
    return True
    #return json.loads(row['extra'])['usagerights'] in ('open', 'copyright_open')

_SQUARE_BRACKETS_RE = re.compile(r"\[[^\]]+\]")
_STAR_SEPARATOR_RE = re.compile(r"^[\* ]+\n", flags=re.MULTILINE)

def clean_gutenberg_text(row: dict) -> dict:
    if row['text'].startswith("www.gutenberg.org/license."):
        row['text'] = "" # Don't bother with cleaning license notice
    else:
        row['text'] = _SQUARE_BRACKETS_RE.sub("", row['text'])
        row['text'] = _STAR_SEPARATOR_RE.sub("", row['text'])
        lines = row['text'].splitlines()
        # Remove unnecessary line breaks in the middle of a sentence.
        # Works for poems if every line starts with a capital letter.
        for i in range(1, len(lines) - 1):
            if not lines[i] or not lines[i+1]:
                continue
            if lines[i][-1] in r".!?:;,\"'" or lines[i+1][0].isupper():
                lines[i] += "\n"
        row['text'] = ''.join(lines[1:]) # TODO: remove publishing info

    return row


# pattern : \_\_\_\_\_ (possibly in bold)
_VBPL_RULER_RE = re.compile(r"(?:\*)*(?:\\_)+(?:\*)*", flags=re.MULTILINE)

def format_vbpl_md(row: dict) -> dict:
    """Improve the Markdown formatting of a VBPL example."""
    content = row['markdown_content']
    content = _VBPL_RULER_RE.sub("", content)
    content = replace_md_links_with_text(content)
    content = content.replace("/.", "") # Weird espace character at the end of a text
    content = content.replace("\xa0", " ") # was nbsp; in HTML
    row['markdown_content'] = content
    return row


class DatasetNames(StrEnum):
    _ignore_ = [
        '_IDS', '_ID_COUNTERS', '_GLOBAL_ID_COUNTER', '_LOCK',
        '_ENABLED', 'ENGLISH', 'VIETNAMESE', 'CODE', 'PLACEHOLDER_URLS',
    ]

    # SmolLM corpus
    cosmopedia_v2 = "HuggingFaceTB/smollm-corpus:cosmopedia-v2"
    fineweb_edu_dedup = "HuggingFaceTB/smollm-corpus:fineweb-edu-dedup"

    starcoder_python_edu = "JanSchTech/starcoderdata-python-edu-lang-score"

    # Annealing datasets
    finemath_4plus = "HuggingFaceTB/finemath:finemath-4plus"
    olmocr_pes2o = "allenai/olmOCR-pes2o-0225"
    stackmathqa = "math-ai/StackMathQA:stackmathqa200k"
    flan_v2 = "SirNeural/flan_v2"
    wikipedia_en = "omarkamali/wikipedia-monthly:20250702.en"

    # Lucie pretraining datasets (excerpt)
    claire_en = "OpenLLM-France/Lucie-Training-Dataset:Claire-en"
    gutenberg_en = "OpenLLM-France/Lucie-Training-Dataset:Gutenberg-en"
    europarl_en = "OpenLLM-France/Lucie-Training-Dataset:Europarl-en"

    lucie_training_code = "OpenLLM-France/Lucie-Training-Dataset:code"

    # Vietnamese datasets
    wikipedia_vi = "omarkamali/wikipedia-monthly:20250702.vi"
    culturax = "uonlp/CulturaX:vi"
    madlad400 = "Symato/madlad-400_vi"
    fineweb2_hq = "epfml/FineWeb2-HQ:vie_Latn"
    binhvq_news_corpus = "bigscience-data/roots_vi_binhvq_news_corpus"
    vbpl = "doanhieung/vbpl"
    mtet = "phongmt184172/mtet"
    ccvj = "ccvj" # In-house dataset

    _LOCK: Lock = None
    _IDS = {}
    _GLOBAL_ID_COUNTER = 0
    _ID_COUNTERS = {}
    _ENABLED: dict[str, set["DatasetNames"]] = {}
    ENGLISH = set()
    VIETNAMESE = set()
    CODE = set()
    PLACEHOLDER_URLS: list[str] = []

    @classmethod
    def _init_cls_vars(cls):
        """Initialize class member variables."""
        cls._LOCK = Lock()
        cls._IDS = {
            name: i + 1
            for i, name in enumerate(cls.__members__)
        }
        cls._ID_COUNTERS = {name: 0 for name in cls._IDS}
        cls._GLOBAL_ID_COUNTER = 0
        cls._ENABLED = {
            SMOLLM2.name: {
                cls.cosmopedia_v2,
                cls.fineweb_edu_dedup,
                cls.starcoder_python_edu,
                cls.olmocr_pes2o,
                cls.finemath_4plus,
                cls.stackmathqa,
                cls.wikipedia_en,

                cls.wikipedia_vi,
                cls.culturax,
                cls.fineweb2_hq,
                cls.binhvq_news_corpus,
                cls.vbpl,
                cls.mtet,
                cls.ccvj,
            },
            LUCIE.name: set(cls.__members__.values())
        }
        cls.ENGLISH = {
            cls.cosmopedia_v2,
            cls.fineweb_edu_dedup,
            cls.finemath_4plus,
            cls.olmocr_pes2o,
            cls.stackmathqa,
            cls.flan_v2,
            cls.wikipedia_en,
            cls.claire_en,
            cls.gutenberg_en,
            cls.europarl_en,
        }
        cls.VIETNAMESE = {
            cls.wikipedia_vi,
            cls.culturax,
            cls.madlad400,
            cls.fineweb2_hq,
            cls.binhvq_news_corpus,
            cls.vbpl,
            cls.mtet,
            cls.ccvj,
        }
        cls.CODE = {
            cls.starcoder_python_edu,
            cls.lucie_training_code,
        }
        cls.PLACEHOLDER_URLS = [
            name.placeholder_domain
            for name in cls.__members__.values()
        ]
    
    @property
    def language(self) -> str:
        cls = type(self)
        if self in cls.ENGLISH:
            return Languages.english
        elif self in cls.VIETNAMESE:
            return Languages.vietnamese
        elif self in cls.CODE:
            return "code"
        raise RuntimeError(f"Unknown language: {self}")


    @property
    def _id(self) -> int:
        """The dataset identifier."""
        return DatasetNames._IDS[self.name]
    
    @property
    def _counter(self) -> int:
        with DatasetNames._LOCK:
            return DatasetNames._ID_COUNTERS[self.name]
    
    def origin_metadata(self, id: int | str = None) -> dict[str, object]:
        """Add additional metadata about the used source dataset.

        Args:
            id (int, optional): The original id of the dataset.

        Returns:
            dict[str, object]: additional metadata about this dataset.
        
        **Note:** this function should always be called after [`generate_row_id`].
        """
        metadata = dict(origin=self.value)
        if id is None:
            metadata['dataset_generated_id'] = self._counter
        else:
            metadata['dataset_generated_id'] = str(id)
        
        return metadata

    def generate_row_id(self) -> int:
        """Generate a new globally-unique id for a row.

        Args:
            id (int, optional): The original id used by the dataset source,
                to be registered and mapped to the new generated row id.

        Returns:
            int: A globally unique row id.
        """
        with DatasetNames._LOCK:
            DatasetNames._ID_COUNTERS[self.name] += 1
            DatasetNames._GLOBAL_ID_COUNTER += 1
            # Appending the dataset id ensures consistency in case the global id is reset
            # e.g. when redownloading a dataset.
            return self._id + 1000 * DatasetNames._GLOBAL_ID_COUNTER
    
    @property
    def placeholder_domain(self) -> str:
        """A placeholder URL prefix to use when the dataset has no web sources."""
        return f"https://{self.name.replace('_', '-')}.example"


DatasetNames._init_cls_vars()

# TODO: refactor with inheritance and add placeholder url according to the dataset
class NormalizeCols:
    """Organize the dataset columns according to a unified format."""

    @staticmethod
    def format_prompt_response(prompt: str, response: str) -> str:
        """Return a text example using a randomized prompt-response template."""
        # TODO: experiment with best prompt templates, possibly use Vietnamese templates
        return random.choices(
            population=[
                f"{prompt}\n{response}",
                f"User: {prompt}\nAssistant: {response}",
                f"User: {prompt} Assistant: {response}",
                f"{prompt} {response}",
            ],
            weights=[
                0.75,
                0.15,
                0.05,
                0.05,
            ]
        )[0]
    
    @staticmethod
    def cosmopedia_v2(row: dict) -> dict:
        id = DatasetNames.cosmopedia_v2.generate_row_id()
        return dict(
            id=id,
            # Don't include prompt
            text=row['text'],
            metadata=dict(
                # Cosmopedia v2 is synthetic data generated from web sources
                url=f"{DatasetNames.cosmopedia_v2.placeholder_domain}/{id}",
                **filter_keys(row, ['audience', 'format', 'seed_data']),
                **DatasetNames.cosmopedia_v2.origin_metadata(),
            ),
        )
    
    @staticmethod
    def fineweb_edu_dedup(row: dict) -> dict:
        # text, id(string), metadata[dump, url, date, file_path, score, int_score]
        return dict(
            id=DatasetNames.fineweb_edu_dedup.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(
                    row['metadata'],
                    ['dump', 'url', 'date', 'file_path', 'score', 'int_score'],
                ),
                **DatasetNames.fineweb_edu_dedup.origin_metadata(str(row['id'])),
            ),
        )

    @staticmethod
    def starcoder_python_edu(row: dict) -> dict:
        return dict(
            id=DatasetNames.starcoder_python_edu.generate_row_id(),
            text=row['content_cleaned'],
            metadata=dict(
                # Fake url, handy for identification purposes
                url=f"https://github.com/{row['max_stars_repo_name']}/{row['max_stars_repo_path']}",
                **filter_keys(row, ['language', 'edu_score']),
                **DatasetNames.starcoder_python_edu.origin_metadata(str(row['id'])),
            ),
        )
    
    @staticmethod
    def finemath_4plus(row: dict) -> dict:
        return dict(
            id=DatasetNames.finemath_4plus.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['url', 'warc_filename', 'score']),
                **DatasetNames.finemath_4plus.origin_metadata(),
            ),
        )
    
    @staticmethod
    def stackmathqa(row: dict) -> dict:
        return dict(
            id=DatasetNames.stackmathqa.generate_row_id(),
            text=NormalizeCols.format_prompt_response(row['Q'], row['A']),
            metadata=dict(
                **filter_keys(
                    row['meta'],
                    ['url', 'timestamp', 'source', 'answer_count', 'answer_id', 'question_score'],
                ),
                **DatasetNames.stackmathqa.origin_metadata(),
            ),
        )
    
    @staticmethod
    def olmocr_pes2o(row: dict) -> dict:
        def remove_null_from_metadata(metadata: dict):
            # Formatting fields of study as a string avoids problems with Arrow column type
            # inference when saving an IterableDataset to Parquet.
            # https://github.com/huggingface/datasets/issues/3738
            metadata['fieldofstudy'] = '; '.join(metadata['fieldofstudy'])
            return metadata
    
        return dict(
            id=DatasetNames.olmocr_pes2o.generate_row_id(),
            text=format_olmocr_pes2o(row['text']),
            metadata=dict(
                # This is not the source URL. No problem for URL filters.
                url=f"https://api.semanticscholar.org/graph/v1/paper/{row['id']}",
                **filter_keys(
                    remove_null_from_metadata(row['metadata']),
                    ['pdf-total-pages', 'fieldofstudy']
                ),
                **DatasetNames.olmocr_pes2o.origin_metadata(str(row['id'])),
            ),
        )
    
    @staticmethod
    def flan_v2(row: dict) -> dict:
        id = DatasetNames.flan_v2.generate_row_id()
        return dict(
            id=id,
            text=NormalizeCols.format_prompt_response(row['inputs'], row['targets']),
            metadata=dict(
                # FLAN v2 is LLM-generated, so it has no real web source
                url=f"{DatasetNames.flan_v2.placeholder_domain}/{id}",
                **filter_keys(
                    row,
                    ['task'],
                ),
                **DatasetNames.flan_v2.origin_metadata(),
            ),
        )

    @staticmethod
    def claire_en(row: dict) -> dict:
        return dict(
            id=DatasetNames.claire_en.generate_row_id(),
            text=row['text'],
            metadata=dict(
                # Transcripts of conversations. Not web content
                url=f"{DatasetNames.claire_en.placeholder_domain}/{row['id']['idx_row']}",
                subset=row['extra']['subset'],
                **DatasetNames.claire_en.origin_metadata(str(row['id']['idx_row'])),
            ),
        ) 
    
    @staticmethod
    def wikipedia_en(row: dict) -> dict:
        return dict(
            id=DatasetNames.wikipedia_en.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['url', 'title']),
                **DatasetNames.wikipedia_en.origin_metadata(str(row['id'])),
            )
        )
    
    @staticmethod
    def gutenberg_en(row: dict) -> dict:
        return dict(
            id=DatasetNames.gutenberg_en.generate_row_id(),
            text=row['text'],
            metadata=dict(
                url=f"{DatasetNames.gutenberg_en.placeholder_domain}/{row['id']}",
                title=row['title'],
                **json.loads(row['author']),
                **DatasetNames.gutenberg_en.origin_metadata(str(row['id'])),
            )
        )

    @staticmethod
    def wikipedia_vi(row: dict) -> dict:
        return dict(
            id=DatasetNames.wikipedia_vi.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['url', 'title']),
                **DatasetNames.wikipedia_vi.origin_metadata(str(row['id'])),
            )
        )

    @staticmethod
    def culturax(row: dict) -> dict:
        return dict(
            id=DatasetNames.culturax.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['url', 'timestamp', 'source']),
                **DatasetNames.culturax.origin_metadata(),
            )
        )

    @staticmethod
    def madlad400(row: dict) -> dict:
        id = DatasetNames.madlad400.generate_row_id()
        return dict(
            id=id,
            text=row['text'].replace(r"\n", "\n"),
            metadata=dict(
                # Unfortunately, MADLAD-400 has not provided its exact sources
                url=f"{DatasetNames.madlad400.placeholder_domain}/{id}",
                **DatasetNames.madlad400.origin_metadata()
            ),
        )

    @staticmethod
    def fineweb2_hq(row: dict) -> dict:
        return dict(
            id=DatasetNames.fineweb2_hq.generate_row_id(),
            text=row['text'],
            metadata=dict(
                # How is quality score computed?
                **filter_keys(row, ['url', 'date', 'dump', 'language_score', 'quality_score']),
                **DatasetNames.fineweb2_hq.origin_metadata(str(row['id'])),
            )
        )

    @staticmethod
    def binhvq_news(row: dict) -> dict:
        id = DatasetNames.binhvq_news_corpus.generate_row_id()
        return dict(
            id=id,
            text=row['text'],
            metadata=dict(
                # NOTE: this is not the true source URL. Only used for URLFilter to accept
                url=f"{DatasetNames.binhvq_news_corpus.placeholder_domain}/{id}",
                **DatasetNames.binhvq_news_corpus.origin_metadata(),
            )
        )

    @staticmethod
    def vbpl(row: dict) -> dict:
        return dict(
            id=DatasetNames.vbpl.generate_row_id(),
            text=row['markdown_content'],
            metadata=dict(
                url=row['url'],
                title=row["Tiêu đề"],
                scope=row["Phạm vi"],
                industry=row["Ngành"],
                sector=row["Lĩnh vực"],
                effective_date=row["Ngày có hiệu lực"],
                validity_status=row["Tình trạng hiệu lực"],
                **DatasetNames.vbpl.origin_metadata(str(row['id']))
            )
        )

    @staticmethod
    def mtet(row: dict) -> dict:
        id = DatasetNames.mtet.generate_row_id()
        return dict(
            id=id,
            # NOTE: this is an fake URL for mTet. Only used for URLFilter to accept
            url=f"{DatasetNames.mtet.placeholder_domain}/{id}",
            text=NormalizeCols.format_prompt_response(row['prompt'], row['response']),
            metadata=DatasetNames.mtet.origin_metadata(),
        )
    
    @staticmethod
    def ccvj(row: dict) -> dict:
        id = DatasetNames.ccvj.generate_row_id()
        return dict(
            id=id,
            text=row['text'],
            metadata=dict(
                url=row.get('url', f"{DatasetNames.ccvj.placeholder_domain}/{id}"),
                title=row.get('title', ''),
                journal_id=row.get('journal_id', ''),
                **DatasetNames.ccvj.origin_metadata(),
            )
        )