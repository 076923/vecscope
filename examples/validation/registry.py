"""Public datasets and public models for the held-out validation of vecscope's findings.

Every dataset here is public on the Hugging Face Hub and none was used to set a threshold:
``role="heldout"`` marks data the defaults were never fitted on. The thresholds were fitted on
SciFact, NFCorpus and ko-strategyqa (see docs/verification.md); refitting them on anything
listed here would turn the held-out set into a training set, so the report only measures.

``role="confirm"`` marks a second held-out set, chosen and frozen in preregistration.md
before any of it was embedded, to check the changes made after the first set was read: the
query_hubness null, judging findings against a reference model, the label-free result change
and the labeled-query budget. confirm.py analyses it; report.py leaves it out.

No private or company data belongs in this file, in the cache it writes, or in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: str  # "mteb" (corpus/queries/qrels configs) or "esci"
    repo: str
    language: str  # ISO 639-1, or "code"
    domain: str
    license: str
    role: str = "heldout"
    corpus: tuple[str, str] = ("corpus", "corpus")  # (config, split)
    queries: tuple[str, str] = ("queries", "queries")
    qrels: tuple[str, str] = ("default", "test")
    locale: str = ""  # esci only
    max_queries: int = 0  # 0 = all judged queries; else a seeded subset
    note: str = ""


def _xpqa(lang: str, code: str) -> Dataset:
    pair = f"{code}-{code}"
    return Dataset(
        name=f"xpqa-{lang}",
        kind="mteb",
        repo="mteb/XPQARetrieval",
        language=lang,
        domain="product answers",
        license="CDLA-Sharing-1.0",
        corpus=(f"{pair}-corpus", "test"),
        queries=(f"{pair}-queries", "test"),
        qrels=(f"{pair}-qrels", "test"),
        note="short product answers (XPQA, Amazon), monolingual pair",
    )


def _miracl(lang: str) -> Dataset:
    return Dataset(
        name=f"miracl-{lang}",
        kind="mteb",
        repo="mteb/MIRACLRetrievalHardNegatives",
        language=lang,
        domain="wikipedia",
        license="CC-BY-SA-4.0",
        corpus=(f"{lang}-corpus", "dev"),
        queries=(f"{lang}-queries", "dev"),
        qrels=(f"{lang}-qrels", "dev"),
        note="MIRACL dev with the MTEB hard-negative corpus",
    )


def _csn(lang: str) -> Dataset:
    return Dataset(
        name=f"codesearchnet-{lang}",
        kind="mteb",
        repo="mteb/CodeSearchNetRetrieval",
        language="code",
        domain="code",
        license="MIT",
        corpus=(f"{lang}-corpus", "test"),
        queries=(f"{lang}-queries", "test"),
        qrels=(f"{lang}-qrels", "test"),
        note=f"docstring -> {lang} function",
    )


def _webfaq(code: str, lang: str) -> Dataset:
    return Dataset(
        name=f"webfaq-{code}",
        kind="mteb",
        repo="mteb/WebFAQRetrieval",
        language=lang,
        domain="web FAQ",
        license="CC-BY-4.0",
        role="confirm",
        corpus=(f"{code}-corpus", "test"),
        queries=(f"{code}-queries", "test"),
        qrels=(f"{code}-qrels", "test"),
        max_queries=1000,
        note="FAQ answers from the web (WebFAQ); 1000 seeded queries",
    )


def _esci(locale: str, lang: str) -> Dataset:
    return Dataset(
        name=f"esci-{locale}",
        kind="esci",
        repo="tasksource/esci",
        language=lang,
        domain="product titles",
        license="Apache-2.0",
        locale=locale,
        max_queries=500,
        note=(
            "Amazon ESCI test, small_version; corpus = product titles judged for 500 seeded "
            "queries; gains as in the KDD Cup 2022 metric: Exact 1, Substitute 0.1, "
            "Complement 0.01, Irrelevant 0"
        ),
    )


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in [
        # short product text in five more languages
        _xpqa("ja", "jpn"),
        _xpqa("de", "deu"),
        _xpqa("fr", "fra"),
        _xpqa("es", "spa"),
        _xpqa("zh", "cmn"),
        _xpqa("hi", "hin"),
        _xpqa("ar", "ara"),
        _esci("us", "en"),
        _esci("es", "es"),
        _esci("jp", "ja"),
        # multilingual Wikipedia passages, 60k to 80k documents
        _miracl("de"),
        _miracl("fr"),
        _miracl("hi"),
        # code search
        _csn("python"),
        _csn("go"),
        _csn("java"),
        Dataset(
            name="cosqa",
            kind="mteb",
            repo="mteb/cosqa",
            language="code",
            domain="code",
            license="MIT",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="web query -> python function",
        ),
        # legal and biomedical
        Dataset(
            name="legal-summarization",
            kind="mteb",
            repo="mteb/legal_summarization",
            language="en",
            domain="legal",
            license="Apache-2.0",
            note="terms-of-service summaries -> source clauses (LegalBench)",
        ),
        Dataset(
            name="trec-covid",
            kind="mteb",
            repo="mteb/trec-covid",
            language="en",
            domain="biomedical",
            license="CORD-19 dataset license (research use)",
            note="171k CORD-19 abstracts, 50 queries with deep graded judgments",
        ),
        # second held-out set (role="confirm"), frozen in preregistration.md
        _webfaq("kor", "ko"),
        _webfaq("tha", "th"),
        _webfaq("vie", "vi"),
        Dataset(
            name="jagovfaqs",
            kind="mteb",
            repo="mteb/JaGovFaqsRetrieval",
            language="ja",
            domain="government FAQ",
            license="CC-BY-4.0",
            role="confirm",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="Japanese government FAQ answers",
        ),
        Dataset(
            name="gerdalir-small",
            kind="mteb",
            repo="mteb/GerDaLIRSmall",
            language="de",
            domain="legal",
            license="MIT",
            role="confirm",
            corpus=("corpus", "corpus"),
            queries=("queries", "queries"),
            qrels=("default", "test"),
            max_queries=1000,
            note="German court decisions citing each other; 1000 seeded queries",
        ),
        Dataset(
            name="cqadupstack-webmasters",
            kind="mteb",
            repo="mteb/cqadupstack-webmasters",
            language="en",
            domain="community QA",
            license="Apache-2.0",
            role="confirm",
            corpus=("corpus", "corpus"),
            queries=("queries", "queries"),
            qrels=("default", "test"),
            note="StackExchange webmasters questions -> duplicate questions",
        ),
        Dataset(
            name="turhistquad",
            kind="mteb",
            repo="mteb/TurHistQuadRetrieval",
            language="tr",
            domain="history",
            license="MIT",
            role="confirm",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="Turkish history passages, SQuAD-style questions",
        ),
    ]
}


def role(name: str) -> str:
    return DATASETS[name].role if name in DATASETS else ""


@dataclass(frozen=True)
class Model:
    id: str
    languages: str  # "en" (English-only) or "multi"
    query_prefix: str = ""
    doc_prefix: str = ""
    query_prompt_name: str = ""  # a prompt defined by the model's sentence-transformers config
    kwargs: dict = field(default_factory=dict)
    dtype: str = "float16"

    @property
    def short(self) -> str:
        return self.id.split("/")[-1].replace("average_word_embeddings_", "")


BGE_EN = "Represent this sentence for searching relevant passages: "
MODELS: list[Model] = [
    # English-only
    Model("sentence-transformers/average_word_embeddings_glove.6B.300d", "en", dtype="float32"),
    Model("sentence-transformers/all-MiniLM-L6-v2", "en"),
    Model("sentence-transformers/all-mpnet-base-v2", "en"),
    Model("BAAI/bge-small-en-v1.5", "en", query_prefix=BGE_EN),
    Model("BAAI/bge-base-en-v1.5", "en", query_prefix=BGE_EN),
    Model("intfloat/e5-base-v2", "en", query_prefix="query: ", doc_prefix="passage: "),
    Model("thenlper/gte-base", "en"),
    # multilingual
    Model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "multi"),
    Model("intfloat/multilingual-e5-base", "multi", query_prefix="query: ", doc_prefix="passage: "),
    Model(
        "intfloat/multilingual-e5-large", "multi", query_prefix="query: ", doc_prefix="passage: "
    ),
    Model("BAAI/bge-m3", "multi"),
    Model("Alibaba-NLP/gte-multilingual-base", "multi", kwargs={"trust_remote_code": True}),
    Model("nlpai-lab/KURE-v1", "multi"),
    Model("Qwen/Qwen3-Embedding-0.6B", "multi", query_prompt_name="query", dtype="bfloat16"),
]
