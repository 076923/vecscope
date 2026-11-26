"""Public datasets and public models for the held-out validation of vecscope's findings.

Every dataset here is public on the Hugging Face Hub and none was used to set a threshold:
``role="heldout"`` marks data the defaults were never fitted on. The thresholds were fitted on
SciFact, NFCorpus and ko-strategyqa (see docs/verification.md); refitting them on anything
listed here would turn the held-out set into a training set, so the report only measures.

``role="confirm"`` marks a second held-out set, chosen and frozen in preregistration.md
before any of it was embedded, to check the changes made after the first set was read: the
query_hubness null, judging findings against a reference model, the label-free result change
and the labeled-query budget. confirm.py analyses it; report.py leaves it out.

``role="third"`` marks a third held-out set, frozen in preregistration_third.md before it was
embedded, for the changes made after the second was read: the reference-overlap signal, the
labeled gate judged against the noise of its queries, and the query_hubness chance rule,
which the second set could not test. third.py analyses it.

``role="fourth"`` marks a fourth held-out set, frozen in preregistration_fourth.md before it
was embedded, for the two changes made after the third was read: the agreed top hits as a
label-free sign of harm, and lowering hub findings at fail when the results agree with the
reference. fourth.py analyses it.

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


def _miracl(lang: str, role: str = "heldout", max_queries: int = 0) -> Dataset:
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
        role=role,
        max_queries=max_queries,
        note="MIRACL dev with the MTEB hard-negative corpus"
        + (f"; {max_queries} seeded queries" if max_queries else ""),
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


def _wiki(lang: str, role: str = "third") -> Dataset:
    return Dataset(
        name=f"wikipedia-{lang}",
        kind="mteb",
        repo="mteb/WikipediaRetrievalMultilingual",
        language=lang,
        domain="wikipedia",
        license="CC-BY-SA-3.0",
        role=role,
        corpus=(f"{lang}-corpus", "test"),
        queries=(f"{lang}-queries", "test"),
        qrels=(f"{lang}-qrels", "test"),
        note="Wikipedia passages with generated questions (WikipediaRetrievalMultilingual)",
    )


def _webfaq(code: str, lang: str, role: str = "confirm", max_queries: int = 1000) -> Dataset:
    return Dataset(
        name=f"webfaq-{code}",
        kind="mteb",
        repo="mteb/WebFAQRetrieval",
        language=lang,
        domain="web FAQ",
        license="CC-BY-4.0",
        role=role,
        corpus=(f"{code}-corpus", "test"),
        queries=(f"{code}-queries", "test"),
        qrels=(f"{code}-qrels", "test"),
        max_queries=max_queries,
        note=f"FAQ answers from the web (WebFAQ); {max_queries} seeded queries",
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
        # third held-out set (role="third"), frozen in preregistration_third.md. The first
        # four have few queries over a large corpus, where chance alone gives query_hubness
        # a high skewness; the others have at most 25k documents for the injected faults.
        _miracl("ko", role="third"),
        _miracl("th", role="third", max_queries=200),
        Dataset(
            name="scidocs",
            kind="mteb",
            repo="mteb/scidocs",
            language="en",
            domain="science",
            license="CC-BY-SA-4.0",
            role="third",
            max_queries=60,
            note="paper titles -> cited abstracts (BEIR SCIDOCS); 60 seeded queries",
        ),
        _webfaq("ind", "id", role="third", max_queries=150),
        Dataset(
            name="arguana",
            kind="mteb",
            repo="mteb/arguana",
            language="en",
            domain="arguments",
            license="CC-BY-SA-4.0",
            role="third",
            note="argument -> best counter-argument (BEIR ArguAna)",
        ),
        Dataset(
            name="stackoverflowqa",
            kind="mteb",
            repo="mteb/StackOverflowQA",
            language="en",
            domain="code QA",
            license="MIT",
            role="third",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="StackOverflow questions -> accepted answers with code",
        ),
        _wiki("bn"),
        _wiki("cs"),
        Dataset(
            name="autorag-ko",
            kind="mteb",
            repo="mteb/AutoRAGRetrieval",
            language="ko",
            domain="finance, public, medical, legal, commerce",
            license="MIT",
            role="third",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="Korean documents of five domains with generated questions",
        ),
        # fourth held-out set (role="fourth"), frozen in preregistration_fourth.md. The first
        # six have at most 25k documents for the injected faults; the others are short FAQ
        # answers, where good models made the hubs left at fail on the second and third
        # sets, and larger corpora.
        Dataset(
            name="cqadupstack-android",
            kind="mteb",
            repo="mteb/cqadupstack-android",
            language="en",
            domain="community QA",
            license="Apache-2.0",
            role="fourth",
            corpus=("corpus", "corpus"),
            queries=("queries", "queries"),
            qrels=("default", "test"),
            note="StackExchange android questions -> duplicate questions",
        ),
        Dataset(
            name="feedbackqa",
            kind="mteb",
            repo="mteb/FeedbackQARetrieval",
            language="en",
            domain="public health and government FAQ",
            license="Apache-2.0",
            role="fourth",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="questions -> FAQ passages of public health and government sites",
        ),
        Dataset(
            name="hunsum2",
            kind="mteb",
            repo="mteb/HunSum2AbstractiveRetrieval",
            language="hu",
            domain="news",
            license="CC-BY-4.0",
            role="fourth",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            note="Hungarian news lead -> article",
        ),
        _wiki("it", role="fourth"),
        _wiki("fa", role="fourth"),
        _webfaq("msa", "ms", role="fourth", max_queries=1000),
        _webfaq("heb", "he", role="fourth", max_queries=1000),
        _webfaq("ell", "el", role="fourth", max_queries=1000),
        _webfaq("hun", "hu", role="fourth", max_queries=1000),
        Dataset(
            name="rubq",
            kind="mteb",
            repo="mteb/RuBQRetrieval",
            language="ru",
            domain="wikipedia QA",
            license="CC-BY-SA-4.0",
            role="fourth",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            max_queries=1000,
            note="Russian questions -> Wikipedia paragraphs; 1000 seeded queries",
        ),
        Dataset(
            name="litsearch",
            kind="mteb",
            repo="mteb/LitSearchRetrieval",
            language="en",
            domain="science",
            license="MIT",
            role="fourth",
            corpus=("corpus", "test"),
            queries=("queries", "test"),
            qrels=("qrels", "test"),
            max_queries=120,
            note="literature search questions -> paper abstracts; 120 seeded queries",
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
