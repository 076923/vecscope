**English** · [한국어](ko/verification.md)

# Verification

vecscope 0.5.0 plus the then-unreleased changes (sweep latency and index size, compare with
supplied neighbors, the CI gate) was verified without changing the library: the upstream tests, exactness against float64
at 100k x 768, an independent recall scorer, the sampling claims in the README, the
performance table, the "Real embeddings" table, and 19 public models against nDCG@10 on
three labeled datasets. This page records what was measured and what changed in vecscope
because of it. The thresholds and documentation in the current release reflect these
results.

- Environment: AMD EPYC 7R13, 8 vCPU, 26 GB RAM, NVIDIA L4; Python 3.12.3, numpy 2.5.4,
  faiss-gpu-cu12 1.14.1, sentence-transformers 3.4.1
- Corpora: six synthetic shapes (`examples/bench_common.py`: clustered, isotropic,
  anisotropic, duplicates, lowrank, zipf), the eight public pre-computed embedding sets of
  the README table, and BEIR SciFact, NFCorpus and ko-strategyqa encoded with 19 models

## Results

| area | result | change made |
|---|---|---|
| exact k-NN (`exact_knn`, `query_knn`, faiss flat CPU and GPU) | 0 rank errors against float64 on six shapes at 100k x 768, 19 models x 3 datasets and 8 public embedding sets; float32 distance error at most 4.1e-7 | none |
| ANN recall scoring | tie slack `1e-6 + 1e-4 x distance` was about 70x float32 rounding and credited an 11th neighbor up to 7e-5 farther than the 10th: recall inflated by +0.0000 to +0.0016 against a strict scorer | slack reduced to `1e-6 + 1e-5 x distance`; `tests/test_scale_exactness.py` checks vecscope against the strict scorer |
| sampled audit, values | the documented behaviour held: distribution findings within 2.3% of the full value; hubness and antihubs a lower bound everywhere; collapse share 0.96 to 1.03 x (fraction x full share) | none |
| sampled audit, verdicts | a full-corpus `fail` or `warn` read `ok` or `warn` on the sample: zipf connectivity 0.372 fail to 0.0008 ok on a 25% sample; clustered hubness 2.60 warn to 1.15 ok; duplicates collapse 0.200 fail to 0.051 warn; outliers over-counted 3.9x on a 10% sample | hubness, antihubs and connectivity verdicts other than `fail`, and outlier verdicts other than `ok`, are marked `inconclusive`; the collapse verdict uses the estimated full share; the gate names inconclusive findings |
| `sample_repeats` | `value_range` often did not contain the full-corpus value, even for very stable findings | documented as the spread between samples, not an interval estimate |
| thresholds | with pipelines reproducing published nDCG@10 within 0.007, `ood_queries` failed 18 of 19 models on NFCorpus and `anisotropy` warned on the best models; on Korean data `query_hubness` failed good multilingual models (nDCG 0.80) and English-only ones (0.04) alike | `anisotropy` (0.5, 0.85) to (0.85, 0.95); `ood_queries` (0.05, 0.20) to (0.5, 0.8); `query_hubness` (2, 4) to (4, 10) |
| README "Real embeddings" table | all 88 verdicts reproduced on the first 40k rows of each set; one printed value off by rounding | Cohere ko `distance_concentration` 1.73 to 1.72 |
| performance table | single runs of `benchmark_scale.py` within 2% of a three-run median in separate processes; tracemalloc peak within 0.04 GiB of the OS peak | table replaced with the measured seconds and memory for 384, 768 and 1,536 dims up to 200k rows |
| sweep | recommendations identical across three seeds; IVF recall at nprobe=1 rose from 0.696 at 100k to 0.955 at 500k on the clustered corpus because its n/200 centers line up with nlist = 4*sqrt(n) | sweep caveat says IVF recall can move either way with size; `benchmark_sweep.py` runs several shapes and takes `--centers` |

## Finding values against nDCG@10 (19 models)

Spearman correlation between each finding's value and nDCG@10 across the 19 models, per
dataset. A consistently negative value means the finding rises as retrieval gets worse.

| finding | SciFact | NFCorpus | ko-strategyqa |
|---|---|---|---|
| query_hubness | −0.57 | −0.46 | −0.90 |
| hubness | −0.52 | −0.48 | −0.76 |
| antihubs | −0.27 | −0.12 | −0.72 |
| rogue_dimensions | −0.38 | −0.38 | −0.05 |
| ood_queries | −0.34 | −0.33 | +0.25 |
| anisotropy | +0.29 | +0.21 | −0.11 |
| distance_concentration | +0.19 | +0.34 | −0.82 |

What this supports:

- `hubness` and `query_hubness` carry a signal about retrieval quality across models. Their
  values were used to move the `query_hubness` thresholds so that good models on Korean data
  (values 2.3 to 9.1) are not failed while English-only models on Korean data (11 to 30) are.
- `ood_queries` measures distribution shift between queries and corpus, which short queries
  against long passages always show. Good models scored 0.0 to 0.65; a wrong model or
  language scored 0.7 to 1.0, or scored 0 because both sides were encoded equally badly. The
  new thresholds flag only the large shifts; the finding is not a quality score.
- `anisotropy` is unrelated to quality in this range. The best retrievers sit at 0.58 to
  0.82, so `warn` now starts at 0.85.
- A pipeline that is wrong in an obvious way (GloVe on BEIR, a Korean-only model on English,
  an English-only model on Korean) raises several findings at once. The count of warnings
  separates such pipelines; a single `fail` does not.

The verification did not cover the MCP server or production query logs, and the real-model
measurements are three datasets in two languages. The sampling and exactness claims were
re-checked on synthetic data only; the retrieval-quality relation was measured on labeled
public data only.

## Reproducing

- `pytest -q` runs the moved verification tests (`tests/test_new_features.py`,
  `tests/test_scale_exactness.py`); `pytest -m slow` adds the 100k x 384 exactness checks.
- `python examples/benchmark_scale.py --rows 10000 50000 100000 --dim 768 --json scale.json`
  and `python examples/benchmark_sweep.py --rows 100000 --dim 768 --corpus clustered zipf
  --json sweep.json` re-measure the tables and record the machine in the JSON.
- `python examples/retrieval_quality.py` runs the labeled-data study for the public datasets.
