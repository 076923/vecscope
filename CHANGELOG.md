# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.9.0] - 2026-10-31

### Changed

- `vecscope calibrate` gates only on statistically supported pairs.
  - Each finding gets a one-sided permutation p-value in the direction it implies (exact up
    to eight states, 20,000 seeded orderings above), Holm-adjusted over every finding tested,
    and a 95% bootstrap interval for its Spearman correlation.
  - A pair whose adjusted p-value is 0.05 or more is `exploratory`: reported, listed under
    `exploratory_thresholds`, and left out of `thresholds` and `--out` unless
    `--include-low-confidence` is given. On five states nothing can be supported; the report
    recommends ten or more and says when there are fewer.
  - `leave_one_out` judges every state with thresholds fitted on the others.
  - States scored on different qrels or k are refused (`--allow-mixed-qrels` overrides).
- The k-NN cache (`--knn-cache`) stores a digest per row (cache format 2) and updates rows
  re-embedded in place.
  - Changed rows, and rows whose cached list held one, are recomputed against every row; the
    other rows only look at the changed and appended rows. Status `updated`, with
    `changed_rows` and `recomputed_rows`.
  - It rebuilds when the rows to recompute would be more than half of all rows, rows were
    removed, or `dim` changed, and says why in `knn_cache.reason`. A format-1 cache is
    rebuilt once.
  - On 100k x 768, re-embedding 1,000 rows recomputed 10,625 of them: 19 s against 95 s for
    a full audit, same verdicts. 10,000 changed rows would take 83 s against 89 s, hence the
    rebuild rule.
- Vector store reads hold float32 arrays as they arrive instead of Python lists of floats.

### Added

- MCP server read limit: files over `--max-read-mb` (default 4096, `VECSCOPE_MAX_READ_MB`,
  0 = no limit) are refused before they are read, and a store stops being read once its
  vectors pass that size. `load_source(..., max_bytes=)` does the same in Python.
- `timeout=` URI option for network stores (default 60 s per request): Qdrant, Milvus and
  Weaviate take it directly; for Chroma it is set on the client's session after it connects.
- The recent-audit memory behind MCP `baseline` is guarded by a lock.
- `examples/agent_case.py` and [docs/agent_case.md](docs/agent_case.md): Claude Code in
  headless mode diagnosing and fixing two ingestion faults in real SciFact embeddings over
  MCP, with the tool calls and numbers of the runs.

### Security

- Store URIs in evidence and error messages drop user info and replace every query value
  other than the known options with `REDACTED`; errors raised by store clients are re-raised
  with the API key and those values scrubbed.

## [0.8.0] - 2026-10-31

### Changed

- **MCP server reads only its own directory by default.** This changes existing setups.
  - Without `--allow-dir` (or `VECSCOPE_ALLOWED_DIRS`) it reads the directory it was
    started in and below; before, every file the process could read was reachable.
  - `--allow-any` (or `VECSCOPE_ALLOW_ANY=1`) restores unrestricted reads; `--allow-dir` wins
    when both are given.
  - On start the server prints to stderr which directories it reads.
- Sampled audits count exact copies over every row.
  - One 16-byte digest per row, not a copy of the matrix: 0.5 s for 200k x 128.
  - `collapse.details.full_corpus_exact_copy_share` and `full_corpus_largest_exact_groups`.
  - The collapse estimate is never below that share, so rare copies a sample misses still
    raise the verdict (2.5% copied rows read `ok` on a 2% sample before and `warn` now).

### Added

- MCP reads each file through one verified handle.
  - The path is checked, opened, and the open handle compared (device and inode) with the
    file the path resolves to inside the allowed directories; a symlink swapped between the
    check and the read is refused.
  - FIFOs and directories are refused without blocking the server.
  - A local database directory (`qdrant+local://` and the like) is checked by path only,
    because its client opens the files itself.
  - `load_vectors`, `load_neighbors`, `load_qrels` and `load_source` take the file's bytes
    as `data=`.
- `vecscope calibrate` and `vecscope.calibrate.calibrate`: thresholds from your own pipeline
  states.
  - Input: `--json` audits of several states run with the same `--queries` and `--qrels`,
    and `--min-quality` for the retrieval metric.
  - Per finding: Spearman with the metric, whether its sign matches the finding's direction,
    the worst acceptable and best unacceptable values, and a (warn, fail) pair only when the
    finding tracks quality and separates the states.
  - `--holdout` audits check the proposal on states or queries it was not fitted on; `--out`
    writes the pairs for `audit --thresholds`.
- Plan: every finding in a step carries `placed_because`, the rule that put it there. A
  reason naming another finding marks a grouping by co-occurrence.
- Plan rule: islands next to a near-duplicate group smaller than the island size join the
  input-integrity step.
- Verification on three more Korean MTEB sets (XPQA kor-kor, AutoRAG, MultiLongDoc ko; 19
  models each), chosen for document length.
  - Long documents: `hubness` or `query_hubness` flags all 9 English-only models.
  - A narrow-domain corpus: `effective_rank` warns on every good model (0.064 to 0.094), a
    domain effect to calibrate rather than a fault.
  - Very short templated answers: `collapse`, `effective_rank` and `query_gap` flag every
    model; documented as a limit of the defaults on such data.

## [0.7.1] - 2026-10-31

### Fixed

- Windows paths are paths, not URI schemes: `C:\...\index.faiss` was read as the scheme `c`.
  - The CLI and Python API refused faiss index files on Windows.
  - The MCP server refused them as network stores.
  - Only `scheme://` now counts as a store URI, and `faiss://` and `+local` URIs keep a drive
    letter (`qdrant+local://C:/data/qd`).
- The incremental k-NN cache (`--knn-cache`) closes the cache file after reading it.
  - Windows refused to replace the still-open file ("Access is denied").

### Changed

- README, Korean README, verification pages, skill and changelog: long paragraphs split into
  short ones and lists; no content changed

## [0.7.0] - 2026-10-31

### Added (from the ood_queries redesign run, docs/verification.md)

- `query_gap` finding (`queries.query_gap_ratio`, thresholds (2.6, 3.0)): the median
  query-to-corpus nearest distance over the median corpus-to-corpus nearest distance
  - On 19 models x SciFact, NFCorpus and ko-strategyqa, the 18 pipelines that reproduce
    published nDCG@10 score 0.94 to 2.52 and none is flagged (`ood_queries` flagged 4)
  - Query sets embedded by another model of the same dimension that lose more than a quarter
    of nDCG@10 are flagged in 250 of 268 cases, at a median of 5 to 8
- `query_shift` finding (`queries.shifted_query_ratio`, thresholds (0.20, 0.30)): the share
  of queries farther from the corpus than the median + 3 MAD of a reference query set
  - It needs a reference: `audit(..., query_baseline=evidence)`, CLI `--baseline` from a run
    with `--queries` (used automatically when the baseline has query distances), MCP
    `query_baseline_path`
  - In the same run it flagged 266 of 268 harmful model swaps and none of the healthy query
    sets (0.00 to 0.09)
- `vecscope.evidence.query_reference(evidence)` returns the (median, MAD) pair a baseline
  provides; `query_audit(..., reference=)` adds the `shifted_*` keys, and its result now
  carries `query_nn_mad` and `query_gap_ratio`
- Retrieval quality on labeled queries (`vecscope.retrieval`, `audit(..., qrels=, ranked=)`,
  CLI `--qrels`/`--ranked`, MCP `qrels_path`/`ranked_path`)
  - nDCG@k, recall@k, MRR@k, hit rate and `duplicate_result_share` of exact cosine search
    over every row (also when the findings are sampled), and of the production top-k with
    `overlap_with_exact`
  - The `retrieval` block carries a `qrels_fingerprint`
  - `diff_evidence` adds `retrieval_changes` and `retrieval_comparable`, and the text diff
    prints them under the finding changes
- `plan` on every Evidence (`vecscope.plan.build_plan`): the non-ok findings grouped by the
  cause they likely share, in fix order
  - Causes: input integrity, dominant dimension, metric, shared direction, weak signal,
    corpus structure, index, query pipeline
  - Each step has the common cause, the smallest change, what to re-measure and why it comes
    at that position
  - Collapse without exact copies next to a non-ok anisotropy is grouped with the shared
    direction, not with deduplication
  - norms, metric_mismatch, outliers and hub findings next to a non-ok rogue_dimensions join
    the dominant-dimension step, whose first change is z-scoring, because normalizing does
    not remove a dominant coordinate (on 21 real pairs with one coordinate scaled 30x,
    z-scoring restored a median 98% of nDCG@10)
  - Query hubness next to corpus hubness joins the hubness step
  - query_gap next to a duplicate problem joins the input-integrity step, because
    near-copies shrink the corpus distance it divides by (60% jittered copies of SciFact
    raised it to `fail` with the queries unchanged)
  - The text output prints it and the MCP instructions tell the agent to follow it
- Vector store adapters (`vecscope.adapters`, optional extras `qdrant`, `chroma`, `milvus`,
  `weaviate`, `stores`): the audit path can be a faiss index file or a store URI
  (`qdrant://`, `chroma://`, `milvus://`, `weaviate://`, and `+local` forms)
  - Rows are read in ascending id order with a fingerprint in `Evidence.source`; findings
    carry `sample_source_ids`
  - `--ann-from-source` scores the store's own index; `--save-ids` writes the ids in row
    order; API keys only from `api_key_env=`
  - Tested against faiss Flat/IVF/HNSW and IDMap files, Qdrant local mode, Chroma persistent
    mode and Milvus Lite; Weaviate against an embedded server (opt-in test,
    `VECSCOPE_TEST_WEAVIATE=1`)
  - A Milvus collection that is not loaded is loaded for the read and released again
  - The MCP server reads network stores only with `--allow-network`
    (`VECSCOPE_ALLOW_NETWORK=1`)
- Incremental exact k-NN (`vecscope.incremental`, `audit(..., knn=)`, CLI `--knn-cache`)
  - Neighbor lists cached with a SHA-256 fingerprint of their rows, reused for the same rows,
    extended for appended rows (new rows against all, old rows against the new ones) and
    rebuilt otherwise; `Evidence.knn_cache` reports which
  - Equal to a full computation except among exact ties
  - On 100k x 768, appending 5k rows audits in 15 s against 94 s for a full audit, 20k rows
    in 47 s against 123 s
- Agent integration: `tests/test_agent_workflow.py` runs audit, plan step, change and
  re-audit over the MCP stdio protocol; README and Korean README give Claude Code, Cursor and
  Codex configurations
- `tests/test_retrieval.py`, `tests/test_plan_incremental.py`, `tests/test_adapters.py`
- `tests/test_query_findings.py`: gap ratio arithmetic, zero without distinct corpus
  neighbors, query_shift against a JSON round-tripped baseline, baseline validation, CLI
  and MCP wiring, the tie slack against the gaps 0.6.0 credited

### Changed

- ANN recall tie slack reduced from `1e-6 + 1e-5 x distance` to `1e-6 + 2e-6 x distance`
  - On 100k x 768 clustered vectors the 0.6.0 slack credited 11th neighbors 1.3e-6 to 2.7e-6
    farther than the 10th (two to six times the measured float32 error of at most 5.9e-7),
    +0.0003 recall against a strict scorer at IVF nprobe=1
  - With 2e-6 all 24 settings of the exactness run (six shapes x HNSW and IVF) are within
    1e-4 of the strict scorer, the 57 real (model, dataset) pairs within 3.1e-4 (0.6.0:
    6.2e-4), and exact ties still match
- The query mismatch injection test checks `query_gap` and `query_shift` alongside
  `ood_queries`; `describe_findings` lists the two new findings
- README, Korean README, skill and verification page: the query section describes
  `query_gap` and `query_shift`
  - It states that a missing query prefix is not detected by any query finding (on SciFact
    and NFCorpus dropping the bge/e5/mxbai prefix moved nDCG@10 by -0.014 to +0.007 and the
    query distances in either direction)
  - It states that models fine-tuned from the same base (bge-large and mxbai, bge-base and
    gte-base, bge-m3 and its Korean fine-tunes) share a space, so swapping them is not always
    a fault
- Verification page: correlations of every finding with nDCG@10 for 19 models on six labeled
  datasets, the intervention study and the agent run
  - Datasets: ArguAna, SCIDOCS and FiQA added, none used to set a threshold
  - Intervention study: centering, deduplication, z-scoring a dominant coordinate, ef_search
    against retrieval quality on 21 real pairs
  - `docs/retrieval_quality.md` regenerated with `query_gap`
- `examples/retrieval_quality.py` loads models without the `dtype` keyword, which
  transformers before 4.56 rejects
- Korean documents: ranges written with `~` are escaped so GitHub does not render them as
  strikethrough
- `query_hubness` explanation, README row and skill state the measured healthy range and
  that it grows with corpus size
  - Good English models 1.5 to 3.4 up to 26k documents, 4.0 to 4.8 on FiQA's 58k
  - The thresholds stay (4, 10), since FiQA was held out from calibration
- The hubness action and the corpus-structure plan step name CSLS against the queries
  (2 cos(q, d) minus d's mean cosine to its 10 nearest queries) instead of "mutual k-NN or
  local scaling"
  - On 21 real pairs it raised nDCG@10 on all 4 with `query_hubness` not ok, while the same
    scaling by corpus neighbors lowered nDCG@10 on 16 of 21
- CI runs the vector store adapter and agent workflow tests against local Qdrant, Chroma and
  Milvus Lite in a separate Linux job with the `stores` extra
- MCP `audit_embeddings`: every response carries an `audit_id` (a content hash)
  - `baseline` takes an id from this server session (the last 32 audits are kept in memory)
    or an evidence file and adds `baseline_diff` with `tolerances`, as `--baseline` does in
    the CLI
  - `summary` names retrieval nDCG, the first plan step and the baseline comparison
  - `verbose=false` also drops explanations of ok findings and the thresholds table
- `corpus_fingerprint` on every Evidence (SHA-256 of the rows)
  - `--baseline` (CLI) and `baseline` (MCP) are the `query_shift` reference only for the same
    corpus or an unchanged prefix of it
  - After a transform the earlier query distances are in another space and flagged every
    query as shifted, so `query_shift` is skipped with a `query_reference_note`
- qrels files that are not text, or have another extension, are refused with the file name
  instead of a decoding error

### Deprecated

- `ood_queries` (`queries.ood_query_ratio`): its cutoff also counts the normal distance
  between short queries and long passages
  - It stays in 0.7.0 with a deprecation note in its explanation and is removed in 0.8.0;
    read `query_gap` instead

## [0.6.0] - 2026-10-30

### Changed (from the verification run, docs/verification.md)

- Thresholds re-calibrated on 19 public models against nDCG@10 on SciFact, NFCorpus and
  ko-strategyqa: `anisotropy.mean_random_cosine` (0.5, 0.85) to (0.85, 0.95), because the
  best retrievers sit at 0.58 to 0.82 with no quality cost; `queries.ood_query_ratio`
  (0.05, 0.20) to (0.5, 0.8), because short queries against long passages put good models at
  0 to 0.65 and only a wrong model or language reaches 0.7 to 1.0; `queries.query_hubness_skewness`
  (2, 4) to (4, 10), because good multilingual models on Korean data sit at 2.3 to 9.1 and
  wrong-language models at 11 to 30. Baselines written with the old thresholds are reported
  as not `comparable`
- ANN recall tie slack reduced from `1e-6 + 1e-4 x distance` to `1e-6 + 1e-5 x distance`: the
  old slack was about 70x float32 rounding and credited an 11th neighbor up to 7e-5 farther
  than the 10th, inflating recall by up to +0.0016 against a strict scorer. Exact ties
  (duplicates) are still interchangeable; `ann_recall` and `sweep` move together
- Sampled audits: hubness, antihubs and connectivity verdicts other than `fail`, and outlier
  verdicts other than `ok`, are marked `inconclusive` (new `Finding.inconclusive`,
  `Evidence.inconclusive_findings`, `gate.inconclusive`, text marker `INCONCLUSIVE (sampled)`),
  because on 200k x 384 a full-corpus connectivity `fail` (0.372) read `ok` (0.0008) on a 25%
  sample and a hubness `warn` read `ok`. The collapse verdict is taken from the estimated full
  share `value / fraction` (`details.estimated_full_share`), so a full-corpus `fail` no longer
  reads `warn` on a sample. `value_range` is documented as the spread between samples, not
  an interval for the full value
- Sweep caveat: HNSW recall falls with corpus size, IVF recall can move either way because
  `nlist` follows the row count (observed 0.696 at 100k to 0.955 at 500k on the clustered
  synthetic corpus, whose n/200 centers line up with nlist)
- README: performance table replaced with measured seconds and memory for 384, 768 and
  1,536 dims up to 200k rows (8-vCPU EPYC 7R13, three-run medians within 2% of single runs,
  tracemalloc within 0.04 GiB of the OS peak); Cohere ko `distance_concentration` 1.73 to 1.72;
  all 88 "Real embeddings" verdicts reproduced; ada-002 anisotropy is `ok` under the new threshold
- Golden evidence regenerated for the threshold change (`anisotropic.npy` is now `warn`)

### Added

- `docs/verification.md`: what was measured and what changed because of it;
  Korean documents moved under `docs/ko/` (`README.md`, `verification.md`,
  `retrieval_quality.md`), linked from each English page and from the Korean README
- Tests moved in from the verification: `tests/test_new_features.py` (sweep latency and
  `index_bytes` against an independent serialization, GPU/CPU agreement, compare with supplied
  neighbors, gate table, `--fail-on none`) and `tests/test_scale_exactness.py` (exact_knn,
  query_knn and faiss flat against float64 on six corpus shapes, sweep recall against an
  independent strict scorer; `-m slow` for 100k rows, `-m gpu` for the GPU baseline), with
  `tests/exactness_helpers.py` and `examples/bench_common.py` (six seeded corpus shapes and a
  machine description)
- `examples/benchmark_sweep.py`: index sweep per corpus shape with `--centers`, machine
  description in the JSON; `benchmark_scale.py` takes `--corpus` and records the machine too

- `compare(..., neighbors_a=, neighbors_b=)`, CLI `--neighbors-a/--neighbors-b`, MCP
  `neighbors_a_path`/`neighbors_b_path`: neighbor lists computed on the full corpus in each
  space (exact or from an index), so a sampled compare keeps full-corpus neighbors. Every
  result carries `neighbor_scope` ("full" or "sample") and `neighbor_source` ("recomputed" or
  "supplied"), the text header shows them, and a sample-scope explanation says that the
  agreement is about the sample's own neighbor graph (on a 2k-row synthetic pair a 500-row
  sample reports top-10 overlap 0.75 where the full corpus gives 0.57)
- `vecscope audit --fail-on none`: never fail on the absolute thresholds, so with
  `--baseline` only regressions decide the exit code. Every audit run ends with a `gate`
  block (JSON) or line (text) that separates the absolute rule from the baseline rules
  (`absolute`, `verdict_regressions`, `value_regressions`, `reasons`, `exit_code`);
  `vecscope.cli.gate_decision` is the library call
- `sweep`: each point carries single-query latency percentiles `latency_ms_p50/p95/p99`
  (`--latency-queries`, MCP `latency_queries`, default 200, 0 to skip) and `index_bytes`, the
  serialized index size; the text table shows p50, p95 and MiB, the recommendation carries
  p95 and size, and two caveats say what the numbers are and are not
- Sampled audits now state in the explanation of hubness, antihubs, collapse, outliers and
  connectivity what the sampled value stands for, with the sample fraction: hubness and
  antihubs are a lower bound (k-occurrence concentrates with density; a clustered 6k-row
  corpus scores skewness 1.6 on all rows, 0.9 on half, 0.0 on a quarter), the collapse share
  is about fraction times the full share, outliers and islands are those inside the sample
- Tests (`tests/test_regression_injection.py`): a mismatched query model trips only
  `ood_queries`; lowering HNSW `ef_search` lowers `ann_recall` monotonically into fail;
  injected duplicates move only `collapse`, in proportion; a 25% and 50% sample keep the
  verdict and stay within 5% on the distribution findings while hubness falls and collapse
  scales with the fraction; compare's neighbor scope and supplied neighbors; the gate
- `examples/benchmark_scale.py`: wall-clock and peak NumPy memory of a full and a sampled
  audit at chosen row counts, as a Markdown table

### Changed

- README performance table of what a sample can say: hubness and antihubs moved from
  "distribution estimate" to "lower bound", collapse states the fraction rule; the
  retrieval-quality section says what the study has not shown (other domains, languages,
  production query logs)

## [0.5.0] - 2026-10-29

### Added

- `diff_evidence(new, old, tolerances=...)` and `vecscope audit --baseline x --tolerances
  tol.json`: a per-finding budget for how much a value may worsen against the baseline.
  Findings beyond it are listed under `value_regressions` and fail the run even when the
  verdict band did not change; findings without a tolerance keep the verdict-only rule. Every
  change row now carries `lower_is_worse`, `worsened`, `worsening` (positive = worse) and
  `tolerance`, so drift inside a band is visible without a tolerance
- `--require-comparable`: exit 2 when the baseline was made with different n, dim, k,
  sampling, thresholds or near-duplicate cutoff, instead of printing the diff with a note
- MCP server path restriction: `vecscope-mcp --allow-dir DIR` (repeatable) or
  `VECSCOPE_ALLOWED_DIRS`; every tool resolves its path arguments and refuses one outside the
  allowed directories before opening it. Unset, nothing changes
- Tests: an end-to-end MCP round trip over stdio through the `mcp` client (initialize, list
  tools, a successful call, a refused path), and one check that `audit`, `compare` and
  `sweep` reject NaN/Inf, oversized values, 1-D input, zero columns and non-numeric dtypes
  with the same messages
- README: per-finding table of what a sampled audit can and cannot say, the compute cost
  behind the memory bound, and a procedure for calibrating thresholds and tolerances on
  labeled queries with a held-out split; the skill file points agents at it

## [0.4.0] - 2026-10-29

### Added

- `--baseline old.json` / `diff_evidence(new, old)`: per-finding value and verdict changes
  against an earlier run, `regressions` for verdicts that got worse, exit 1 on regression.
  Evidence JSON carries `schema_version` (1) and `near_dup_cos`; a baseline from another
  schema is rejected, and one made with a different near-duplicate cutoff or threshold set is
  marked not `comparable` like a different k, so a changed setting is not read as a regression
- Reference tests (`tests/test_reference.py`): the k-NN kernel against a naive full distance
  matrix for all metrics, hubness skewness against SciPy, effective rank of rank-r data,
  hand-computed ann_recall and collapse ratios, and audit invariance to row order, rotation
  and uniform scale; golden evidence for the example matrices (`tests/golden/`, regenerated
  by `examples/make_examples.py`, which now exposes `make_examples()`)
- MCP `sweep_index` takes `n_queries` like the CLI `--n-queries`
- `ann_recall` reports `rows_with_fewer_than_k_ids_ratio` and says in the explanation when the
  ANN file supplied fewer than k ids after dropping self ids, which caps recall below 1;
  `effective_rank` details carry `max_attainable_rank` and the explanation says when the row
  count, not the dimension, bounds the rank; `effective_rank` and `query_hubness` carry their
  numbers in `details`

### Changed

- The exact k-NN pass sizes its row block from the row count so peak memory stays near 1 GiB
  for the distance matrix at any size (100k x 768: 1.9 GiB instead of 4.1 GiB); results do
  not depend on the block size except among exactly tied distances. The README performance
  table lists memory only
- `anisotropy.mean_random_cosine` (and the value after centering) is now the exact mean over
  all pairs, computed in closed form from the sum vector, instead of a 20,000-pair sample.
  Values move by at most the old sampling error (about 0.003) and no longer depend on row
  order; calibration tables are unchanged at three decimals. The noise baseline of
  `distance_concentration` uses the same estimator (the old one was biased by up to 2% below
  200 rows)
- Exact-copy grouping compares rows by direction and by length relative to the data's scale
  (log scale, rounded) instead of at an absolute 1e-6, so vectors of size 1e-7 are no longer
  one duplicate group and short rows with their own direction are not copies of each other.
  Rows that differ by float32 noise can still split on a rounding boundary; they remain
  near-duplicates through the cosine cutoff, only the exact-copy statistics undercount them
- `compare(sample=0)` means every row, as in `audit`; both share one `resolve_sample`
  validator. The CLI and MCP already mapped 0 to None, only the library call differed
- The text output aligns the finding name column to the longest name
  (`distance_concentration`)

### Fixed

Verdicts and thresholds:

- A threshold override with `warn == fail` kept the default direction only by accident: for
  lower-is-worse keys (`ann_recall.recall_at_k`, `metric_mismatch.topk_overlap`,
  `anisotropy.effective_rank_ratio`, `distance_concentration.contrast_over_noise`) it flipped
  the verdict. The direction now always comes from the defaults
- `verify_by` for antihubs and rogue_dimensions names the metric the verdict uses
  (`antihub_excess`, `top1_dim_variance_ratio`); the `distance_concentration` explanation
  quotes the contrast measured on `baseline_rows`, which is the number the verdict uses
  (`relative_contrast_at_baseline_rows` in details); `distance_concentration` called directly
  without `x` on more than 4000 rows raises instead of mislabelling its baseline rows
- `anisotropy.effective_rank` of a constant input is 0, not 1 (float32 rounding left a
  non-zero spectrum)

ANN recall and sweep:

- `ann_recall` and `vecscope sweep` count hits by distance instead of by id: both sides are
  recomputed in float64 from the vectors and returned ids are assigned to exact neighbor
  slots by a maximum matching within a tie tolerance relative to the neighbor distance.
  Exact neighbors at one distance are interchangeable, so duplicate groups larger than k no
  longer cap recall (an exact index on a corpus with 8 copies per item scored 0.72 and FAIL,
  now 1.0), while a missed closer neighbor stays a miss
- `sweep` validates the corpus and the queries before casting to float32 (NaN, Inf and values
  beyond float32 get the "rescale" message instead of garbage recall); `--device gpu` without
  a GPU build prints a one-line error and exits 2 instead of a traceback
- An ANN neighbor array with zero columns is rejected with a clear error instead of
  producing NaN recall

Duplicates and degenerate corpora:

- `collapse.near_duplicate_ratio` counts every exact copy of another row, including zero
  vectors, whose cosine with their own copies is 0; an all-zero matrix used to pass collapse
  as ok. Their `sample_pairs` partner is another member of the exact group. The per-cluster
  `near_duplicate_ratio` and `compare` use the same rule, applied in float64, so the cluster
  map shows where zero-vector duplicates live. `nn_cosine_median` / `nn_cosine_p95` are
  clamped to 1
- `outliers` and `distance_concentration` report 0 with `distinct_neighbors: false` when every
  row's nearest neighbor is a copy of it, instead of measuring float32 rounding noise;
  `ood_queries` reports 0 with `corpus_has_distinct_neighbors: false` when every corpus row
  has an exact copy (there is no reference distance), instead of flagging every query
- `compare` treats two nearest partners that are exact copies of each other in both spaces
  as the same partner, so copy groups no longer show up as disagreements; it computes the
  copy groups of each space once instead of twice

Row ids and determinism:

- Row ids in `details.sample_ids`, `details.sample_pairs`, the cluster report and
  `compare`'s `sample_disagreements` are ids of the input rows when the audit was sampled;
  they used to index the subsample, so an agent opened the wrong rows
- `audit(seed=...)` is forwarded to the anisotropy SVD subset and the distance-concentration
  subsample as well as to row sampling; ordered id lists (`sample_ids`, `rogue_dims`,
  `ood_sample_ids`, cluster samples) use stable sorts; among exactly tied neighbor distances
  the selection is NumPy's, identical between runs on one NumPy build
- `audit(near_dup_cos=np.float32(...))` no longer leaves a NumPy scalar in the JSON

Input validation and loaders:

- Values above 1e16 in magnitude are rejected with a message to rescale: they are finite,
  but their squares overflow float32 and used to give silently wrong neighbors. The check
  runs before the float32 cast in `audit`, `compare` and `sweep`, and JSON/JSONL/parquet/text
  vectors are parsed as float64, so float64 values beyond float32 get the "rescale" message
  instead of "NaN or Inf"; the int64 minimum no longer slips past the check; vector norms are
  computed in float64 so large-but-safe values do not overflow during normalization
- Query vectors are validated before the corpus k-NN pass (a bad query file fails at once);
  a query file with zero rows is rejected (it produced NaN values and invalid JSON)
- Input with zero columns is rejected in `audit`, `compare`, `sweep` and the loaders (it
  crashed with a ZeroDivisionError, and `compare` reported perfect agreement); `compare`
  rejects `k` >= rows and `near_dup_cos` outside (0, 1] like `audit`; a threshold pair given
  as a string, and a malformed finding passed to `diff_evidence`, are rejected with a message
- Neighbor files: ids are read as exact integers everywhere (parquet ids above 2**24 were
  rounded; `.json`/`.jsonl` ids with a fractional part are rejected instead of truncated;
  `12.0` in CSV/TSV loads as an integer); `null`, `inf`, boolean arrays and unsigned ids
  beyond the int64 range are rejected as non-integer instead of failing with a TypeError
- Embedding files: an empty `.npy` or a truncated `.npz` is reported as a user error (exit 2 /
  ToolError) instead of an EOFError or BadZipFile traceback; a zip archive named `.npy` (or an
  array named `.npz`) is a clear error instead of an AttributeError; string or datetime
  dtypes, scalar JSON rows, nested lists, objects inside rows, booleans, empty JSON and empty
  CSV/TSV/TXT files get clear errors instead of a TypeError, a 3-D array or a NumPy warning
- A missing neighbor file and a `--baseline` / `--thresholds` file that is not JSON report a
  clear error (exit 2); the `--clusters` cap (one per 10 rows) is stated in the help texts
- `examples/make_examples.py` run outside the repository writes only the matrices: it checks
  for the vecscope source tree instead of any `tests/` directory next to it, so a downloaded
  copy no longer writes `tests/golden/` into a stranger's project

## [0.3.0] - 2026-10-28

### Added

- `sample_repeats` (`audit(sample_repeats=...)`, `--sample-repeats`, MCP `sample_repeats`):
  when sampling applies, re-audits independent seeded samples and attaches `value_range` and
  `verdict_stable` to each finding plus a top-level `unstable_findings` list
- Retrieval-quality study (`docs/retrieval_quality.md`, `examples/retrieval_quality.py`):
  every finding against nDCG@10 on BEIR SciFact and NFCorpus for eight public models and for
  controlled defects; README section summarizing what the findings do and do not predict
- `collapse` details now describe the duplicate structure: `near_duplicate_groups`,
  `largest_near_duplicate_groups`, `largest_exact_duplicate_groups` and `sample_pairs`
  (row, row, cosine; one pair per group, largest groups first), so an agent can tell
  duplicate items (many small groups) from an embedding failure (a few huge groups) without
  extra scripts. Groups link every neighbor within the cutoff among the k audited, so a
  jittered boilerplate cluster is one group rather than many nearest-neighbor pairs. The
  explanation and the suggested action say which is which; the action previously pointed at
  a `sample_ids` field that collapse never had
- `vecscope compare A B` / `compare()` / MCP `compare_embeddings`: agreement between two
  embeddings of the same rows (nearest-neighbor and top-k overlap, near-duplicate share in
  each space and in both, same duplicate partner, largest disagreements); no verdicts
- Sweep recommendations carry `margin` (recall minus target) and are marked `tight` with a
  `safer_alternative` when the margin is under `--margin` (default 0.02); results carry
  `caveats` (in-corpus queries, recall falls with corpus size, qps is relative) which the CLI
  prints and the MCP docstring tells the agent to relay. qps is the median of
  `--timing-repeats` runs (default 3) instead of a single timing
- Project metadata: description, keywords and Changelog/Documentation URLs on PyPI
- README: troubleshooting for an MCP server that stops connecting after the Python
  environment is rebuilt, with a `python -m vecscope.mcp_server` registration

### Changed

- `anisotropy` thresholds (0.3, 0.6) -> (0.5, 0.85): the best models on labeled data sit at
  0.58 to 0.81 with no ranking cost, and a shared offset up to 0.98 left nDCG@10 unchanged
- `distance_concentration` now reports `contrast_over_noise`: the relative contrast divided
  by that of seeded isotropic noise at the same dimension (`details` keep `relative_contrast`
  and `noise_relative_contrast`). The raw contrast of pure noise falls from 1.43 at 128-d to
  1.10 at 1536-d, so no fixed threshold worked across models; the ratio is 1.0 for noise at
  any width. Thresholds (1.4, 1.1): noise that halves nDCG@10 gives 1.05 and noise that
  costs 4% gives 1.27; sixteen real model/corpus pairs sit at 1.43 to 2.9. Threshold key renamed to `distance_concentration.contrast_over_noise`; an override
  that still uses the old key gets an error naming the new one. The nearest-neighbor
  distance also shrinks as rows are added, so the data is measured on a seeded subsample of
  the same size as the noise baseline (4000 rows; `details.baseline_rows`): pure noise is
  1.0 at any n instead of drifting to 1.1 at 100k rows
- `ood_queries` drops duplicate corpus rows with the same relative cutoff as the other
  findings instead of a fixed 1e-4
- MCP `sweep_index` exposes `margin` and `timing_repeats`
- `examples/healthy.npy` and the healthy test scenarios now have cluster structure; pure
  isotropic noise is reported as having no neighbor signal, which is correct

### Fixed

- Text loaders (`.csv`, `.tsv`, `.txt`) read UTF-8 with a BOM and always return 2-D arrays
  (a single-row file used to load as 1-D)

## [0.2.0] - 2026-10-27

### Added

- `thresholds` override (`audit(thresholds=...)`, `--thresholds file.json`, MCP `thresholds`);
  effective thresholds are echoed in the JSON result
- Every finding carries `likely_causes`, `suggested_action` and `verify_by`; the text output
  adds a `try:` line for findings that are not `ok`
- Scenario test suite (`tests/test_scenarios.py`): 13 synthetic defects, 11 degenerate inputs,
  determinism
- README: scope statement (vector search implementation, not relevance), threshold caveats,
  sampling caveats, performance table
- MCP: server instructions with the diagnose/change/re-verify loop, read-only tool
  annotations, `sample` and `verbose` parameters, a one-line `summary` first in every response,
  and a `describe_findings` tool

### Changed

- Agent skill rewritten around a diagnose → smallest change → re-audit → compare loop
- `antihubs` now reports the excess over the random baseline e^-k (`antihub_excess`), so small
  k no longer trips it on healthy data; `rogue_dimensions` now reports the top dimension's
  variance as a multiple of the uniform 1/dim share (`top1_dim_variance_ratio`, warn 10x,
  fail 50x) instead of an absolute share that always warned on low-dimensional vectors
- `ann_neighbors` combined with sampling now raises instead of being silently dropped

### Fixed

- `connectivity` capped label propagation at 1000 rounds and reported one long-chain component
  as thousands of islands; it now converges fully (pointer jumping)
- `vecscope sweep` removed "self" by position, so duplicate or zero rows capped recall below
  1.0; self is now removed by id for both exact and approximate lists
- ANN neighbor files are validated (shape, integer ids, id range); ids that point at the item
  itself are ignored; ids are loaded as integers so values above 2^24 survive
- Parquet loader picks a list-of-numbers column by type (no longer grabs `embedding_model`)
- JSON/JSONL loader skips non-numeric lists (tags), rejects empty files and ragged rows with
  clear errors
- `audit` validates `k`, `sample`, `clusters`, `near_dup_cos`, `thresholds` (type, finiteness)
  and rejects complex/object dtypes with clear errors instead of crashing deep inside NumPy
- `sweep` validates `k < n`, `n_queries >= 1`, `target_recall`, query dimensionality
- MCP tools raise `ToolError`, so the real message (missing file, bad k) reaches the agent
  instead of "Error executing tool"
- CLI prints a one-line error and exits 2 on bad input instead of a traceback

## [0.1.1] - 2026-10-26

### Fixed

- MCP server failed to import with `mcp` 2.x, which renamed `FastMCP` to `MCPServer`; both
  1.x and 2.x are now supported and the server is smoke-tested in CI

### Added

- Python 3.14 classifier and a `.gitattributes` that normalizes line endings

## [0.1.0] - 2026-10-25

### Added

- `vecscope audit` CLI with text and `--json` output, `--fail-on` gate and non-zero exit on FAIL
- Findings: hubness, antihubs, anisotropy, effective rank, collapse, rogue dimensions, norms,
  metric mismatch, distance concentration, outliers, k-NN graph connectivity, ANN recall
- Seeded sampling (`--sample`, default 100k rows) so large matrices stay tractable
- MCP server (`vecscope-mcp`) exposing `audit_embeddings`
- Query-side findings (`--queries`): out-of-distribution queries and query hubness
- Per-cluster report with sample ids (`--clusters N`)
- `vecscope sweep` / MCP `sweep_index`: HNSW and IVF parameter sweep with recall, build time and
  QPS, plus a recommended configuration (optional `faiss-cpu` extra); IVF and the exact
  baseline run on the GPU automatically with a faiss GPU build (`--device auto|cpu|gpu`)
- Loaders for `.npy`, `.npz`, `.parquet`, `.jsonl`, `.json`, `.csv`
- Agent skill file describing when to run the audit and how to act on verdicts
- Each finding reports `healthy_range`, `warn_at`, `fail_at`; `anisotropy` also reports the
  mean-vector norm and the random-pair cosine after centering
- Thresholds calibrated on public embeddings from seven models over eight datasets (OpenAI ada-002 and 3-large,
  Cohere embed-v3, all-MiniLM-L6-v2, bge-m3, mxbai-embed-large-v1, gte-base-en-v1.5);
  `examples/real_models.py` reproduces the table in the README
