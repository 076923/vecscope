<div align="center">

[English](../../README.md) · **한국어**

# vecscope

### 에이전트를 위한 임베딩 공간 진단 도구

**벡터 인덱스를 신뢰하기에 앞서, 에이전트로 검색 공간의 상태부터 점검하세요.**

[![PyPI version](https://img.shields.io/pypi/v/vecscope?style=flat-square&color=3775a9)](https://pypi.org/project/vecscope/)
[![Python](https://img.shields.io/pypi/pyversions/vecscope?style=flat-square)](https://pypi.org/project/vecscope/)
[![CI](https://img.shields.io/github/actions/workflow/status/076923/vecscope/ci.yml?style=flat-square&label=CI)](https://github.com/076923/vecscope/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-f4c430?style=flat-square)](../../LICENSE)

[빠른 시작](#빠른-시작) · [진단 항목](#진단-항목) · [질의와 군집](#질의와-군집) · [인덱스 스윕](#인덱스-스윕) · [두 임베딩 비교](#두-임베딩-비교) · [검색 품질과 수정 순서](#검색-품질과-수정-순서) · [벡터 저장소](#벡터-저장소에서-읽기) · [에이전트와 함께 사용](#에이전트와-함께-사용) · [CI에서 사용](#ci에서-사용) · [입력 형식](#입력-형식) · [실제 임베딩 검증](#실제-임베딩-검증) · [FAQ](#faq)

<code>pip install vecscope</code>

</div>

---

에이전트는 검색 증강 생성(RAG), 의미 기반 검색(semantic search), 추천 검색을 쉽게 구축하지만,
정작 자신이 검색하는 임베딩 공간은 들여다보지 못합니다.

그러므로 검색 품질 문제는 버그처럼 드러나지 않고 다음과 같은 증상으로만 나타납니다.

- 무엇을 물어도 같은 청크(chunk) 몇 개가 돌아옵니다.
- 유사도 점수가 전부 0.8 근처에 몰려 있습니다.
- 상위 k개가 중복 항목으로 채워집니다.
- 전수 탐색(brute-force)으로는 검색되는 항목이 HNSW에서는 빠집니다.

이런 증상은 정립된 진단 지표로 측정할 수 있습니다. `vecscope`는 그 지표들을
명령 하나, JSON 하나로 묶었습니다. 결과에는 다음이 함께 들어 있습니다.

- 측정값
- `ok` / `warn` / `fail` 판정
- 에이전트가 다음 조치를 정할 때 참고할 설명

NumPy만 활용하여 로컬에서 동작하고, 모델이나 네트워크 연결이 필요 없으며, 같은 입력에는
언제나 같은 결과를 출력합니다.

`vecscope`가 진단하는 대상은 **벡터 검색 구현**입니다. 모델이 만든 공간의 기하, 코퍼스의
상태, 거리 척도, 인덱스, 질의 파이프라인이 여기에 해당합니다.

검색 결과의 관련성은 측정하지 않습니다. 질문에 맞는 문서가 실제로 검색되는지는 정답이 있는
질의로 따로 평가해야 하며, 감사 결과가 모두 `ok`라는 것은 검색을 방해하는 요인이 없다는
근거이지 검색 품질이 좋다는 증명은 아닙니다.

## 빠른 시작

### 1. 설치

```bash
pip install vecscope
```

### 2. 임베딩 파일 준비

`vecscope`는 행마다 임베딩 벡터가 하나씩 담긴 파일을 읽습니다. 모델의 `encode`가 반환하는
`(n, dim)` 배열을 `.npy`, `.parquet`, `.jsonl`, `.json`, `.csv` 중 하나로 저장하면 됩니다
([전체 형식](#입력-형식)).

```python
import numpy as np
embeddings = model.encode(texts)          # 형태 (n, dim), 예: (3000, 128)
np.save("embeddings.npy", embeddings)
```

준비된 임베딩이 없다면 합성 행렬 세 개(정상, 비등방성, 중복)를 만들어 시험해 볼 수 있습니다.

```bash
python -c "import urllib.request as u; u.urlretrieve('https://raw.githubusercontent.com/076923/vecscope/main/examples/make_examples.py', 'make_examples.py')"
python make_examples.py
```

### 3. 감사 실행

```bash
vecscope audit embeddings.npy
```

```text
vecscope 0.7.1  n=3000 dim=128 k=10  verdict=FAIL
  [ok  ] hubness                skewness=1.1661  (ok < 2, warn >= 2, fail >= 4)
                               k-occurrence skewness 1.17; top 1% of items take 3.9% of all neighbor slots (max N_k=51, expected 10).
  [WARN] antihubs               antihub_excess=0.0523  (ok < 0.05, warn >= 0.05, fail >= 0.2)
                               5.2% of items are never anyone's neighbor (random data at k=10 would give 0.0%); the excess is unreachable through k-NN retrieval.
                               try: inspect a sample of never-retrieved items; drop empty inputs, then apply the hubness fix
  [FAIL] anisotropy             mean_random_cosine=0.8732  (ok < 0.5, warn >= 0.5, fail >= 0.85)
                               random pairs have mean cosine 0.873 (isotropic ~0); mean unit vector has norm 0.93; centering would bring the cosine to -0.000.
                               try: subtract the corpus mean vector and re-normalize, applying the same transform to queries; details.mean_random_cosine_after_centering shows the expected result
  [ok  ] effective_rank         effective_rank_ratio=0.5407  (ok >= 0.1, warn < 0.1, fail < 0.05)
                               effective rank 69.2 of 128 attainable (128 dims; 54.1%); low values mean most dimensions carry nothing.
  [ok  ] collapse               near_duplicate_ratio=0.0000  (ok < 0.02, warn >= 0.02, fail >= 0.1)
                               0.0% of items have a neighbor with cosine >= 0.99 or an exact copy (exact duplicates 0.0%); no row has a near-duplicate.
  [ok  ] rogue_dimensions       top1_dim_variance_ratio=1.3070  (ok < 10, warn >= 10, fail >= 50)
                               dimension 94 alone holds 1.0% of variance, 1.3x the uniform share of 0.78%; top 3 dims hold 3.0%. Rogue dimensions dominate cosine.
  [ok  ] norms                  norm_cv=0.0000  (ok < 0.1, warn >= 0.1, fail >= 0.3)
                               vector norms vary with CV 0.000 (median 1.000, range 1.000-1.000); 0.00% are zero vectors. Unequal norms bias inner-product search toward long vectors.
  [ok  ] metric_mismatch        topk_overlap=1.0000  (ok >= 0.9, warn < 0.9, fail < 0.7)
                               cosine and inner-product top-10 share 100.0% of neighbors; 100.0% of items get identical lists. Low overlap means the index metric changes the results.
  [ok  ] distance_concentration contrast_over_noise=2.2784  (ok >= 1.4, warn < 1.4, fail < 1.1)
                               on 3000 rows a random item is 3.27x farther than the nearest neighbor (random distance 0.127); isotropic noise at 128-d and the same rows would give 1.44x, so the contrast is 2.28x that of noise. On all rows the contrast is 3.27x (median NN distance 0.039). Near 1.0 means distances carry no signal.
  [ok  ] outliers               outlier_ratio=0.0030  (ok < 0.01, warn >= 0.01, fail >= 0.05)
                               0.30% of items have no neighbor within cosine distance 0.050 (median 0.039); see details.sample_ids. Typically junk, empty or foreign-language content.
  [ok  ] connectivity           island_ratio=0.0000  (ok < 0.01, warn >= 0.01, fail >= 0.1)
                               0.00% of items sit in 0 small island(s) of the k-NN graph (fewer than 15 items; 60 components in total). Graph-based ANN (HNSW) attaches islands to the wrong region or never reaches them.
```

### 4. 결과 읽는 법

- 첫 줄의 `verdict`가 전체 판정입니다. 항목별 판정 가운데 가장 낮은 등급을 따릅니다.
- 항목마다 측정값, `ok`로 판정되는 범위, 그 수치가 검색에서 무엇을 뜻하는지 설명하는 문장이
  차례대로 출력됩니다. `ok`가 아닌 항목에는 먼저 시도할 변경을 적은 `try:` 줄이 붙습니다.
- 판정이 `FAIL`이면 종료 코드가 1이므로 같은 명령을 CI 게이트로 쓸 수 있습니다.
- `--json`을 붙이면 같은 내용을 에이전트나 스크립트가 읽기 쉬운 JSON으로 출력합니다.

## 진단 항목

| 항목 | 지표 | ok | warn | fail | fail 판정의 의미 |
| --- | --- | --- | --- | --- | --- |
| `hubness` | k-출현 빈도(k-occurrence) 분포의 왜도(skewness) | < 2.0 | 2.0\~4.0 | ≥ 4.0 | 소수의 항목이 대부분의 질의에서 k-최근접 이웃(k-NN)에 포함되어 검색 결과 상위를 독점한다. |
| `antihubs` | 어떤 질의에서도 이웃으로 선택되지 않는 항목의 비율에서 무작위 기준선 e^-k를 뺀 값 | < 5% | 5\~20% | ≥ 20% | 코퍼스(corpus)의 일부에 k-NN 검색으로 도달할 수 없다. |
| `anisotropy` | 무작위 쌍의 평균 코사인 유사도 | < 0.85 | 0.85\~0.95 | ≥ 0.95 | 벡터가 좁은 원뿔 영역에 몰려 점수가 좁은 구간에 압축된다. 검색 성능이 좋은 모델도 이보다 훨씬 낮은 값에서 nDCG 손실 없이 동작하므로 극단적인 값만 표시한다. |
| `effective_rank` | 공분산 스펙트럼의 참여 비율(participation ratio)을 도달 가능한 랭크(차원 수, 행 수가 차원 수 이하이면 행 수 − 1)로 나눈 값 | ≥ 0.10 | 0.05\~0.10 | < 0.05 | 대부분의 차원이 정보를 담고 있지 않다. |
| `collapse` | 코사인 유사도 0.99 이상인 이웃이 있거나 정확히 같은 행이 있는(0 벡터 포함) 근접 중복(near-duplicate) 비율. details에 중복 묶음과 예시 쌍이 있다. | < 2% | 2\~10% | ≥ 10% | 상위 k개가 동일 항목의 복사본으로 채워진다. |
| `ann_recall` | 정확한 k-NN 대비 근사 최근접 이웃(ANN) 인덱스의 재현율(recall@k). 거리 기준으로 세므로 완전 동점인 중복 항목은 서로 바꿔 세며, 동점 허용 오차는 1e-6 + 2e-6 × 거리(float32 반올림 오차의 2\~3배)다 | ≥ 0.95 | 0.85\~0.95 | < 0.85 | 모델이 아니라 인덱스에서 결과가 누락되고 있다. |
| `rogue_dimensions` | 분산이 가장 큰 차원의 분산이 균등 분배(1/dim) 대비 몇 배인지 | < 10배 | 10\~50배 | ≥ 50배 | 소수의 좌표가 모든 코사인 점수를 결정한다. |
| `norms` | 벡터 길이(norm)의 변동 계수(coefficient of variation) | < 0.10 | 0.10\~0.30 | ≥ 0.30 | 내적(inner product) 검색이 긴 벡터를 우대한다. 길이가 0인 벡터의 비율도 함께 보고한다. |
| `metric_mismatch` | 코사인과 내적의 상위 k개 겹침 비율 | ≥ 0.90 | 0.70\~0.90 | < 0.70 | 인덱스의 거리 척도(metric)에 따라 반환 결과가 달라진다. |
| `distance_concentration` | 상대 대비(무작위 거리 / 최근접 이웃 거리)를 같은 차원·같은 행 수의 등방성 잡음이 갖는 값으로 나눈 비율 | ≥ 1.4 | 1.1\~1.4 | < 1.1 | 최근접 이웃이 무작위 항목과 거의 같은 거리에 있다. 1.0이면 순수 잡음이다. |
| `outliers` | 최근접 이웃이 비정상적으로 먼 이상치(outlier) 항목의 비율 | < 1% | 1\~5% | ≥ 5% | 불량 항목, 빈 문서, 다른 언어로 된 항목으로 id를 `details`에 나열한다. |
| `connectivity` | k-NN 그래프의 작은 섬에 속한 항목 비율 | < 1% | 1\~10% | ≥ 10% | 군집(cluster)으로 보기에는 너무 작은 고립 집단. HNSW가 잘못 연결하거나 도달하지 못한다. |
| `query_gap` | 질의→코퍼스 최근접 거리의 중앙값을 코퍼스→코퍼스 최근접 거리의 중앙값으로 나눈 값 | < 2.6 | 2.6\~3.0 | ≥ 3.0 | 질의와 코퍼스가 같은 공간에 있지 않다. 대표적인 원인은 질의를 다른 모델로 임베딩한 경우로, 5\~10이 나온다. 공개 nDCG@10을 재현한 파이프라인은 0.94\~2.52이고, 질의 접두어 누락으로는 이 값이 일정하게 움직이지 않는다. |
| `query_shift` | 기준 질의 집합의 중앙값 + 3 MAD보다 코퍼스에서 먼 질의의 비율(기준선 필요) | < 20% | 20\~30% | ≥ 30% | 기준 시점 이후 질의 파이프라인이 바뀌었다. 다른 모델이나 버전, 또는 코퍼스가 다루지 않는 주제의 질의가 원인이다. 정상 질의 집합은 0\~9%다. |
| `ood_queries` | 지원 중단, `query_gap`을 볼 것: 코퍼스 항목 간 거리의 99% 지점보다 코퍼스에서 더 먼 질의의 비율 | < 50% | 50\~80% | ≥ 80% | `query_gap`과 같은 것을 묻지만 짧은 질의와 긴 문서 사이의 정상적인 거리까지 세므로(좋은 모델도 BEIR에서 0\~65%) 공개 nDCG를 재현한 파이프라인 18개 중 4개를 잘못 경고했다. |
| `query_hubness` | 질의 결과에서 코퍼스 항목의 k-출현 빈도 왜도 | < 4.0 | 4.0\~10.0 | ≥ 10.0 | 소수의 항목이 대부분의 질의에 답한다. 이 값은 모델 간 nDCG@10과 꾸준히 상관하며(Spearman −0.5\~−0.9), 언어가 맞지 않는 모델은 11\~30이 나온다. 코퍼스가 클수록 커져서 좋은 영어 모델도 문서 2.6만 개까지는 1.5\~3.4, 5.8만 개에서는 4.0\~4.8이므로 큰 코퍼스에서는 기준선과 비교한다. |

JSON의 각 항목에는 `healthy_range`, `warn_at`, `fail_at`과 세부 수치를 담은 `details`가
들어 있습니다. `anisotropy`의 `details`에는 같은 벡터를 중심화했을 때의 코사인 값까지 있어서,
중심화만으로 해결되는 문제인지 에이전트가 바로 판단할 수 있습니다.

일부 항목은 해당 입력을 넘겼을 때만 나타납니다.

- `ann_recall`: `--ann`으로 ANN 이웃 파일을 넘겼을 때
- `query_gap`, `query_hubness`, `ood_queries`: `--queries`로 질의 임베딩을 넘겼을 때
- `query_shift`: 여기에 더해 `--baseline`이 질의와 함께 감사한 이전 실행을 가리킬 때

각 항목에는 `likely_causes`(가능한 원인), `suggested_action`(먼저 시도할 조치),
`verify_by`(확인 방법)도 함께 들어 있습니다. 이는 처방이 아니라 가설입니다.

- 가장 작은 변경을 적용합니다.
- 다시 감사합니다.
- 실제 질의로 검색 결과를 전후 비교하고 나서 채택합니다.

[에이전트 스킬](../../.agents/skills/vecscope/SKILL.md)에 이 절차가 정리되어 있습니다.

임곗값은 [공개 모델 7개](#실제-임베딩-검증)로 보정한 뒤 두 언어의 라벨 데이터세트 6개에서 모델 19종으로
다시 확인한 기본값입니다([검증 결과](verification.md)). 진단의 출발점이지 보편적인
품질 기준은 아닙니다.

임베딩 공간은 모델, 언어, 도메인, 차원 수에 따라 달라지므로 좋은 모델에서도 `warn`은 흔하고,
감사 결과가 모두 `ok`라고 해서 검색 품질이 보장되지는 않습니다.

실행할 때마다 원하는 쌍을 덮어쓸 수 있습니다.

```bash
vecscope audit vectors.npy --thresholds my_thresholds.json
# my_thresholds.json: {"anisotropy.mean_random_cosine": [0.4, 0.7], "hubness.skewness": [2.5, 5]}
```

키 목록은 `src/vecscope/evidence.py`의 `THRESHOLDS`에 있고, 모든 JSON 결과의 `thresholds`
필드에도 그대로 실립니다.

## 질의와 군집

질의 임베딩을 함께 넘기면 질의가 코퍼스와 같은 공간에 있는지도 검사합니다.

- 질의 쪽에 다른 모델을 쓴 경우는 `query_gap`으로 드러납니다.
- 같은 청크 몇 개가 모든 질의에 답하는 현상은 `query_hubness`로 드러납니다. 임베딩 모델이
  코퍼스의 언어를 다루지 못하는 경우도 `query_hubness`가 잡습니다.
- 질의가 정상이던 시점의 실행을 기준선으로 넘기면, `query_shift`가 지금 질의 가운데 그때보다
  코퍼스에서 멀어진 질의가 얼마나 되는지 알려 줍니다. 일부 트래픽만 다른 모델이나 새 주제로
  바뀐 경우도 여기서 잡힙니다.

```bash
vecscope audit corpus.npy --queries queries.npy
vecscope audit corpus.npy --queries good_queries.npy --json > baseline.json
vecscope audit corpus.npy --queries todays_queries.npy --baseline baseline.json   # query_shift 추가
```

질의 접두어 누락(bge, e5)은 이 판정들로 잡히지 않습니다. SciFact와 NFCorpus에서 접두어를 빼도
nDCG@10은 −0.014\~+0.007만 바뀌었고, 질의 거리는 방향 없이 움직였습니다([검증 결과](verification.md)).
접두어는 질의를 임베딩하는 코드에서 확인하십시오.

`--clusters 10`을 붙이면 결과 끝에 군집별 요약이 추가됩니다. 군집마다 다음을 보여 줍니다.

- 크기
- 밀집도
- 근접 중복 비율
- 허브 점유율
- 에이전트가 직접 확인할 수 있는 샘플 id

판정은 하지 않으며, 문제가 *어느 영역에* 있는지 알려 주는 용도입니다.

```text
  clusters (3):
     id   size  share  cos2ctr  neardup  hubshare  sample_ids
      2   1113  37.1%    0.217     0.0%     37.0%  [1282, 1611, 496, 283, 2041]
      0   1023  34.1%    0.232     0.0%     34.2%  [1855, 2932, 1786, 1025, 2476]
      1    864  28.8%    0.222     0.0%     28.9%  [2505, 647, 2740, 2499, 2965]
```

## 인덱스 스윕

`vecscope sweep`은 HNSW와 IVF 인덱스를 몇 가지 파라미터 조합으로 구축한 뒤, 조합마다 다음을
실제 코퍼스에서 측정합니다.

- 정확한 k-NN 대비 재현율(recall@k)
- 구축 시간
- 초당 질의 수(QPS)
- 단일 질의 지연 시간의 백분위수(p50/p95/p99)
- 직렬화한 인덱스 크기

그리고 목표 재현율(기본값 0.95)을 만족하는 조합 가운데 가장 빠른 것을 추천하고, 목표 대비
여유(`margin`)를 함께 보고합니다.

여유가 0.02 미만이면 추천을 `tight`로 표시하고 여유가 충분한 조합 중 가장 빠른 것을 함께
제시합니다. 같은 파라미터라도 코퍼스가 커지면 재현율이 떨어지기 때문입니다. 5만 행 표본에서
목표를 간신히 넘긴 조합은 전체 코퍼스에서 대개 목표에 못 미칩니다.

`faiss-cpu`를 추가로 설치해야 합니다.

```bash
pip install "vecscope[index]"
vecscope sweep embeddings.npy -k 10 --target-recall 0.95
```

```text
  index params                                        recall  build s       qps  p50 ms  p95 ms     MiB  dev
 *hnsw  M=8, ef_construction=100, ef_search=16         0.987     0.35    237954   0.034   0.051     0.6  cpu
 *hnsw  M=16, ef_construction=100, ef_search=32        0.998     1.23    165139   0.084   0.108     0.8  cpu
 *ivf   nlist=76, nprobe=64                            0.999     0.14     33251   0.087   0.095     0.4  cpu
  recommendation: hnsw (M=8, ef_construction=100, ef_search=16)  recall=0.987  qps=237954  p95=0.051ms  index=0.6MiB  margin=+0.037  (fastest configuration with recall@5 >= 0.95)
  note: recall is measured on 200 queries drawn from the corpus itself; real query traffic can differ, pass it as queries when available
  note: HNSW recall at fixed parameters falls as the corpus grows, so a sweep on a sample is optimistic for HNSW; IVF nlist is set from the row count (4*sqrt(n)), so its recall at a given nprobe can move either way with size. Re-run at production size before fixing parameters
  note: queries_per_second is the median of 3 runs on this machine with one query batch; it ranks settings against each other and is not a serving throughput
  note: latency_ms_p50/p95/p99 time 200 queries one at a time in this process, with no network, concurrency or serving stack; compare settings with them, do not quote them as service latency
  note: index_bytes is the serialized index (vectors plus graph or lists); the process needs more while building, and a product-quantized index would be smaller
```

질의는 기본적으로 코퍼스에서 고정 시드로 표본을 뽑아 쓰고, `--queries`를 지정하면 실제 질의로
측정합니다. 질의 로그가 있다면 꼭 넘기십시오. 코퍼스에서 뽑은 질의로 고른 최적 조합이 실제
트래픽에서는 최적이 아닐 수 있습니다.

- 조합마다 세 번 측정해 중앙값을 씁니다(`--timing-repeats`).
- `--latency-queries`개의 질의(기본값 200, 0이면 생략)는 한 건씩 따로 검색해 지연 시간
  백분위수를 구합니다.
- `index_bytes`는 직렬화한 인덱스 크기라서 속도뿐 아니라 메모리 기준으로도 조합을 고를 수
  있습니다.

끝의 note 줄은 JSON에 `caveats`로 들어갑니다. MCP 도구 이름은 `sweep_index`이며 같은
옵션(`n_queries`, `target_recall`, `device`, `margin`, `timing_repeats`, `latency_queries`)을
매개변수로 받습니다.

스윕이 답하는 질문은 하나입니다. 이 코퍼스와 이 질의에서 어떤 파라미터가 목표 재현율을
만족하고, 그 비용이 얼마인가입니다. 수치는 측정한 장비와 프로세스 안에서의 상대값이므로,
고른 조합은 운영 환경에서 지연 시간·처리량·메모리를 따로 재어 확정하십시오.

GPU용 faiss(Linux에서는 `pip install faiss-gpu-cu12`)가 설치되어 있으면 IVF 인덱스와 정답
기준이 되는 정확 검색을 자동으로 GPU에서 실행하고, 출력 첫 줄에 사용한 장치를 표시합니다.
HNSW는 faiss에 GPU 구현이 없어 항상 CPU에서 실행됩니다. `--device cpu|gpu`로 장치를 직접
지정할 수도 있습니다.

## 두 임베딩 비교

같은 항목을 두 모델로 임베딩했거나, 한 모델을 바꾸기 전후로 임베딩했다면 `vecscope compare`로
두 공간이 이웃 관계에 얼마나 합의하는지 볼 수 있습니다. 다음을 보고합니다.

- 최근접 이웃과 top-k 목록이 일치하는 비율
- 각 공간과 양쪽 모두에서의 근접 중복 비율
- 양쪽이 같은 중복 상대를 고르는지
- 가장 크게 어긋나는 행

판정은 하지 않습니다. 용도는 항목 문제와 모델 문제를 가르는 것입니다.

- 두 공간이 같은 상대와 함께 잡은 10%의 collapse는 항목의 성질(같은 이미지가 여러 id로
  등록된 경우 등)입니다.
- 한 공간에서만 잡힌 중복은 그 모델을 가리킵니다.

```bash
vecscope compare clip.npy lens.npy -k 10          # 두 파일의 행은 같은 항목, 같은 순서여야 합니다
```

```text
vecscope 0.7.1  compare  n=3000 dim=128/128 k=10  neighbors=recomputed/full
  nn1_agreement=0.7050  topk_overlap=0.8537  topk_overlap_p10=0.7000  rows_with_no_overlap=0.0000
  near_dup A=0.0400  B=0.0400  both=0.0400  only_A=0.0000  only_B=0.0000  same_nearest_among_both=1.0000
    the two spaces share the nearest neighbor for 70.5% of rows and 85.4% of their top-10 lists
    on average (0.0% of rows share nothing). Near-duplicates: 4.0% in A, 4.0% in B, 4.0% in both,
    and 100.0% of those agree on the partner. Duplicates flagged by both spaces with the same
    partner are a property of the items; duplicates flagged by one space only point at that model.
  largest disagreements (row: nearest in A / nearest in B, top-k overlap):
    1769: 1340 / 1340  0.50
    1810: 1229 / 1229  0.50
```

일치율을 인용하기 전에 `neighbor_scope`부터 확인하십시오.

행 수가 `--sample`(기본값 10만)을 넘으면 고정 시드로 뽑은 표본을 비교하는데, 기본 동작은
이웃을 *표본 안에서* 다시 계산합니다. 진짜 이웃이 표본 밖에 있는 항목은 대신 다른 이웃과
비교되고, 이런 대체 이웃은 진짜 이웃보다 더 자주 일치합니다.

실제로 2,000행짜리 합성 공간 두 개에서 500행 표본의 top-10 일치율은 0.75였지만 전체
코퍼스에서는 0.57이었습니다. 이런 결과에는 `neighbor_scope: sample`이 붙고 설명문에도 그
사실이 적힙니다.

행만 표본으로 뽑되 전체 코퍼스 기준의 순위를 비교하려면, 각 공간에서 전체 코퍼스로 계산한
이웃 목록을 `(n x >= k)` id 파일로 넘기십시오. 정확한 k-NN 결과든 운영 인덱스의 결과든
상관없고, 자기 자신 id와 음수 채움값은 자동으로 제거됩니다.

```bash
vecscope compare clip.npy lens.npy -k 10 --neighbors-a clip_knn.npy --neighbors-b lens_knn.npy
```

그러면 결과에 `neighbors=supplied/full`이 표시됩니다. 행은 표본이어도 이웃은 전체 코퍼스
기준이며, 일치율은 넘긴 두 목록을 있는 그대로 비교한 값입니다. MCP 도구
이름은 `compare_embeddings`(`neighbors_a_path`, `neighbors_b_path`)입니다.

## 검색 품질과 수정 순서

진단 항목은 벡터 공간을 설명할 뿐이고, 맞는 문서가 돌아오는지는 정답 라벨이 있어야 알 수
있습니다.

질의의 관련성 판정(qrels)을 넘기면 감사 결과에 `retrieval` 블록이 붙습니다.

- 모든 행에 대한 정확한 코사인 검색의 nDCG@k, recall@k, MRR@k, 적중률(진단 항목을 표본으로
  감사할 때도 전체 행 기준)
- `duplicate_result_share`: 결과 칸 가운데 더 높은 순위 결과의 근접 중복이 차지한 비율

`--ranked`를 주면 운영 인덱스가 돌려준 top-k도 정확한 검색 옆에서 함께 채점하고
`overlap_with_exact`를 붙입니다.

```bash
vecscope audit corpus.npy --queries queries.npy --qrels qrels.tsv
vecscope audit corpus.npy --queries queries.npy --qrels qrels.tsv --ranked prod_topk.npy
```

`qrels`는 다음 중 하나입니다.

- 한 줄에 `질의 행, 코퍼스 행, 등급`을 담은 `.tsv`/`.csv`(머리글은 건너뜀)
- `{"질의 행": {"코퍼스 행": 등급}}` 형태의 `.json`
- 한 줄에 객체 하나인 `.jsonl`

BEIR의 `query-id`/`corpus-id`/`score` 이름도 읽습니다. 행 번호는 질의 파일과 코퍼스
파일에서의 위치입니다.

`--baseline`과 함께 쓰면 비교 결과에 진단 항목 변화와 나란히 `retrieval_changes`가
붙습니다. 그래서 어떤 변경을, 고치려던 진단 항목과 검색 품질 양쪽으로 판단할 수 있습니다.

```text
  retrieval vs baseline:
    exact   duplicate_result_share   0.0512 -> 0.0000  (-0.0512)
    exact   hit_rate_at_k            1.0000 -> 1.0000  (+0.0000)
    exact   mrr_at_k                 1.0000 -> 1.0000  (+0.0000)
    exact   ndcg_at_k                1.0000 -> 1.0000  (+0.0000)
    exact   recall_at_k              1.0000 -> 1.0000  (+0.0000)
```

두 실행의 qrels나 k가 다르면 `retrieval_comparable`이 false입니다.

모든 감사 결과에는 `plan`도 들어 있습니다. 정상이 아닌 진단 항목을 공통 원인 가설별로 묶고
고칠 순서대로 늘어놓은 것입니다. 각 단계에는 다음이 들어 있습니다.

- 해당 항목과 값
- 확인할 공통 원인
- 그 가설을 시험할 가장 작은 변경
- 다시 잴 숫자
- 그 순서에 놓인 이유

```text
  plan (fix in this order; each step is a hypothesis to test):
    1. metric: metric_mismatch=0.0329 fail
       cause: vector norms vary, so cosine and inner product rank differently
       first: decide the index metric; L2-normalize if cosine is intended
       check: norm_cv near 0 and topk_overlap 1.0 on re-audit
    2. shared_direction: anisotropy=0.926 warn, antihubs=0.061 warn, collapse=0.108 fail
       cause: a large component shared by all vectors (a mean offset or a few high-variance coordinates) that compresses scores and creates hubs
       first: subtract the corpus mean and re-normalize (anisotropy.details shows the expected cosine after centering); if rogue_dimensions fails, z-score dimensions instead. Apply the same transform to queries and rebuild the index
       check: anisotropy and rogue_dimensions below warn_at, hubness.skewness down; retrieval ndcg_at_k not lower than before
```

위 예시는 정규화하지 않은 벡터에 공통 오프셋을 더한 합성 코퍼스(2,000행, 32차원)이고, 앞의
비교 결과는 두 단계를 모두 적용한 코퍼스를 정규화만 한 코퍼스와 비교한 것입니다. 라벨이 있는
질의 80개가 쉬워 nDCG는 1.0에 머물렀고, 변화는 중복 비율에서만 보입니다.

순서는 수정 사이의 의존 관계를 따릅니다.

- 0 벡터나 복사본 같은 깨진 입력은 모든 통계를 왜곡하므로 가장 먼저입니다.
- 그다음은 지배적인 좌표입니다. rogue 차원 하나가 모든 거리를 정하고 그에 따라 벡터 길이,
  이상치, 허브까지 움직이므로, 정규화보다 차원 표준화가 먼저입니다.
- 그다음이 거리 척도입니다.
- 그다음이 이웃 그래프를 다시 만드는 중심화 같은 변환입니다.
- 그다음이 모델입니다.
- 코퍼스 구조와 인덱스가 뒤를 잇습니다.
- 질의 쪽은 고친 코퍼스를 기준으로 재므로 마지막입니다.

공통 방향 때문에 생긴 근접 중복(모든 코사인이 기준을 넘지만 정확한 복사본은 없음)은 중복
제거가 아니라 공통 방향 단계로 묶습니다. 중복과 함께 나온 `query_gap`은 중복 단계에 넣습니다.
근접 복제가 `query_gap`이 나누는 코퍼스 거리를 줄이므로, 질의를 의심하기 전에 중복을 제거하고
다시 읽어야 하기 때문입니다.

규칙은 고정되어 있고 `src/vecscope/plan.py`에 있습니다. 각 단계는 다시 잴 숫자로 시험할
가설이지 진단이 아닙니다.

실제 임베딩에서 이 조치들이 검색 품질을 어떻게 바꿨는지는 [검증 결과](verification.md#개입-실험)에
있습니다.

## 벡터 저장소에서 읽기

경로 자리에 임베딩 파일 대신 faiss 인덱스 파일이나 벡터 데이터베이스를 지정할 수 있습니다. 행은
id 오름차순으로 읽으므로 내용이 같으면 두 번 읽어도 같은 행렬이 되고, 결과에는 출처와 읽은
내용의 지문이 기록됩니다. `details.sample_ids`의 행 번호에는 저장소의 id인 `sample_source_ids`가
함께 붙습니다.

```bash
pip install "vecscope[qdrant]"     # 또는 [chroma], [milvus], [weaviate], faiss는 [index], 전부는 [stores]
vecscope audit index.faiss --ann-from-source --sample 0
vecscope audit "qdrant://localhost:6333/docs?vector=dense" --save-ids ids.json
vecscope audit "qdrant+local:///data/qdrant?collection=docs"
vecscope audit "chroma+local:///data/chroma?collection=docs"
vecscope audit "milvus+local:///data/milvus.db?collection=docs"
vecscope audit "milvus://milvus.internal:19530/docs?api_key_env=MILVUS_TOKEN"
vecscope audit "weaviate://localhost:8080/Docs?grpc_port=50051"
```

`--ann-from-source`는 저장된 벡터마다 저장소 자체의 인덱스에 질의해 그 결과를 `--ann`처럼
채점합니다. 이웃 목록을 따로 내보내지 않고도 운영 중인 인덱스의 재현율을 잴 수 있습니다.

옵션은 질의 문자열로 넘깁니다.

- `vector=`(이름 붙은 벡터나 Milvus 필드)
- `https=1`
- `pk=`(Milvus 기본 키)
- `api_key_env=NAME`(키는 그 환경 변수에서 읽고, URI에 직접 적은 키는 쓰지 않음)

읽기만 하며, 로드되지 않은 Milvus 컬렉션만 읽는 동안 로드했다가 다시 해제합니다. MCP 서버는
로컬 저장소를 `--allow-dir` 범위 안에서 읽고, 네트워크 저장소는 `--allow-network`로 시작했을
때만 읽습니다.

## 대규모 코퍼스: 증분 감사

전체 행 감사는 행 수의 제곱에 비례합니다. 코퍼스가 뒤에 행을 덧붙이는 식으로 커질 때
`--knn-cache`는 실행 사이에 정확한 이웃 목록을 보관하고, 새 행이 바꾸는 부분만 계산합니다. 새
행과 전체 행 사이, 기존 행과 새 행 사이만 계산합니다.

```bash
vecscope audit day1.npy --sample 0 --knn-cache knn.npz    # created: 전체 행 계산
vecscope audit day2.npy --sample 0 --knn-cache knn.npz    # extended: 덧붙인 행만 계산
```

캐시에는 계산에 쓴 행의 SHA-256 지문이 들어 있습니다.

- 같은 행이면 그대로 씁니다.
- 캐시의 행이 입력 앞부분과 그대로 같으면 확장합니다.
- 그 밖의 경우(행이 바뀌거나 순서가 바뀌었거나 `dim`이 다른 경우)에는 새로 만듭니다.

이웃 목록은 거리가 정확히 같은 동점을 빼면 전체 계산과 같고, 진단 값은 float32 반올림
범위에서 일치합니다. Python에서는 `vecscope.incremental.cached_knn_for_audit(x, k, path)`가
`audit(x, knn=...)`에 넘길 목록을 돌려줍니다.

10만 행 × 768차원([성능](#성능) 표와 같은 장비)에서 잰 결과입니다.

- 5,000행을 덧붙였을 때 캐시 확장에 11초, 캐시로 감사하는 데 4초가 걸렸고 전체 감사는
  94초였습니다.
- 2만 행을 덧붙이면 43초와 4초, 전체 감사는 123초였습니다.

판정은 모두 전체 감사와 같았고 값의 최대 상대 차이는 7e-5입니다. 캐시를 처음 만드는 비용은
전체 감사와 비슷합니다(여기서는 83초).

## 에이전트와 함께 사용

`vecscope`에는 읽기 전용 도구 네 개(`audit_embeddings`, `sweep_index`, `compare_embeddings`,
`describe_findings`)를 제공하는 MCP 서버가 들어 있습니다. MCP를 지원하는 클라이언트라면
어디에든 등록할 수 있습니다.

```bash
pip install "vecscope[mcp]"
claude mcp add vecscope -- vecscope-mcp --allow-dir /data/embeddings     # Claude Code
```

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp", "args": ["--allow-dir", "/data/embeddings"] } } }
```

JSON 형식은 Cursor에서는 `.cursor/mcp.json`, Claude Code 프로젝트에서는 `.mcp.json`에 넣습니다.
Codex는 `~/.codex/config.toml`의 TOML을 읽습니다.

```toml
[mcp_servers.vecscope]
command = "vecscope-mcp"
args = ["--allow-dir", "/data/embeddings"]
```

진단, 변경, 재검증으로 이어지는 흐름은 MCP stdio 프로토콜 위에서 테스트하고
(`tests/test_agent_workflow.py`), Claude Code headless 모드(`claude -p`에 위 서버 등록)로도 끝까지
돌려 봤습니다.

Claude Code는 공통 오프셋이 있는 코퍼스를 감사한 뒤 계획대로 정규화와 중심화를 하고 다시
감사했으며, 요청받지 않은 나머지 단계는 건드리지 않고 멈췄습니다. Cursor와 Codex는 같은
프로토콜을 쓰지만 직접 돌려 보지는 않았습니다.

등록한 뒤에는 에이전트에게 다음과 같이 요청합니다.

```text
검색기를 튜닝하기 전에 data/chunks.npy 임베딩을 감사하고, 먼저 조치할 항목을 알려줘.
```

도구는 읽기 전용이지만 에이전트가 지정한 경로라면 무엇이든 읽습니다. 에이전트가 보면 안 되는
데이터가 있는 장비에서는 서버를 임베딩이 있는 디렉터리로 제한하십시오. 그 밖의 경로는(`..`과
심볼릭 링크를 풀어 확인한 뒤) 파일을 열기 전에 거부됩니다.

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp", "args": ["--allow-dir", "/data/embeddings"] } } }
```

`--allow-dir`는 여러 번 줄 수 있고, 환경 변수 `VECSCOPE_ALLOWED_DIRS`(Unix는 `:`, Windows는
`;`로 구분)도 같은 역할을 합니다. 둘 다 없으면 서버 프로세스가 읽을 수 있는 모든 파일에 접근할
수 있습니다.

[`.agents/skills/vecscope/SKILL.md`](../../.agents/skills/vecscope/SKILL.md)의 에이전트
스킬에는 감사를 실행할 시점과 판정별 조치(벡터 중심화(centering) 또는 모델 교체, 중복 제거,
재순위화(re-ranking), 인덱스 재구축)가 정리되어 있습니다. 같은 절차가 서버 instructions에도
들어 있어 MCP 연결만 있는 에이전트도 동일하게 동작합니다.

- `sweep_index`는 [인덱스 스윕](#인덱스-스윕)을 실행합니다.
- `compare_embeddings`는 [두 임베딩 비교](#두-임베딩-비교)를 실행합니다.
- `describe_findings`는 모든 항목의 임곗값, 원인, 조치를 나열합니다.

응답은 한 줄짜리 `summary`(ok가 아닌 항목, 검색 nDCG, 계획의 첫 단계, 재감사라면 기준선과의
비교)와 `audit_id`로 시작합니다.

변경 뒤 다시 감사할 때 그 id를 `baseline`으로 넘기면, 응답에 CLI의 `--baseline`과 같은
`baseline_diff`(판정과 값의 변화, `regressions`, `improvements`, 검색 품질 변화)가 붙습니다.
에이전트가 이전 응답을 파일로 저장할 필요가 없습니다. 서버는 최근 감사 32개를 메모리에 두며,
저장한 응답이나 `--json` 파일도 받습니다.

`verbose=false`를 주면 `ok` 항목의 세부 내용, 설명, 조치를 생략합니다. MCP를 지원하지 않는
에이전트라면 CLI를 `--json`으로 실행해 같은 결과를 읽으면 됩니다.

서버가 연결되지 않고 "vecscope-mcp not found"가 나오면 대개 설치했던 Python 환경이 새로
만들어진 경우입니다. 환경을 다시 구성하는 호스팅 노트북이나 클러스터에서 흔히 일어납니다.

클라이언트가 쓰는 인터프리터에서 `pip show vecscope`로 확인하고 그 환경에 다시 설치하십시오.
처음부터 그 인터프리터를 통해 서버를 등록해 두면 오류 메시지가 어느 환경의 문제인지
바로 드러납니다.

```json
{ "mcpServers": { "vecscope": { "command": "/path/to/env/bin/python", "args": ["-m", "vecscope.mcp_server"] } } }
```

## 라이브러리로 사용

```python
import numpy as np
from vecscope import audit

embeddings = np.load("embeddings.npy")    # (n, dim) 형태의 float 배열

ev = audit(embeddings, k=10)
ev.verdict                                # "ok" | "warn" | "fail"
ev.findings[0].value                      # 첫 번째 항목의 측정값
ev.to_dict()                              # --json과 동일한 전체 결과
```

## CI에서 사용

JSON 출력은 실행 간에 안정적이고 `schema_version`을 담고 있어, 기준선으로 커밋해 두고
인덱스를 다시 만들 때마다 비교할 수 있습니다. `--baseline`은 항목별 변화를 출력하고,
어느 항목의 판정이라도 나빠지면(또는 `--fail-on` 조건에 걸리면) 종료 코드 1을 돌려줍니다.
이미 있던 `warn` 때문이 아니라 실제 퇴행이 있을 때만 파이프라인이 실패합니다.

```bash
vecscope audit embeddings.npy --json > baseline.json          # 처음 한 번, 검토 후 커밋
vecscope audit embeddings.npy --baseline baseline.json        # 다시 만들 때마다
```

```text
  baseline baseline.json: verdict ok -> fail  (n, dim, k, sampling, thresholds or near-dup cutoff differ: values and verdicts are not directly comparable)
    hubness                1.0454 -> 0.8594  (-0.1860)  ok -> ok
    antihubs               0.0383 -> 0.0105  (-0.0278)  ok -> ok
    anisotropy             0.0105 -> 0.0127  (+0.0022)  ok -> ok
    collapse               0.0000 -> 0.8000  (+0.8000)  ok -> fail  WORSE
    ...
  regressions: collapse
```

위 예시는 행 수가 다른 `healthy.npy`와 `collapsed.npy`를 비교한 것이라 첫 줄에 주의 문구가
붙었습니다. 같은 코퍼스를 같은 설정으로 다시 만든 경우에는 붙지 않습니다.

`--json`과 함께 쓰면 비교 결과가 `baseline_diff`로 붙고, 라이브러리에서는
`diff_evidence(new, old)`입니다. 다른 `schema_version`으로 쓴 기준선은 다시 만들라는 안내와
함께 거부됩니다.

기준선과 검사 모두 `--sample 0`으로 전체 행을 감사해 두 실행이 같은 행을 보게 하십시오. 두
실행 모두 `--queries`를 넘기면 기준선의 질의 거리로 `query_shift`도 계산합니다([질의와
군집](#질의와-군집) 참고). 기준선 자체에는 비교 대상이 없었으므로 비교 결과에서는
`only_in_new`에 나타납니다.

판정은 거친 신호입니다. 측정값이 임곗값을 넘지 않은 채 몇 달 동안 서서히 나빠질 수 있습니다.
그래서 변화 행마다 다음을 함께 기록합니다.

- `worsened`: 값이 나쁜 방향으로 움직였는지
- `worsening`: 얼마나 나빠졌는지(해당 지표의 단위로, 양수가 악화)

`--tolerances`를 주면 이 변화량을 판정 단계와 무관한 게이트로 쓸 수 있습니다.

```bash
# tolerances.json: {"collapse": 0.01, "ann_recall": 0.02, "hubness": 0.3}
vecscope audit embeddings.npy --baseline baseline.json --tolerances tolerances.json --require-comparable
```

- 허용량보다 더 나빠진 항목은 `value_regressions`에 오르고(텍스트 출력에서는
  `VALUE WORSE by 0.0300 > 0.01`), 판정이 그대로여도 종료 코드 1을 돌려줍니다.
- 허용량을 주지 않은 항목은 판정 변화로만 평가합니다.
- 값이 좋아진 항목은 어떤 허용량에서도 퇴행이 아닙니다.

허용량은 커밋된 기준선 대비 한 번의 재구축에 허용하는 예산이므로, 검토를 거쳐 받아들인
변경이 있으면 기준선을 다시 커밋하십시오.

결과에는 `comparable`도 들어 있습니다. n, 차원, k, 샘플링, 임곗값, 근접 중복 기준이 기준선과
다르면 false가 됩니다. `--require-comparable`을 주면 이 경우 주의 문구 대신 오류(종료 코드
2)로 처리합니다. 설정이 다른 두 실행의 차이는 코퍼스가 아니라 설정을 재는 것이므로
게이트에서는 이 정책이 맞습니다. 라이브러리에서는 `diff_evidence(new, old, tolerances={...})`입니다.

실행을 실패로 만드는 규칙은 두 가지이고, 출력에서 둘을 구분해 보여 줍니다.

- 하나는 절대 규칙으로, 임곗값 기준 판정입니다(기본값 `--fail-on fail`, 더 엄격하게는
  `--fail-on warn`).
- 다른 하나는 기준선 규칙으로, 판정이 나빠진 `regressions`와 값이 허용량 이상 나빠진
  `value_regressions`입니다.

기본 임곗값은 공개 모델로 보정한 값이라, 정상적인 코퍼스가 어떤 항목에서 늘 `warn`이나
`fail`에 걸리는 도메인이라면 절대 규칙으로 막으면 안 됩니다. 이때는
`--fail-on none --baseline baseline.json`으로 검토를 마친 기준선 대비 변화만 판단하십시오.

실행 결과 끝에는 어느 규칙이 결과를 결정했는지 적은 `gate` 블록(JSON) 또는 줄(텍스트)이
붙습니다.

```text
  gate: FAIL; verdict fail reaches --fail-on fail; verdict worse than baseline: collapse
  gate: pass (baseline)
```

이 게이트는 우선 추세 추적 용도로 쓰십시오. 어느 정도의 변화가 "지나친지"는 진단값이 실제
질의의 검색 품질과 어떤 관계인지에 달려 있습니다.

[검색 품질 연구](#진단-항목이-검색-품질을-예측합니까)가 공개 데이터세트에서 그 관계를 보여 주고,
[임곗값을 도메인에 맞추기](#임곗값을-도메인에-맞추기)는 병합을 차단하기 전에 자신의 데이터로
그 관계를 재는 방법을 설명합니다.

## CLI 옵션

```bash
vecscope audit vectors.npy -k 20                       # 항목당 이웃 수 (기본값 10)
vecscope audit vectors.parquet --json                  # JSON 형식 출력
vecscope audit vectors.npy --ann hnsw_neighbors.npy    # ANN 인덱스의 재현율을 정확한 k-NN과 비교
vecscope audit vectors.npy --queries queries.npy       # 질의 측 진단 추가
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv   # 정확한 검색의 nDCG/recall/MRR 추가
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv --ranked topk.npy   # 운영 랭킹도 함께 채점
vecscope audit index.faiss --ann-from-source --sample 0     # faiss 인덱스나 저장소를 읽고 자체 이웃을 채점
vecscope audit "qdrant://host:6333/docs" --save-ids ids.json   # 벡터 저장소 URI, 행 순서대로 id 저장
vecscope audit vectors.npy --sample 0 --knn-cache knn.npz   # 실행 사이에 정확한 이웃 목록을 재사용/확장
vecscope audit vectors.npy --clusters 10               # 군집별 요약 추가 (최대 n/10개)
vecscope audit vectors.npy --thresholds th.json         # (warn, fail) 임곗값 덮어쓰기
vecscope audit vectors.npy --fail-on warn              # warn 판정도 종료 코드 1로 처리
vecscope audit vectors.npy --fail-on none --baseline old.json   # 절대 임곗값은 무시하고 기준선 대비 퇴행만 실패 처리
vecscope audit vectors.npy --baseline old.json         # 이전 --json 결과와 비교, 퇴행이 있으면 종료 코드 1
vecscope audit vectors.npy --baseline old.json --tolerances tol.json   # 값이 허용량보다 나빠져도 종료 코드 1
vecscope audit vectors.npy --baseline old.json --require-comparable    # 기준선의 설정이 다르면 종료 코드 2
vecscope audit vectors.npy --near-dup 0.98             # 근접 중복으로 간주할 코사인 임곗값
vecscope audit vectors.npy --sample 0                  # 샘플링 없이 전체 행을 감사 (기본값은 10만 행)
vecscope audit vectors.npy --sample-repeats 3          # 표본 3개를 감사해 값의 범위와 판정 안정성을 보고
vecscope sweep vectors.npy --target-recall 0.95        # HNSW/IVF 파라미터 스윕 (faiss 추가 설치)
vecscope sweep vectors.npy --device gpu                # IVF를 GPU에서 실행 (GPU용 faiss)
vecscope sweep vectors.npy --margin 0.05               # 이 여유 미만이면 추천을 tight로 표시 (기본값 0.02)
vecscope sweep vectors.npy --n-queries 2000 --timing-repeats 5   # 시간을 잴 질의 수(기본값 1000), 설정별 측정 횟수(기본값 3)
vecscope sweep vectors.npy --latency-queries 0         # 단일 질의 지연 백분위수 생략 (기본값 200개 질의)
vecscope compare a.npy b.npy -k 10                     # 같은 행을 가진 두 임베딩의 이웃 일치도
vecscope compare a.npy b.npy --neighbors-a a_knn.npy --neighbors-b b_knn.npy   # 행은 표본, 이웃은 전체 코퍼스
```

## 입력 형식

임베딩 파일에는 행마다 벡터가 하나씩 들어갑니다. 어떤 형식이든 감사 전에 `(n, dim)` 형태의
2차원 `float32` 배열로 바뀝니다. 값은 유한해야 합니다. NaN, Inf, 그리고 제곱이 `float32`
범위를 넘는 절댓값 1e16 초과 값은 안내 문구와 함께 거부됩니다. 아래 예시의 `embeddings`가
바로 이 배열이며, 모델의 `encode` 출력이 대표적인 예시입니다.

```python
embeddings = model.encode(texts)        # 형태 (n, dim), 예: (3000, 128)
embeddings[0]                           # 첫 번째 항목의 벡터
# array([ 0.0312, -0.1187,  0.0459, ...], dtype=float32)
```

| 확장자 | 구조 | 예 |
| --- | --- | --- |
| `.npy` | 2차원 숫자 배열 | `np.save("vectors.npy", embeddings)` |
| `.npz` | 아카이브의 첫 번째 배열을 사용 | `np.savez("vectors.npz", emb=embeddings)` |
| `.parquet`, `.pq` | 컬럼 이름에 `emb` 또는 `vec`가 포함된 float 리스트 컬럼. id, 텍스트 등 다른 컬럼은 무시 | `pd.DataFrame({"id": ids, "embedding": list(embeddings)}).to_parquet("vectors.parquet")` |
| `.jsonl` | 한 줄에 JSON 객체 하나. 첫 번째 리스트 필드를 벡터로 사용 | `{"id": 7, "text": "...", "embedding": [0.1, -0.3, ...]}` |
| `.json` | 위와 같은 객체의 배열, 또는 리스트의 배열 | `[[0.1, -0.3, ...], [0.2, 0.5, ...]]` |
| `.csv`, `.tsv`, `.txt` | 한 줄에 벡터 하나, 숫자만 포함. 헤더나 id 컬럼 없음 | `0.1,-0.3,0.8,...` |

`--ann` 파일(MCP에서는 `ann_path`)은 선택 입력으로, `(n, k)` 형태의 정수 배열입니다. `i`번째
행에는 ANN 인덱스가 `i`번째 항목의 이웃으로 돌려준 `k`개의 id가 들어가며, 행 순서는 임베딩
파일과 같아야 합니다. 위 형식 중 어느 것이든 쓸 수 있습니다.

id가 행 위치를 가리키므로 바로 이 행들로 만든 인덱스에서 나온 파일이어야 하며, 따라서 모든
행을 감사해야 합니다. 행 수가 `--sample` 크기(기본값 10만 행)를 넘으면 `--sample 0`을 지정해야
하고, 지정하지 않으면 감사가 거부합니다.

항목 자기 자신을 가리키는 id는 무시되므로, 자기 자신을 먼저 돌려주는 인덱스라면 k + 1개를
질의하십시오.

`--queries` 파일(MCP에서는 `queries_path`)도 선택 입력으로, 같은 형식 중 하나로 저장한 질의
임베딩입니다. 코퍼스와 `dim`이 같아야 합니다.

`query_shift`의 기준은 `--queries`와 함께 실행한 감사의 `--json` 출력입니다(`--baseline`,
MCP에서는 `baseline`이나 `query_baseline_path`).

`--baseline`과 `baseline`은 그 기준선이 같은 코퍼스나 그대로인 앞부분에서 감사된 경우에만
기준으로 씁니다(행의 SHA-256인 `corpus_fingerprint`로 확인). 정규화나 중심화 뒤에는 이전 질의
거리가 다른 공간의 값이므로 `query_shift`를 건너뛰고 `query_reference_note`에 이유를 적습니다.

## 성능

정확한 k-NN은 행 수의 제곱에 비례합니다. 행 수가 두 배가 되면 시간은 대략 네 배가 되고,
10만 행 전체 감사(`--sample 0`)는 노트북에서 수 분이 걸립니다.

k-NN 계산은 행 수와 무관하게 거리 행렬을 1GiB 안팎으로 유지하도록 행 블록 크기를 조절하고,
나머지 계산은 행렬의 float32 복사본 몇 벌을 쓰므로 최대 메모리는 행 수 × 차원에 비례해 행렬
크기의 약 네 배까지 커집니다.

감사의 실행 시간과 Python이 추적한 최대 메모리는 다음과 같습니다(입력
행렬 자체는 별도로 더 필요합니다).

| 행 수 | 차원 | 초 | 최대 메모리 (입력 제외) |
|---|---|---|---|
| 10,000 | 384 | 1.3 | 0.40 GiB |
| 10,000 | 768 | 2.0 | 0.41 GiB |
| 10,000 | 1,536 | 3.6 | 0.44 GiB |
| 50,000 | 384 | 17 | 1.63 GiB |
| 50,000 | 768 | 23 | 1.70 GiB |
| 50,000 | 1,536 | 36 | 1.84 GiB |
| 100,000 | 384 | 65 | 1.71 GiB |
| 100,000 | 768 | 86 | 1.85 GiB |
| 100,000 | 1,536 | 130 | 3.45 GiB |
| 200,000 | 768 | 338 | 3.46 GiB |
| 200,000, 기본 표본 10만 | 768 | 85 | 2.14 GiB |

8 vCPU AMD EPYC 7R13에서 BLAS 스레드 8개로 잰 값입니다(NumPy 단일 프로세스, 군집 구조의 합성
코퍼스, k=10). 별도 프로세스에서 3회 반복한 중앙값과 이 1회 측정의 차이는 2% 이내였고, OS가
본 최대 메모리와의 차이도 0.04 GiB 이내였습니다.

시간은 5만 행 이상에서 행 수가 두 배가 될 때 3.5\~3.9배 늘어 n²에 가깝습니다. 메모리는
384·768차원에서는 5만 행 이후 거의 변하지 않다가(k-NN 블록 크기가 n에 따라 줄어들기 때문)
1,536차원과 20만 행에서 한 단계 올라갑니다. 기본값인 10만 행 표본은 어떤 코퍼스든 비용을
10만 행 수준으로 묶어 둡니다.

`--queries`는 질의 묶음마다 코퍼스를 한 번 더 훑고, `metric_mismatch`는 벡터 길이가 다를 때만
k-NN을 한 번 더 계산합니다. 표본 감사는 표본에 대한 설명이므로, 섬과 이상치를 하나도 빠뜨리지
않아야 한다면 `--sample 0`을 지정하십시오.

`python examples/benchmark_scale.py --rows 50000 100000 --dim 768`을 실행하면 원하는 규모에서
이 표(실행 시간과 감사 중 NumPy가 할당한 최대 메모리)를 직접 다시 잴 수 있습니다.
`examples/benchmark_sweep.py`는 인덱스 스윕을 같은 방식으로 잽니다. 두 스크립트 모두
`--corpus` 옵션(clustered, isotropic, anisotropic, duplicates, lowrank, zipf)을 받고 실행 환경
정보를 JSON에 함께 기록합니다.

블록 계산이 제한하는 것은 메모리이지 연산량이 아닙니다. 정확한 k-NN은 여전히 n² × 차원에
비례하는 곱셈이 필요하므로, 768차원 100만 행은 10만 행의 약 100배가 듭니다. 그 규모의
코퍼스는 기본 표본(또는 `--sample-repeats`로 표본 여러 개)을 감사하고, `--sample 0`은 전체 행이
꼭 필요한 항목에만 쓰십시오. 어느 항목이 그런지는 다음과 같습니다.

| 항목 | 표본으로 얻는 것 | 전체 행이 필요한 경우 |
|---|---|---|
| anisotropy, effective_rank, rogue_dimensions, norms, metric_mismatch, distance_concentration | 분포 추정치. 20만 × 384 코퍼스에서 표본 10만·5만·2만 모두 전체 값과 2.3% 이내로 같았고 판정도 같았음 | 없음 |
| hubness, antihubs | *하한값*. k-출현 빈도는 밀도가 높을수록 쏠리므로 같은 코퍼스라도 표본이 작으면 hubness가 낮게 나옴(20만 행 전체 2.6 `warn`, 4분의 1 표본 1.15, 10분의 1 표본 0.16). `fail`이 아닌 판정은 inconclusive로 표시 | 코퍼스 자체의 hubness를 잴 때 |
| collapse | 표본 안에서 확인된 중복 *비율*. 중복 쌍의 두 행이 모두 뽑혀야 세므로 표본 비율 × 전체 비율의 0.96\~1.03배가 나옴. 판정은 이를 되돌려 추정한 전체 비율로 내림 | 모든 중복을 세거나 나열할 때 |
| outliers | 작은 표본에서 실제보다 많이 잡힘(10분의 1 표본에서 3.9배). `ok`가 아닌 판정은 inconclusive로 표시 | 모든 이상치를 찾을 때 |
| connectivity | 표본으로 만든 k-NN 그래프의 섬. 코퍼스의 섬은 대부분 보이지 않음(전체 행의 37%가 섬인 코퍼스가 4분의 1 표본에서는 0.08%). `fail`이 아닌 판정은 inconclusive로 표시 | 항상. 표본으로는 판단할 수 없음 |
| ann_recall | 표본에서는 불가. 행 번호가 ANN 파일과 맞아야 함 | 항상 (`--sample 0`) |
| query_gap, query_shift, query_hubness, ood_queries | 표본에서 구한 기준 거리. 질의는 모두 평가. query_gap은 표본이 작을수록 함께 커지는 두 거리의 비율이고, query_shift는 표본 크기가 같은 기준선과만 비교 | 없음 |

그래서 표본 감사는 다음과 같이 처리합니다.

- 표에서 inconclusive로 표시한 판정들이 `inconclusive_findings` 목록으로 들어갑니다.
- 그 판정들은 텍스트 출력에서는 `INCONCLUSIVE (sampled)`로, gate 줄에서는 이름으로
  표시됩니다.
- collapse 판정은 표본 비율로 되돌려 추정한 전체 비율(`details.estimated_full_share`)로
  내립니다.
- 각 항목의 설명문에도 이 주의 사항이 숫자 바로 옆에 붙습니다.

20만 행 수치는
[검증 결과](verification.md)에서 가져왔고, `tests/test_regression_injection.py`가 같은 성질을
6천 행 규모에서 매번 검사합니다.

모든 수치는 감사한 행에 대해 정확합니다. vecscope는 정확한 이웃 탐색을 근사 탐색으로
대체하지 않으며, 근사가 들어가는 곳은 행의 선택뿐입니다.

## 동작 원리

`vecscope`는 행렬 전체의 정확한 k-NN을 한 번만 계산하고, 모든 진단을 그 결과에서 끌어냅니다.

- k-출현 빈도 분포에서 hubness와 antihubs를 구합니다.
- 최근접 이웃 거리에서 collapse, outliers, distance_concentration을 구합니다.
- k-NN 그래프에서 connectivity를 구합니다.
- 무작위 쌍과 공분산 스펙트럼에서 anisotropy, effective_rank, rogue_dimensions를 구합니다.
- 벡터 길이에서 norms를 구합니다.
- metric_mismatch는 내적 기준 k-NN을 한 번 더 계산해 비교하되, 벡터 길이가 모두 같으면
  건너뜁니다.
- ANN 이웃 파일이 있으면 정확한 이웃과 비교해 재현율을 구합니다.

질의 임베딩이 있으면 코퍼스를 대상으로 정확 검색을 한 번 더 수행해 질의 측 항목을 구합니다.

같은 입력에는 언제나 같은 결과가 나옵니다. 시드가 고정되어 있고, 이웃은 근사 없이 정확하게
계산하며, 학습된 모델이나 네트워크 통신도 수행하지 않습니다. 같은 파일을 두 번 감사하면 JSON이
완전히 일치합니다.

정확한 k-NN은 행 수의 제곱에 비례하는 비용이 들기 때문에, 10만 행을 넘는 입력은 `--sample 0`을
지정하지 않는 한 고정 시드로 뽑은 표본만 감사합니다.

## 실제 임베딩 검증

임곗값은 미리 계산되어 공개된 임베딩으로 보정했습니다. 대상은 OpenAI ada-002와
text-embedding-3-large, Cohere embed-v3(한국어·영어 위키백과), all-MiniLM-L6-v2, bge-m3,
mxbai-embed-large-v1, gte-base-en-v1.5의 7개 모델, 8개 데이터세트입니다.

아래 표는 각 데이터세트의 앞 4만 행을 측정한 값이며, 판정 표시가 없는 칸은 `ok`입니다.
`python examples/real_models.py`로 재현할 수 있고 모델을 내려받을 필요는 없습니다.

| finding | Cohere v3 (한국어 위키) | Cohere v3 (영어 위키) | ada-002 (dbpedia) | 3-large (dbpedia) | MiniLM-L6 (위키) | bge-m3 (위키) | mxbai-large (위키) | gte-base (arXiv) |
|---|---|---|---|---|---|---|---|---|
| hubness | 1.47 | 1.16 | 1.56 | 1.21 | 1.59 | 1.85 | 1.43 | 2.28<br>warn |
| antihubs | 0.009 | 0.005 | 0.013 | 0.005 | 0.022 | 0.026 | 0.019 | 0.013 |
| anisotropy | 0.333 | 0.228 | 0.692 | 0.086 | 0.026 | 0.307 | 0.330 | 0.210 |
| effective_rank | 0.158 | 0.178 | 0.090<br>warn | 0.148 | 0.285 | 0.142 | 0.087<br>warn | 0.136 |
| collapse | 0.002 | 0.002 | 0.000 | 0.000 | 0.001 | 0.000 | 0.017 | 0.000 |
| rogue_dimensions | 3.2 | 3.2 | 3.0 | 2.5 | 1.7 | 1.8 | 2.0 | 1.8 |
| norms | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| metric_mismatch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 |
| distance_concentration | 1.72 | 1.74 | 1.74 | 1.43 | 1.78 | 1.84 | 2.21 | 1.62 |
| outliers | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.000 |
| connectivity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

각 세트의 앞 4만 행으로 감사를 다시 돌리면 판정 88개가 모두 재현됩니다. `warn`은 gte-base의
hubness와 ada-002·mxbai-large의 `effective_rank`뿐입니다.

ada-002의 anisotropy(0.692)는 `warn` 임곗값 0.85보다 낮습니다. 이 모델의 비등방성은 널리
알려진 특성입니다. 평균 단위 벡터의 길이가 0.83이라 무관한 문서끼리도 코사인 유사도가 0.7
근처에서 시작합니다. 후속 모델인 3-large는 같은 코퍼스에서 등방성을 보입니다(길이 0.29). 어느
모델이든 중심화하면 무작위 쌍의 코사인 유사도가 0.00으로 떨어지므로, 이 값을 세부 수치로 함께
보고합니다.

텍스트 임베딩에서 hubness 왜도 1.2\~1.9는 정상 범위라 `warn` 임곗값을 2.0으로 잡았고, arXiv
초록을 임베딩한 gte-base만 이를 살짝 넘습니다. anisotropy 임곗값(0.85, 0.95)은 아래의 검색 품질
연구에서 정했습니다.

## 진단 항목이 검색 품질을 예측합니까?

부분적으로, 그리고 특정한 방식으로 그렇습니다.
[docs/ko/retrieval_quality.md](retrieval_quality.md)는 라벨이 있는 BEIR 데이터세트 두 개(SciFact,
NFCorpus)에서 정확한 코사인 검색으로 nDCG@10을 구하고, 공개 모델 8종과 가장 좋은 모델에 의도적으로
주입한 결함에 대해 모든 항목을 nDCG@10과 대조합니다. `python examples/retrieval_quality.py`로
재현할 수 있습니다.

- **같은 파이프라인 안에서는 항목이 손상을 따라갑니다.** 불량 차원 주입, 계수 축소, 잡음 추가,
  행 중복을 가하면 해당 항목이 `fail`로 바뀌면서(원래 `ok`에서, 코퍼스에 중복이 조금 있던 경우에는
  `warn`에서) nDCG@10이 떨어지고, 결함 수준이 커질수록 단조롭게 나빠집니다.

  관계없는 모델로 임베딩한 질의는 `query_gap`이 8\~10(`fail`, nDCG@10은 0 근처)입니다. 공간을
  공유하는 bge-base 질의를 gte-base 코퍼스에 넣으면 2.9\~3.9가 나오지만 nDCG@10은 유지됩니다.

  거의 같은 복제가 많아도 `query_gap`이 오릅니다. `query_gap`이 나누는 코퍼스 거리를 복제가
  줄이기 때문이며, 계획은 이를 중복과 같은 단계에 넣어 중복 제거 뒤 다시 읽게 합니다.
- **좋은 모델들 사이에서는 기하 항목이 모델 순위를 매기지 못합니다.** 두 데이터세트 모두에서 가장
  좋은 모델(gte-base, bge, e5)이 anisotropy가 *가장 높고*([검증 결과](verification.md)의 모델 19종까지 포함하면 0.58\~0.82) effective_rank는 평범합니다.

  약한 모델을 가르는 것은 `norms`와 `metric_mismatch`(정규화되지 않은 출력), 그리고 `hubness`로,
  nDCG@10과의 Spearman |rho|가 0.4\~0.8입니다.
- **anisotropy만으로는 순위가 나빠지지 않습니다.** 공통 오프셋을 더해 무작위 쌍 코사인을 0.98까지
  올려도 nDCG@10은 그대로입니다. 점수가 좁은 구간에 압축되므로 절대 코사인 임곗값과 점수 보정은
  깨지지만, 신호 자체가 사라질 때 움직이는 항목은 `distance_concentration`입니다.

`anisotropy`, `query_hubness`, `query_gap`, `query_shift` 임곗값은 라벨이 있는 데이터세트 세 개에서
모델 19종과 주입한 질의 결함으로 정하고 데이터세트 세 개를 더해 다시 확인한 [검증](verification.md)에서 나왔습니다.

검증이 다루지 않은 것은 다른 도메인과 운영 질의 로그에서 `warn`과 `fail`이 품질 저하를 얼마나
잘 예측하는가입니다. 자신의 도메인에서 그 관계를 확인하기 전까지는 판정을 들여다볼 계기로,
기준선 대비 퇴행을 검색 결과를 비교할 계기로 삼고, 품질 점수로 읽지 마십시오.

### 임곗값을 도메인에 맞추기

기본 임곗값은 공개 모델과 공개 데이터세트으로 보정한 값입니다. 자신의 코퍼스에서 `fail`이
나왔다면 들여다볼 이유이지 검색이 나쁘다는 증명이 아니고, 모두 `ok`라고 해서 맞는 문서가
검색된다는 보장도 아닙니다.

임곗값이나 허용량을 병합 차단 게이트로 쓰기 전에, 그것이 자신의 데이터에서 무엇을 예측하는지
먼저 재십시오.

1. 도메인의 정답 질의를 모읍니다(관련 문서 ID가 달린 질의 수백 개면 충분합니다). 그리고
   임곗값을 고르는 데 쓸 부분과 검증용으로 따로 둘 부분으로 나눕니다.
2. 파이프라인의 후보 상태(모델, 청크 분할, 변환, 인덱스 설정)마다 `vecscope audit --json`을
   실행하고, 같은 상태에서 같은 질의로 검색 품질(nDCG@k, Recall@k, MRR)을 계산합니다.
   `examples/retrieval_quality.py`가 공개 데이터세트에 대해 둘을 함께 수행하며, 자신의 데이터를
   가리키도록 바꿀 수 있습니다.
3. 상태들에 걸쳐 각 진단값과 품질 지표의 관계를 봅니다. 자신의 도메인에서 품질과 함께
   움직이지 않는 항목은 기본값이 무엇이든 게이트로 쓰지 않습니다.
4. 품질이 허용 범위였던 상태들을 기준으로 `--thresholds`와 `--tolerances`를 정합니다. 이때
   첫 번째 부분만 사용합니다.
5. 따로 둔 검증용 질의로 그 임곗값이 허용 상태와 나머지를 구분하는지 확인합니다. 같은 질의로
   맞추고 검증한 임곗값은 실제보다 좋아 보입니다.

`suggested_action`에도 같은 절차가 적용됩니다. 권장 조치는 원인에 대한 가설이고, 중심화,
표준화, 차원 절단은 검색 순위를 바꿉니다.

- 원본 벡터와 검색 결과를 기준선으로 보관합니다.
- 변환은 복사본에 적용합니다.
- 양쪽에서 품질과 지연 시간, 메모리를 함께 잽니다.
- 항목이 `ok`로 바뀌었을 때가 아니라 정답 질의의 품질이 좋아졌을 때만 반영하십시오.

## FAQ

<details>
<summary><strong>벡터가 외부로 전송됩니까?</strong></summary>

아니요. NumPy만 사용하고 모든 계산이 로컬에서 이루어집니다. 모델을 내려받거나 네트워크
연결을 하지 않습니다.

</details>

<details>
<summary><strong>얼마나 큰 행렬까지 처리할 수 있습니까?</strong></summary>

정확한 k-NN은 행 수의 제곱에 비례하는 비용이 듭니다. 그래서 기본 설정에서는 입력이 10만 행을
넘으면 고정 시드로 뽑은 표본만 감사하고, 그 사실을 출력에 표시합니다(`sampled: true`,
`n_total`).

표본 결과는 표본에 대한 설명이지 전체 코퍼스에 대한 확정 판정이 아니므로, 드문 군집이나 작은
섬, 소수의 이상치는 놓칠 수 있습니다.

`details`의 행 번호(`sample_ids`, `sample_pairs`, 군집 보고서)는 표본을 뽑았을 때도 항상 입력
파일의 행 번호입니다. 전체 행을 감사하려면 `--sample 0`을 지정하십시오. 비용은 [성능](#성능)
절을 참고하십시오.

표본 자체가 수치를 얼마나 흔드는지 보려면 `--sample-repeats 3`(MCP: `sample_repeats`)을
지정하십시오. 그러면 서로 다른 시드로 뽑은 표본 3개를 감사해 모든 항목에 `value_range`(표본
간 최솟값\~최댓값)와 `verdict_stable`을 붙이고, 최상위에 `unstable_findings` 목록을 둡니다.

표본에 따라 판정이 바뀌는 항목은 코퍼스의 성질이 아니라 경계선에 걸친 값으로 읽어야 합니다.
비용은 감사 3회분입니다.

`value_range`는 표본 사이의 흩어짐을 보여 줄 뿐, 전체 코퍼스 값이
그 안에 있다고 보장하지는 않습니다. hubness는 어느 표본에서나 전체보다 낮게 나오고, 아주
안정적인 항목도 전체 값이 표본 3개의 범위를 조금 벗어날 수 있습니다.

표본에서 모든 항목이 같은 의미를 갖지는 않습니다. hubness는 낮게, outliers는 많게 잡히고
섬은 대부분 보이지 않습니다. 항목별로 표본이 무엇을 말해 주는지, 어떤 판정이
`inconclusive`로 표시되는지는 [성능](#성능) 절의 표에 있습니다.

</details>

<details>
<summary><strong>운영 중인 모델이 anisotropy에서 fail 판정을 받았습니다. 모델에 문제가 있는 것입니까?</strong></summary>

꼭 그렇지는 않습니다. 널리 쓰이는 텍스트 임베딩 모델 상당수가 구조적으로 강한 비등방성을
띱니다.

이 판정은 코사인 점수가 좁은 구간에 몰려 변별력이 떨어진다는 뜻이고, 중심화,
백색화(whitening), 재순위화로 대부분 해결됩니다. `details`에 중심화 후 코사인 값이 함께
보고되므로 중심화만으로 충분한지 확인할 수 있습니다. 판정은 에이전트가 참고할 근거이지
모델의 점수가 아닙니다.

</details>

<details>
<summary><strong>어떤 에이전트에서 쓸 수 있습니까?</strong></summary>

로컬 MCP 서버를 지원하는 에이전트라면 모두 쓸 수 있고, 셸 명령을 실행하고 JSON을 읽을 수
있는 에이전트라면 MCP 없이도 쓸 수 있습니다. 스킬 파일에 두 방식이 모두 설명되어 있습니다.

</details>

## 문서

- [에이전트 스킬](../../.agents/skills/vecscope/SKILL.md): 감사를 실행할 시점과 판정별 조치
- [검색 품질 연구](retrieval_quality.md): 라벨 데이터의 nDCG@10과 모든 진단 항목을 대조한 결과
- [검증 결과](verification.md): 정확성, recall 채점, 표본 감사, 질의 항목, 개입 실험, 에이전트 연동, 라벨 데이터세트 6개에서 모델 19종을 측정한 결과
- [예제](../../examples/): 합성 행렬 생성 스크립트, 보정 스크립트, 검색 품질 연구 스크립트
- [변경 이력](../../CHANGELOG.md)

## 기여

이슈와 PR을 환영합니다. 테스트 방법과 기여 기준은 [CONTRIBUTING.md](../../CONTRIBUTING.md)에,
보안 취약점 제보 방법은 [SECURITY.md](../../SECURITY.md)에 있습니다.

## 라이선스

[MIT](../../LICENSE)
