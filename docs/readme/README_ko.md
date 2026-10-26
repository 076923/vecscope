<div align="center">

[English](../../README.md) · **한국어**

# vecscope

### 에이전트를 위한 임베딩 공간 진단 도구

**벡터 인덱스를 신뢰하기 전에, 에이전트가 검색 공간부터 점검하게 하세요.**

[![PyPI version](https://img.shields.io/pypi/v/vecscope?style=flat-square&color=3775a9)](https://pypi.org/project/vecscope/)
[![Python](https://img.shields.io/pypi/pyversions/vecscope?style=flat-square)](https://pypi.org/project/vecscope/)
[![CI](https://img.shields.io/github/actions/workflow/status/076923/vecscope/ci.yml?style=flat-square&label=CI)](https://github.com/076923/vecscope/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-f4c430?style=flat-square)](../../LICENSE)

[빠른 시작](#빠른-시작) · [진단 항목](#진단-항목) · [질의와 군집](#질의와-군집) · [인덱스 스윕](#인덱스-스윕) · [에이전트와 함께 사용](#에이전트와-함께-사용) · [입력 형식](#입력-형식) · [실제 임베딩 검증](#실제-임베딩-검증) · [FAQ](#faq)

<code>pip install vecscope</code>

</div>

---

에이전트는 검색 증강 생성(RAG), 의미 기반 검색(semantic search), 추천 검색을 매일 만들지만,
정작 자신이 검색하는 임베딩 공간은 들여다보지 못합니다. 그래서 검색 품질 문제는 버그처럼
드러나지 않고 증상으로만 나타납니다. 무엇을 물어도 같은 청크(chunk) 몇 개가 돌아오고, 유사도
점수는 전부 0.8 언저리에 몰려 있으며, 상위 k개가 중복 항목으로 채워지고, 전수
탐색(brute-force)으로는 찾히는 항목이 HNSW에서는 빠집니다.

이런 증상은 저마다 학계에서 정립된 진단 지표로 측정할 수 있습니다. `vecscope`는 그 지표들을
명령 하나, JSON 하나로 묶었습니다. 결과에는 측정값과 `ok` / `warn` / `fail` 판정, 그리고
에이전트가 다음 조치를 정할 때 참고할 설명이 함께 들어 있습니다. NumPy만으로 로컬에서
동작하고, 모델이나 네트워크 연결이 필요 없으며, 같은 입력에는 언제나 같은 결과를 냅니다.

## 빠른 시작

### 1. 설치

```bash
pip install vecscope
```

### 2. 임베딩 파일 준비

`vecscope`는 행마다 임베딩 벡터가 하나씩 담긴 파일을 읽습니다. 모델의 `encode`가 돌려주는
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
vecscope 0.1.0  n=3000 dim=128 k=10  verdict=FAIL
  [FAIL] hubness          skewness=6.9023  (ok < 2, warn >= 2, fail >= 4)
                         k-occurrence skewness 6.90; top 1% of items take 16.3% of all neighbor slots (max N_k=356, expected 10).
  [FAIL] antihubs         antihub_ratio=0.2180  (ok < 0.05, warn >= 0.05, fail >= 0.2)
                         21.8% of items are never anyone's neighbor; they are unreachable through k-NN retrieval.
  [FAIL] anisotropy       mean_random_cosine=0.9782  (ok < 0.3, warn >= 0.3, fail >= 0.6)
                         random pairs have mean cosine 0.978 (isotropic ~0); mean unit vector has norm 0.99; centering would bring the cosine to -0.000.
  [ok  ] effective_rank   effective_rank_ratio=0.9521  (ok >= 0.1, warn < 0.1, fail < 0.05)
                         effective rank 121.9 of 128 dims (95.2%); low values mean most dimensions carry nothing.
  [ok  ] collapse         near_duplicate_ratio=0.0000  (ok < 0.02, warn >= 0.02, fail >= 0.1)
                         0.0% of items have a neighbor with cosine >= 0.99 (exact duplicates 0.0%).
  [ok  ] rogue_dimensions top1_dim_variance_share=0.0084  (ok < 0.05, warn >= 0.05, fail >= 0.2)
                         dimension 7 alone holds 0.8% of variance (uniform would be 0.78%); top 3 dims hold 2.5%. Rogue dimensions dominate cosine.
  [ok  ] norms            norm_cv=0.0000  (ok < 0.1, warn >= 0.1, fail >= 0.3)
                         vector norms vary with CV 0.000 (median 1.000, range 1.000-1.000); 0.00% are zero vectors. Unequal norms bias inner-product search toward long vectors.
  [ok  ] metric_mismatch  topk_overlap=1.0000  (ok >= 0.9, warn < 0.9, fail < 0.7)
                         cosine and inner-product top-10 share 100.0% of neighbors; 100.0% of items get identical lists. Low overlap means the index metric changes the results.
  [ok  ] distance_concentration relative_contrast=1.5260  (ok >= 1.2, warn < 1.2, fail < 1.05)
                         a random item is only 1.53x farther than the nearest neighbor (median NN distance 0.014, random 0.022). Near 1.0 means distances carry no signal.
  [ok  ] outliers         outlier_ratio=0.0037  (ok < 0.01, warn >= 0.01, fail >= 0.05)
                         0.37% of items have no neighbor within cosine distance 0.018 (median 0.014); see details.sample_ids. Typically junk, empty or foreign-language content.
  [ok  ] connectivity     island_ratio=0.0000  (ok < 0.01, warn >= 0.01, fail >= 0.1)
                         0.00% of items sit in 0 small island(s) of the k-NN graph (fewer than 15 items; 1 components in total). Graph-based ANN (HNSW) attaches islands to the wrong region or never reaches them.
```

### 4. 결과 읽는 법

- 첫 줄의 `verdict`가 전체 판정입니다. 항목별 판정 가운데 가장 낮은 등급을 따릅니다.
- 항목마다 측정값, `ok`로 판정되는 범위, 그 수치가 검색에서 무엇을 뜻하는지 설명하는 문장이
  차례로 나옵니다.
- 판정이 `FAIL`이면 종료 코드가 1이므로 같은 명령을 CI 게이트로 쓸 수 있습니다.
- `--json`을 붙이면 같은 내용을 에이전트나 스크립트가 읽기 좋은 JSON으로 출력합니다.

## 진단 항목

| 항목 | 지표 | ok | warn | fail | fail 판정의 의미 |
| --- | --- | --- | --- | --- | --- |
| `hubness` | k-출현 빈도(k-occurrence) 분포의 왜도(skewness) | < 2.0 | 2.0~4.0 | ≥ 4.0 | 소수의 항목이 대부분의 질의에서 k-최근접 이웃(k-NN)에 포함되어 검색 결과 상위를 독점한다 |
| `antihubs` | 어떤 질의에서도 이웃으로 선택되지 않는 항목의 비율 | < 5% | 5~20% | ≥ 20% | 코퍼스(corpus)의 일부에 k-NN 검색으로 도달할 수 없다 |
| `anisotropy` | 무작위 쌍의 평균 코사인 유사도 | < 0.30 | 0.30~0.60 | ≥ 0.60 | 벡터가 좁은 원뿔 영역에 집중되어 모든 쌍이 유사하게 보인다 |
| `effective_rank` | 참여 비율(participation ratio)을 차원 수로 나눈 값 | ≥ 0.10 | 0.05~0.10 | < 0.05 | 대부분의 차원이 정보를 담고 있지 않다 |
| `collapse` | 코사인 유사도 0.99 이상인 근접 중복(near-duplicate) 비율 | < 2% | 2~10% | ≥ 10% | 상위 k개가 동일 항목의 복사본으로 채워진다 |
| `ann_recall` | 정확한 k-NN 대비 근사 최근접 이웃(ANN) 인덱스의 재현율(recall@k) | ≥ 0.95 | 0.85~0.95 | < 0.85 | 모델이 아니라 인덱스에서 결과가 누락되고 있다 |
| `rogue_dimensions` | 분산이 가장 큰 단일 차원의 분산 비율 | < 5% | 5~20% | ≥ 20% | 소수의 좌표가 모든 코사인 점수를 결정한다 |
| `norms` | 벡터 길이(norm)의 변동 계수(coefficient of variation) | < 0.10 | 0.10~0.30 | ≥ 0.30 | 내적(inner product) 검색이 긴 벡터를 우대한다. 길이가 0인 벡터의 비율도 함께 보고한다 |
| `metric_mismatch` | 코사인과 내적의 상위 k개 겹침 비율 | ≥ 0.90 | 0.70~0.90 | < 0.70 | 인덱스의 거리 척도(metric)에 따라 반환 결과가 달라진다 |
| `distance_concentration` | 무작위 거리 / 최근접 이웃 거리 | ≥ 1.2 | 1.05~1.2 | < 1.05 | 최근접 이웃이 무작위 항목과 거의 같은 거리에 있다 |
| `outliers` | 최근접 이웃이 비정상적으로 먼 이상치(outlier) 항목의 비율 | < 1% | 1~5% | ≥ 5% | 불량 항목, 빈 문서, 다른 언어로 된 항목. id를 `details`에 나열한다 |
| `connectivity` | k-NN 그래프의 작은 섬에 속한 항목 비율 | < 1% | 1~10% | ≥ 10% | 군집(cluster)으로 보기에는 너무 작은 고립 집단. HNSW가 잘못 연결하거나 도달하지 못한다 |
| `ood_queries` | 코퍼스 항목 간 거리의 99% 지점보다 코퍼스에서 더 먼 질의의 비율 | < 5% | 5~20% | ≥ 20% | 질의와 코퍼스가 같은 공간에 있지 않다. 다른 모델, 질의 접두어 누락, 코퍼스가 다루지 않는 주제 |
| `query_hubness` | 질의 결과에서 코퍼스 항목의 k-출현 빈도 왜도 | < 2.0 | 2.0~4.0 | ≥ 4.0 | 소수의 항목이 대부분의 질의에 답한다 |

JSON의 각 항목에는 `healthy_range`, `warn_at`, `fail_at`과 세부 수치를 담은 `details`가
들어 있습니다. `anisotropy`의 `details`에는 같은 벡터를 중심화했을 때의 코사인 값까지 있어서,
중심화만으로 해결되는 문제인지 에이전트가 바로 판단할 수 있습니다. `ann_recall`은 `--ann`으로
ANN 이웃 파일을 넘겼을 때, `ood_queries`와 `query_hubness`는 `--queries`로 질의 임베딩을
넘겼을 때만 나타납니다.

임곗값은 `src/vecscope/evidence.py`의 `THRESHOLDS`에 모여 있으며
[실제 임베딩](#실제-임베딩-검증)으로 보정했습니다.

## 질의와 군집

질의 임베딩을 함께 넘기면 질의가 코퍼스와 같은 공간에 있는지도 검사합니다. 비대칭 모델을
질의 접두어 없이 쓴 경우, 질의 쪽에 다른 모델을 쓴 경우, 코퍼스가 다루지 않는 주제를 묻는
경우가 모두 `ood_queries`로 드러나고, 같은 청크 몇 개가 모든 질의에 답하는 현상은
`query_hubness`로 드러납니다.

```bash
vecscope audit corpus.npy --queries queries.npy
```

`--clusters 10`을 붙이면 결과 끝에 군집별 요약이 추가됩니다. 각 군집의 크기, 밀집도, 근접 중복
비율, 허브 점유율과 함께 에이전트가 직접 확인할 수 있는 샘플 id를 보여 줍니다. 판정은 하지
않으며, 문제가 *어느 영역에* 있는지 알려 주는 용도입니다.

```text
  clusters (3):
     id   size  share  cos2ctr  neardup  hubshare  sample_ids
      1   1014  33.8%    0.111     0.0%     34.5%  [1533, 2934, 800, 1585, 824]
      2   1001  33.4%    0.109     0.0%     33.4%  [2550, 1171, 4, 1496, 668]
      0    985  32.8%    0.111     0.0%     32.0%  [1910, 2516, 2859, 124, 2392]
```

## 인덱스 스윕

`vecscope sweep`은 HNSW와 IVF 인덱스를 몇 가지 파라미터 조합으로 구축한 뒤, 조합마다 정확한
k-NN 대비 재현율(recall@k), 구축 시간, 초당 질의 수(QPS)를 실제 코퍼스에서 측정합니다. 그리고
목표 재현율(기본값 0.95)을 만족하는 조합 가운데 가장 빠른 것을 추천합니다. `faiss-cpu`를
추가로 설치해야 합니다.

```bash
pip install "vecscope[index]"
vecscope sweep embeddings.npy -k 10 --target-recall 0.95
```

```text
  index params                                       recall  build s       qps
 *hnsw  M=16, ef_construction=100, ef_search=32       0.994     0.31    141532
  ivf   nlist=76, nprobe=32                           0.772     0.02     70925
 *ivf   nlist=76, nprobe=64                           0.975     0.02     41427
  recommendation: hnsw (M=32, ef_construction=100, ef_search=16)  recall=0.985  qps=128425
```

질의는 기본적으로 코퍼스에서 고정 시드로 표본을 뽑아 쓰고, `--queries`를 지정하면 실제 질의로
측정합니다. MCP 도구 이름은 `sweep_index`입니다.

GPU용 faiss(Linux에서는 `pip install faiss-gpu-cu12`)가 설치되어 있으면 IVF 인덱스와 정답
기준이 되는 정확 검색을 자동으로 GPU에서 실행하고, 출력 첫 줄에 사용한 장치를 표시합니다.
HNSW는 faiss에 GPU 구현이 없어 항상 CPU에서 실행됩니다. `--device cpu|gpu`로 장치를 직접
지정할 수도 있습니다.

## 에이전트와 함께 사용

`vecscope`에는 `audit_embeddings` 도구를 제공하는 MCP 서버가 들어 있습니다. Claude Code,
Cursor, Codex처럼 MCP를 지원하는 클라이언트라면 어디에든 등록할 수 있습니다.

```bash
pip install "vecscope[mcp]"
```

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp" } } }
```

등록한 뒤에는 에이전트에게 이렇게 요청하면 됩니다.

```text
검색기를 튜닝하기 전에 data/chunks.npy 임베딩을 감사하고, 먼저 조치할 항목을 알려줘.
```

[`.agents/skills/vecscope/SKILL.md`](../../.agents/skills/vecscope/SKILL.md)의 에이전트
스킬에는 감사를 실행할 시점과 판정별 조치(벡터 중심화(centering) 또는 모델 교체, 중복 제거,
재순위화(re-ranking), 인덱스 재구축)가 정리되어 있습니다. 두 번째 도구 `sweep_index`는
[인덱스 스윕](#인덱스-스윕)을 실행합니다. MCP를 지원하지 않는 에이전트라면 CLI를 `--json`으로
실행해 같은 결과를 읽으면 됩니다.

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

## CLI 옵션

```bash
vecscope audit vectors.npy -k 20                       # 항목당 이웃 수 (기본값 10)
vecscope audit vectors.parquet --json                  # JSON 형식 출력
vecscope audit vectors.npy --ann hnsw_neighbors.npy    # ANN 인덱스의 재현율을 정확한 k-NN과 비교
vecscope audit vectors.npy --queries queries.npy       # 질의 측 진단 추가
vecscope audit vectors.npy --clusters 10               # 군집별 요약 추가
vecscope audit vectors.npy --fail-on warn              # warn 판정도 종료 코드 1로 처리
vecscope audit vectors.npy --near-dup 0.98             # 근접 중복으로 간주할 코사인 임곗값
vecscope audit vectors.npy --sample 0                  # 샘플링 없이 전체 행을 감사 (기본값은 10만 행)
vecscope sweep vectors.npy --target-recall 0.95        # HNSW/IVF 파라미터 스윕 (faiss 추가 설치)
vecscope sweep vectors.npy --device gpu                # IVF를 GPU에서 실행 (GPU용 faiss)
```

## 입력 형식

임베딩 파일에는 행마다 벡터가 하나씩 들어갑니다. 어떤 형식이든 감사 전에 `(n, dim)` 형태의
2차원 `float32` 배열로 바뀝니다. 아래 예시의 `embeddings`가 바로 이 배열이며, 모델의 `encode`
출력이 대표적인 예입니다.

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
파일과 같아야 합니다. 위 형식 중 어느 것이든 쓸 수 있습니다. id가 행 위치를 가리키므로 바로 이
행들로 만든 인덱스에서 나온 파일이어야 하고, `--sample`이 적용되면 행 id가 어긋나므로 ANN
파일은 무시됩니다.

`--queries` 파일(MCP에서는 `queries_path`)도 선택 입력으로, 같은 형식 중 하나로 저장한 질의
임베딩입니다. 코퍼스와 `dim`이 같아야 합니다.

## 동작 원리

`vecscope`는 행렬 전체의 정확한 k-NN을 한 번만 계산하고, 모든 진단을 그 결과에서 끌어냅니다.
k-출현 빈도 분포에서 hubness와 antihubs를, 최근접 이웃 거리에서 collapse, outliers,
distance_concentration을, k-NN 그래프에서 connectivity를, 무작위 쌍과 공분산 스펙트럼에서
anisotropy, effective_rank, rogue_dimensions를, 벡터 길이에서 norms를 구합니다.
metric_mismatch는 내적 기준 k-NN을 한 번 더 계산해 비교하되, 벡터 길이가 모두 같으면
건너뜁니다. ANN 이웃 파일이 있으면 정확한 이웃과 비교해 재현율을 구하고, 질의 임베딩이 있으면
코퍼스를 대상으로 정확 검색을 한 번 더 수행해 질의 측 항목을 구합니다.

같은 입력에는 언제나 같은 결과가 나옵니다. 시드가 고정되어 있고, 이웃은 근사 없이 정확하게
계산하며, 학습된 모델이나 네트워크 통신도 쓰지 않습니다. 같은 파일을 두 번 감사하면 JSON이
완전히 일치합니다.

정확한 k-NN은 행 수의 제곱에 비례하는 비용이 들기 때문에, 10만 행을 넘는 입력은 `--sample 0`을
지정하지 않는 한 고정 시드로 뽑은 표본만 감사합니다.

## 실제 임베딩 검증

임곗값은 미리 계산되어 공개된 임베딩으로 보정했습니다. 대상은 OpenAI ada-002와
text-embedding-3-large, Cohere embed-v3, all-MiniLM-L6-v2, bge-m3, mxbai-embed-large-v1,
gte-base-en-v1.5의 8개 모델입니다. 아래 표는 각 데이터셋의 앞 4만 행을 측정한 값이며, 판정
표시가 없는 칸은 `ok`입니다. `python examples/real_models.py`로 재현할 수 있고 모델을 내려받을
필요는 없습니다.

| finding | Cohere v3 (한국어 위키) | Cohere v3 (영어 위키) | ada-002 (dbpedia) | 3-large (dbpedia) | MiniLM-L6 (위키) | bge-m3 (위키) | mxbai-large (위키) | gte-base (arXiv) |
|---|---|---|---|---|---|---|---|---|
| hubness | 1.47 | 1.16 | 1.56 | 1.21 | 1.59 | 1.85 | 1.43 | 2.28<br>warn |
| antihubs | 0.009 | 0.005 | 0.013 | 0.005 | 0.022 | 0.026 | 0.020 | 0.013 |
| anisotropy | 0.333<br>warn | 0.228 | 0.692<br>**fail** | 0.086 | 0.026 | 0.307<br>warn | 0.330<br>warn | 0.210 |
| effective_rank | 0.158 | 0.178 | 0.090<br>warn | 0.148 | 0.285 | 0.143 | 0.087<br>warn | 0.136 |
| collapse | 0.002 | 0.002 | 0.000 | 0.000 | 0.001 | 0.000 | 0.017 | 0.000 |
| rogue_dimensions | 0.003 | 0.003 | 0.002 | 0.002 | 0.004 | 0.002 | 0.002 | 0.002 |
| norms | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| metric_mismatch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 |
| distance_concentration | 3.00 | 3.00 | 2.53 | 1.86 | 2.93 | 2.77 | 3.99 | 2.30 |
| outliers | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.000 |
| connectivity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

`fail`은 ada-002의 anisotropy 하나뿐이며, 이 모델의 비등방성은 널리 알려진 특성입니다. 평균
단위 벡터의 길이가 0.83이라 무관한 문서끼리도 코사인 유사도가 0.7 언저리에서 시작합니다.
후속 모델인 3-large는 같은 코퍼스에서 등방성을 보입니다(길이 0.29). 세 모델 모두 중심화하면
무작위 쌍의 코사인 유사도가 0.00으로 떨어지므로, 이 값을 세부 수치로 함께 보고합니다. 텍스트
임베딩에서 hubness 왜도 1.2~1.9는 정상 범위라 `warn` 임곗값을 2.0으로 잡았고, arXiv 초록을
임베딩한 gte-base만 이를 살짝 넘습니다. 품질이 좋은 모델 세 개도 anisotropy `warn`
구간(0.30~0.33)에 들어가지만 정도가 가볍고, `details`의 중심화 후 수치를 보면 중심화만으로
해소됩니다.

## FAQ

<details>
<summary><strong>벡터가 외부로 전송됩니까?</strong></summary>

아니요. NumPy만 사용하고 모든 계산이 로컬에서 이루어집니다. 모델을 내려받거나 네트워크
연결을 열지 않습니다.

</details>

<details>
<summary><strong>얼마나 큰 행렬까지 처리할 수 있습니까?</strong></summary>

정확한 k-NN은 행 수의 제곱에 비례하는 비용이 듭니다. 그래서 기본 설정에서는 입력이 10만 행을
넘으면 고정 시드로 뽑은 표본만 감사하고, 그 사실을 출력에 표시합니다. 전체 행을 감사하려면
`--sample 0`을 지정하십시오.

</details>

<details>
<summary><strong>운영 중인 모델이 anisotropy에서 fail 판정을 받았습니다. 모델에 문제가 있는 것입니까?</strong></summary>

꼭 그렇지는 않습니다. 널리 쓰이는 텍스트 임베딩 모델 상당수가 구조적으로 강한 비등방성을
띱니다. 이 판정은 코사인 점수가 좁은 구간에 몰려 변별력이 떨어진다는 뜻이고, 중심화,
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
- [예제](../../examples/): 합성 행렬 생성 스크립트와 실제 임베딩 보정 스크립트
- [변경 이력](../../CHANGELOG.md)

## 기여

이슈와 PR을 환영합니다. 테스트 방법과 기여 기준은 [CONTRIBUTING.md](../../CONTRIBUTING.md)에,
보안 취약점 제보 방법은 [SECURITY.md](../../SECURITY.md)에 있습니다.

## 라이선스

[MIT](../../LICENSE)
