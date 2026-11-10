[English](../usage.md) · **한국어**

# 사용법

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
nDCG@10은 −0.014~+0.007만 바뀌었고, 질의 거리는 방향 없이 움직였습니다([검증 결과](verification.md)).
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
- IVF의 nlist는 `--nlist`로 정하지 않으면 행 수를 따릅니다(4*sqrt(n)). 기존 인덱스의 nprobe를
  조정하려면 그 인덱스의 nlist를 넘기십시오.

끝의 note 줄은 JSON에 `caveats`로 들어갑니다. MCP 도구 이름은 `sweep_index`이며 같은
옵션(`n_queries`, `target_recall`, `device`, `margin`, `timing_repeats`, `latency_queries`,
`nlist`)을 매개변수로 받습니다.

스윕이 답하는 질문은 하나입니다. 이 코퍼스와 이 질의에서 어떤 파라미터가 목표 재현율을
만족하고, 그 비용이 얼마인가입니다. 수치는 측정한 장비와 프로세스 안에서의 상댓값이므로,
고른 조합은 운영 환경에서 지연 시간·처리량·메모리를 따로 재어 확정하십시오.

GPU용 faiss(`pip install "vecscope[index-gpu]"`, CUDA 12.6 이상인 Linux x86_64용 Meta 공식 `faiss-gpu`)가
설치되어 있으면 IVF 인덱스와 정답
기준이 되는 정확 검색을 자동으로 GPU에서 실행하고, 출력 첫 줄에 사용한 장치를 표시합니다.
HNSW는 faiss에 GPU 구현이 없어 항상 CPU에서 실행됩니다. `--device cpu|gpu`로 장치를 직접
지정할 수도 있습니다.
`index`와 `index-gpu`는 둘 다 `faiss` 모듈을 설치하므로 하나만 설치합니다. `stores`에는 `index`가
포함되어 있으므로 GPU 장비에서는 저장소(store) extra를 하나씩 지정합니다(`"vecscope[index-gpu,qdrant]"`). 드라이버가 오래된 환경에서는 `index-gpu` 대신 커뮤니티 빌드인
`faiss-gpu-cu12`(CUDA 12.1)나 `faiss-gpu-cu11`을 직접 설치합니다.

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
vecscope x.y.z  compare  n=3000 dim=128/128 k=10  neighbors=recomputed/full
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

## 벡터 저장소에서 읽기

경로 자리에 임베딩 파일 대신 faiss 인덱스 파일이나 벡터 데이터베이스를 지정할 수 있습니다.

- 행은 id 오름차순으로 읽으므로 내용이 같으면 두 번 읽어도 같은 행렬이 됩니다.
- 결과에는 출처와 읽은 내용의 지문이 기록됩니다.
- `details.sample_ids`의 행 번호에는 저장소의 id인 `sample_source_ids`가 함께 붙습니다.

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
- `timeout=`(네트워크 저장소 요청 하나의 제한 시간, 초 단위, 기본값 60. 유한한 양수여야 하고 최대
  3600)

읽기만 하며, 로드되지 않은 Milvus 컬렉션만 읽는 동안 로드했다가 다시 해제합니다.

비밀값은 출력에 남지 않습니다.

- 결과에는 사용자 정보를 지우고, 위 옵션이 아닌 질의 값은 `REDACTED`로 바꾼 URI가 기록됩니다.
- 저장소 클라이언트의 오류는 API 키와 그 값들을 지운 뒤 다시 던집니다.
- Chroma HTTP 클라이언트는 자체 제한 시간 설정이 없어서, 연결한 뒤 세션에 `timeout=`을
  겁니다. 클라이언트를 만들 때 하는 연결 확인에는 제한 시간이 없습니다.

MCP 서버는 로컬 저장소를 [허용된 디렉터리](mcp.md#서버가-읽을-수-있는-범위) 안에서 읽고, 네트워크
저장소는 `--allow-network`로 시작했을 때만 읽습니다.

## 대규모 코퍼스: 증분 감사

전체 행 감사는 행 수의 제곱에 비례합니다. 코퍼스가 뒤에 행을 덧붙이거나 일부 행을 제자리에서
다시 임베딩할 때, `--knn-cache`는 실행 사이에 정확한 이웃 목록을 보관하고 그 행들이 바꾸는
부분만 계산합니다.

```bash
vecscope audit day1.npy --sample 0 --knn-cache knn.npz    # created: 전체 행 계산
vecscope audit day2.npy --sample 0 --knn-cache knn.npz    # extended: 덧붙인 행만 계산
vecscope audit day3.npy --sample 0 --knn-cache knn.npz    # updated: 바뀐 행과 그 이웃만 계산
```

캐시에는 계산에 쓴 행마다 해시가 들어 있습니다.

- 같은 행이면 그대로 씁니다(reused).
- 행이 덧붙고 캐시의 행은 그대로이면 확장합니다(extended).
- 캐시의 행 일부가 새 벡터로 바뀌었으면 갱신합니다(updated). 바뀐 행과, 이웃 목록에 바뀐 행이
  들어 있던 행은 전체 행과 다시 계산합니다. 나머지 행은 바뀐 행과 덧붙인 행만 봅니다.
- 그 밖의 경우에는 새로 만듭니다. 다시 계산할 행이 전체의 절반을 넘거나(갱신 비용이 새로
  만드는 것과 비슷해짐), 행이 지워졌거나, `dim`이 다른 경우입니다. 이유는
  `knn_cache.reason`에 적힙니다.

변경은 퍼집니다. k=10이면 바뀐 행 하나가 다른 행 약 10개의 이웃 목록에 들어 있습니다. 무작위로
1%를 바꾸면 약 10%를, 5%를 바꾸면 약 절반을 다시 계산하므로, 갱신은 작고 흩어진 재임베딩과
행 추가에서 이득이 있습니다.

이웃 목록은 거리가 정확히 같은 동점을 빼면 전체 계산과 같고, 진단 값은 float32 반올림
범위에서 일치합니다. Python에서는 `vecscope.incremental.cached_knn_for_audit(x, k, path)`가
`audit(x, knn=...)`에 넘길 목록을 돌려줍니다.

10만 행 × 768차원([성능](performance.md#성능) 표와 같은 장비)에서 잰 결과입니다.

- 5,000행을 덧붙였을 때 캐시 확장에 11초, 캐시로 감사하는 데 4초가 걸렸고 전체 감사는
  94초였습니다.
- 2만 행을 덧붙이면 43초와 4초, 전체 감사는 123초였습니다.

판정은 모두 전체 감사와 같았고 값의 최대 상대 차이는 7e-5입니다. 캐시를 처음 만드는 비용은
전체 감사와 비슷합니다(여기서는 83초).

10만 행 중 1,000행을 제자리에서 다시 임베딩하면 10,625행을 다시 계산했습니다. 캐시 갱신에
15초, 캐시로 감사하는 데 5초가 걸려 전체 감사(95초)보다 빨랐고 판정도 같았습니다. 1만 행을
바꾸면 갱신이 68,933행을 다시 계산해 83초가 걸려 전체 감사(89초)와 비슷하므로, 이때는 캐시를
새로 만듭니다. 이 기준으로 다시 돌리자 1만 행 변경에서 캐시를 새로 만들었고, 87초로 전체 감사(87초)와
같았습니다.

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
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv   # 정확한 검색의 nDCG/recall/MRR 추가
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv --ranked topk.npy   # 운영 랭킹도 함께 채점
vecscope audit index.faiss --ann-from-source --sample 0     # faiss 인덱스나 저장소를 읽고 자체 이웃을 채점
vecscope audit "qdrant://host:6333/docs" --save-ids ids.json   # 벡터 저장소 URI, 행 순서대로 id 저장
vecscope audit vectors.npy --sample 0 --knn-cache knn.npz   # 실행 사이에 정확한 이웃 목록을 재사용/확장/갱신
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
vecscope sweep vectors.npy --nlist 128                 # 운영 인덱스의 목록 수에서 nprobe 스윕
vecscope compare a.npy b.npy -k 10                     # 같은 행을 가진 두 임베딩의 이웃 일치도
vecscope compare a.npy b.npy --neighbors-a a_knn.npy --neighbors-b b_knn.npy   # 행은 표본, 이웃은 전체 코퍼스
vecscope calibrate fit_*.json --min-quality 0.45 --out th.json   # 자신의 파이프라인 상태들로 게이트 임곗값 산출
vecscope calibrate fit_*.json --min-quality 0.45 --holdout hold_*.json --include-low-confidence   # 홀드아웃 상태로 확인, 탐색용 쌍도 함께 저장
```

## 입력 형식

임베딩 파일에는 행마다 벡터가 하나씩 들어갑니다. 어떤 형식이든 감사 전에 `(n, dim)` 형태의
2차원 `float32` 배열로 바뀝니다.

값은 유한해야 합니다. NaN, Inf, 그리고 제곱이 `float32` 범위를 넘는 절댓값 1e16 초과 값은
안내 문구와 함께 거부됩니다.

아래 예시의 `embeddings`가 바로 이 배열이며, 모델의 `encode` 출력이 대표적인 예시입니다.

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

형식별로 읽을 때 드는 메모리입니다. float32 벡터 50,000개 x 384차원(행렬 73 MiB)에서 쟀습니다.

| 형식 | 읽는 방식 | 최대 메모리 |
| --- | --- | --- |
| `.npy` | 저장된 그대로 | 0.10 GiB |
| `.csv` | 여러 줄씩 끊어서 | 0.18 GiB |
| `.jsonl` | 한 줄씩 읽어 행 블록 단위로 변환 | 0.34 GiB |
| `.parquet` | 벡터 열만 배치 단위로 읽어 배열 하나에 채움 | 0.39 GiB |
| `.json` | 파일 전체를 한 번에 파싱 | 1.13 GiB |

큰 코퍼스에는 `.npy`, `.csv`, `.parquet`를 쓰세요. `.json`은 전체를 한 번에 파싱하므로 행렬
크기의 몇 배가 듭니다.

선택 입력은 다음과 같습니다.

- **`--ann`**(MCP에서는 `ann_path`): `(n, k)` 형태의 정수 배열. `i`번째 행에는 ANN 인덱스가
  `i`번째 항목의 이웃으로 돌려준 `k`개의 id가 들어가며, 행 순서는 임베딩 파일과 같아야 합니다.
  위 형식 중 어느 것이든 쓸 수 있습니다.
- **`--queries`**(MCP에서는 `queries_path`): 같은 형식 중 하나로 저장한 질의 임베딩. 코퍼스와
  `dim`이 같아야 합니다.
- **`query_shift`의 기준**: `--queries`와 함께 실행한 감사의 `--json` 출력(`--baseline`,
  MCP에서는 `baseline`이나 `query_baseline_path`).

ANN id는 행 위치를 가리키므로 바로 이 행들로 만든 인덱스에서 나온 파일이어야 하고, 모든 행을
감사해야 합니다. 행 수가 `--sample` 크기(기본값 10만 행)를 넘으면 `--sample 0`을 지정해야 하며,
지정하지 않으면 감사가 거부합니다. 항목 자기 자신을 가리키는 id는 무시되므로, 자기 자신을 먼저
돌려주는 인덱스라면 k + 1개를 질의하십시오.

`query_shift`의 기준은 같은 코퍼스나 그대로인 앞부분에서 감사된 경우에만 씁니다(행의
SHA-256인 `corpus_fingerprint`로 확인). 정규화나 중심화 뒤에는 이전 질의 거리가 다른 공간의
값이므로 `query_shift`를 건너뛰고 `query_reference_note`에 이유를 적습니다.
