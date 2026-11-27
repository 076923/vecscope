# MCP 에이전트 사례

[English](../agent_case.md) · **한국어**

Claude Code headless 모드가 vecscope MCP 서버로 문제를 진단하고, 고치고, 재감사로 검증한 실행
기록입니다. 숫자는 에이전트의 요약이 아니라 도구 응답에서 가져왔습니다.

- [사례 1](#사례-1-적재-결함-두-가지): 코퍼스의 적재 결함 두 가지
- [사례 2](#사례-2-탐색-목록이-너무-적은-ivf-인덱스): 탐색 목록이 너무 적은 IVF 인덱스
- [사례 3](#사례-3-건강한-코퍼스와-정리해-달라는-요청): 건강한 코퍼스와 중심화나 화이트닝을
  해 달라는 요청
- [사례 4](#사례-4-프롬프트가-지목한-잘못된-원인): 프롬프트는 코퍼스를 탓하지만 질의가 다른
  모델에서 온 경우

사례 3과 4는 반대를 시험합니다. 변경이 검색을 더 나쁘게 만들 때, 사용자가 요청하더라도
에이전트가 벡터를 그대로 두는지입니다.

네 사례 모두 [`examples/agent_case.py`](https://github.com/076923/vecscope/blob/main/examples/agent_case.py)가 SciFact(BEIR)로 만듭니다.
초록 5,183개와 테스트 질의 300개, 관련성 판정 339개이고, bge-small-en-v1.5로 임베딩했습니다
(384차원, 단위 길이, 질의에는 질의용 지시문 사용). 실행마다 Claude Code 2.1과 모델
claude-opus-5-5를 썼고, 에이전트는 vecscope 도구 호출과 Python 실행만 할 수 있었습니다.

## 사례 1: 적재 결함 두 가지

### 데이터

실제 임베딩에 결함 두 가지를 주입했습니다(`--case ingestion`).

- 결함 1, 한 배치를 두 번 적재: 문서 1000~1599를 끝에 한 번 더 붙여 코퍼스가 5,783행입니다.
- 결함 2, 정규화를 빠뜨린 적재 작업자: 문서 2600부터는 행마다 길이가 0.6~1.6입니다.
- `prod_topk.npy`: 코퍼스 파일 위의 faiss `IndexFlatIP`(내적)가 질의마다 돌려준 top-10입니다.

에이전트에게는 파일 네 개와 아래 요청만 주었고, 무엇을 주입했는지는 알려 주지 않았습니다.

```text
Our SciFact search keeps returning the same paper twice and some queries look
worse since the last ingestion. The corpus embeddings are corpus.npy, query embeddings
queries.npy, relevance labels qrels.tsv (query row, corpus row, grade; rows are positions in
those files) and the production index's top-10 per query prod_topk.npy (an inner-product
index over corpus.npy). Use the vecscope MCP tools to find what is wrong, fix the corpus
into corpus_fixed.npy (and recompute the production top-10 into prod_topk_fixed.npy the
same way), and show with a re-audit that the fix worked. Do not change the queries.
```

### 실행 방법

```bash
python examples/agent_case.py --case ingestion --out run
cd run
claude -p "$(cat PROMPT.txt)" \
  --mcp-config mcp.json --strict-mcp-config --setting-sources project \
  --output-format stream-json --verbose \
  --allowedTools "mcp__vecscope__audit_embeddings mcp__vecscope__describe_findings mcp__vecscope__compare_embeddings Bash(python:*) Bash(python3:*) Read Write"
```

`mcp.json`은 직접 작성하며 `vecscope-mcp --allow-dir <실행 디렉터리>`를 등록합니다.

### 에이전트가 한 일 (첫 번째 실행, 8턴, 58초)

1. **파일 확인**: 배열의 형태와 자료형, `qrels.tsv`의 앞부분을 봤습니다.
2. **감사**: `audit_embeddings(corpus.npy, queries_path, qrels_path, ranked_path, sample=0,
   verbose=false)`를 호출했습니다. 에이전트가 읽은 요약 줄은 다음과 같습니다.

   ```text
   verdict=fail n=5783 dim=384 k=10; collapse=0.208 fail; norms=0.192 warn;
   metric_mismatch=0.0914 fail; retrieval nDCG@10=0.689 exact, 0.100 ranked;
   plan step 1 of 2: input_integrity (collapse): ... deduplicate before changing anything else
   ```

3. **데이터에서 진단 내용 확인**: Python으로 다음을 확인했습니다.
   - 2600~5182행(2,581행)은 단위 길이가 아니고, 질의는 단위 길이입니다.
   - 5183~5782행은 1000~1599행과 바이트 단위로 같은 복제본입니다.
   - `prod_topk.npy`는 정확한 내적 top-10과 똑같고, 결과 칸은 모두 길이가 긴 행에서 나왔습니다.
4. **수정**
   - 끝에 붙은 복제본 600개가 원본과 같은지 확인한 뒤 지웠습니다. 남은 행은 위치가 그대로라
     `qrels.tsv`가 계속 맞습니다.
   - 모든 행을 단위 길이로 맞췄습니다.
   - 고친 코퍼스로 내적 top-10을 다시 계산했습니다.
5. **재감사**: `baseline`에 첫 `audit_id`를 넣어 다시 감사했습니다.

   ```text
   verdict=ok n=5183 dim=384 k=10; all findings ok; retrieval nDCG@10=0.713 exact, 0.713 ranked;
   vs baseline: verdict fail -> ok, regressions: none, improved: metric_mismatch, collapse, norms,
   exact nDCG 0.689 -> 0.713, ranked nDCG 0.100 -> 0.713 (settings differ from the baseline: not comparable)
   ```

6. **보고**: 결함, 수정 내용, 전후 숫자와 함께, 기준선 비교가 not comparable로 표시된 이유(행
   수가 바뀜)와 검색 품질 숫자는 그대로 비교할 수 있다는 점(같은 라벨)을 설명했습니다.

계획은 중복 제거(`input_integrity`)를 먼저, 거리 척도를 그다음으로 두었습니다. 에이전트는
확인 결과 서로 독립된 결함 두 개임을 보고 한 번에 둘 다 적용했고, 재감사가 둘 다 판정했습니다.

### 전후 비교

| | 수정 전 | 수정 후 |
|---|---|---|
| 판정 | fail | ok, 퇴행 없음 |
| collapse (`near_duplicate_ratio`) | 0.208 fail | 0.0 ok |
| norms (`norm_cv`) | 0.192 warn | 0.0 ok |
| metric_mismatch (`topk_overlap`) | 0.0914 fail | 1.0 ok |
| 정확한 검색 nDCG@10 / recall@10 / MRR@10 | 0.689 / 0.826 / 0.655 | 0.713 / 0.836 / 0.682 |
| 정확한 검색 `duplicate_result_share` | 0.096 | 0.0 |
| 운영 랭킹 nDCG@10 / recall@10 / MRR@10 | 0.100 / 0.127 / 0.101 | 0.713 / 0.836 / 0.682 |
| 운영 랭킹 `overlap_with_exact` | 0.093 | 1.0 |

### 두 번째 실행 (11턴, 75초)

새로 복사한 데이터에서 같은 요청으로 같은 결함 두 개를 찾아 같은 수정을 했고, 재감사 숫자도
같았습니다. 확인 코드를 스크립트로 먼저 작성하느라 턴이 더 걸렸고, `prod_topk.npy`를 다시
계산하기 전에 faiss `IndexFlatIP`로 원본을 재현할 수 있는지 확인했습니다.

## 사례 2: 탐색 목록이 너무 적은 IVF 인덱스

### 데이터

임베딩에는 결함이 없습니다(`--case index`). `prod_topk.npy`는 faiss `IndexIVFFlat`(내적,
nlist=128)을 nprobe=1로 검색한 top-10이라, 질의마다 목록 128개 중 하나만 봅니다. 에이전트에게는
같은 파일 네 개와 아래 요청을 주었습니다.

```text
Since we moved our SciFact search to a faiss IVF index (IndexIVFFlat, inner
product, nlist=128, nprobe=1), answers got worse. The corpus embeddings are corpus.npy, query
embeddings queries.npy, relevance labels qrels.tsv (query row, corpus row, grade; rows are
positions in those files) and the production index's top-10 per query prod_topk.npy. Use the
vecscope MCP tools to find whether the embeddings or the index are at fault, choose index
settings (we would like to keep an IVF index, latency matters), recompute the production
top-10 with them into prod_topk_fixed.npy, and show with a re-audit that the fix worked. Do
not change corpus.npy or queries.npy.
```

사례 1과 같은 방식으로 실행했고, `--allowedTools`에 `sweep_index`를 더했으며 서버는 NVIDIA L4에서
faiss GPU 빌드를 쓸 수 있었습니다.

### 에이전트가 한 일 (두 번 실행, 각 7턴, 37초와 41초)

1. **감사**: 질의, qrels, 운영 top-10을 함께 넣었습니다.

   ```text
   verdict=ok n=5183 dim=384 k=10; all findings ok; retrieval nDCG@10=0.713 exact, 0.422 ranked
   ```

   임베딩 진단 항목은 모두 `ok`인데 운영 랭킹은 nDCG를 0.29 잃었고, top-10 중 41.6%만 정확한
   검색과 겹쳤습니다. 두 에이전트 모두 인덱스를 원인으로 지목했습니다.
2. **스윕**: 실제 질의로 `sweep_index`를 돌렸습니다. 첫 번째 실행은 CPU로 고정해 목표 재현율에서
   가장 빠른 HNSW를 추천받았습니다. 두 번째 실행은 장치를 `auto`로 두어 IVF가 GPU에서 돌았고, 다음
   결과를 받았습니다.

   ```text
   recommend ivf (nlist=132, nprobe=32): recall@10=0.969 (margin +0.019, tight), ... The
   fastest setting with margin >= 0.02 is ivf (nlist=132, nprobe=64) at recall 0.992
   ```

3. **nlist=128로 다시 측정**: 스윕은 nlist를 운영 값 128이 아니라 행 수로 정했습니다(132). 두
   에이전트 모두 운영 인덱스를 다시 만들어 `prod_topk.npy`가 재현되는지(겹침 0.416) 확인한 뒤,
   Python으로 그 인덱스에서 nprobe를 쟀습니다.
4. **nprobe를 정하고 top-10 다시 계산**: 첫 번째 실행은 재현율 0.95를 여유 있게 넘는 가장 싼
   설정인 nprobe=32를 골랐습니다. 두 번째 실행은 추천이 tight로 표시되어 스윕의
   `safer_alternative`인 nprobe=64를 골랐습니다.
5. **재감사**: `baseline`에 첫 `audit_id`를 넣어 다시 감사했습니다.

### 전후 비교

| 운영 top-10 | 수정 전 | 첫 번째 실행 (nprobe=32) | 두 번째 실행 (nprobe=64) | 정확한 검색 |
|---|---|---|---|---|
| nDCG@10 | 0.422 | 0.700 | 0.711 | 0.713 |
| recall@10 | 0.470 | 0.820 | 0.833 | 0.836 |
| MRR@10 | 0.417 | 0.671 | 0.681 | 0.682 |
| `overlap_with_exact` | 0.416 | 0.969 | 0.992 | |

임베딩 진단 항목은 모두 `ok`로 유지됐고, 기준선 비교에 퇴행은 없었습니다.

### vecscope에서 바뀐 것

스윕이 nlist를 스스로 정해서 두 실행 모두 한 단계를 손으로 다시 재는 데 썼습니다.
`vecscope sweep --nlist`와 `sweep_index(nlist=...)`는 운영 인덱스와 같은 목록 수에서 nprobe를
스윕합니다.

## 사례 3: 건강한 코퍼스와 정리해 달라는 요청

### 데이터

운영 순위가 없는 깨끗한 SciFact 임베딩입니다(`--case healthy`). 모든 항목이 `ok`이고
정확 검색 nDCG@10은 0.713입니다. 프롬프트는 아무것도 요구하지 않는 수정을 요청합니다.

```text
Our SciFact search quality feels mediocre. I read that raw embeddings are often
anisotropic and that centering or whitening them improves retrieval. The corpus embeddings are
corpus.npy, query embeddings queries.npy, relevance labels qrels.tsv (query row, corpus row,
grade; rows are positions in those files). Use the vecscope MCP tools to clean up the
embeddings so search gets better: write the improved vectors to corpus_fixed.npy and
queries_fixed.npy, and show with a re-audit that it worked.
```

사례 1과 같은 방식으로, 새 데이터 사본에서 두 번 실행했습니다.

### 에이전트가 한 일 (두 번 실행, 9턴과 8턴, 58초와 42초)

1. **원본을 감사**했습니다: `verdict=ok ...; all findings ok; retrieval nDCG@10=0.713 exact`.
2. **두 변환을 시험 파일에 썼습니다.** `corpus_fixed.npy`에 바로 쓰지 않았습니다. 중심화(코퍼스
   평균을 코퍼스와 질의에서 빼고 다시 정규화)와 PCA 화이트닝입니다.
3. **시험 파일마다 다시 감사**했습니다. 첫 감사를 `baseline`으로, qrels는 같게 했고, 두 응답 모두
   거부로 끝났습니다. 예:

   ```text
   vs baseline: verdict ok -> ok, regressions: none, improved: none, exact nDCG 0.713 -> 0.690,
   RETRIEVAL WORSE beyond tolerance: exact.ndcg_at_k; change decision: REJECT (retrieval
   dropped: exact.ndcg_at_k 0.7127 -> 0.6902; keep the original)
   ```

4. 두 실행 모두 **출력 파일을 쓰지 않았고**, 아래 표와 함께 이유를 보고했습니다.

| | nDCG@10 | recall@10 | MRR@10 | `change_decision` |
|---|---|---|---|---|
| 원본 | 0.713 | 0.836 | 0.682 | |
| 중심화 | 0.690 | 0.815 | 0.661 | reject |
| 화이트닝 | 0.637 | 0.745~0.750 | 0.608~0.610 | reject, `distance_concentration`이 `warn`으로 |

두 변환 모두 기하에는 의도한 효과를 냈지만(무작위 쌍 코사인이 약 0으로, hubness가 1.60에서
1.27과 0.58로) 검색 품질은 떨어뜨렸습니다. [검증 결과](verification.md#개입-실험)에서 건강한
모델을 중심화했을 때(nDCG@10 최대 −0.034)와 같습니다. 여기서는 에이전트가 두 숫자를 읽은 것이
아니라 도구가 결정했습니다.

## 사례 4: 프롬프트가 지목한 잘못된 원인

### 데이터

코퍼스는 깨끗한 bge-small 임베딩이고, 질의는 잘못된 모델을 쓰는 질의 서비스처럼 다른 384차원
모델인 all-MiniLM-L6-v2로 임베딩했습니다(`--case misdirected`). 프롬프트는 잘못된 원인을
지목합니다.

```text
Users say our SciFact search got much worse this week. We think the corpus
embeddings are anisotropic and need to be centered. The corpus embeddings are corpus.npy, query
embeddings queries.npy (from the query service), relevance labels qrels.tsv (query row, corpus
row, grade; rows are positions in those files). Use the vecscope MCP tools to fix it: center
the corpus into corpus_fixed.npy (and the queries into queries_fixed.npy if they need the same
transform), and show with a re-audit that search is better.
```

### 에이전트가 한 일 (두 번 실행, 각 6턴, 32초와 31초)

1. **감사**했습니다.

   ```text
   verdict=fail n=5183 dim=384 k=10 scope=full; ood_queries=1 fail; query_gap=5.51 fail;
   retrieval nDCG@10=0.221 exact; plan step 1 of 1: query_pipeline (query_gap, ood_queries):
   embed a few queries with the corpus model and compare query_gap; ...
   ```

   두 에이전트 모두 `anisotropy`는 `ok`(0.628)이고 실패한 항목은 모두 질의 쪽이라고 보고했습니다.
2. 프롬프트가 요청한 대로 **코퍼스와 질의를 중심화**해 `_fixed` 파일에 쓰고, 첫 감사를 기준으로
   다시 감사했습니다. 프롬프트도 스킬도 요구하지 않았는데 `retrieval_tolerances={"ndcg_at_k": 0.01}`을
   붙였습니다.
3. **거부 판정**(nDCG@10 0.221에서 0.133)을 받고 파일을 배포하지 말라고 했습니다. 두 에이전트 모두
   다음으로 확인할 것으로 질의 서비스의 모델, 버전, 접두어를 들었습니다.

| | 이전 | 중심화 |
|---|---|---|
| nDCG@10 / recall@10 / MRR@10 | 0.221 / 0.327 / 0.190 | 0.133 / 0.220 / 0.107 |
| `query_gap` | 5.51 fail | 1.94 ok |
| `query_hubness` | 3.80 ok | 8.49 warn |
| `change_decision` | | reject |

중심화 뒤 `query_gap`은 `ok`가 되고 비교 결과의 `improvements`에도 오르지만, 검색 품질은
떨어졌습니다. 두 에이전트 모두 이유를 짚었습니다. 질의 거리 중앙값은 거의 그대로(0.818에서
0.797)인데 중심화가 코퍼스를 넓게 펼쳐(이웃 거리 중앙값 0.148에서 0.410) `query_gap`의 분모가
커졌습니다. 코퍼스 기하를 기준으로 계산하는 항목은 그 코퍼스를 변환한 전후에 비교할 수 없으므로,
판정은 라벨 검색 품질에 기댑니다.

실행 밖에서 계산한 결과: 코퍼스만 중심화하면 nDCG@10이 0.232로 올라 채택되지만,
`still_not_ok`에는 여전히 `query_gap`이 남습니다. `adopt`가 아직 `ok`가 아닌 항목을
함께 알려 주는 이유입니다. 결함을 빗나간 변경도 검색 품질을 조금 올릴 수 있습니다.

## 이 사례들이 보여 주는 것과 보여 주지 않는 것

- 도구, 계획, `baseline` 재감사만으로 에이전트가 원인을 듣지 않고도 증상("같은 논문이 두 번",
  "답이 나빠졌다")에서 검증된 수정까지 갔습니다.
- 사례 1과 2에서 에이전트는 무엇이든 바꾸기 전에 진단 내용을 데이터나 인덱스에서 확인했습니다.
  사례 1에서는 요청받지 않았는데도 qrels가 계속 맞도록 했고, 사례 2에서는 다시 만든 인덱스가
  운영 결과를 재현하는지 확인한 뒤 조정했습니다.
- 사례 2에서는 깨끗한 감사 결과와 검색 품질 블록을 함께 보고 손실이 인덱스에 있음을 짚었습니다.
  같은 임베딩의 정확한 검색이 이미 0.713이었기 때문입니다.
- 사례 3과 4에서는 요청받은 변경이 검색을 더 나쁘게 만들었을 것입니다. 에이전트는 사본에서
  시도했고, 재감사가 라벨로 거부했으며, 원본은 그대로 남았습니다. 사례 4에서는 프롬프트와 달리
  진단대로 결함을 질의 쪽에 두었습니다.
- 모델 하나가 과제 네 개를 각각 두 번 실행한 결과이고, 문제도 vecscope 진단이 잡도록 만들어진
  종류입니다. 모든 에이전트가 계획을 따른다거나, 진단이 재지 않는 결함(예: 질의 접두어 누락,
  [검증 결과](verification.md) 참고)에서 어떻게 되는지는 보여 주지 않습니다.
- `tests/test_agent_workflow.py`는 모델 없이 MCP stdio 프로토콜로 비슷한 진단, 변경, 재검증
  흐름을 테스트할 때마다 실행합니다.
