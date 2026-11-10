[English](../quality.md) · **한국어**

# 검색 품질과 임곗값

## 검색 품질과 수정 순서

진단 항목은 벡터 공간을 설명할 뿐이고, 맞는 문서가 돌아오는지는 정답 라벨이 있어야 알 수
있습니다.

질의의 관련성 판정(qrels)을 넘기면 감사 결과에 `retrieval` 블록이 붙습니다.

- 모든 행에 대한 정확한 코사인 검색의 nDCG@k, recall@k, MRR@k, 적중률(진단 항목을 표본으로
  감사할 때도 전체 행 기준)
- `duplicate_result_share`: 결과 칸 가운데 더 높은 순위 결과의 근접 중복이 차지한 비율

`--ranked`를 주면 운영 인덱스가 돌려준 top-k도 정확한 검색 옆에서 함께 채점하고
`overlap_with_exact`를 붙입니다.

`--ranked`에는 질의를 처리하는 단계가 무엇이든 그 최종 top-k를 넣으면 됩니다. ANN 인덱스,
필터를 건 저장소 질의, 하이브리드 검색, 재정렬기 모두 됩니다. 정확한 검색 기준선은 필터,
키워드 점수, 재정렬 없이 모든 행에 대한 코사인입니다.

- 파이프라인에 이런 단계가 있으면 `overlap_with_exact`는 ANN 재현율만이 아니라 파이프라인이
  벡터 검색에서 얼마나 벗어났는지를 잽니다.
- vecscope는 단계별로 따로 채점하지 않습니다. 품질을 잃는 단계를 찾으려면 각 단계의 출력을
  차례로 `--ranked`에 넣으세요.

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

관련 문서가 없으면 nDCG와 recall이 정의되지 않으므로, 점수는 관련(등급 > 0) 판정이 하나
이상 있는 질의만으로 평균합니다. 관련 판정이 없거나 판정 자체가 없는 질의는
`unscored_queries`로 세고 설명에 적습니다.

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

### 수정 순서

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

몇몇 항목은 그 항목을 만들 수 있는 원인의 단계로 옮겨 묶습니다.

- 공통 방향 때문에 생긴 근접 중복(모든 코사인이 기준을 넘지만 정확한 복사본은 없음)은 중복
  제거가 아니라 공통 방향 단계로 묶습니다.
- 중복과 함께 나온 `query_gap`은 중복 단계에 넣습니다. 근접 복제가 `query_gap`이 나누는 코퍼스
  거리를 줄이므로, 질의를 의심하기 전에 중복을 제거하고 다시 읽어야 하기 때문입니다.
- 섬 기준보다 작은 근접 중복 묶음과 함께 나온 섬(connectivity)도 중복 단계에 넣습니다. 그 복제
  묶음이 섬일 수 있기 때문입니다.

규칙은 고정되어 있고 `src/vecscope/plan.py`에 있습니다. 각 단계는 다시 잴 숫자로 시험할
가설이지 진단이 아닙니다.

단계 안의 항목마다 `placed_because`가 붙고, 둘 중 하나입니다.

- "this group's cause covers it"
- 그 항목을 그 단계에 넣게 한 다른 항목의 판정. 예: "rogue_dimensions is not ok; one
  coordinate moves this finding too". 함께 나타났다는 이유로 묶은 것이므로 단계의 재측정
  숫자로 확인하십시오.

실제 임베딩에서 이 조치들이 검색 품질을 어떻게 바꿨는지는 [검증 결과](verification.md#개입-실험)에
있습니다.

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

텍스트 임베딩에서 hubness 왜도 1.2~1.9는 정상 범위라 `warn` 임곗값을 2.0으로 잡았고, arXiv
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

  관계없는 모델로 임베딩한 질의는 `query_gap`이 8~10(`fail`, nDCG@10은 0 근처)입니다. 공간을
  공유하는 bge-base 질의를 gte-base 코퍼스에 넣으면 2.9~3.9가 나오지만 nDCG@10은 유지됩니다.

  거의 같은 복제가 많아도 `query_gap`이 오릅니다. `query_gap`이 나누는 코퍼스 거리를 복제가
  줄이기 때문이며, 계획은 이를 중복과 같은 단계에 넣어 중복 제거 뒤 다시 읽게 합니다.
- **좋은 모델들 사이에서는 기하 항목이 모델 순위를 매기지 못합니다.** 두 데이터세트 모두에서 가장
  좋은 모델(gte-base, bge, e5)이 anisotropy가 *가장 높고*([검증 결과](verification.md)의 모델 19종까지 포함하면 0.58~0.82) effective_rank는 평범합니다.

  약한 모델을 가르는 것은 `norms`와 `metric_mismatch`(정규화되지 않은 출력), 그리고 `hubness`로,
  nDCG@10과의 Spearman |rho|가 0.4~0.8입니다.
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

`vecscope calibrate`는 2단계의 감사 결과로 3~5단계를 실행합니다. 상태마다 맞추기용 질의로 한
번, 검증용 질의로 한 번 감사합니다.

```bash
vecscope audit state_a.npy --queries q_fit.npy --qrels qrels_fit.tsv --json > fit_a.json
vecscope audit state_a.npy --queries q_hold.npy --qrels qrels_hold.tsv --json > hold_a.json
# ... 모든 상태에 대해 같은 방식으로 실행한 뒤:
vecscope calibrate fit_*.json --min-quality 0.45 --holdout hold_*.json --out th.json
vecscope audit vectors.npy --thresholds th.json
```

항목마다 다음을 보여 줍니다.

- 상태들에 걸친 검색 지표(`--metric`, 기본값 nDCG@k)와의 Spearman 상관, 그 95% 부트스트랩
  구간, 그리고 부호가 그 항목이 뜻하는 방향과 맞는지
- 그 방향의 단측 순열 검정 p값. 검정한 모든 항목에 대해 Holm 방식으로 보정합니다.
- 허용 상태 중 가장 나쁜 값과 나머지 상태 중 가장 좋은 값
- 항목이 품질을 따라가고 두 묶음을 가를 때만 제안하는 (warn, fail) 쌍. 두 값 사이 간격의 3분의
  1과 3분의 2 지점에 둡니다.

자신의 데이터에서 품질과 함께 움직이지 않거나 반대로 움직이는 항목에는 쌍을 제안하지 않습니다.
기본값은 들여다볼 계기로만 쓰십시오.

항목 약 15개를 한꺼번에 검정하면, 상태가 적을 때 그중 하나는 우연히 품질을 따라가는 것처럼
보입니다. 그래서 보정한 p값이 0.05 미만인 쌍만 `--out`에 씁니다.

- 나머지 쌍은 `exploratory`(탐색용)로 보고하고 게이트 파일에서는 뺍니다.
  `--include-low-confidence`를 주면 함께 씁니다.
- 상태가 5개면 어떤 항목도 이 기준을 넘을 수 없습니다(보정 전 단측 p값의 최솟값이 1/120).
  상태는 10개 이상을 권장하며, 그보다 적으면 보고서에 그렇게 적힙니다.

제안을 맞추는 데 쓰지 않은 상태로 두 가지를 확인합니다.

- `leave_one_out`은 상태마다 그 하나를 빼고 맞춘 임곗값으로 뺀 상태를 판정합니다.
- 홀드아웃 줄은 `--holdout` 상태 중 게이트가 잡은 불합격 상태와 합격 상태에 낸 오경보 수를
  셉니다. 다른 데이터세트의 상태를 홀드아웃으로 넘기면 임곗값이 그 데이터에도 통하는지 볼 수
  있습니다.

상태는 3개 이상이어야 하고 `--min-quality` 양쪽에 모두 있어야 합니다. 서로 다른 qrels나 k로
채점한 상태들은 품질 값을 비교할 수 없으므로 거부합니다(`--allow-mixed-qrels`로 허용).

`suggested_action`에도 같은 절차가 적용됩니다. 권장 조치는 원인에 대한 가설이고, 중심화,
표준화, 차원 절단은 검색 순위를 바꿉니다.

- 원본 벡터와 검색 결과를 기준선으로 보관합니다.
- 변환은 복사본에 적용합니다.
- 양쪽에서 품질과 지연 시간, 메모리를 함께 잽니다.
- 항목이 `ok`로 바뀌었을 때가 아니라 정답 질의의 품질이 좋아졌을 때만 반영하십시오.
