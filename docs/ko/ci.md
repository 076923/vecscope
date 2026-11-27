[English](../ci.md) · **한국어**

# CI에서 사용

JSON 출력은 실행 간에 안정적이고 `schema_version`을 담고 있어, 기준선으로 커밋해 두고
인덱스를 다시 만들 때마다 비교할 수 있습니다.

`--baseline`은 항목별 변화를 출력하고, 어느 항목의 판정이라도 나빠지면(또는 `--fail-on` 조건에 걸리면) 종료 코드 1을 돌려줍니다.
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
군집](usage.md#질의와-군집) 참고). 기준선 자체에는 비교 대상이 없었으므로 비교 결과에서는
`only_in_new`에 나타납니다.

## 변화량 허용치

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
다르면 false가 됩니다.

`--require-comparable`을 주면 이 경우 주의 문구 대신 오류(종료 코드 2)로 처리합니다. 설정이
다른 두 실행의 차이는 코퍼스가 아니라 설정을 재는 것이므로 게이트에서는 이 정책이 맞습니다.
라이브러리에서는 `diff_evidence(new, old, tolerances={...})`입니다.

## 검색 품질 게이트

진단 항목은 공간의 기하를 설명할 뿐, 검색이 맞는 문서를 돌려주는지는 말하지 않습니다. 모든
진단값이 그대로인데 검색이 망가지는 변경이 있습니다. 행과 문서의 대응을 잃어버린 재색인, 내
질의에서 순위를 더 못 매기는 모델 업데이트, 관련 문서를 빼 버리는 필터가 그렇습니다. 라벨이
달린 질의가 있으면 그것으로 직접 게이트를 거십시오.

```bash
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 --json > baseline.json
# 변경 후
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 \
  --baseline baseline.json --retrieval-tolerance ndcg_at_k=0.01 --retrieval-tolerance recall_at_k=0.02
```

- `--retrieval-tolerance METRIC=DELTA`(여러 번 지정 가능)는 기준선 대비 지표가 떨어져도 되는
  폭입니다. 지표는 `ndcg_at_k`, `recall_at_k`, `mrr_at_k`, `hit_rate_at_k`,
  `duplicate_result_share`(올라가는 쪽이 나쁨), `overlap_with_exact`입니다. 접두어 없이 쓰면
  정확 검색에, 두 실행 모두 `--ranked`를 넘겼다면 그 순위에도 적용되고, `exact.`나 `ranked.`를
  붙이면 그쪽에만 적용됩니다.
- 허용량보다 더 떨어진 지표는 `retrieval_regressions`에 오르고(`exact.ndcg_at_k` 형식, 텍스트
  출력은 `WORSE by 0.0213 > 0.01`), 진단 결과와 상관없이 종료 코드 1을 돌려줍니다.
- 확인할 수 없는 허용량도 실행을 실패시키며 `retrieval_unchecked`에 오릅니다. 기준선에 검색
  품질 블록이 없거나, qrels나 k가 다르거나(`retrieval_comparable`이 false), `ranked.` 허용량인데
  어느 한쪽에 순위가 없는 경우입니다. 요청한 검사가 돌지 않았다면 통과가 아닙니다.
  `--require-comparable`을 주면 대신 오류(종료 코드 2)가 됩니다.
- 이 옵션은 `--baseline`과 해당 실행의 `--qrels`가 있어야 합니다.

qrels의 행 번호는 두 실행에서 같은 문서를 가리켜야 합니다. 행을 지우거나 순서를 바꾸는
변경(중복 제거, 다른 export로 재구축)은 새 행에 맞게 qrels를 다시 써야 합니다. 다른 문서를
가리키게 된 qrels는 이 게이트가 잡으려는 바로 그 품질 붕괴로 나타납니다. 라이브러리에서는
`diff_evidence(new, old, retrieval_tolerances={"ndcg_at_k": 0.01})`입니다.

### 라벨 질의는 몇 개면 되는지

게이트는 라벨 질의 전체의 평균을 읽으므로, 질의가 적으면 흔들립니다. 공개 파이프라인 99개에
결함 594건을 넣고(`examples/validation/label_budget.py`), 라벨 질의 일부만 무작위로 뽑아
nDCG 허용량 0.01을 걸었을 때 전체 질의로 본 답과 비교한 결과입니다.

| 라벨 질의 수 | 해로운 결함 검출 | 0.02~0.05 하락 검출 | 0.05 초과 하락 검출 | 해롭지 않은 변경 실패 처리 |
|---|---|---|---|---|
| 25 | 83% | 80% | 97% | 33% |
| 50 | 89% | 89% | 100% | 29% |
| 100 | 92% | 95% | 100% | 24% |
| 200 | 95% | 99% | 100% | 17% |
| 400 | 97% | 100% | 100% | 10% |

400개 행은 라벨 질의가 400개 이상인 파이프라인만 셉니다. 임베딩 전에 사전 등록한 두 번째 공개
데이터 세트에서는 라벨 질의 100개로 해로운 결함 95건 중 87%(0.02 넘는 하락은 97%)를 잡았고, 해롭지
않은 변경 25건 중 26%를 실패 처리했습니다([검증](verification.md#사전-등록한-두-번째-held-out-세트)).
라벨 질의 100개 정도면 nDCG를 2포인트 넘게 깎는 결함은 거의 다 잡습니다. 0.01~0.02 사이의
하락은 수백 개가 필요합니다. 오경보는 전체 질의로 본 하락이 0.01을 살짝 밑돈 변경입니다. 작은
질의 집합은 이를 실제 0.01 하락과 구분하지 못하므로, 작은 집합에서 게이트가 실패하면 "망가졌다"가
아니라 "더 많은 질의로 다시 확인하라"로 읽으십시오.

### 라벨 질의의 잡음을 감안해 하락 판단하기

`--retrieval-confidence LEVEL`을 주면 위의 조언을 게이트가 직접 판정합니다. 두 실행 모두 판정
대상 질의마다 값을 저장하고(`retrieval.per_query`), diff는 질의 행끼리 짝지어 비교합니다.

```bash
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 \
  --baseline baseline.json --retrieval-tolerance ndcg_at_k=0.01 --retrieval-confidence 0.95
```

- **fail**(`retrieval_regressions`): 평균 하락이 허용량을 넘고, 그 수준의 단측 구간으로 봐도 0보다
  커서 실제 하락인 경우입니다.
- **pass**: 구간의 위쪽 끝이 허용량 안에 있는 경우입니다.
- **inconclusive**(`retrieval_inconclusive`): 나머지 경우입니다. 실행은 1로 끝나고 텍스트에
  `INCONCLUSIVE`가 표시됩니다. `retrieval_confidence_checks`에 구간과, 라벨 질의가 몇 개쯤 있으면
  판가름 날지가 들어갑니다. `change_decision`은 `reject`가 아니라 `needs_review`입니다.

첫 held-out 세트의 결함 주입에서 라벨 질의 100개로 이 규칙을 적용했습니다. 해로운 변경이 통과한
경우는 1%였습니다(평균만 볼 때는 8%). 해롭지 않은 변경을 실패시킨 경우는 14%였습니다(평균만 볼
때는 24%). 나머지는 판단 불가로, 해로운 변경의 17%와 해롭지 않은 변경의 62%가 여기 해당했습니다.
그 세트의 해롭지 않은 결함은 대부분 0.01을 살짝 밑도는 손해라 라벨 100개로는 판가름이 나지
않습니다. 이 규칙은 그 세트로 설계했고, 임베딩 전에 동결한 세 번째 held-out 세트에서
확인됐습니다. 해로운 변경의 0.4%가 통과했고(평균만 볼 때 3.9%), 해롭지 않은 변경의 10.1%가
실패했으며(23.7%), 해로운 변경의 12%와 해롭지 않은 변경의 68%가 판단 불가였습니다
([검증](verification.md#사전-등록한-세-번째-held-out-세트)). 이 버전 이전에 만든 기준선에는 질의별
값이 없어서 평균 규칙으로
판정합니다.

## 결과가 바뀌면 안 되는 변경

라벨이 없으면 공간의 어떤 통계도 해로운 변경과 해롭지 않은 변경을 구분하지 못합니다. 같은 결함
주입에서, 변경이 바꾼 top-k 결과의 비율은 해로운 변경과 해롭지 않은 변경을 우연 수준으로밖에
가르지 못했습니다(AUC 0.58). 대신 결과가 움직였다는 사실은 보여 주며, 어떤 진단 항목도 움직이지
않을 때도 그렇습니다. id 매핑을 잃으면 진단값은 하나도 바뀌지 않지만, 옮겨진 행을 찾던 질의의
결과는 모두 바뀝니다. 그래서 같은 결과를 돌려줘야 하는 파이프라인(재색인, 리팩터링, 라이브러리나
하드웨어 업그레이드)은 결과 자체로 게이트를 거십시오.

```bash
vecscope audit embeddings.npy --queries queries.npy --result-sample 500 --json > baseline.json
# 변경 후
vecscope audit embeddings.npy --queries queries.npy --result-sample 500 \
  --baseline baseline.json --fail-on none --result-tolerance 0
```

- `--result-sample M`은 질의 M개와 코퍼스 행 M개의 정확 top-k를 JSON에 저장합니다. 행은 행 번호의
  해시로 고르므로, 뒤에 추가된 행은 표본의 일부만 차지하고 커진 코퍼스도 기준선과 표본 대부분을
  공유합니다.
- diff의 `result_agreement`에는 쪽(`queries`, `rows`)마다 `changed_share`(다른 행으로 바뀐 top-k
  칸의 비율)와 `unchanged_items`(top-k가 그대로인 항목의 비율)가 들어갑니다.
- `--result-tolerance SHARE`는 어느 쪽이든 그보다 많이 바뀌면(`result_regressions`), 그리고 어느
  한 실행에 표본이 없으면(`result_unchecked`) 실행을 실패시킵니다. `--baseline`과
  `--result-sample`이 있어야 합니다.
- 결과를 바꾸려는 변경(새 모델, 변환)은 이 수치로 판단하지 않습니다. 라벨이 없으면
  `change_decision`은 `needs_review`이고 바뀐 비율을 근거로 적습니다. 더 나아졌는지는 위의 라벨
  게이트가 정합니다.

### 라벨 없이: 기준 모델과 합의한 상위 결과를 잃었는가

기준 모델이 라벨을 대신할 수 있습니다. 같은 행과 질의를 독립적으로 좋은 두 번째 공개 모델로
임베딩하고, 기준선을 만들 때 넘깁니다.

```bash
vecscope audit embeddings.npy --queries queries.npy --result-sample 200 \
  --reference ref.npy --reference-queries ref_q.npy --json > baseline.json
# 변경 후: 기준 모델은 기준선에만 있으면 됩니다
vecscope audit embeddings.npy --queries queries.npy --result-sample 200 \
  --baseline baseline.json --fail-on none --reference-tolerance 0.05
```

- 기준선은 표본 질의와 행마다 기준 모델의 1위 결과를 저장합니다(`result_sample.reference_first`).
  그 결과가 파이프라인의 top-k에도 있으면 두 모델이 합의한 것이고, 이를 합의된 상위 결과라고
  부릅니다.
- 비교 결과에는 쪽마다 `result_agreement.<side>.reference_top_hit`가 들어갑니다. 합의된 상위
  결과가 있던 항목 수, 그중 새 실행이 더 이상 돌려주지 않는 수, 잃은 비율과 그 z입니다. 두 실행
  모두 기준 모델이 있으면 `result_agreement.<side>.reference`에 전후 겹침도 참고로 들어갑니다.
- `--reference-tolerance SHARE`는 잃은 비율이 허용량을 넘고 표준오차 2배도 넘으면 그 쪽을
  `reference_regressions`에 넣습니다. 질의가 있으면 질의 쪽으로 판정합니다. 기준 모델 없이 만든
  기준선은 `reference_unchecked`입니다. 둘 다 1로 끝납니다.
- 허용량보다 작아도 잃은 비율이 0.03과 표준오차 2배를 넘으면 `reference_review`에 넣습니다
  (텍스트: `REVIEW: a small loss is possible, check with labels`). 게이트를 실패시키지는 않고,
  `change_decision`이 작은 손실이 있을 수 있다고 적습니다. 네 held-out 세트의 결함에서, 0.05가
  통과시킨 nDCG@10 0.02 넘는 손실 117건 중 80건과 해롭지 않은 변경 244건 중 29건을 여기에
  넣었습니다. 임베딩 전에 동결하고 다른 기준 모델로 돌린 다섯 번째 세트([검증](verification.md#사전-등록한-다섯-번째-held-out-세트))에서는
  그런 손실 23건 중 12건과 해롭지 않은 변경 47건 중 3건을 넣었습니다. 여섯 번째 세트에서는 28건 중
  13건과 61건 중 4건으로, 기대한 절반에 못 미쳤습니다.
- 허용량이 없어도 `change_decision`은 합의된 상위 결과를 잃었는지(0.05 기준), 조금 잃었는지(0.03
  초과), 지켰는지 적습니다.

목록 전체가 아니라 상위 결과를 보는 이유는 이렇습니다. 차원 절단이나 양자화 같은 해롭지 않은
변경은 두 모델이 애초에 합의하지 않던 아래 순위를 뒤섞고, 해로운 변경은 두 모델이 모두 1위로
꼽은 결과를 잃습니다. 이전 버전은 목록 전체 겹침의 하락으로 판정했는데, 사전 등록한 세 번째
held-out 세트에서 해롭지 않은 결함의 44%를 표시했습니다([검증](verification.md#사전-등록한-세-번째-held-out-세트)).
세 held-out 세트의 결함 주입(중복 제외)에서 합의된 상위 결과를 잃은 비율은 해로운 변경과
해롭지 않은 변경을 AUC 0.91로 갈랐습니다. 목록 전체 겹침의 하락은 0.86, 바뀐 결과 비율은
0.51이었습니다. 0.05로 두면 세트마다 nDCG@10을 0.02 넘게 떨어뜨린 결함의 72~83%와 해롭지 않은
결함의 5~8%를 표시했습니다. 이 세트들은 설계에 쓴 데이터입니다. 임베딩 전에 동결한 네 번째
held-out 세트([검증](verification.md#사전-등록한-네-번째-held-out-세트))에서는 AUC 0.97로 갈랐고,
0.02 넘게 손해 본 결함의 72%와 해롭지 않은 결함 48개 중 1개(2%)를 표시했습니다. 목록 전체 겹침의
하락은 48개 중 21개를 표시했습니다. 기준 모델을 bge-m3 대신 gte-multilingual-base로 바꾼 다섯 번째
세트([검증](verification.md#사전-등록한-다섯-번째-held-out-세트))에서는 AUC 0.93으로 갈랐고, 0.02 넘게
손해 본 결함의 78%와 해롭지 않은 결함 47개 중 1개를 표시했습니다. 다시 bge-m3를 쓴 여섯 번째
세트([검증](verification.md#사전-등록한-여섯-번째-held-out-세트))에서는 AUC 0.93, 74%, 61개 중
1개였습니다. 잃었다는 신호는 "해로울 가능성이 높으니
라벨로 확인하라"로 읽고, 측정된 손해로 읽지 마십시오.

- 아래 순위에 흩어진 작은 손해는 놓칩니다. nDCG@10을 0.01~0.02 떨어뜨린 차원 절단·양자화
  결함 대부분이 여기 해당합니다. 그중 일부는 `reference_review`가 검토 대상으로 올립니다.
- 잃은 행 결함(영벡터 5%)은 절반쯤 잡습니다. 이런 행은 진단 항목이 잡습니다.
- 두 모델 모두에게 틀린 행은 보지 못합니다. 중복된 행은 기준 모델에도 나타나며, 이런 행은
  `collapse`가 찾습니다.
- 기준 모델은 시험 대상 파이프라인과 독립적으로, 원본 텍스트에서 만들어야 합니다. 같은 잘못된
  내보내기에서 만든 기준 모델은 같은 결함을 공유합니다.
- 변경이 잃은 합의 상위 결과만 세고 새로 얻은 것은 세지 않으므로, 새 임베딩 모델에 대한 판정이
  아닙니다. 새 모델은 다른 곳에서 더 찾더라도 합의 결과 일부를 잃습니다. 사전 등록한 여섯 번째
  세트에서 nDCG@10을 0.02 넘게 잃는 모델 교체 80건 중 72건을 표시했지만, 오히려 좋아진 교체 27건 중
  5건도 표시했습니다([검증](verification.md#사전-등록한-여섯-번째-held-out-세트)). 합의 결과를 많이
  잃는 후보를 거르는 데 쓰고, 결정은 `--retrieval-confidence`로 하십시오(라벨 질의 100개로 해로운
  교체 0.04% 통과, 해롭지 않은 교체 0.07% 실패). 기준 모델과 같은 계열의 후보(기준 모델의 미세조정
  모델)는 원래부터 기준 모델과 합의하므로 이 방법으로 판단할 수 없습니다. 예시:
  [임베딩 모델 교체](case_model_change.md).

## 어느 규칙이 실패시켰는지

실행을 실패로 만드는 규칙은 두 가지이고, 출력에서 둘을 구분해 보여 줍니다.

- 하나는 절대 규칙으로, 임곗값 기준 판정입니다(기본값 `--fail-on fail`, 더 엄격하게는
  `--fail-on warn`).
- 다른 하나는 기준선 규칙으로, 판정이 나빠진 `regressions`, 값이 허용량 이상 나빠진
  `value_regressions`, 라벨 검색 지표가 허용량 이상 떨어진 `retrieval_regressions`, 검색 품질
  허용량을 확인하지 못한 `retrieval_unchecked`, 표본 검색 결과가 `--result-tolerance` 이상 바뀐
  `result_regressions`, 어느 실행에 `--result-sample`이 없던 `result_unchecked`,
  `--retrieval-confidence`에서 라벨 질의로 판가름 나지 않은 `retrieval_inconclusive`, 합의된 상위
  결과를 `--reference-tolerance` 넘게 잃은 `reference_regressions`, 기준선에 기준 모델 결과가
  없던 `reference_unchecked`입니다.

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

[검색 품질 연구](quality.md#진단-항목이-검색-품질을-예측합니까)가 공개 데이터세트에서 그 관계를 보여 주고,
[임곗값을 도메인에 맞추기](quality.md#임곗값을-도메인에-맞추기)는 병합을 차단하기 전에 자신의 데이터로
그 관계를 재는 방법을 설명합니다.

## GitHub Action

이 저장소는 GitHub Action으로도 쓸 수 있습니다. 워크플로가 지정한 태그의 vecscope를 설치하고,
`vecscope audit --json`을 실행해 진단 결과 표를 작업 요약(job summary)에 쓰며, CLI의 종료 코드로
작업을 실패시킵니다.

```yaml
- uses: 076923/vecscope@v0          # 또는 릴리스 고정: @v0.13.0
  with:
    path: embeddings.npy
    baseline: baseline.json         # 커밋해 둔 --json 출력, 선택
    tolerances: tolerances.json     # 선택
    queries: queries.npy            # 선택, qrels와 함께 쓰면 라벨 검색 품질
    qrels: qrels.json
    retrieval-tolerance: ndcg_at_k=0.01 recall_at_k=0.02
    sample: "0"                     # 기준선과 비교할 때는 전체 행을 감사
    require-comparable: "true"
```

| 입력 | 기본값 | CLI 옵션 |
|---|---|---|
| `path` | 필수 | 감사할 파일, 인덱스, 저장소 URI |
| `baseline`, `tolerances`, `thresholds`, `queries`, `qrels` | 비어 있음 | `--baseline`, `--tolerances`, `--thresholds`, `--queries`, `--qrels` |
| `retrieval-tolerance` | 비어 있음 | `METRIC=DELTA` 하나마다 `--retrieval-tolerance` 하나, 공백이나 쉼표로 구분 |
| `fail-on` | `fail` | `--fail-on` |
| `k`, `sample` | `10`, `100000` | `-k`, `--sample` |
| `require-comparable` | `false` | `--require-comparable` |
| `args` | 비어 있음 | 그 밖의 옵션, 공백으로 구분 |
| `output` | `vecscope-evidence.json` | 진단 JSON을 쓸 경로 |
| `extras` | 비어 있음 | 함께 설치할 extras, 예: faiss 파일은 `index`, Qdrant는 `qdrant` |
| `python-version` | `3.13` | 비우면 러너에 이미 있는 Python을 씁니다 |

출력은 `verdict`(`ok`, `warn`, `fail`, 진단 JSON이 없으면 `error`), `exit-code`,
`regressions`(쉼표로 구분한 항목 이름), `retrieval-regressions`(쉼표로 구분한 `exact.ndcg_at_k`
같은 지표, 확인하지 못한 허용량은 그 뒤에 `unchecked:<지표>`), `evidence`(JSON 경로)입니다. 두
실행 모두 라벨이 있으면 작업 요약에 검색 품질 표가 붙습니다. `evidence`는 아티팩트로
올려 보관하거나, 검토를 거쳐 다음 기준선으로 커밋하십시오.

`@v0`은 0.x 릴리스마다 새 버전으로 옮겨지고, 0.x의 마이너 릴리스는 진단 항목이나 임곗값을 바꿀 수
있습니다. 결과가 안정적이어야 하는 게이트에서는 전체 버전(`@v0.13.0`)으로 고정하십시오.
