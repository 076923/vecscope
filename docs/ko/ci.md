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

## 어느 규칙이 실패시켰는지

실행을 실패로 만드는 규칙은 두 가지이고, 출력에서 둘을 구분해 보여 줍니다.

- 하나는 절대 규칙으로, 임곗값 기준 판정입니다(기본값 `--fail-on fail`, 더 엄격하게는
  `--fail-on warn`).
- 다른 하나는 기준선 규칙으로, 판정이 나빠진 `regressions`, 값이 허용량 이상 나빠진
  `value_regressions`, 라벨 검색 지표가 허용량 이상 떨어진 `retrieval_regressions`, 검색 품질
  허용량을 확인하지 못한 `retrieval_unchecked`입니다.

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
