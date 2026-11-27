# 임베딩 모델 교체

[English](../case_model_change.md) · **한국어**

multilingual-e5-base로 상품 검색을 운영하는 팀이 다른 모델이 더 나은지 판단해야 한다고 합시다. 이
페이지는 전체 라벨이 있는 공개 데이터에서 그 결정을 재현하고, 라벨 질의가 있을 때와 없을 때
vecscope가 후보마다 무엇이라고 했는지 기록합니다. 행은 사전 등록한 held-out 세트([검증](verification.md#사전-등록한-held-out-세트))에서 나왔고
`examples/validation/case_model_change.py`가 씁니다(`case_model_change.json`). 순감소 임곗값은 이 행을
포함한 데이터로 골랐으므로, 이 페이지는 규칙이 작동하는 모습을 보여 줄 뿐 held-out 시험이 아닙니다.
그 시험은 [검증](verification.md#합의한-상위-결과)에 있습니다.

## 데이터

[XPQA](https://huggingface.co/datasets/mteb/XPQARetrieval)는 Amazon 상품에 대한 실제 질문과 그
답변으로, 상품 카탈로그처럼 짧은 텍스트입니다(중앙값 48~67자). 아래 세 언어를 씁니다.

| 세트 | 답변(코퍼스) | 라벨 있는 질의 | 기준선 nDCG@10 |
|---|---|---|---|
| 이탈리아어(ita-ita) | 1,272 | 663 | 0.703 |
| 폴란드어(pol-pol) | 1,579 | 785 | 0.480 |
| 포르투갈어(por-por) | 1,622 | 800 | 0.448 |

기준선은 multilingual-e5-base입니다. 후보는 검증의 나머지 공개 모델 12종이고, 자주 하는 실수이므로
영어 전용 모델도 넣었습니다. bge-m3는 기준 모델이라 후보가 아닙니다. 정답은 라벨 있는 모든 질의에서
잰 nDCG@10 변화입니다.

## 실행 방법

현재 모델에서 한 번, 기준 모델과 결과 표본을 함께:

```bash
vecscope audit e5_docs.npy --queries e5_queries.npy --qrels qrels.json \
  --reference bgem3_docs.npy --reference-queries bgem3_queries.npy \
  --result-sample 200 --json > baseline.json
```

그다음 후보마다 같은 행과 질의로:

```bash
vecscope audit cand_docs.npy --queries cand_queries.npy --qrels qrels.json \
  --result-sample 200 --baseline baseline.json --fail-on none \
  --reference-tolerance 0.03 \
  --retrieval-tolerance ndcg_at_k=0.01 --retrieval-confidence 0.95
```

- `--reference-tolerance 0.03`은 라벨 없는 판정입니다. 예전 모델과 기준 모델이 합의한 상위 결과를
  후보가 얻은 것을 빼고도 잃었는가?(순감소가 0.03 초과면 `reference_regressions`, 0.015 초과면
  `reference_review`, 둘 다 표준오차 2배 초과)
- `--retrieval-confidence 0.95`는 라벨 판정입니다. 라벨 질의가 적은 팀이 볼 결과를 보이려고 라벨 질의
  100개를 시드로 뽑은 부분집합 200개에서 실행했고, 표는 통과, 판단 불가, 실패의 비율입니다.

## 결과

이탈리아어:

| 후보 | 언어 | nDCG@10 변화 | 잃은 합의 상위 결과 | 순감소 | 라벨 없는 판정 | 라벨 게이트 통과 / 판단 불가 / 실패 |
|---|---|---|---|---|---|---|
| KURE-v1 | 다국어 | +0.063 | 0.0% | -0.095 | — | 96% / 4% / 0% |
| multilingual-e5-large | 다국어 | +0.040 | 2.2% | -0.020 | — | 78% / 22% / 0% |
| Qwen3-Embedding-0.6B | 다국어 | +0.012 | 5.0% | -0.010 | — | 28% / 69% / 3% |
| gte-multilingual-base | 다국어 | -0.011 | 6.6% | +0.025 | — | 4% / 88% / 8% |
| bge-base-en-v1.5 | 영어 | -0.067 | 15.5% | +0.105 | `reference_regressions` | 0% / 33% / 67% |
| gte-base | 영어 | -0.073 | 15.5% | +0.120 | `reference_regressions` | 0% / 26% / 74% |
| bge-small-en-v1.5 | 영어 | -0.101 | 20.4% | +0.145 | `reference_regressions` | 0% / 5% / 95% |
| e5-base-v2 | 영어 | -0.125 | 18.8% | +0.150 | `reference_regressions` | 0% / 1% / 99% |
| all-MiniLM-L6-v2 | 영어 | -0.157 | 30.4% | +0.250 | `reference_regressions` | 0% / 0% / 100% |
| all-mpnet-base-v2 | 영어 | -0.162 | 26.5% | +0.220 | `reference_regressions` | 0% / 0% / 100% |
| paraphrase-multilingual-MiniLM-L12-v2 | 다국어 | -0.187 | 22.7% | +0.180 | `reference_regressions` | 0% / 0% / 100% |
| glove.6B.300d | 영어 | -0.470 | 70.7% | +0.630 | `reference_regressions` | 0% / 0% / 100% |

폴란드어:

| 후보 | 언어 | nDCG@10 변화 | 잃은 합의 상위 결과 | 순감소 | 라벨 없는 판정 | 라벨 게이트 통과 / 판단 불가 / 실패 |
|---|---|---|---|---|---|---|
| KURE-v1 | 다국어 | +0.038 | 0.0% | -0.195 | — | 58% / 42% / 0% |
| multilingual-e5-large | 다국어 | +0.030 | 6.4% | -0.010 | — | 56% / 44% / 0% |
| Qwen3-Embedding-0.6B | 다국어 | -0.043 | 17.8% | +0.045 | — | 0% / 48% / 52% |
| gte-multilingual-base | 다국어 | -0.047 | 15.3% | +0.025 | — | 0% / 50% / 50% |
| e5-base-v2 | 영어 | -0.132 | 31.8% | +0.185 | `reference_regressions` | 0% / 0% / 100% |
| gte-base | 영어 | -0.141 | 34.4% | +0.200 | `reference_regressions` | 0% / 0% / 100% |
| bge-base-en-v1.5 | 영어 | -0.170 | 38.2% | +0.220 | `reference_regressions` | 0% / 0% / 100% |
| bge-small-en-v1.5 | 영어 | -0.177 | 38.2% | +0.260 | `reference_regressions` | 0% / 0% / 100% |
| paraphrase-multilingual-MiniLM-L12-v2 | 다국어 | -0.196 | 34.4% | +0.215 | `reference_regressions` | 0% / 0% / 100% |
| all-mpnet-base-v2 | 영어 | -0.244 | 46.5% | +0.325 | `reference_regressions` | 0% / 0% / 100% |
| all-MiniLM-L6-v2 | 영어 | -0.257 | 46.5% | +0.330 | `reference_regressions` | 0% / 0% / 100% |
| glove.6B.300d | 영어 | -0.387 | 79.0% | +0.595 | `reference_regressions` | 0% / 0% / 100% |

포르투갈어:

| 후보 | 언어 | nDCG@10 변화 | 잃은 합의 상위 결과 | 순감소 | 라벨 없는 판정 | 라벨 게이트 통과 / 판단 불가 / 실패 |
|---|---|---|---|---|---|---|
| KURE-v1 | 다국어 | +0.048 | 0.6% | -0.200 | — | 78% / 22% / 0% |
| Qwen3-Embedding-0.6B | 다국어 | +0.034 | 6.4% | -0.075 | — | 45% / 55% / 0% |
| gte-multilingual-base | 다국어 | +0.027 | 5.7% | -0.040 | — | 40% / 60% / 0% |
| multilingual-e5-large | 다국어 | -0.037 | 12.7% | +0.070 | `reference_regressions` | 1% / 50% / 49% |
| gte-base | 영어 | -0.062 | 22.9% | +0.100 | `reference_regressions` | 0% / 42% / 58% |
| e5-base-v2 | 영어 | -0.070 | 17.8% | +0.095 | `reference_regressions` | 0% / 18% / 82% |
| bge-small-en-v1.5 | 영어 | -0.074 | 26.1% | +0.120 | `reference_regressions` | 0% / 28% / 72% |
| bge-base-en-v1.5 | 영어 | -0.076 | 26.1% | +0.135 | `reference_regressions` | 0% / 23% / 77% |
| all-mpnet-base-v2 | 영어 | -0.113 | 30.6% | +0.150 | `reference_regressions` | 0% / 2% / 98% |
| paraphrase-multilingual-MiniLM-L12-v2 | 다국어 | -0.125 | 24.2% | +0.120 | `reference_regressions` | 0% / 2% / 98% |
| all-MiniLM-L6-v2 | 영어 | -0.128 | 29.9% | +0.170 | `reference_regressions` | 0% / 2% / 98% |
| glove.6B.300d | 영어 | -0.322 | 74.5% | +0.575 | `reference_regressions` | 0% / 0% / 100% |

## 보여 주는 것

- **라벨 없는 신호는 후보의 순위를 매깁니다.** 순감소는 손실과 함께 커졌습니다. 영어 전용 모델은
  0.095~0.63, paraphrase-multilingual-MiniLM-L12-v2는 0.12~0.22였고, 개선한 후보는 모두 0 아래였습니다.
  nDCG@10을 0.06 이상 잃은 후보는 모두 표시했습니다.
- **여기서 개선은 하나도 표시하지 않았습니다.** nDCG@10이 좋아진 후보는 모두 순감소가 0 아래였고,
  합의 상위 결과를 5.7~6.4% 잃은 후보(폴란드어 multilingual-e5-large, 포르투갈어
  Qwen3-Embedding-0.6B와 gte-multilingual-base)도 마찬가지였습니다. 새 모델은 예전 모델의 합의 결과
  일부를 잃으면서 다른 곳에서 기준 모델의 첫 결과를 더 찾습니다.
- **다국어 모델 사이의 작은 손실은 놓칩니다.** 0.037~0.047을 잃은 다국어 후보 3개 중 1개(포르투갈어
  multilingual-e5-large, 순감소 0.070)만 표시했습니다. 폴란드어의 Qwen3-Embedding-0.6B와
  gte-multilingual-base(순감소 0.045, 0.025, 표준오차 2배 이내)는 표시도 검토 대상도 아니었습니다.
  held-out 데이터도 같아서, 그런 손실 13건 중 1건만 표시했습니다. 라벨 없는 결과가 깨끗하다고
  nDCG@10이 유지됐다는 증거는 아닙니다.
- **라벨 게이트는 방향을 거의 틀리지 않았습니다.** 라벨 질의 100개로, 0.03 이상 잃은 후보는 부분집합의
  1% 이하에서만 통과했고, 개선은 3% 이하에서만 실패했습니다(이탈리아어 Qwen3-Embedding-0.6B,
  +0.012). 작은 차이는 판단 불가가 많았습니다. 폴란드어 e5-large(+0.030)는 부분집합의 56%에서
  통과했고, 순감소가 놓친 폴란드어 손실 두 건은 절반쯤 실패하고 나머지는 판단 불가였으며 한 번도
  통과하지 않았습니다. 판단 불가인 실행은 결론에 필요한 라벨 질의 수를 알려 줍니다
  (`retrieval_confidence_checks`).
- **KURE-v1은 기준 모델의 미세조정 모델이라 기준 모델에게는 완벽해 보입니다.** 세트의 모든
  데이터세트에서 합의 상위 결과를 0.0~0.6%만 잃었습니다. 실제로도 가장 좋은 후보였지만 합의 상위
  결과로는 그것을 알 수 없습니다. 기준 모델과 같은 계열의 모델은 이 신호로 판단할 수 없습니다.
- **절대 판정은 교체할 이유가 아닙니다.** multilingual-e5-base는 세 세트 모두에서 `fail`이었고
  (`ood_queries`, `collapse`, `query_gap`) 여기서 여전히 나은 축에 드는 모델입니다. 기준 모델과 낮춤
  규칙도 이 항목들을 풀지 않았습니다. 변경은 실행 하나의 판정이 아니라 기준선 비교로 결정합니다.

## 이렇게 하십시오

1. 같은 행과 질의를 후보마다, 그리고 다른 계열의 기준 모델로 임베딩합니다.
2. 현재 모델을 기준 모델과 `--result-sample`로 한 번 감사하고 JSON을 보관합니다.
3. `reference_regressions`에 든 후보를 뺍니다. 이 데이터에서는 모두 nDCG@10을 0.037 이상 잃었습니다.
   나머지는 안전한 것이 아니라 후보로 남긴 것입니다. 표시되지 않은 폴란드어 후보 두 개는 0.043과
   0.047을 잃었습니다.
4. 나머지는 라벨 질의와 `--retrieval-confidence 0.95`로 결정합니다. 판단 불가가 나오면 교체 전에 게이트가
   요구하는 수만큼 질의에 라벨을 붙입니다.

## 보여 주지 못하는 것

- XPQA는 상품 검색의 대역입니다. 공개 출처 하나, 세 언어, 짧은 답변입니다. 운영 질의 로그와 카탈로그는
  측정하지 않았습니다.
- 기준선은 multilingual-e5-base 한 모델입니다.
- 라벨 게이트는 라벨 질의가 실제 트래픽의 무작위 표본이라고 가정합니다.
