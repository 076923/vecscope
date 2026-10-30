[English](../verification.md) · **한국어**

# 검증 결과

vecscope 0.5.0과 당시 미공개 변경분(sweep의 지연 시간·인덱스 크기, 이웃 목록을 받는 compare,
CI 게이트)을 라이브러리 코드는 손대지 않은 채 검증했습니다. 검증 대상은 기존 테스트 전부,
100k × 768 규모에서 float64 기준 정확성, 별도로 작성한 recall 채점기, README의 표본 감사 설명,
성능 표, "실제 임베딩" 표, 그리고 라벨이 있는 데이터셋 3개에서 공개 모델 19종의 nDCG@10과
진단값의 관계입니다. 무엇을 쟀고 그 결과로 vecscope의 무엇이 바뀌었는지 정리합니다. 현재
릴리스의 임곗값과 문서는 이 결과를 반영한 것입니다.

- 실행 환경: AMD EPYC 7R13 8 vCPU, RAM 26 GB, NVIDIA L4. Python 3.12.3, numpy 2.5.4,
  faiss-gpu-cu12 1.14.1, sentence-transformers 3.4.1
- 데이터: 합성 분포 6종(`examples/bench_common.py`의 clustered, isotropic, anisotropic,
  duplicates, lowrank, zipf), README 표에 쓰인 공개 임베딩 8종, 모델 19종으로 인코딩한 BEIR
  SciFact·NFCorpus·ko-strategyqa

## 결과 요약

| 영역 | 확인한 사실 | 반영한 변경 |
|---|---|---|
| 정확한 k-NN (`exact_knn`, `query_knn`, faiss flat CPU·GPU) | 100k × 768 분포 6종, 모델 19종 × 데이터셋 3개, 공개 임베딩 8종 모두에서 float64와 비교해 순위 오류가 없었습니다. float32 거리 오차는 최대 4.1e-7입니다. | 없음 |
| ANN recall 채점 | 동점 허용 오차 `1e-6 + 1e-4 × 거리`가 float32 반올림 오차의 약 70배였습니다. 그래서 10번째 이웃보다 최대 7e-5 먼 11번째 이웃까지 적중으로 세어, 엄격한 채점기보다 recall이 최대 +0.0016 높게 나왔습니다. | 허용 오차를 `1e-6 + 1e-5 × 거리`로 줄였습니다. `tests/test_scale_exactness.py`가 엄격한 채점기와 대조합니다. |
| 표본 감사의 값 | 문서 설명과 일치했습니다. 분포형 항목은 전체 값과 2.3% 이내로 같았고, hubness와 antihubs는 모든 경우 전체 값보다 낮았으며, collapse 비율은 표본 비율 × 전체 비율의 0.96~1.03배였습니다. | 없음 |
| 표본 감사의 판정 | 전체 코퍼스에서 `fail`이나 `warn`인 항목이 표본에서는 `ok`나 `warn`으로 내려갔습니다. zipf 분포의 connectivity는 0.372 fail이 25% 표본에서 0.0008 ok가 됐고, clustered의 hubness는 2.60 warn이 1.15 ok, duplicates의 collapse는 0.200 fail이 0.051 warn이 됐습니다. outliers는 10% 표본에서 3.9배 많게 셌습니다. | hubness·antihubs·connectivity의 `fail`이 아닌 판정과 outliers의 `ok`가 아닌 판정을 `inconclusive`로 표시합니다. collapse 판정은 추정한 전체 비율로 내립니다. 게이트 출력에 inconclusive 항목 이름을 적습니다. |
| `sample_repeats` | 매우 안정적인 항목에서도 `value_range` 안에 전체 코퍼스 값이 없는 경우가 잦았습니다. | 표본 사이의 흩어짐일 뿐 전체 값의 구간 추정이 아니라고 문서에 적었습니다. |
| 임곗값 | 공개 nDCG@10을 0.007 이내로 재현한 정상 파이프라인에서도 `ood_queries`는 NFCorpus 모델 19개 중 18개가 fail이었고 `anisotropy`는 최상위 모델에 warn이 붙었습니다. 한국어 데이터에서는 `query_hubness`가 nDCG 0.80인 다국어 모델과 0.04인 영어 전용 모델을 구분하지 못하고 둘 다 fail로 판정했습니다. | `anisotropy` (0.5, 0.85) → (0.85, 0.95), `ood_queries` (0.05, 0.20) → (0.5, 0.8), `query_hubness` (2, 4) → (4, 10) |
| README "실제 임베딩" 표 | 각 데이터의 앞 4만 행에서 판정 88개가 모두 재현됐습니다. 표기값 하나가 반올림 경계에서 달랐습니다. | Cohere ko의 `distance_concentration` 1.73 → 1.72 |
| 성능 표 | `benchmark_scale.py` 1회 측정이 별도 프로세스 3회 중앙값과 2% 이내로 같았고, tracemalloc 최대치는 OS 기준 최대 메모리와 0.04 GiB 이내였습니다. | 384·768·1,536차원, 20만 행까지 실측한 시간과 메모리로 표를 바꿨습니다. |
| sweep | 추천 설정이 seed 3개에서 같았습니다. clustered 분포에서 IVF nprobe=1의 recall이 100k에서 0.696, 500k에서 0.955로 오히려 올랐는데, 군집 수 n/200이 nlist = 4√n과 겹치기 때문입니다. | sweep caveat에 IVF recall은 규모에 따라 어느 쪽으로든 움직일 수 있다고 적었습니다. `benchmark_sweep.py`는 여러 분포를 돌리고 `--centers` 옵션을 받습니다. |

## 진단값과 nDCG@10의 관계 (모델 19종)

데이터셋마다 모델 19종에 대해 각 항목의 값과 nDCG@10 사이의 Spearman 상관을 구했습니다.
음수가 일관되게 나오면 검색이 나빠질수록 값이 커진다는 뜻입니다.

| 항목 | SciFact | NFCorpus | ko-strategyqa |
|---|---|---|---|
| query_hubness | −0.57 | −0.46 | −0.90 |
| hubness | −0.52 | −0.48 | −0.76 |
| antihubs | −0.27 | −0.12 | −0.72 |
| rogue_dimensions | −0.38 | −0.38 | −0.05 |
| ood_queries | −0.34 | −0.33 | +0.25 |
| anisotropy | +0.29 | +0.21 | −0.11 |
| distance_concentration | +0.19 | +0.34 | −0.82 |

여기서 확인되는 점은 다음과 같습니다.

- `hubness`와 `query_hubness`는 모델 사이의 검색 품질 차이를 반영합니다. 한국어 데이터에서
  좋은 모델(값 2.3~9.1)은 통과하고 영어 전용 모델(11~30)은 fail이 되도록 `query_hubness`
  임곗값을 조정했습니다.
- `ood_queries`는 질의와 코퍼스의 분포 차이를 재는 항목이고, 짧은 질의와 긴 문서 사이에는
  그 차이가 늘 있습니다. 좋은 모델은 0.0~0.65, 모델이나 언어가 잘못된 경우는 0.7~1.0이
  나오거나, 양쪽이 똑같이 잘못 인코딩되면 0이 나옵니다. 새 임곗값은 큰 차이만 표시하며,
  이 항목은 품질 점수가 아닙니다.
- `anisotropy`는 이 범위에서 품질과 관계가 없습니다. 가장 좋은 모델들이 0.58~0.82에 있으므로
  `warn`은 0.85부터 시작합니다.
- 누가 봐도 잘못된 조합(BEIR에 GloVe, 영어 데이터에 한국어 전용 모델, 한국어 데이터에 영어
  전용 모델)은 여러 항목이 한꺼번에 경고를 냅니다. 그런 조합은 경고 개수로 가려낼 수 있지만
  `fail` 하나만으로는 판단할 수 없습니다.

MCP 서버와 운영 질의 로그는 검증 범위에 들어 있지 않습니다. 실제 모델 측정은 두 언어의
데이터셋 3개로 했고, 표본 감사와 정확성은 합성 데이터에서, 검색 품질과의 관계는 라벨이 있는
공개 데이터에서 확인했습니다.

## 재현 방법

- `pytest -q`는 검증 과정에서 추가된 테스트(`tests/test_new_features.py`,
  `tests/test_scale_exactness.py`)를 실행합니다. `pytest -m slow`를 주면 100k × 384 정확성
  검사도 실행합니다.
- `python examples/benchmark_scale.py --rows 10000 50000 100000 --dim 768 --json scale.json`과
  `python examples/benchmark_sweep.py --rows 100000 --dim 768 --corpus clustered zipf --json
  sweep.json`으로 표를 다시 잴 수 있습니다. 실행 환경 정보가 JSON에 함께 기록됩니다.
- `python examples/retrieval_quality.py`는 공개 데이터셋으로 검색 품질 연구를 실행합니다.
