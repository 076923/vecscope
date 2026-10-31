# 에이전트 사례: MCP로 적재 결함 두 가지를 찾고 고치기

[English](../agent_case.md) · **한국어**

Claude Code headless 모드가 vecscope MCP 서버로 코퍼스를 진단하고, 고치고, 다시 감사해 수정을
검증한 실행 기록입니다. 아래 숫자는 에이전트의 요약이 아니라 실행 중 도구가 돌려준 응답에서
가져왔습니다.

## 데이터

실제 임베딩에 결함 두 가지를 주입했고, [`examples/agent_case.py`](../../examples/agent_case.py)가
만듭니다.

- SciFact(BEIR): 초록 5,183개와 테스트 질의 300개, 관련성 판정 339개. bge-small-en-v1.5로
  임베딩했습니다(384차원, 단위 길이, 질의에는 질의용 지시문 사용).
- 결함 1, 한 배치를 두 번 적재: 문서 1000\~1599를 끝에 한 번 더 붙여 코퍼스가 5,783행입니다.
- 결함 2, 정규화를 빠뜨린 적재 작업자: 문서 2600부터는 행마다 길이가 0.6\~1.6입니다.
- `prod_topk.npy`: 코퍼스 파일 위의 내적 인덱스가 질의마다 돌려준 top-10입니다. faiss
  `IndexFlatIP`가 돌려주는 것과 같습니다.

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

## 실행 방법

```bash
python examples/agent_case.py --out run
cd run
claude -p "$(cat PROMPT.txt)" \
  --mcp-config mcp.json --strict-mcp-config --setting-sources project \
  --output-format stream-json --verbose \
  --allowedTools "mcp__vecscope__audit_embeddings mcp__vecscope__describe_findings mcp__vecscope__compare_embeddings Bash(python:*) Bash(python3:*) Read Write"
```

- `mcp.json`은 직접 작성하며 `vecscope-mcp --allow-dir <실행 디렉터리>`를 등록합니다.
- 에이전트는 vecscope 도구 호출과 Python 실행만 할 수 있었습니다.
- Claude Code 2.1, 모델 claude-opus-5-5.

## 에이전트가 한 일 (첫 번째 실행, 8턴, 58초)

1. **파일 확인**: 배열의 형태와 자료형, `qrels.tsv`의 앞부분을 봤습니다.
2. **감사**: `audit_embeddings(corpus.npy, queries_path, qrels_path, ranked_path, sample=0,
   verbose=false)`를 호출했습니다. 에이전트가 읽은 요약 줄은 다음과 같습니다.

   ```text
   verdict=fail n=5783 dim=384 k=10; collapse=0.208 fail; norms=0.192 warn;
   metric_mismatch=0.0914 fail; retrieval nDCG@10=0.689 exact, 0.100 ranked;
   plan step 1 of 2: input_integrity (collapse): ... deduplicate before changing anything else
   ```

3. **데이터에서 진단 내용 확인**: Python으로 다음을 확인했습니다.
   - 2600\~5182행(2,581행)은 단위 길이가 아니고, 질의는 단위 길이입니다.
   - 5183\~5782행은 1000\~1599행과 바이트 단위로 같은 복제본입니다.
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

6. **보고**: 결함, 수정 내용, 전후 숫자를 보고했습니다. 기준선 비교가 not comparable로
   표시된 이유도 설명했습니다. 행 수가 바뀌었기 때문이고, 두 실행이 같은 정답 데이터를 썼으므로
   검색 품질 숫자는 그대로 비교할 수 있다는 것입니다.

계획은 중복 제거(`input_integrity`)를 먼저, 거리 척도를 그다음으로 두었습니다. 에이전트는
확인 결과 서로 독립된 결함 두 개임을 보고 한 번에 둘 다 적용했고, 재감사로 둘 다 판정했습니다.

## 전후 비교

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

## 두 번째 실행 (11턴, 75초)

데이터를 새로 복사해 같은 요청으로 다시 실행했습니다. 같은 결함 두 개를 찾아 같은 수정을
했고, 재감사 숫자도 같았습니다. 확인 코드를 스크립트로 먼저 작성하느라 턴이 더 걸렸습니다.
`prod_topk.npy`를 다시 계산하기 전에 faiss `IndexFlatIP`로 원본을 재현할 수 있는지도
확인했습니다.

## 이 사례가 보여 주는 것과 보여 주지 않는 것

- 도구, 계획, `baseline` 재감사만으로 에이전트가 원인을 듣지 않고도 증상("같은 논문이 두 번,
  질의 품질 저하")에서 검증된 수정까지 갔습니다.
- 에이전트는 무엇이든 바꾸기 전에 각 진단을 데이터에서 확인했고, 요청받지 않았는데도 qrels가
  계속 맞도록 했습니다.
- 모델 하나가 과제 하나를 두 번 실행한 결과이고, 결함도 vecscope 진단이 잡도록 만들어진
  종류입니다. 모든 에이전트가 계획을 따른다거나, 진단이 재지 않는 결함(예: 질의 접두어 누락,
  [검증 결과](verification.md) 참고)에서 어떻게 되는지는 보여 주지 않습니다.
- `tests/test_agent_workflow.py`는 모델 없이 MCP stdio 프로토콜로 비슷한 진단, 변경, 재검증
  흐름을 테스트할 때마다 실행합니다.
