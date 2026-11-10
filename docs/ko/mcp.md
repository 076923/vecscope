[English](../mcp.md) · **한국어**

# 에이전트와 함께 사용

`vecscope`에는 읽기 전용 도구 네 개(`audit_embeddings`, `sweep_index`, `compare_embeddings`,
`describe_findings`)를 제공하는 MCP 서버가 들어 있습니다. MCP를 지원하는 클라이언트라면
어디에든 등록할 수 있습니다.

```bash
pip install vecscope
claude mcp add vecscope -- vecscope-mcp --allow-dir /data/embeddings     # Claude Code
claude mcp add vecscope -- uvx vecscope mcp --allow-dir /data/embeddings # 설치하지 않고 실행
```

`vecscope mcp`와 `vecscope-mcp`는 옵션까지 같은 서버입니다.
[MCP 레지스트리(MCP Registry)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.076923/vecscope)에
`io.github.076923/vecscope`로 등록되어 있습니다.

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

VS Code는 작업 공간의 `.vscode/mcp.json`을 읽으며, `mcpServers` 대신 `servers` 아래에 등록합니다.
VS Code는 셸의 PATH를 그대로 보지 못할 수 있으므로 `vecscope-mcp`의 전체 경로를 쓰십시오(Windows는
`where vecscope-mcp`, 그 밖의 환경은 `which vecscope-mcp`로 확인합니다).

```json
{
  "servers": {
    "vecscope": {
      "type": "stdio",
      "command": "/path/to/vecscope-mcp",
      "args": ["--allow-dir", "${workspaceFolder}/data"]
    }
  }
}
```

항목 위의 **Start** 링크나 `MCP: List Servers`로 서버를 시작하고, Chat 창의 Agent 모드에서 도구를
사용합니다.

등록한 뒤에는 에이전트에게 다음과 같이 요청합니다.

```text
검색기를 튜닝하기 전에 data/chunks.npy 임베딩을 감사하고, 먼저 조치할 항목을 알려줘.
```

## 도구와 응답

- `audit_embeddings`는 CLI와 같은 입력으로 감사를 실행합니다.
- `sweep_index`는 [인덱스 스윕](usage.md#인덱스-스윕)을 실행합니다.
- `compare_embeddings`는 [두 임베딩 비교](usage.md#두-임베딩-비교)를 실행합니다.
- `describe_findings`는 모든 항목의 임곗값, 원인, 조치를 나열합니다.

응답은 한 줄짜리 `summary`와 `audit_id`로 시작합니다. summary에는 ok가 아닌 항목, 검색 nDCG,
계획의 첫 단계, 재감사라면 기준선과의 비교가 들어갑니다.

변경 뒤 다시 감사할 때 그 id를 `baseline`으로 넘기십시오. 응답에 CLI의 `--baseline`과 같은
`baseline_diff`(판정과 값의 변화, `regressions`, `improvements`, 검색 품질 변화)가 붙으므로,
에이전트가 이전 응답을 파일로 저장할 필요가 없습니다. 서버는 최근 감사 32개를 메모리에 두며,
저장한 응답이나 `--json` 파일도 받습니다.

`verbose=false`를 주면 `ok` 항목의 세부 내용, 설명, 조치를 생략합니다. MCP를 지원하지 않는
에이전트라면 CLI를 `--json`으로 실행해 같은 결과를 읽으면 됩니다.

[`.agents/skills/vecscope/SKILL.md`](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md)의 에이전트
스킬에는 감사를 실행할 시점과 판정별 조치(벡터 중심화(centering) 또는 모델 교체, 중복 제거,
재순위화(re-ranking), 인덱스 재구축)가 정리되어 있습니다. 같은 절차가 서버 instructions에도
들어 있어 MCP 연결만 있는 에이전트도 동일하게 동작합니다.

## 서버가 읽을 수 있는 범위

도구는 읽기 전용이지만 에이전트가 지정한 경로라면 무엇이든 읽으므로, 서버가 그 범위를
제한합니다. 허용된 디렉터리 밖의 경로는(`..`과 심볼릭 링크를 풀어 확인한 뒤) 파일을 열기 전에
거부됩니다.

- 기본값은 서버를 시작한 디렉터리와 그 아래뿐입니다.
- `--allow-dir DIR`을 주면 위 등록 예시처럼 그 디렉터리를 대신 씁니다. 여러 번 줄 수 있고,
  환경 변수 `VECSCOPE_ALLOWED_DIRS`(Unix는 `:`, Windows는 `;`로 구분)도 같은 역할을 합니다.
- `--allow-any`(또는 `VECSCOPE_ALLOW_ANY=1`)는 제한을 풀어 서버 프로세스가 읽을 수 있는 모든
  파일에 접근하게 합니다. 자신의 컴퓨터에서만 쓰십시오.
- 네트워크 벡터 저장소는 여기에 더해 `--allow-network`가 있어야 합니다. 읽기 전용이라고 해서
  접속이 무해한 것은 아니므로, 서버의 네트워크 접근이 의도한 범위일 때만 켜십시오.

파일은 연 뒤에 확인한 핸들 하나로 읽습니다. 일반 파일이어야 하고, 경로가 가리키는 허용 디렉터리
안의 파일과 같은 파일이어야 합니다. 확인과 읽기 사이에 바꿔치기한 심볼릭 링크, FIFO, 디렉터리는
거부합니다.

로컬 데이터베이스 디렉터리(`qdrant+local://` 등)는 클라이언트가 파일을 직접 열기 때문에 경로만
확인합니다.

요청 하나가 서버 메모리를 고갈시키지 못하게 합니다.

- `--max-read-mb`(기본값 4096, `VECSCOPE_MAX_READ_MB`, 0이면 제한 없음)보다 큰 파일은 읽기 전에
  거부합니다.
- 저장소도 벡터가 같은 크기를 넘으면 읽기를 멈춥니다.
- `--max-rows`와 `--max-dim`(`VECSCOPE_MAX_ROWS`, `VECSCOPE_MAX_DIM`, 기본값 0이면 제한 없음)은
  바이트 크기와 상관없이 행이나 열이 그보다 많은 행렬을 거부합니다. 질의 파일도 같은 기준으로
  확인하고, 저장소는 `--max-rows`에서 읽기를 멈춥니다.
- 파일은 확인한 핸들에서 바로 파싱하고, 먼저 메모리에 통째로 복사하지 않습니다. 500,000 x 384
  `.npy` 파일(0.72 GiB)을 MCP로 읽을 때 최대 메모리는 CLI와 같은 0.79 GiB입니다.
- 읽기나 감사 중에 `MemoryError`가 나면 호출을 조용히 끝내지 않고, 빠져나갈 방법(표본, 더 작은
  입력, 위의 상한)을 적은 도구 오류로 에이전트에게 돌려줍니다.

병렬 호출과 비용이 큰 설정에도 한도가 있습니다.

- 서버는 도구 호출마다 스레드를 따로 씁니다. 감사, 스윕, 비교는 `--max-concurrent` 슬롯(기본값 1,
  `VECSCOPE_MAX_CONCURRENT`) 하나를 차지하고, 나머지 호출은 기다립니다. 에이전트가 감사를 세 번
  동시에 호출해도 메모리에는 코퍼스 하나만 올라갑니다.
- 호출의 비용을 몇 배로 키우는 설정은 호출마다 상한이 있습니다. `k`와 `clusters`는 1,000,
  `sample_repeats`와 `timing_repeats`는 10, `n_queries`는 100,000, `latency_queries`는 10,000입니다.
  CLI에는 상한이 없습니다.

감사, 스윕, 비교는 호출마다 서버의 stderr에 한 줄을 남기고, MCP 클라이언트는 이를 로그에
보관합니다. 도구 이름, 읽은 파일이나 저장소(인증정보는 가림), 판정이나 오류, 슬롯 대기를 포함한
걸린 시간이 들어갑니다.

```text
vecscope-mcp: audit_embeddings path=/data/run/corpus.npy qrels_path=/data/run/qrels.tsv: fail in 2.1 s
```

`baseline`이 가리키는 최근 감사 기록은 락으로 보호합니다. stdio 서버는 자신을 시작한 클라이언트
하나만 상대하므로, 에이전트를 여럿 병렬로 돌리면 각자 서버를 띄웁니다.

## 테스트한 클라이언트

진단, 변경, 재검증으로 이어지는 흐름은 MCP stdio 프로토콜 위에서 테스트하고
(`tests/test_agent_workflow.py`), Claude Code headless 모드(`claude -p`에 위 서버 등록)로도 끝까지
돌려 봤습니다.

Claude Code는 공통 오프셋이 있는 코퍼스를 감사한 뒤 계획대로 정규화와 중심화를 하고 다시
감사했으며, 요청받지 않은 나머지 단계는 건드리지 않고 멈췄습니다.

VS Code의 GitHub Copilot(Agent 모드, 위의 `.vscode/mcp.json`)으로는 `examples/make_examples.py`가
만드는 예시 행렬을 대상으로 직접 시험했습니다.

- `healthy.npy`, `anisotropic.npy`, `collapsed.npy`는 각각 ok, warn, fail로 판정됐고 수치는
  `vecscope audit`과 같았으며, 에이전트는 `plan`의 첫 단계를 권했습니다.
- `anisotropic.npy`를 중심화하고 다시 정규화한 뒤, 에이전트는 첫 `audit_id`를 `baseline`으로
  넘겨 다시 감사했고 anisotropy와 antihubs를 개선 항목으로 보고했습니다.
- `--allow-dir` 밖의 파일은 서버가 거부했고, 에이전트는 거부를 우회하려 하지 않았습니다.
- Copilot은 이 정도 크기의 응답(`verbose`를 켠 상태로 약 13,000~15,000자)을 파일로 옮겨 나눠
  읽습니다. 감사마다 도구 호출이 몇 번 더 들지만 답은 달라지지 않았습니다.

Cursor와 Codex는 같은 프로토콜을 쓰지만 직접 돌려 보지는 않았습니다.

[docs/ko/agent_case.md](agent_case.md)에는 실제 SciFact 임베딩으로 만든 사례 두 개를 각각 두 번
실행한 기록이 있습니다.

- **적재 결함 두 가지**: 한 배치를 두 번 적재했고, 정규화를 빠뜨린 작업자가 있었습니다.
  에이전트는 증상만 듣고 두 결함을 찾아 코퍼스를 고쳤고, 첫 `audit_id`를 기준으로 다시
  감사했습니다. 운영 nDCG@10은 0.100에서 0.713이 됐습니다.
- **nprobe=1로 검색하는 IVF 인덱스**: 감사 결과 임베딩은 건강했고, 운영 랭킹은 정확한 검색보다
  nDCG가 0.29 낮았습니다. 에이전트는 인덱스를 스윕해 nprobe를 골랐고, 운영 nDCG@10은 0.422에서
  0.700과 0.711이 됐습니다.

## 서버가 연결되지 않을 때

"vecscope-mcp not found"가 나오면 대개 설치했던 Python 환경이 새로 만들어진 경우입니다.
환경을 다시 구성하는 호스팅 노트북이나 클러스터에서 흔히 일어납니다.

클라이언트가 쓰는 인터프리터에서 `pip show vecscope`로 확인하고 그 환경에 다시 설치하십시오.
그다음 그 인터프리터를 통해 서버를 등록하면 오류 메시지가 어느 환경의 문제인지 바로 드러납니다.

```json
{ "mcpServers": { "vecscope": { "command": "/path/to/env/bin/python", "args": ["-m", "vecscope.mcp_server"] } } }
```
