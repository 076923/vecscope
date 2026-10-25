# Contributing to vecscope

Thanks for helping agents see embedding spaces. This file explains how to work on the code
and what a good change looks like.

## Setup

```bash
git clone https://github.com/076923/vecscope
cd vecscope
python -m pip install -e ".[dev,mcp,index]"
pytest -q
ruff check src tests examples
```

`python examples/make_examples.py` writes three small matrices into `examples/` so you can try
`vecscope audit examples/anisotropic.npy` right away.

## Scope

`vecscope` produces **deterministic evidence** about an embedding matrix or a vector index for
an agent to act on. A change fits when it:

- makes a retrieval failure visible that an agent could not see before, or
- makes an existing finding more accurate, cheaper, or easier to act on.

It does not fit when it needs a learned component, a network call, or a model download at
audit time, or when it makes two runs on the same input produce different output.

## Adding a finding

1. Put the measurement in `src/vecscope/diagnostics/<name>.py` as a pure function returning a
   `dict` of plain Python numbers. Reuse the shared exact k-NN from `audit` when you can.
2. Register the threshold pair in `THRESHOLDS` and build the `Finding` in
   `src/vecscope/evidence.py`. The explanation string must state the number, what it means, and
   what "healthy" looks like.
3. Add a test with a synthetic case that fails and a healthy case that stays `ok`.
4. Describe the finding in the table in `README.md` and the action in
   `.agents/skills/vecscope/SKILL.md`.

## Pull requests

Keep PRs focused on one change. Fill in the template, especially the evidence section.
Dependencies stay at NumPy for the core; optional extras are fine for loaders and the MCP server.
