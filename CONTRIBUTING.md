# Contributing to vecscope

Thanks for helping agents see embedding spaces. This file explains how to work on the code
and what a good change looks like. Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
git clone https://github.com/076923/vecscope
cd vecscope
python -m pip install -e ".[dev,index]"
pytest -q
ruff check src tests examples
ruff format --check src tests examples
```

`python examples/make_examples.py` writes three small matrices into `examples/` so you can try
`vecscope audit examples/anisotropic.npy` right away.

`pre-commit install` runs `ruff --fix`, `ruff format` and the whitespace, YAML and TOML checks from
`.pre-commit-config.yaml` on every commit (`pip install pre-commit` first). `.editorconfig`
sets the indentation and line endings for editors that read it.

## Documentation site

<https://076923.github.io/vecscope/> is built by [Zensical](https://zensical.org) from
`README.md`, `CHANGELOG.md`, `docs/` and `docs/ko/`, using the navigation in `zensical.toml`.
`.github/workflows/docs.yml` deploys it on every push to `main`. Build it locally before
changing a document:

```bash
python -m pip install --group docs
zensical build --strict    # fails on a broken link or anchor
zensical serve             # preview at http://127.0.0.1:8000
```

Link to repository files that are not in `docs/` (the skill, examples, `LICENSE`) by their
GitHub URL, as the READMEs do, so the link works both on GitHub and on the site.

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
3. Add a test with a synthetic case that fails and a healthy case that stays `ok`. When a
   change moves the value of an existing finding on the example matrices, regenerate
   `tests/golden/` with `python examples/make_examples.py` and say so in the PR.
4. Describe the finding in the table in `README.md` and `docs/ko/README.md`, and the
   action in `.agents/skills/vecscope/SKILL.md`.

## Pull requests

Keep PRs focused on one change. Fill in the template, especially the evidence section.
The diagnostics compute with NumPy only; `mcp` is a dependency for the server alone, and
optional extras are fine for loaders.

## Releasing

Releases are published to PyPI by `.github/workflows/release.yml` through PyPI trusted
publishing; no token is stored anywhere. PyPI never lets a version number be reused, so bump
the version before tagging.

1. Bump `__version__` in `src/vecscope/__init__.py` and both `version` fields in `server.json`
   (a test checks that they match).
2. Move the entries under `Unreleased` in `CHANGELOG.md` into a new `## [x.y.z] - YYYY-MM-DD`
   section.
3. Commit and push to `main`, and wait for CI to pass.
4. Tag and push the tag. The workflow builds and publishes to PyPI, then creates the GitHub
   release from the changelog section with the built files attached, and moves the major tag
   (`v0`) that `uses: 076923/vecscope@v0` reads:

   ```bash
   git tag -a vx.y.z -m "vx.y.z"
   git push origin vx.y.z
   ```

5. On the first release with `action.yml`, edit the GitHub release and tick "Publish this
   Action to the GitHub Marketplace" (needs two-factor authentication). Check on the next
   release that the Marketplace listing shows the new version.
6. Check `pip install vecscope==x.y.z` in a clean environment.
7. Publish the new version to the MCP Registry once PyPI has it: `mcp-publisher login github`,
   then `mcp-publisher publish` from the repository root.
