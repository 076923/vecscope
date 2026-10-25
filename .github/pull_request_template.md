## What this changes

<!-- One paragraph. If it adds or changes a finding, say what symptom it makes visible. -->

## Evidence

<!-- For a new or changed diagnostic: the synthetic case it detects, the healthy case it leaves
     at `ok`, and the reference (paper or doc) for the measurement, if any. -->

## Checklist

- [ ] `pytest -q` passes
- [ ] `ruff check src tests examples` passes
- [ ] Output is still deterministic (fixed seeds, no learned components)
- [ ] README / SKILL.md updated if the CLI, MCP tool or verdicts changed
- [ ] CHANGELOG.md has an entry under Unreleased
