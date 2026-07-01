# Contributing

This is a research-code repository. Contributions should prioritize reproducibility, clear documentation, and small, reviewable changes.

## Suggested Workflow

1. Create a feature branch.
2. Keep generated outputs out of Git unless they are intentionally curated artifacts.
3. Run basic checks before proposing changes.
4. Document new scripts, command-line arguments, and required input files.

## Basic Checks

```bash
python -m compileall .
git status
git diff --stat
```

## Documentation Expectations

When adding a new experiment or metric, include:

- What the script does.
- Required inputs.
- How to run it.
- Where outputs are written.
- Whether outputs should be committed.
