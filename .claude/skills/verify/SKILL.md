---
name: verify
description: Run full quality checks and test suite. Use before submitting PRs or after significant changes.
---

Run the project's quality checks and test suite to verify changes are correct.

## Steps

1. **Quality check** (formatting + linting):
   ```bash
   make quality
   ```
   If this fails, run `make style` to auto-fix formatting, then re-run `make quality`.

2. **Test suite**:
   ```bash
   make test
   ```
   If specific tests fail, re-run with verbose output:
   ```bash
   pytest -ra tests/path/to/test_file.py -k 'test_name' -v
   ```

3. **Report results** — summarize pass/fail status. If anything failed, diagnose and fix before marking done.

## Notes

- Quality checks enforce SPDX copyright headers, black formatting, isort import ordering, and flake8 linting.
- GPU tests will be skipped if no accelerator is available — this is expected in CPU-only environments.
- Distributed tests (`@torchrun`) spawn subprocesses and may need a multi-GPU setup.
