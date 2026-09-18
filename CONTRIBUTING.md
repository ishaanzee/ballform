# Contributing to Ballform

Thanks for improving Ballform. Contributions to mechanics metrics, tracking reliability, camera-angle handling, tests, documentation, and privacy are welcome.

## Development setup

```bash
uv sync --extra dev --python 3.12
uv run pytest
uv run uvicorn app.main:app --reload
```

Use Python 3.11 or 3.12. Keep changes focused, add tests for scoring or networking behavior, and run the complete test suite before opening a pull request.

## Privacy and test data

Never commit real user footage, generated job directories, pairing URLs, access tokens, downloaded model weights, or biometric/pose exports. Use synthetic fixtures or footage for which you have explicit redistribution permission. A pull request containing personal footage will not be accepted.

## Licensing

By submitting a contribution, you agree that it may be distributed under the repository's AGPL-3.0-only license. New dependencies must have licenses compatible with AGPL-3.0 and should be documented in `NOTICE.md` when they materially affect distribution.

## Pull requests

- Explain the user-visible behavior and tradeoffs.
- Include tests for new deterministic logic.
- Separate model or threshold changes from unrelated interface changes.
- Do not silently turn local processing into a cloud dependency.
