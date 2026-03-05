# Repository Guidelines

## Project Structure & Module Organization
- `jobscout/`: core pipeline logic (resume parsing, job fetching/parsing, filtering, scoring, scheduling, email delivery).
- `backend/`: FastAPI API layer, adapters, storage, security middleware, metrics, and API models.
- `tests/`: primary pytest suite for scoring, sources, adapters, semantic matching, and backend behavior.
- Top-level `test_*.py` files: broader integration/stateless flow checks kept at repo root.
- `scripts/`: developer utilities (for example `preload_semantic_model.py`, `lint.sh`).
- `config.example.yaml` and `resume.example.txt`: safe templates for local setup.

## Build, Test, and Development Commands
- Install dependencies:
  - `pip install -r requirements.txt`
  - `pip install -r backend/requirements.txt`
- Run CLI flow once: `python jobscout_cli.py --config config.yaml`
- Run scheduler mode: `python jobscout_cli.py --schedule`
- Run API locally: `python -m uvicorn backend.main:app --reload --port 8000`
- Run tests: `pytest tests/ -v`
- Run coverage report: `pytest tests/ --cov=jobscout --cov-report=html`
- Run lightweight lint/sanity checks: `bash scripts/lint.sh`

## Coding Style & Naming Conventions
- Follow Python conventions: 4-space indentation, `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep module/class/function docstrings concise and functional.
- Prefer explicit type hints on public functions and data boundaries.
- Use `logging` (not `print`) for runtime diagnostics; follow existing structured log style.
- Keep files focused: orchestration in `jobscout/main.py`, API concerns in `backend/main.py`, shared domain logic in `jobscout/`.

## Testing Guidelines
- Framework: `pytest` with fixtures and parametrized assertions where useful.
- Naming: `test_<feature>.py`, `Test<Feature>` classes, and descriptive test names such as `test_role_alignment_affects_apply_readiness`.
- Add or update tests for every behavior change in scoring, filtering, parsing, or API responses.
- No strict coverage gate is configured; maintain or improve coverage in touched modules.

## Commit & Pull Request Guidelines
- Follow the observed Conventional Commit style: `feat:`, `fix:`, `refactor:`, `build:` with optional scope (example: `feat(job_parser): ...`).
- Keep commits atomic and message subjects imperative and specific.
- PRs should include:
  - change summary and rationale,
  - linked issue (if available),
  - commands run (tests/lint) and outcomes,
  - sample request/response payloads for API-facing changes.

## Security & Configuration Tips
- Do not commit secrets or personal artifacts (`config.yaml`, real resume files, API keys).
- Use environment variables for sensitive settings (`OPENAI_API_KEY`, `API_KEY`, `METRICS_API_KEY`, SMTP credentials).
- Keep `CORS_ORIGINS` restricted to trusted domains in non-local environments.
